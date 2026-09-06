"""Live vendor ModelClient + budget guard for the P1/P2 codegen ablation (WS-C).

Turns the blocked codegen path into a runnable one WITHOUT reinventing vendor plumbing: like the
Scientist's ``PhaseDModelClient``, this wraps the Librarian's battle-tested backends
(``agents.librarian.pipeline.real_client.make_backend`` — lazy-imported, so importing this module
and injecting a fake backend in tests never touches an SDK).

Two phases (the codegen ``ModelClient`` protocol is ``name`` + ``generate(prompt, *, seed) -> str``):

* ``dev``      -> ``librarian.model_stack.phase_d`` (Gemini 3.1-flash-lite + Mistral-small, FREE).
                  NON-reportable; the plumbing smoke test.
* ``reported`` -> ``librarian.model_stack.phase_f`` (Claude Sonnet 4.6 + Gemini 3.5-flash).
                  The reported run, gated by the contract's $30/16k sub-cap (§3/§9).

BUDGET (contract §3/§9, APPROVED 2026-08-06): the per-call output ceiling (16,000 tokens) and
the cumulative sub-cap ($30 of the Phase-F cap) are ENFORCED here, not merely recorded. ``BudgetGuard``
threads the per-call ceiling into every vendor call, LOGS each call's token usage + estimated spend,
and RAISES ``BudgetExceededError`` the moment cumulative spend crosses the cap. Because the ablation
is cache-first (``runner.generate_once``), a re-run is zero-spend.

The USD estimate uses pre-registered per-model prices (``p1_codegen.budget.prices_usd_per_1m``): the
$30 cap is the binding safety rail on a ~10-call, ~$1-3 experiment, and the prices are conservative
estimates confirmed at the reported-run approval. The dev (free) pair is priced 0 — its
spend is still logged (tokens), it just never trips the cap.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_THRESHOLDS = _REPO_ROOT / "docs" / "thresholds.yaml"


class BudgetExceededError(RuntimeError):
    """Raised when a call would push cumulative estimated spend past the sub-cap, or when a
    model has no registered price (a silent unpriced call could not be capped)."""


@dataclass(frozen=True)
class SpendRecord:
    model_id: str
    prompt_tokens: int
    completion_tokens: int
    usd_estimate: float


class BudgetGuard:
    """Enforces the contract §3/§9 sub-cap: a per-call output-token ceiling and a cumulative
    USD cap, with a per-call spend log. Shared across the run's clients (one guard, one cap)."""

    def __init__(self, *, usd_cap: float, per_call_output_token_cap: int,
                 prices_usd_per_1m: dict[str, dict]):
        self.usd_cap = float(usd_cap)
        self.per_call_output_token_cap = int(per_call_output_token_cap)
        self._prices = dict(prices_usd_per_1m)
        self.records: list[SpendRecord] = []

    @property
    def spent_usd(self) -> float:
        return round(sum(r.usd_estimate for r in self.records), 6)

    def output_cap_for(self, configured: int) -> int:
        """The per-call output-token ceiling actually used: the smaller of the client's
        configured value and the contract cap (so a call can never exceed the cap)."""
        return min(int(configured), self.per_call_output_token_cap)

    def _price(self, model_id: str) -> dict:
        try:
            return self._prices[model_id]
        except KeyError as exc:
            raise BudgetExceededError(
                f"no price registered for model {model_id!r} in "
                "p1_codegen.budget.prices_usd_per_1m — cannot bound spend, refusing (fail loud)"
            ) from exc

    def estimate_usd(self, model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
        p = self._price(model_id)
        return round(
            (int(prompt_tokens or 0) / 1e6) * float(p["input"])
            + (int(completion_tokens or 0) / 1e6) * float(p["output"]),
            6,
        )

    def charge(self, model_id: str, prompt_tokens: int, completion_tokens: int) -> SpendRecord:
        """Record one call's spend and HALT (raise) if cumulative crosses the cap. The record is
        appended before the check so ``spent_usd`` stays truthful about what was actually spent."""
        usd = self.estimate_usd(model_id, prompt_tokens, completion_tokens)
        rec = SpendRecord(model_id, int(prompt_tokens or 0), int(completion_tokens or 0), usd)
        self.records.append(rec)
        if self.spent_usd > self.usd_cap + 1e-9:
            raise BudgetExceededError(
                f"cumulative estimated spend ${self.spent_usd:.4f} crossed the "
                f"${self.usd_cap:.2f} sub-cap (contract §3/§9) after a {model_id} call — halting")
        return rec

    def summary(self) -> dict:
        return {
            "usd_cap": self.usd_cap,
            "per_call_output_token_cap": self.per_call_output_token_cap,
            "spent_usd_estimate": self.spent_usd,
            "calls": len(self.records),
            "records": [asdict(r) for r in self.records],
        }


class LiveCodegenClient:
    """A live codegen ``ModelClient`` (``name`` + ``generate(prompt, *, seed) -> str``).

    Wraps ONE Librarian vendor backend. Pass ``backend`` to inject a fake (tests never import the
    Librarian or an SDK); otherwise the backend is built lazily from (vendor, model_id, api_key).
    The per-call output ceiling is the smaller of ``max_output_tokens`` and the budget cap; each
    call's token usage is charged to the shared ``budget``. ``last_model_version`` records the
    vendor's returned SKU for the reported-run match (contract §3)."""

    def __init__(self, *, vendor, model_id, max_output_tokens, api_key=None, budget=None,
                 temperature=0.0, min_interval_s=0.0, max_retries=6, backoff_base=2.0,
                 backoff_cap=30.0, backend=None):
        self.name = model_id
        self.vendor = vendor
        self._budget = budget
        cap = int(max_output_tokens)
        self.max_output_tokens = budget.output_cap_for(cap) if budget is not None else cap
        if backend is not None:
            self._backend = backend
        else:
            from agents.librarian.pipeline.real_client import make_backend  # lazy — SDK on build
            # json_mode=False: codegen needs FREE TEXT (a ```python-fenced script per the output
            # contract), NOT the JSON the Librarian/Scientist forcing would return.
            self._backend = make_backend(vendor, model_id, api_key, temperature=temperature,
                                         json_mode=False)
        self._min_interval_s = float(min_interval_s)
        self._max_retries = int(max_retries)
        self._backoff_base = float(backoff_base)
        self._backoff_cap = float(backoff_cap)
        self._last_call_ts = 0.0
        self.last_model_version: str | None = None
        # Mechanical operational counters (mirrors RealModelClient / PhaseDModelClient).
        self.model_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_retries = 0

    def generate(self, prompt: str, *, seed: int) -> str:
        """One structured vendor call, paced + retried, capped + budget-charged. ``seed`` rides
        through for cache provenance (reproducibility is by cache-replay, not vendor determinism —
        codegen is temperature 0 anyway)."""
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            self._pace()
            try:
                text, version = self._backend.generate(prompt, self.max_output_tokens)
                self._last_call_ts = time.monotonic()
                self.model_calls += 1
                self.total_retries += attempt
                self.last_model_version = version
                u = getattr(self._backend, "last_usage", None) or {}
                pt = int(u.get("prompt") or 0)
                ct = int(u.get("completion") or 0)
                self.total_prompt_tokens += pt
                self.total_completion_tokens += ct
                if self._budget is not None:
                    self._budget.charge(self.name, pt, ct)  # may raise BudgetExceededError
                return text or ""
            except BudgetExceededError:
                raise                                       # never retry past the cap
            except Exception as exc:
                from agents.librarian.pipeline.real_client import (  # lazy — only on a live error
                    _is_non_retryable,
                    _retry_after_seconds,
                )
                last_exc = exc
                if _is_non_retryable(exc):
                    raise
                delay = _retry_after_seconds(exc)
                if delay is None:
                    delay = min(self._backoff_cap, self._backoff_base ** attempt)
                time.sleep(delay)
        raise RuntimeError(
            f"codegen client {self.name!r} exhausted {self._max_retries} retries") from last_exc

    def operational_usage(self) -> dict:
        return {
            "model_calls": self.model_calls,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "retries": self.total_retries,
            "last_model_version": self.last_model_version,
        }

    def _pace(self) -> None:
        if self._min_interval_s <= 0:
            return
        elapsed = time.monotonic() - self._last_call_ts
        if elapsed < self._min_interval_s:
            time.sleep(self._min_interval_s - elapsed)


def load_budget(thresholds_path: Path | None = None) -> BudgetGuard:
    """Build the run's ``BudgetGuard`` from ``p1_codegen.budget`` (fail-loud — a silent default
    would run generation against an unchosen cap)."""
    doc = yaml.safe_load((thresholds_path or _THRESHOLDS).read_text(encoding="utf-8"))
    try:
        block = doc["p1_codegen"]["budget"]
        return BudgetGuard(
            usd_cap=block["usd_cap"],
            per_call_output_token_cap=block["per_call_output_token_cap"],
            prices_usd_per_1m=block["prices_usd_per_1m"],
        )
    except (KeyError, TypeError) as exc:
        raise KeyError(
            "thresholds.yaml has no complete p1_codegen.budget block "
            "(usd_cap / per_call_output_token_cap / prices_usd_per_1m)") from exc


def build_codegen_factory(*, budget: BudgetGuard, temperature: float = 0.0):
    """A ``client_factory`` (``model_spec -> LiveCodegenClient``) closing over the shared budget.
    Fail-loud when a model's API key is unset — never a silent stub."""
    def factory(model: dict) -> LiveCodegenClient:
        env = model.get("api_key_env", "")
        api_key = os.environ.get(env)
        if not api_key:
            raise RuntimeError(
                f"codegen model {model.get('model_id')!r} needs its API key in ${env} (not set). "
                f"Populate .env / export it and `pip install` the vendor SDK.")
        return LiveCodegenClient(
            vendor=model["vendor"], model_id=model["model_id"], api_key=api_key,
            max_output_tokens=budget.per_call_output_token_cap, budget=budget,
            temperature=temperature,
            min_interval_s=float(model.get("min_interval_s", 0.0)))
    return factory
