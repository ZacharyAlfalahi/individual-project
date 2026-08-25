"""Rung-3 phase_d model wiring — the real free-dev vendor pair (D4 phase_d: Gemini
3.1-flash-lite + Mistral-small). Turns the deferred rung 3 (llm_source.DeferredModelClient) into a
buildable LIVE client.

REUSE, not reinvent. The vendor SDK plumbing (lazy imports, JSON decode, non-retryable
classification, server retry-after parsing) is the Librarian's battle-tested code
(`agents/librarian/pipeline/real_client`). The two agents call the IDENTICAL phase_d SKUs, and
that module is SDK-free to import (its vendor imports are lazy). Those reuses are LAZY-imported
here (inside the methods that need them), so importing this module — and injecting a fake backend
in tests — never touches the Librarian or any SDK. A future refactor could lift the shared vendor
plumbing into `shared/llm/`; noted as minor tech debt, not blocking.

NON-REPORTABLE by design. phase_d is the FREE DEV pair (D4/D33). Reportable generative figures
require phase_f (Claude Sonnet 4.6 + Gemini 3.5-flash) + SKU/cost authorization — a SEPARATE, still-open
gate (`librarian.model_stack.phase_f`). Wiring phase_d does NOT make the generative arm reportable.

LIVE-RUN PREREQUISITES (fail loud). A live call needs (i) the vendor SDK installed (`google-genai`,
`mistralai`) and (ii) the API key in the env named by `api_key_env` (GEMINI_API_KEY / MISTRAL_API_KEY).
Absent either, `build_phase_d_clients` raises an actionable error rather than silently degrading.

REPRODUCIBILITY is by cache-replay, not vendor determinism. The seed-keyed ResponseCache (R5)
freezes each (prompt, model, seed) response, so the generation temperature can be > 0 to give the
k-seed generation-quality arm genuine seed-to-seed diversity while every response stays on disk.
`seed` is therefore accepted (for the protocol + cache provenance) but not threaded to the vendor —
a documented follow-up; the cache is the reproducibility mechanism, not a vendor seed.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import yaml

_THRESHOLDS_PATH = Path(__file__).resolve().parents[3] / "docs" / "thresholds.yaml"


class PhaseDModelClient:
    """A live `ModelClient` (Scientist protocol: `name` + `generate(prompt, *, seed) -> str`).

    Wraps ONE Librarian vendor backend. Pass `backend` to inject a fake (tests never import the
    Librarian or an SDK); otherwise the backend is built lazily from (vendor, model_id, api_key)."""

    def __init__(self, *, vendor, model_id, max_output_tokens, api_key=None, temperature=0.0,
                 min_interval_s=0.0, max_retries=6, backoff_base=2.0, backoff_cap=30.0,
                 backend=None):
        self.name = model_id
        self.vendor = vendor
        self.max_output_tokens = int(max_output_tokens)
        if backend is not None:
            self._backend = backend
        else:
            from agents.librarian.pipeline.real_client import make_backend  # lazy — SDK on build
            self._backend = make_backend(vendor, model_id, api_key, temperature=temperature)
        self._min_interval_s = float(min_interval_s)
        self._max_retries = int(max_retries)
        self._backoff_base = float(backoff_base)
        self._backoff_cap = float(backoff_cap)
        self._last_call_ts = 0.0
        # WS-8 (§4.7) mechanical operational counters (mirrors RealModelClient).
        self.model_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_retries = 0

    def generate(self, prompt: str, *, seed: int) -> str:
        """One structured vendor call, paced + retried. Returns raw text (the generation loop in
        sources.py counts invalid/duplicate; it never regenerates). `seed` rides through for cache
        provenance (R5) — see the module note on reproducibility-by-cache."""
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            self._pace()
            try:
                text, _version = self._backend.generate(prompt, self.max_output_tokens)
                self._last_call_ts = time.monotonic()
                self.model_calls += 1
                self.total_retries += attempt
                u = getattr(self._backend, "last_usage", None)
                if u:
                    if u.get("prompt") is not None:
                        self.total_prompt_tokens += u["prompt"]
                    if u.get("completion") is not None:
                        self.total_completion_tokens += u["completion"]
                return text or ""
            except Exception as exc:  # classify: re-raise the un-retryable, back off on the rest
                # lazy import: only reached on a live vendor error, never at module import time.
                from agents.librarian.pipeline.real_client import (
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
            f"phase_d client {self.name!r} exhausted {self._max_retries} retries") from last_exc

    def operational_usage(self) -> dict:
        """Mechanical token/call/retry totals (WS-8 / §4.7), mirroring RealModelClient."""
        return {
            "model_calls": self.model_calls,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "retries": self.total_retries,
        }

    def _pace(self) -> None:
        """Free tiers cap requests/sec; sleep so calls are >= min_interval_s apart (0 = no pacing)."""
        if self._min_interval_s <= 0:
            return
        elapsed = time.monotonic() - self._last_call_ts
        if elapsed < self._min_interval_s:
            time.sleep(self._min_interval_s - elapsed)


def load_model_stack(path: Path = _THRESHOLDS_PATH) -> dict:
    """Read `scientist.model_stack` from thresholds.yaml — HARD-RAISE if absent (never default a
    model config: same fail-loud discipline as the entry-rule + reporting-delay loaders)."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    try:
        ms = data["scientist"]["model_stack"]
        _ = (ms["phase_d"]["model_a"], ms["phase_d"]["model_b"])
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "scientist.model_stack.phase_d.{model_a,model_b} missing from thresholds.yaml — "
            "phase_d is unconfigured (fail loud, do not default)") from exc
    return ms


def _client_from(spec: dict, ms: dict) -> PhaseDModelClient:
    env = spec["api_key_env"]
    api_key = os.environ.get(env)
    if not api_key:
        raise RuntimeError(
            f"phase_d model {spec['model_id']!r} needs its API key in ${env} (not set). Export it "
            f"and `pip install google-genai mistralai` for a live run. phase_d is the free DEV pair "
            f"— output is NON-reportable; reportable figures require phase_f + SKU/cost authorization.")
    return PhaseDModelClient(
        vendor=spec["vendor"], model_id=spec["model_id"],
        max_output_tokens=ms.get("max_output_tokens", 2048), api_key=api_key,
        temperature=ms.get("temperature", 0.0),
        min_interval_s=spec.get("min_interval_s", ms.get("min_interval_s", 0.0)),
        max_retries=ms.get("max_retries", 6))


def build_phase_d_clients(
    path: Path = _THRESHOLDS_PATH,
) -> tuple[PhaseDModelClient, PhaseDModelClient]:
    """Build the two live phase_d clients (model_a = Gemini, model_b = Mistral) from thresholds.yaml.
    Fail loud (RuntimeError) if the config OR an API key is absent — never a silent stub. Each is a
    ModelClient; a caller runs the llm_researcher rung once per model (the pair is not merged here —
    dual-agreement is a Librarian EXTRACTION concept, not a generation one)."""
    ms = load_model_stack(path)
    return (_client_from(ms["phase_d"]["model_a"], ms),
            _client_from(ms["phase_d"]["model_b"], ms))
