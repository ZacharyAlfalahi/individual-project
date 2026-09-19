"""P1 codegen (WS-C) — the pre-flight spend gate for live generation (design decision 2026-09-11).

Policy: REPLAY FIRST. Every (strategy, model) whose prompt is already in the content-addressed
``ResponseCache`` (key = model + seed + prompt) costs nothing. BEFORE any live client exists, the run
computes the cache MISSES it would need and an upper-bound spend estimate per vendor:

  input tokens  = max(``INPUT_TOKENS_UPPER_BOUND`` (8,000), the call's prompt UTF-8 bytes /
                  ``INPUT_BYTES_PER_TOKEN`` (2)) — real tokenisers average ~3-4 characters per token on
                  prose, JSON and code, so the size-based term over-counts
  output tokens = ``p1_codegen.budget.per_call_output_token_cap`` per call — the output ceiling the live
                  client enforces on every call (never a cached-length average, which can under-count)
  USD           = ``p1_codegen.budget.prices_usd_per_1m`` via ``BudgetGuard.estimate_usd`` (one price
                  source; an unpriced model fails loud there)

and REFUSES — typed, before any call — when a vendor's estimate exceeds its cap (a priced vendor with
no cap is refused too: its spend could not be bounded). Outside the bound: a failed attempt a vendor
bills and the client then retries, and any reasoning tokens a vendor bills beyond the output ceiling.
Passing the gate does not lift the run-time ``BudgetGuard``, which meters vendor-reported usage.

A model with ZERO misses is served by ``CacheReplayClient``: no credentials, no SDK, no network.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Callable, Iterable

import yaml

from agents.scientist.researcher.cache import ResponseCache
from evaluation.codegen.live_client import BudgetExceededError, BudgetGuard, build_codegen_factory

_THRESHOLDS = Path(__file__).resolve().parents[2] / "docs" / "thresholds.yaml"


def load_spend_policy(thresholds_path: Path | None = None) -> tuple[float, int]:
    """``(per_vendor_usd_ceiling, input_tokens_floor)`` from ``p1_codegen.budget`` — fail-loud, since
    a silent default would price a live run against an unregistered cap."""
    doc = yaml.safe_load((thresholds_path or _THRESHOLDS).read_text(encoding="utf-8"))
    try:
        block = doc["p1_codegen"]["budget"]
        ceiling, floor = block["per_vendor_usd_ceiling"], block["input_tokens_floor"]
    except (KeyError, TypeError) as exc:
        raise KeyError("thresholds.yaml has no complete p1_codegen.budget spend policy "
                       "(per_vendor_usd_ceiling / input_tokens_floor)") from exc
    # Validate BEFORE converting: float(True) is 1.0 and int(0.5) is 0, so coercion would turn a
    # malformed entry into a plausible policy — an unbounded ceiling or a floor that is not there.
    if isinstance(ceiling, bool) or not isinstance(ceiling, (int, float)) \
            or not math.isfinite(ceiling) or ceiling < 0:
        raise ValueError("p1_codegen.budget.per_vendor_usd_ceiling must be a finite non-negative "
                         f"number, got {ceiling!r}")
    if isinstance(floor, bool) or not isinstance(floor, int) or floor <= 0:
        raise ValueError(f"p1_codegen.budget.input_tokens_floor must be a positive integer, got {floor!r}")
    return float(ceiling), floor


#: Per-vendor spend ceiling and the floor on the prompt-token bound, both registered in thresholds.yaml.
PER_VENDOR_USD_CEILING, INPUT_TOKENS_UPPER_BOUND = load_spend_policy()
#: Conservative UTF-8 bytes per prompt token for the size-based bound (real tokenisers average ~3-4).
INPUT_BYTES_PER_TOKEN = 2
#: The ablation is one-shot at seed 0 (runner.generate_once default; I3).
SEED = 0


class UnexpectedCacheMissError(BudgetExceededError):
    """A replay-only model was asked for a live call: the cache no longer matches the pre-flight.
    Subclasses ``BudgetExceededError`` because that is the one error the scored ablation re-raises
    (halts) instead of recording as a generation_error — an un-pre-flighted call must stop the run."""


class CacheReplayClient:
    """Codegen ``ModelClient`` for a model whose every prompt is a cache hit. ``generate_once`` never
    calls it on a hit; any call means the pre-flight is stale, so it halts instead of going live."""

    def __init__(self, model_id: str):
        self.name = model_id
        self.last_model_version: str | None = None

    def generate(self, prompt: str, *, seed: int) -> str:
        raise UnexpectedCacheMissError(
            f"cache replay client for {self.name!r} was asked to generate (seed={seed}); the "
            "pre-flight recorded no miss for this model — refusing an unbounded live call")


def cache_census(
    strategies: Iterable[str],
    models: list[dict],
    cache: ResponseCache,
    *,
    prompt_fn: Callable[[str], str],
    seed: int = SEED,
) -> tuple[list[dict], list[dict]]:
    """``(hits, misses)`` over every (strategy, model) the ablation would generate, looked up with
    the same key ``generate_once`` uses. Each row: strategy, model_id, vendor, prompt_sha256,
    input_tokens (the call's prompt-token bound)."""
    hits: list[dict] = []
    misses: list[dict] = []
    for strategy in strategies:
        prompt = prompt_fn(strategy)
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        for model in models:
            row = {"strategy": strategy, "model_id": model["model_id"],
                   "vendor": model.get("vendor"), "prompt_sha256": digest,
                   "input_tokens": input_token_bound(prompt)}
            (hits if cache.get(prompt, model["model_id"], seed) is not None else misses).append(row)
    return hits, misses


def input_token_bound(prompt: str) -> int:
    """Upper bound on one call's prompt tokens: the size-based bound, never below the floor."""
    return max(INPUT_TOKENS_UPPER_BOUND, math.ceil(len(prompt.encode("utf-8")) / INPUT_BYTES_PER_TOKEN))


def estimate_spend(misses: list[dict], budget: BudgetGuard) -> dict:
    """Upper-bound USD for the misses, per model and per vendor (see module docstring)."""
    cap = budget.per_call_output_token_cap
    per_model: dict[str, dict] = {}
    for row in misses:
        rec = per_model.setdefault(row["model_id"], {
            "vendor": row["vendor"], "calls": 0, "input_tokens": 0, "output_tokens_per_call": cap,
            "output_tokens_basis": "per_call_output_token_cap (the live client's enforced output ceiling)",
            "usd": 0.0})
        rec["calls"] += 1
        rec["input_tokens"] += row["input_tokens"]
        rec["usd"] += budget.estimate_usd(row["model_id"], row["input_tokens"], cap)
    per_vendor: dict[str, float] = {}
    for rec in per_model.values():
        rec["usd"] = round(rec["usd"], 6)
        per_vendor[rec["vendor"]] = round(per_vendor.get(rec["vendor"], 0.0) + rec["usd"], 6)
    return {"per_model": per_model, "per_vendor_usd": per_vendor,
            "total_usd": round(sum(per_vendor.values()), 6)}


def cap_violations(per_vendor_usd: dict[str, float], caps_usd: dict[str, float]) -> list[str]:
    """Human-readable violations: a vendor estimate above its cap, or a non-zero estimate for a
    vendor with no cap."""
    out = []
    for vendor, usd in sorted(per_vendor_usd.items()):
        cap = caps_usd.get(vendor)
        if cap is None:
            if usd > 0:
                out.append(f"{vendor}: estimated ${usd:.4f} but no spend cap is set for this vendor")
        elif usd > cap + 1e-9:
            out.append(f"{vendor}: estimated ${usd:.4f} exceeds the ${cap:.2f} cap")
    return out


def run_preflight(
    strategies: Iterable[str],
    models: list[dict],
    *,
    cache_root: Path,
    budget: BudgetGuard,
    caps_usd: dict[str, float],
    prompt_fn: Callable[[str], str],
    seed: int = SEED,
) -> dict:
    """The pre-flight record (written into the results JSON as-is). ``refused`` is True iff any cap
    is violated; ``live_models`` are the models that need a live client (>= 1 miss)."""
    hits, misses = cache_census(list(strategies), models, ResponseCache(cache_root),
                                prompt_fn=prompt_fn, seed=seed)
    estimate = estimate_spend(misses, budget)
    violations = cap_violations(estimate["per_vendor_usd"], caps_usd)
    return {
        "policy": ("replay cached responses first; refuse before any live call if a vendor's "
                   "upper-bound estimate exceeds its cap"),
        "cache_root": str(cache_root),
        "seed": seed,
        "input_tokens_floor": INPUT_TOKENS_UPPER_BOUND,
        "input_bytes_per_token": INPUT_BYTES_PER_TOKEN,
        "output_tokens_per_call": budget.per_call_output_token_cap,
        "caps_usd": dict(caps_usd),
        "cache": {"hits": len(hits), "misses": len(misses),
                  "hit_rows": hits, "miss_rows": misses},
        "estimate": estimate,
        "live_models": sorted({m["model_id"] for m in misses}),
        "violations": violations,
        "refused": bool(violations),
    }


def replay_first_factory(
    live_models: list[str],
    budget: BudgetGuard,
    *,
    live_factory: Callable[[dict], object] | None = None,
):
    """A ``client_factory`` honouring the replay-first policy: a model the pre-flight found
    fully cached gets a ``CacheReplayClient`` (no credentials, no SDK, no network); only a
    model with at least one miss gets a live client. With no live models the vendor factory
    is never built, so a pure replay needs no keys. ``live_factory`` injects a fake in tests."""
    live = None
    if live_models:
        live = live_factory if live_factory is not None else build_codegen_factory(
            budget=budget, temperature=0.0)

    def factory(model: dict):
        if model["model_id"] in live_models:
            return live(model)
        return CacheReplayClient(model["model_id"])

    return factory


def metered_by_model(budget: BudgetGuard) -> dict[str, dict]:
    """Actual metered usage (vendor-reported tokens -> estimated USD) summed per model from the
    run's BudgetGuard records. Empty for a pure replay."""
    out: dict[str, dict] = {}
    for rec in budget.records:
        agg = out.setdefault(rec.model_id, {"calls": 0, "prompt_tokens": 0,
                                            "completion_tokens": 0, "usd_estimate": 0.0})
        agg["calls"] += 1
        agg["prompt_tokens"] += rec.prompt_tokens
        agg["completion_tokens"] += rec.completion_tokens
        agg["usd_estimate"] = round(agg["usd_estimate"] + rec.usd_estimate, 6)
    return out
