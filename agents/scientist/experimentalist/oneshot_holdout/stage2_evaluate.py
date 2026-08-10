"""Stage 2 — holdout evaluation, per SC-SCI-13 / SC-SCI-11 (one-shot holdout §3).

one-shot holdout is the sole caller that loads the holdout panels behind the gate and hands the in-memory
survivor / parent / benchmark series to the CANONICAL, pure computation function
``shared.stats.holdout_inference_window_sensitivity`` (primary paired NW-HAC t + each series'
own NW-HAC alpha, plus the labelled-diagnostic block bootstrap, on the registered 45-month
window AND the tagged ≤2024-12 sub-window in one call). This module adds only the SC-SCI-11 P3
posterior on the survivor alpha and shapes the descriptive record — it introduces no new
statistical convention.

FIREWALL (§1.4): the evaluator reaches statistics ONLY through the sanctioned ``shared.stats``
surface — never ``agents.auditor`` directly — and runs no lattice/attribution inference on the
holdout (per-survivor series statistics only). ``test_oneshot_holdout_no_lattice_imports`` pins that no one-shot holdout
module imports ``agents.auditor.*`` directly. The output schema has NO pass/fail field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from shared.evaluation.thresholds import load_holdout_bootstrap_diagnostic_config
from shared.stats import (
    HoldoutInferenceWindow,
    alpha_se_from_t,
    holdout_inference_window_sensitivity,
    posterior_summary,
)

from .windows import Window


@dataclass(frozen=True)
class BenchmarkResult:
    benchmark: str
    full: dict                       # HoldoutInferenceWindow.to_dict() + {"posterior": ...}
    sensitivity: dict                # same shape for the tagged ≤2024-12 sub-window


@dataclass(frozen=True)
class SurvivorInput:
    survivor_id: str
    returns: pd.Series               # month-end indexed monthly returns (evaluation basis)
    parent_returns: pd.Series
    is_extension_1: bool = False


@dataclass(frozen=True)
class SurvivorResult:
    survivor_id: str
    is_extension_1: bool
    benchmarks: dict[str, BenchmarkResult]
    # Deliberately no pass/fail / advancement field (§3): descriptive only.


def _posterior_for(window: HoldoutInferenceWindow, priors: Mapping[str, float]) -> dict | None:
    """P3 posterior (SC-SCI-11) on the survivor alpha, or None when the alpha t is degenerate
    (the canonical function returns None on an unestimable window — never laundered)."""
    if window.survivor_alpha is None or window.survivor_alpha_t is None or window.survivor_alpha_t == 0.0:
        return None
    se = alpha_se_from_t(window.survivor_alpha, window.survivor_alpha_t)
    summ = posterior_summary(window.survivor_alpha, se, priors)
    return {
        "point": summ.point,
        "se": summ.se,
        "priors": {
            row.label: {
                "prior_sigma": row.prior_sigma,
                "post_mean": row.post_mean,
                "post_sd": row.post_sd,
                "ci_low": row.ci_low,
                "ci_high": row.ci_high,
                "p_positive": row.p_positive,
            }
            for row in summ.posteriors
        },
    }


def _window_record(window: HoldoutInferenceWindow, priors: Mapping[str, float]) -> dict:
    record = window.to_dict()
    record["posterior"] = _posterior_for(window, priors)
    return record


def evaluate_survivors(
    survivors: list[SurvivorInput],
    benchmarks: Mapping[str, pd.DataFrame],
    window: Window,
    sub_window: Window,
    priors: Mapping[str, float],
    *,
    base_seed: int = 82026,
    config_path=None,
) -> list[SurvivorResult]:
    """Evaluate every survivor on both windows under every benchmark, in one pass, via the
    canonical ``holdout_inference_window_sensitivity``. Every series is first CLIPPED to the
    registered ``window`` so the "full" statistic is exactly ``window.n_months`` — seed / pre-window
    months can never leak into the holdout statistic. ``sub_window.end`` supplies the tagged ≤ cutoff.
    The block lengths / replicate count / floor / lag come from the fail-loud config."""
    if not benchmarks:
        raise ValueError("at least one benchmark factor set is required (BBW-4 primary)")
    cfg = load_holdout_bootstrap_diagnostic_config(config_path)
    cutoff = pd.Period(sub_window.end, "M").to_timestamp("M")
    lo = pd.Period(window.start, "M").to_timestamp("M")
    hi = pd.Period(window.end, "M").to_timestamp("M")

    def _clip_series(s: pd.Series) -> pd.Series:
        idx = pd.to_datetime(s.index)
        return s[(idx >= lo) & (idx <= hi)]

    def _clip_factors(f: pd.DataFrame) -> pd.DataFrame:
        idx = pd.to_datetime(f["date"])
        return f[(idx >= lo) & (idx <= hi)]

    clipped_benchmarks = {b: _clip_factors(fr) for b, fr in benchmarks.items()}
    results: list[SurvivorResult] = []
    for si, survivor in enumerate(survivors):
        surv = _clip_series(survivor.returns)
        parent = _clip_series(survivor.parent_returns)
        per_benchmark: dict[str, BenchmarkResult] = {}
        for bi, (bname, factors) in enumerate(clipped_benchmarks.items()):
            full, sub = holdout_inference_window_sensitivity(
                surv,
                parent,
                factors,
                subwindow_cutoff=cutoff,
                block_lengths=tuple(cfg.block_lengths),
                n_replicates=cfg.n_replicates,
                min_effective_blocks=cfg.min_effective_blocks,
                seed=base_seed + si * 100 + bi,
            )  # nw_lags (2) and alpha (0.05) are pinned defaults in the canonical function
            per_benchmark[bname] = BenchmarkResult(
                benchmark=bname,
                full=_window_record(full, priors),
                sensitivity=_window_record(sub, priors),
            )
        results.append(SurvivorResult(survivor.survivor_id, survivor.is_extension_1, per_benchmark))
    return results
