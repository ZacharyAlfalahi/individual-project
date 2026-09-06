"""
shared/stats/holdout_bootstrap.py — SC-SCI-13 holdout-inference block-bootstrap diagnostic.

PURPOSE. Amendment SC-SCI-13 (docs/scientist_protocol.yaml) adds a block-bootstrap
confidence interval to the 45-month holdout inference as a LABELLED DIAGNOSTIC ONLY. The
PRIMARY inference stays the paired Newey-West HAC t (survivor minus parent) and each
series' own NW-HAC alpha vs the BBW-4 factor set; the bootstrap is a distribution-free
cross-check on those, never a decision rule. It narrowly supersedes amendment A7's "no
holdout bootstrap intervals" rider FOR LABELLED-DIAGNOSTIC USE ONLY (see
docs/shared_evaluation_amendments_index.md, row A8): the 10-effective-block floor is
DISCLOSED and LABELLED, not satisfied — a block length is never chosen to clear the
arithmetic; below-floor CIs ship tagged `below_floor`. Every result carries
`is_confirmatory=False`; BH-FDR remains the sole decision rule.

PURITY / NO HOLDOUT. Every function here is pure: it takes in-memory pandas Series /
DataFrames and returns frozen dataclasses. It performs no I/O, reads no config, and never
names a path. It therefore cannot, by construction, read `/data/holdout/`. The only code
that would ever load the real holdout panel and call this is the one-shot holdout frozen-holdout
builder (`agents/scientist/experimentalist/oneshot_holdout/`, built and wired via `stage2_evaluate.py`,
not yet executed against the real holdout panel); until then this module
is exercised on synthetic data only. Enforced by tests/unit/test_holdout_bootstrap_inertness.py.

REUSE (wrap, don't move). Block indices from the Auditor's synchronised fixed-block
bootstrap (`circular_block_indices`, `effective_blocks`); the HAC t / OLS alpha from the
quant library (`summarize_returns`, `regress_on_benchmark`); month alignment from
`shared/evaluation/alignment.align_monthly`. No new statistical convention is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agents.auditor.checks.bootstrap import circular_block_indices, effective_blocks
from agents.quant.library.characteristic_sort import regress_on_benchmark, summarize_returns
from shared.evaluation.alignment import align_monthly

# Stable per-statistic codes so a bootstrap cell's generator is derived reproducibly and
# independently from (master seed, statistic, block length) — order-invariant.
_PAIRED = "paired_mean_difference"
_SURVIVOR_ALPHA = "survivor_alpha"
_PARENT_ALPHA = "parent_alpha"
_STAT_CODES = {_PAIRED: 0, _SURVIVOR_ALPHA: 1, _PARENT_ALPHA: 2}

# The pinned lag: floor(T ** 0.25) = 2 at T = 45 (SC-SCI-13 clause 1).
_PINNED_NW_LAGS = 2


def _none_if_nan(x: float | None) -> float | None:
    if x is None:
        return None
    return None if not np.isfinite(x) else float(x)


def _percentile_ci(draws: np.ndarray, alpha: float) -> tuple[float, float]:
    """Two-sided percentile interval, matching BootstrapResult.interval's convention."""
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _floor_label(eff: int, min_effective_blocks: int) -> str:
    return "floor_met" if eff >= min_effective_blocks else "below_floor"


# ---------------------------------------------------------------------------
# Typed results (frozen, numeric-only, hand-written to_dict — house style of
# shared/stats/pbo.py and posterior.py). No verdict boolean anywhere.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockBootstrapCI:
    """One labelled-diagnostic bootstrap CI for one (statistic x block length) cell."""

    statistic: str                 # paired_mean_difference | survivor_alpha | parent_alpha
    block_length_months: int       # 3 or 6
    n_replicates: int              # B (1000)
    effective_blocks: int          # floor(T / ell): 45/{3,6}=15/7
    min_effective_blocks: int      # the floor this CI is LABELLED against (10)
    floor_label: str               # floor_met | below_floor (disclosure, never a gate)
    point: float | None            # the statistic on the un-resampled sample
    ci_low: float | None           # percentile CI (alpha/2, 1-alpha/2)
    ci_high: float | None
    alpha: float                   # 0.05 (two-sided)
    seed: int                      # master seed; cell rng = f(seed, statistic, block length)
    is_confirmatory: bool = False  # ALWAYS False (test-pinned). BH-FDR is the sole rule.

    def to_dict(self) -> dict:
        return {
            "statistic": self.statistic,
            "block_length_months": self.block_length_months,
            "n_replicates": self.n_replicates,
            "effective_blocks": self.effective_blocks,
            "min_effective_blocks": self.min_effective_blocks,
            "floor_label": self.floor_label,
            "point": self.point,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "alpha": self.alpha,
            "seed": self.seed,
            "is_confirmatory": self.is_confirmatory,
        }


@dataclass(frozen=True)
class HoldoutInferenceWindow:
    """PRIMARY HAC-t statistics + the DIAGNOSTIC bootstrap CIs for one window."""

    window_label: str              # full_45m
    n_obs: int                     # T (paired overlap; 45)
    nw_lags_used: int              # 2 (pinned)
    # PRIMARY — the confirmatory statistics; the bootstrap never replaces these.
    paired_mean_difference: float | None
    paired_t_hac: float | None
    survivor_alpha: float | None
    survivor_alpha_t: float | None
    parent_alpha: float | None
    parent_alpha_t: float | None
    # DIAGNOSTIC — 3 statistics x len(block_lengths) CIs; never confirmatory.
    bootstrap_cis: tuple[BlockBootstrapCI, ...]

    def to_dict(self) -> dict:
        return {
            "window_label": self.window_label,
            "n_obs": self.n_obs,
            "nw_lags_used": self.nw_lags_used,
            "paired_mean_difference": self.paired_mean_difference,
            "paired_t_hac": self.paired_t_hac,
            "survivor_alpha": self.survivor_alpha,
            "survivor_alpha_t": self.survivor_alpha_t,
            "parent_alpha": self.parent_alpha,
            "parent_alpha_t": self.parent_alpha_t,
            "bootstrap_cis": tuple(ci.to_dict() for ci in self.bootstrap_cis),
        }


# ---------------------------------------------------------------------------
# Internal helpers — alignment (reusing the tested align_monthly) and design.
# ---------------------------------------------------------------------------


def _paired_diff_series(survivor: pd.Series, parent: pd.Series) -> pd.Series:
    """(survivor - parent) over the common months. Reuses align_monthly's tested
    month-key intersection (dedup + NaN-drop) so the paired T matches what the pipeline
    elsewhere reports; mirrors alignment.paired_difference's internal reindex."""
    al = align_monthly(survivor, parent, a_label="survivor", b_label="parent")
    common = list(al.common_month_index)
    if not common:
        return pd.Series(dtype=float)

    def _norm(s: pd.Series) -> pd.Series:
        idx = pd.to_datetime(s.index).astype("datetime64[ns]")
        s2 = pd.Series(s.to_numpy(), index=idx)
        s2 = s2[s2.notna()]
        return s2[~s2.index.duplicated(keep="last")]

    def _keys(index: pd.Index) -> list[str]:
        return [pd.Timestamp(t).strftime("%Y-%m-%d") for t in index]

    sa, sb = _norm(survivor), _norm(parent)
    a_c = pd.Series(sa.to_numpy(), index=_keys(sa.index)).reindex(common).to_numpy()
    b_c = pd.Series(sb.to_numpy(), index=_keys(sb.index)).reindex(common).to_numpy()
    return pd.Series(a_c - b_c, index=pd.to_datetime(common))


def _assemble_design(y: pd.Series, factors: pd.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """Assemble (y_arr, X, T) once, mirroring regress_on_benchmark's join/dropna
    (characteristic_sort.py:788-813). X has an intercept in column 0, so beta[0] is alpha."""
    factor_cols = [c for c in factors.columns if c != "date"]
    if not factor_cols:
        return None, None, 0
    y_df = pd.DataFrame({"date": pd.to_datetime(y.index), "_y": y.to_numpy()})
    fac = factors.copy()
    fac["date"] = pd.to_datetime(fac["date"])
    merged = y_df.merge(fac, on="date", how="inner").dropna(subset=["_y"] + factor_cols)
    T = len(merged)
    if T == 0:
        return None, None, 0
    y_arr = merged["_y"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(T)] + [merged[c].to_numpy(dtype=float) for c in factor_cols])
    return y_arr, X, T


# ---------------------------------------------------------------------------
# Public bootstrap functions — pure, never raise.
# ---------------------------------------------------------------------------


def paired_difference_bootstrap(
    diff: pd.Series,
    *,
    statistic_label: str = _PAIRED,
    block_lengths: tuple[int, ...],
    n_replicates: int,
    min_effective_blocks: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[BlockBootstrapCI, ...]:
    """Block-bootstrap CI for the paired mean difference. Resamples the 1-D difference
    series with the circular fixed-block bootstrap and recomputes the mean per replicate.
    Never raises: a degenerate series (empty / length < 2) yields typed None CIs."""
    d = diff.dropna()
    v = d.to_numpy(dtype=float)
    T = int(v.size)
    point = _none_if_nan(float(v.mean())) if T > 0 else None

    out: list[BlockBootstrapCI] = []
    for ell in block_lengths:
        ell = int(ell)
        eff = effective_blocks(T, ell) if T > 0 else 0
        label = _floor_label(eff, min_effective_blocks)
        if T < 2:
            out.append(
                BlockBootstrapCI(
                    statistic=statistic_label, block_length_months=ell,
                    n_replicates=int(n_replicates), effective_blocks=int(eff),
                    min_effective_blocks=int(min_effective_blocks), floor_label=label,
                    point=point, ci_low=None, ci_high=None, alpha=alpha, seed=int(seed),
                )
            )
            continue
        rng = np.random.default_rng((int(seed), _STAT_CODES[statistic_label], ell))
        draws = np.empty(int(n_replicates), dtype=float)
        for b in range(int(n_replicates)):
            idx = circular_block_indices(T, ell, rng)
            draws[b] = v[idx].mean()
        lo, hi = _percentile_ci(draws, alpha)
        out.append(
            BlockBootstrapCI(
                statistic=statistic_label, block_length_months=ell,
                n_replicates=int(n_replicates), effective_blocks=int(eff),
                min_effective_blocks=int(min_effective_blocks), floor_label=label,
                point=point, ci_low=lo, ci_high=hi, alpha=alpha, seed=int(seed),
            )
        )
    return tuple(out)


def own_alpha_bootstrap(
    y: pd.Series,
    factors: pd.DataFrame,
    *,
    statistic_label: str,
    block_lengths: tuple[int, ...],
    n_replicates: int,
    min_effective_blocks: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[BlockBootstrapCI, ...]:
    """Block-bootstrap CI for one series' own OLS alpha vs `factors`. Assembles the design
    once, then block-resamples rows and refits OLS alpha per replicate with lstsq (never
    raises on a singular resample, unlike inv). Never raises: too few overlapping months
    yields typed None CIs."""
    y_arr, X, T = _assemble_design(y, factors)
    k = 0 if X is None else X.shape[1]
    point = None
    if X is not None and T > k:
        beta, *_ = np.linalg.lstsq(X, y_arr, rcond=None)
        point = _none_if_nan(float(beta[0]))

    out: list[BlockBootstrapCI] = []
    for ell in block_lengths:
        ell = int(ell)
        eff = effective_blocks(T, ell) if T > 0 else 0
        label = _floor_label(eff, min_effective_blocks)
        if X is None or T <= k:
            out.append(
                BlockBootstrapCI(
                    statistic=statistic_label, block_length_months=ell,
                    n_replicates=int(n_replicates), effective_blocks=int(eff),
                    min_effective_blocks=int(min_effective_blocks), floor_label=label,
                    point=point, ci_low=None, ci_high=None, alpha=alpha, seed=int(seed),
                )
            )
            continue
        rng = np.random.default_rng((int(seed), _STAT_CODES[statistic_label], ell))
        draws = np.empty(int(n_replicates), dtype=float)
        for b in range(int(n_replicates)):
            idx = circular_block_indices(T, ell, rng)
            beta, *_ = np.linalg.lstsq(X[idx], y_arr[idx], rcond=None)
            draws[b] = beta[0]
        lo, hi = _percentile_ci(draws, alpha)
        out.append(
            BlockBootstrapCI(
                statistic=statistic_label, block_length_months=ell,
                n_replicates=int(n_replicates), effective_blocks=int(eff),
                min_effective_blocks=int(min_effective_blocks), floor_label=label,
                point=point, ci_low=lo, ci_high=hi, alpha=alpha, seed=int(seed),
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# Holdout-inference wrapper — the FUNCTION form of SC-SCI-13 clause 3.
# (The one-shot holdout orchestrator that loads the real holdout panel and calls this —
# agents/scientist/experimentalist/oneshot_holdout/stage2_evaluate.py — is built and wired, not yet executed.)
# ---------------------------------------------------------------------------


def _one_window(
    survivor: pd.Series,
    parent: pd.Series,
    factors: pd.DataFrame,
    *,
    window_label: str,
    block_lengths: tuple[int, ...],
    n_replicates: int,
    min_effective_blocks: int,
    nw_lags: int,
    months_per_year: int,
    seed: int,
    alpha: float,
) -> HoldoutInferenceWindow:
    diff = _paired_diff_series(survivor, parent)
    n_obs = int(diff.dropna().size)

    # PRIMARY — paired HAC t + each series' own NW-HAC alpha (pinned lag).
    if n_obs > 0:
        paired = summarize_returns(diff, nw_lags, months_per_year)
        paired_mean = _none_if_nan(paired["average"])
        paired_t = _none_if_nan(paired["t_stat"])
    else:
        paired_mean = paired_t = None
    surv_reg = regress_on_benchmark(survivor, factors, nw_lags)
    par_reg = regress_on_benchmark(parent, factors, nw_lags)

    # DIAGNOSTIC — never confirmatory. Independent per-statistic generators from `seed`.
    cis: list[BlockBootstrapCI] = []
    cis.extend(paired_difference_bootstrap(
        diff, statistic_label=_PAIRED, block_lengths=block_lengths,
        n_replicates=n_replicates, min_effective_blocks=min_effective_blocks,
        seed=seed, alpha=alpha,
    ))
    cis.extend(own_alpha_bootstrap(
        survivor, factors, statistic_label=_SURVIVOR_ALPHA, block_lengths=block_lengths,
        n_replicates=n_replicates, min_effective_blocks=min_effective_blocks,
        seed=seed, alpha=alpha,
    ))
    cis.extend(own_alpha_bootstrap(
        parent, factors, statistic_label=_PARENT_ALPHA, block_lengths=block_lengths,
        n_replicates=n_replicates, min_effective_blocks=min_effective_blocks,
        seed=seed, alpha=alpha,
    ))

    return HoldoutInferenceWindow(
        window_label=window_label,
        n_obs=n_obs,
        nw_lags_used=int(nw_lags),
        paired_mean_difference=paired_mean,
        paired_t_hac=paired_t,
        survivor_alpha=_none_if_nan(surv_reg["alpha"]),
        survivor_alpha_t=_none_if_nan(surv_reg["alpha_t"]),
        parent_alpha=_none_if_nan(par_reg["alpha"]),
        parent_alpha_t=_none_if_nan(par_reg["alpha_t"]),
        bootstrap_cis=tuple(cis),
    )


def holdout_inference(
    survivor: pd.Series,
    parent: pd.Series,
    factors: pd.DataFrame,
    *,
    block_lengths: tuple[int, ...] = (3, 6),
    n_replicates: int,
    min_effective_blocks: int,
    nw_lags: int = _PINNED_NW_LAGS,
    months_per_year: int = 12,
    seed: int,
    alpha: float = 0.05,
) -> HoldoutInferenceWindow:
    """PRIMARY HAC-t + DIAGNOSTIC bootstrap on the registered 45-month holdout window
    (SC-SCI-13 clause 3).

    PURE: takes in-memory series/frames only; performs no I/O and cannot read the holdout.
    """
    return _one_window(
        survivor, parent, factors, window_label="full_45m",
        block_lengths=block_lengths, n_replicates=n_replicates,
        min_effective_blocks=min_effective_blocks, nw_lags=nw_lags,
        months_per_year=months_per_year, seed=int(seed), alpha=alpha,
    )
