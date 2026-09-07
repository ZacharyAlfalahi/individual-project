"""
BBW (2019) factor definitions — the four-factor harness wiring.

This module holds only the DEFINITIONS (which axis sorts, which axis stripes,
which legs, how CRF composes); it authors NO algorithmic code — it configures
the audited characteristic_sort engine and composes its outputs. Correction-
agnostic: the same configs run on the uncorrected and corrected panels the data
layer emits. IO and family selection live in scripts/build_bbw_factors.py.

Three independent 5×5 bivariate sorts, RATING always one axis (spec §3):

    rating × VaR5  → DRF (VaR axis)   and CRF_VaR  (rating axis)
    rating × gamma → LRF (gamma axis)  and CRF_ILLIQ(rating axis)
    rating × REV   → REV (REV axis)    and CRF_REV (rating axis)

Each grid yields TWO factors by swapping score↔control, so six engine runs total.
Independent sorts (Fama-French style), value-weighted by par (§2.4), monthly
rebalance / one-month hold (§3.3).

Leg directions (§3.4), with rating numeric 1=AAA…22=D so group 4 = lowest rating
= highest credit risk:
    DRF        = high-VaR − low-VaR           (score=var_5pct, long 4, short 0)
    LRF        = high-gamma − low-gamma        (score=gamma,    long 4, short 0)
    REV        = losers − winners             (score=rev,      long 0, short 4)
    CRF_VaR    = low-rating − high-rating      (score=rating,   long 4, short 0) across VaR stripes
    CRF_ILLIQ  = low-rating − high-rating      (score=rating,   long 4, short 0) across gamma stripes
    CRF_REV    = low-rating − high-rating      (score=rating,   long 4, short 0) across REV stripes

CRF composite = (CRF_VaR + CRF_ILLIQ + CRF_REV) / 3, equal-weighted (§3.5).
The BBW model factors are MKTB (build_mktb.py), DRF, LRF, CRF; REV is the
standalone characteristic-axis leg of the REV grid (also feeds CRF via CRF_REV).
"""

from __future__ import annotations

import pandas as pd

from .characteristic_sort import run_characteristic_sort

# name -> (score axis, control axis, long group, short group)
#
# Column-name reconciliation (v1.4): the CRF gold's REV control axis is the concept
# ``prior_1m_excess_return``, which the frozen D27 concept->column table binds to
# column ``xret`` (str depends on that binding). So the adapter compiles the CRF_REV
# leg's control to ``xret``; ``crf_rev`` here is reconciled ``rev``->``xret`` to keep
# the golden rulebook byte-equal with the adapter (never edit the frozen concept
# table). CRF companion: a live CRF run reads the BBW panel's reversal signal under
# column ``xret``, which build_bbw_factors exposes via a ``rev``->``xret`` alias;
# G2 compares rulebook dicts only, so this does not gate the buildable scope. The
# standalone ``rev`` factor (score=``rev``) is in no gold/G2 and is left unchanged.
BBW_FACTOR_CONFIGS: dict[str, dict] = {
    "drf":       {"score": "var_5pct", "control": "rating",   "long_group": 4, "short_group": 0},
    "lrf":       {"score": "gamma",    "control": "rating",   "long_group": 4, "short_group": 0},
    "rev":       {"score": "rev",      "control": "rating",   "long_group": 0, "short_group": 4},
    "crf_var":   {"score": "rating",   "control": "var_5pct", "long_group": 4, "short_group": 0},
    "crf_illiq": {"score": "rating",   "control": "gamma",    "long_group": 4, "short_group": 0},
    "crf_rev":   {"score": "rating",   "control": "xret",     "long_group": 4, "short_group": 0},
}

CRF_COMPONENTS = ("crf_var", "crf_illiq", "crf_rev")


def factor_rulebook(name: str, signal_lag: int = 0) -> dict:
    """Build the engine rulebook for a named BBW factor. 5×5 independent
    bivariate sort, value-weighted by par. signal_lag is the lib_gap toggle
    (0 = as-published contemporaneous alignment); the lead/lag injection toggle
    is a separate panel-level transform (Chunk 4), not a rulebook field."""
    if name not in BBW_FACTOR_CONFIGS:
        raise KeyError(f"unknown BBW factor {name!r}; known: {sorted(BBW_FACTOR_CONFIGS)}")
    c = BBW_FACTOR_CONFIGS[name]
    return {
        "score": c["score"],
        "control": c["control"],
        "groups": 5,
        "control_groups": 5,
        "weighting": "by_size",
        "long_group": c["long_group"],
        "short_group": c["short_group"],
        "signal_lag": signal_lag,
        "nw_lags": None,
    }


def run_bbw_factor(panel: pd.DataFrame, name: str, signal_lag: int = 0) -> dict:
    """Run one BBW factor through the characteristic-sort engine. The panel must
    carry the score and control columns named in the factor's config (rating,
    var_5pct, gamma, rev) plus cusip/date/ret/size."""
    return run_characteristic_sort(panel, factor_rulebook(name, signal_lag=signal_lag))


def compose_crf(component_monthly: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """CRF = equal-weighted mean of the three CRF_* monthly long-short series,
    aligned on date (§3.5). `component_monthly` maps each CRF component name to
    its engine `monthly_returns` frame (needs columns date, strategy_ret).

    A month is included only when all three components are present (inner-join),
    so the composite is never a partial average.

    NB the STANDALONE variant. For the eval pipeline the AUTHORITATIVE CRF composite
    is the adapter->runner `equal_average` path (`agents/quant/config/runner.py`
    `_combine_equal_average`), whose adaptive divisor (skipna by-date mean) differs
    from this inner-join ONLY on months where a component is missing; on the frozen
    BBW window all three share universe/formation every month, so they agree. G2
    compares rulebook dicts + the combiner dict, never return series, so this
    difference does not affect the byte-equality gate.
    """
    missing = [c for c in CRF_COMPONENTS if c not in component_monthly]
    if missing:
        raise KeyError(f"compose_crf missing components: {missing}")

    merged = None
    for comp in CRF_COMPONENTS:
        s = component_monthly[comp][["date", "strategy_ret"]].rename(
            columns={"strategy_ret": comp}
        )
        merged = s if merged is None else merged.merge(s, on="date", how="inner")

    merged["crf"] = merged[list(CRF_COMPONENTS)].mean(axis=1)
    return merged[["date", "crf"]].sort_values("date").reset_index(drop=True)
