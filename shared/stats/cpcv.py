"""
shared/stats/cpcv.py — Combinatorial Purged Cross-Validation (López de Prado), net-new (R1).

GEOMETRY. Partition T observations into N contiguous groups; every choice of `test_groups`
groups is one out-of-sample fold, giving C(N, test_groups) folds and
phi = test_groups * C(N, test_groups) / N backtest paths. Training observations that overlap a
test fold are PURGED, and observations immediately after each contiguous test block are
EMBARGOED, so serial dependence cannot leak between train and test.

CONVENTION (spec §10.2 / R1). Every Sharpe comes from `summarize_returns` (ddof=1,
mean/sd * sqrt(months_per_year)) — the SAME functional the wrapped Auditor stats use — so a
CPCV Sharpe and the primary alpha-vs-BBW-4 Sharpe are directly comparable. No new convention.

EMBARGO IN MONTHS, NOT BLOCKS (protocol amendment A4). embargo is an absolute month count
(the Experimentalist passes holding_period, floored at 1), decoupled from N so the geometry
survives a later change of group count. purge is the per-candidate information span
(signal_lookback + holding_period + conditioning_lag). Both are positional distances because
the panel is monthly (one observation = one month).

FIT-FREE STRATEGIES (deliberate scope note). The Scientist's candidates are deterministic — the
wall + monotonicity fix every config ex ante, so there is no per-fold model fit. The LdP *path*
reconstruction is informative only when a prediction depends on which fold trained it; for a
fixed strategy the phi paths coincide. The informative CPCV output here is therefore the
DISTRIBUTION of OOS Sharpes over the C(N,k) folds; `n_paths` is reported as the LdP path count
for the record. The G4 qualification criterion over this distribution is set in the
Experimentalist build, not here — this module is the validated geometry + stats.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

from agents.quant.library.characteristic_sort import summarize_returns


def partition_groups(n_obs: int, n_groups: int) -> list[np.ndarray]:
    """Contiguous, near-equal groups of positional indices (earlier groups absorb the
    remainder, matching numpy.array_split)."""
    if n_groups < 2:
        raise ValueError(f"n_groups must be >= 2; got {n_groups}")
    if n_obs < n_groups:
        raise ValueError(f"n_obs ({n_obs}) < n_groups ({n_groups}) — cannot partition")
    return list(np.array_split(np.arange(n_obs), n_groups))


def n_backtest_paths(n_groups: int, test_groups: int) -> int:
    """phi = test_groups * C(N, test_groups) / N  (López de Prado). Integer for all valid N,k."""
    return test_groups * comb(n_groups, test_groups) // n_groups


def _purge_embargo(train: np.ndarray, test: np.ndarray, *, purge: int, embargo: int) -> np.ndarray:
    """Drop train observations that (a) lie within `purge` months of ANY test observation
    (symmetric label overlap) or (b) fall within `embargo` months AFTER the end of a contiguous
    test block (forward embargo). Returns the surviving train indices (sorted)."""
    if train.size == 0:
        return train
    test_sorted = np.sort(test)
    keep = np.ones(train.shape, dtype=bool)

    if test_sorted.size and purge >= 0:
        nearest = np.abs(train[:, None] - test_sorted[None, :]).min(axis=1)
        keep &= nearest > purge

    if embargo > 0 and test_sorted.size:
        # Ends of contiguous test blocks: a test index whose successor is not also a test index.
        is_end = np.append(np.diff(test_sorted) != 1, True)
        block_ends = test_sorted[is_end]
        delta = train[:, None] - block_ends[None, :]
        in_embargo = ((delta > 0) & (delta <= embargo)).any(axis=1)
        keep &= ~in_embargo

    return train[keep]


def cpcv_folds(
    n_obs: int, *, n_groups: int, test_groups: int, purge: int, embargo: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """All C(n_groups, test_groups) (train, test) index folds, with purge + embargo applied to
    each train set. Test indices are the union of the chosen groups; train is everything else
    minus purged/embargoed observations."""
    groups = partition_groups(n_obs, n_groups)
    all_pos = np.arange(n_obs)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for combo in combinations(range(n_groups), test_groups):
        test = np.sort(np.concatenate([groups[g] for g in combo]))
        train_full = np.setdiff1d(all_pos, test, assume_unique=True)
        train = _purge_embargo(train_full, test, purge=purge, embargo=embargo)
        folds.append((train, test))
    return folds


@dataclass(frozen=True)
class CPCVResult:
    n_groups: int
    test_groups: int
    n_folds: int                       # C(N, k) — 28 for (8, 2)
    n_paths: int                       # phi = k*C(N,k)/N — 7 for (8, 2); LdP path count
    purge: int
    embargo: int
    fold_sharpes: tuple[float, ...]    # OOS Sharpe per fold (ddof=1 convention)
    median_sharpe: float
    min_sharpe: float
    frac_positive: float               # fraction of finite folds with Sharpe > 0
    mean_train_months: float           # audit: training surviving purge+embargo, averaged

    def to_dict(self) -> dict:
        return {
            "n_groups": self.n_groups,
            "test_groups": self.test_groups,
            "n_folds": self.n_folds,
            "n_paths": self.n_paths,
            "purge": self.purge,
            "embargo": self.embargo,
            "median_sharpe": self.median_sharpe,
            "min_sharpe": self.min_sharpe,
            "frac_positive": self.frac_positive,
            "mean_train_months": self.mean_train_months,
        }


def cpcv_evaluate(
    returns,
    *,
    n_groups: int = 8,
    test_groups: int = 2,
    purge: int,
    embargo: int,
    months_per_year: int = 12,
) -> CPCVResult:
    """Evaluate a monthly return series over the CPCV folds. Returns the OOS-Sharpe distribution
    (one per fold, via `summarize_returns` so the convention matches the primary test) plus
    audit fields. Operates positionally — the caller passes a clean monthly series."""
    s = pd.Series(returns).reset_index(drop=True)
    n = len(s)
    folds = cpcv_folds(n, n_groups=n_groups, test_groups=test_groups, purge=purge, embargo=embargo)

    sharpes: list[float] = []
    train_counts: list[int] = []
    for train, test in folds:
        oos = s.iloc[test]
        summ = summarize_returns(oos, nw_lags=None, months_per_year=months_per_year)
        sharpes.append(float(summ["sharpe"]))
        train_counts.append(int(train.size))

    arr = np.array(sharpes, dtype=float)
    finite = arr[np.isfinite(arr)]
    return CPCVResult(
        n_groups=n_groups,
        test_groups=test_groups,
        n_folds=len(folds),
        n_paths=n_backtest_paths(n_groups, test_groups),
        purge=purge,
        embargo=embargo,
        fold_sharpes=tuple(sharpes),
        median_sharpe=float(np.median(finite)) if finite.size else float("nan"),
        min_sharpe=float(finite.min()) if finite.size else float("nan"),
        frac_positive=float((finite > 0).mean()) if finite.size else float("nan"),
        mean_train_months=float(np.mean(train_counts)) if train_counts else float("nan"),
    )
