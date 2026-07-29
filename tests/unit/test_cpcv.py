"""
Tests for shared/stats — the R1 wrap (identity: nothing moved) and the net-new CPCV geometry
+ Sharpe convention, validated on hand-computable and autocorrelated synthetic data.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import shared.stats as ss  # noqa: E402
from agents.auditor.checks import bootstrap as a_boot  # noqa: E402
from agents.auditor.checks import economic as a_econ  # noqa: E402
from agents.auditor.checks import fdr as a_fdr  # noqa: E402
from agents.quant.library.characteristic_sort import summarize_returns  # noqa: E402
from shared.stats.cpcv import (  # noqa: E402
    cpcv_evaluate,
    cpcv_folds,
    n_backtest_paths,
    partition_groups,
)


# ---- R1: wrap, don't move (identity, not a copy) ------------------------------------------

def test_wrapped_stats_are_the_same_objects():
    assert ss.benjamini_hochberg is a_fdr.benjamini_hochberg
    assert ss.run_fdr is a_fdr.run_fdr
    assert ss.deflated_sharpe_ratio is a_econ.deflated_sharpe_ratio
    assert ss.run_bootstrap is a_boot.run_bootstrap
    assert ss.circular_block_indices is a_boot.circular_block_indices


# ---- CPCV geometry (known answers) --------------------------------------------------------

def test_path_count_and_fold_count():
    assert n_backtest_paths(8, 2) == 7            # phi = 2*C(8,2)/8 = 2*28/8
    assert len(cpcv_folds(209, n_groups=8, test_groups=2, purge=0, embargo=0)) == 28


def test_partition_is_contiguous_and_near_equal():
    groups = partition_groups(209, 8)
    sizes = [len(g) for g in groups]
    assert sum(sizes) == 209
    assert max(sizes) - min(sizes) <= 1           # near-equal
    assert np.array_equal(np.concatenate(groups), np.arange(209))  # contiguous, ordered


def test_partition_rejects_bad_shapes():
    with pytest.raises(ValueError):
        partition_groups(5, 8)                    # n_obs < n_groups
    with pytest.raises(ValueError):
        partition_groups(100, 1)                  # n_groups < 2


# ---- Purge + embargo (hand-computable) ----------------------------------------------------

def test_purge_embargo_hand_example():
    # n_obs=10, N=5 -> groups [0,1][2,3][4,5][6,7][8,9]; single test group; purge=embargo=1.
    folds = cpcv_folds(10, n_groups=5, test_groups=1, purge=1, embargo=1)
    train0, test0 = folds[0]                      # combo (0,) -> test {0,1}
    assert set(test0) == {0, 1}
    # purge removes 2 (|2-1|=1); embargo (block end 1) also targets 2 -> kept {3..9}.
    assert set(train0) == {3, 4, 5, 6, 7, 8, 9}

    train2, test2 = folds[2]                       # combo (2,) -> test {4,5}
    assert set(test2) == {4, 5}
    # purge removes 3 (|3-4|=1) and 6 (|6-5|=1); embargo targets 6 -> kept {0,1,2,7,8,9}.
    assert set(train2) == {0, 1, 2, 7, 8, 9}


def test_embargo_and_purge_are_monotone():
    r = _ar1_returns(209)
    base = cpcv_evaluate(r, purge=0, embargo=0).mean_train_months
    more_embargo = cpcv_evaluate(r, purge=0, embargo=6).mean_train_months
    more_purge = cpcv_evaluate(r, purge=6, embargo=0).mean_train_months
    assert more_embargo <= base
    assert more_purge < base                       # purge is symmetric -> always bites


# ---- Sharpe convention (must match summarize_returns, ddof=1) ------------------------------

def test_fold_sharpe_uses_summarize_returns_ddof1():
    r = _ar1_returns(209)
    folds = cpcv_folds(209, n_groups=8, test_groups=2, purge=0, embargo=0)
    _, test = folds[0]
    oos = pd.Series(r).reset_index(drop=True).iloc[test]
    expected = summarize_returns(oos, nw_lags=None, months_per_year=12)["sharpe"]
    # hand computation with ddof=1
    hand = (oos.mean() / oos.std(ddof=1)) * math.sqrt(12)
    res = cpcv_evaluate(r, n_groups=8, test_groups=2, purge=0, embargo=0)
    assert res.fold_sharpes[0] == pytest.approx(expected)
    assert res.fold_sharpes[0] == pytest.approx(hand)


# ---- Autocorrelated synthetic (sanity vs the known full-sample Sharpe) ---------------------

def test_cpcv_on_autocorrelated_series():
    r = _ar1_returns(209)                          # positive mean, phi=0.3 serial dependence
    full = summarize_returns(pd.Series(r), nw_lags=None, months_per_year=12)["sharpe"]
    assert full > 0                                # known: constructed positive
    res = cpcv_evaluate(r, n_groups=8, test_groups=2, purge=3, embargo=6, months_per_year=12)
    assert res.n_folds == 28 and res.n_paths == 7
    assert math.isfinite(res.median_sharpe)
    assert res.median_sharpe > 0                   # folds are subsamples of a positive series
    assert 0.0 <= res.frac_positive <= 1.0
    # purge+embargo must remove some training relative to the naive complement (~157 months).
    assert res.mean_train_months < 209 - 52


def _ar1_returns(n: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    eps = rng.normal(0.0, 1.0, n)
    x = np.empty(n)
    x[0] = eps[0]
    phi = 0.3
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return 0.01 + 0.02 * x                          # mean ~1%/month, sd from the AR(1) shocks
