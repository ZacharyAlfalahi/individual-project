"""
Tests for shared/stats/pbo.py — the net-new CSCV probability of backtest overfitting (WS-D):
pure-noise calibration near 1/2, a dominant candidate at exactly 0, a hand-enumerated
six-fold regime flip with a known exact answer, and the fail-loud contract.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shared.stats.pbo import PBOResult, pbo_cscv  # noqa: E402


def _noise_matrix(seed: int, t: int = 200, n: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0.0, 1.0, (t, n)), columns=[str(j) for j in range(n)])


# ---- Calibration: pure noise has no persistent winner -> PBO near 1/2 ----------------------

def test_pure_noise_pbo_near_half():
    res = pbo_cscv(_noise_matrix(0))
    assert isinstance(res, PBOResult)
    assert res.n_folds == 28                       # C(8, 2) over the default geometry
    assert 0.2 <= res.pbo <= 0.8
    mean_pbo = float(np.mean([pbo_cscv(_noise_matrix(seed)).pbo for seed in range(20)]))
    assert 0.35 <= mean_pbo <= 0.65


# ---- A genuinely dominant candidate is never flagged as overfitting ------------------------

def test_dominant_candidate_pbo_zero():
    rng = np.random.default_rng(0)
    t = 200
    cols = {"0": 0.05 + 0.001 * rng.normal(size=t)}
    for j in range(1, 10):
        cols[str(j)] = 0.01 * rng.normal(size=t)
    res = pbo_cscv(pd.DataFrame(cols))
    assert res.pbo == 0.0
    # "0" wins in-sample in every fold; the tally is complete over candidates.
    assert res.is_best_counts["0"] == res.n_folds
    assert sum(res.is_best_counts.values()) == res.n_folds
    assert res.median_oos_relative_rank == pytest.approx(10 / 11)  # OOS-best too, every fold


# ---- Hand-computed regime flip (verified against cpcv_folds' actual partition) -------------

def test_hand_computed_regime_flip():
    """T=8, n_groups=4, test_groups=2, purge=embargo=0. partition_groups(8, 4) gives
    G0={0,1}, G1={2,3}, G2={4,5}, G3={6,7}; combinations order gives six folds with test
    pairs (G0,G1), (G0,G2), (G0,G3), (G1,G2), (G1,G3), (G2,G3) and train = the complement
    (purge=0 removes nothing beyond the test set itself).

    A = (1.0, 1.1, 0.9, 1.05, -1.0, -1.1, -0.9, -1.05) is positive on G0,G1 and negative
    on G2,G3; B = -A mirrors it. Sharpe sign = mean sign, so per fold (means in exact
    arithmetic; A-mean shown, B-mean is its negation):

      fold 0: test G0+G1, train G2+G3. A IS mean -1.0125 -> IS-best B; OOS A +1.0125, so
              B is OOS-worst: omega = 1/3, lambda = ln(1/2) < 0.
      fold 1: test G0+G2, train G1+G3. IS and OOS means are both exactly 0 for A and B
              (regimes cancel); OOS tie -> average rank 1.5 -> omega = 1/2, lambda = 0.
      fold 2: test G0+G3, train G1+G2. A IS mean -0.0375 -> IS-best B; OOS A +0.0375 ->
              omega = 1/3, lambda < 0.
      fold 3: test G1+G2, train G0+G3. A IS mean +0.0375 -> IS-best A; OOS A -0.0375 ->
              omega = 1/3, lambda < 0.
      fold 4: test G1+G3, train G0+G2. IS means exactly 0 -> tie -> A (first column);
              OOS also cancels -> lambda >= 0 (never negative).
      fold 5: test G2+G3, train G0+G1. A IS mean +1.0125 -> IS-best A; OOS A -1.0125 ->
              omega = 1/3, lambda < 0.

    Folds 0, 2, 3, 5 have lambda < 0; folds 1 and 4 do not -> PBO = 4/6 = 2/3 exactly.
    (Scratch-verified against the code: the exact-zero folds carry a ~5.6e-17 float
    summation residue on one side, which resolves fold 4 to lambda = +ln 2 rather than 0;
    the residue can never make either lambda negative, and both IS ties resolve to A, so
    pbo, the sign pattern, and the tally below are stable.)
    """
    a_vals = (1.0, 1.1, 0.9, 1.05, -1.0, -1.1, -0.9, -1.05)
    frame = pd.DataFrame({"A": a_vals, "B": [-v for v in a_vals]})
    res = pbo_cscv(frame, n_groups=4, test_groups=2)
    assert res.n_folds == 6
    assert res.pbo == 2 / 3
    assert [lam < 0 for lam in res.lambdas] == [True, False, True, True, False, True]
    assert res.is_best_counts == {"A": 4, "B": 2}


# ---- Fail-loud contract --------------------------------------------------------------------

def test_fail_loud():
    rng = np.random.default_rng(0)

    one_candidate = pd.DataFrame({"solo": rng.normal(size=40)})
    with pytest.raises(ValueError, match="candidate"):
        pbo_cscv(one_candidate)

    with_nan = _noise_matrix(0, t=40, n=3)
    with_nan.iloc[5, 1] = np.nan                   # candidate "1"
    with pytest.raises(ValueError, match=r"NaN.*'1'"):
        pbo_cscv(with_nan)

    constant = pd.DataFrame({"noise": rng.normal(size=40), "const": np.full(40, 0.01)})
    with pytest.raises(ValueError, match=r"Sharpe.*'const'"):
        pbo_cscv(constant)
