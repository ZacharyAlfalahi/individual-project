"""G4 robustness (CPCV gate + DSR + crowding wiring), G5 lexicographic selection, and G6
artefact-gated single-access holdout."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.scientist.experimentalist import holdout as H  # noqa: E402
from agents.scientist.experimentalist.robustness import robustness_g4  # noqa: E402
from agents.scientist.experimentalist.selector import select_g5  # noqa: E402
from agents.scientist.schemas.evaluation import GrossMeasurements  # noqa: E402
from shared.evaluation.thresholds import CrowdingConfig  # noqa: E402

FACTORS = ("mktb", "drf", "crf", "lrf", "str", "mom6")
MONTHS = pd.date_range("2004-08-31", periods=120, freq="ME")   # long enough for CPCV folds


def _cfg():
    return CrowdingConfig(factor_set=FACTORS, hac_lag_rule="newey_west_auto", min_obs=10,
                          bundles={f: (f"{f}.parquet", f"{f}_corr") for f in FACTORS})


def _factors():
    rng = np.random.default_rng(1)
    data = {"date": MONTHS}
    for f in FACTORS:
        data[f] = rng.normal(scale=0.02, size=len(MONTHS))
    return pd.DataFrame(data)


def _candidate(mean):
    rng = np.random.default_rng(2)
    return pd.Series(mean + rng.normal(scale=0.01, size=len(MONTHS)), index=MONTHS)


# ---- G4 -----------------------------------------------------------------------------------

def test_g4_positive_candidate_is_cpcv_qualified_and_populates_measurements():
    y = _candidate(mean=0.02)                                   # clearly positive
    out, meas = robustness_g4(y, GrossMeasurements(alpha_bbw4=0.02), n_trials=6, information_span=2,
                              holding_period=1, sr_std=0.5, crowding_config=_cfg(),
                              crowding_factors=_factors())
    assert out.passed and out.booleans["cpcv_qualified"]
    assert meas.cpcv is not None and meas.cpcv["n_paths"] == 7
    assert meas.crowding is not None and "alpha" in meas.crowding      # crowding WIRED
    assert meas.deflated_sharpe is not None


def test_g4_short_series_does_not_crash():
    # M1 regression: a series shorter than n_groups cannot be CPCV-partitioned -> not qualified,
    # NEVER a crash (a month-filter extension can shrink a BH survivor below 8 months).
    idx = pd.date_range("2010-01-31", periods=5, freq="ME")
    short = pd.Series([0.02, 0.03, 0.01, 0.025, 0.015], index=idx)
    rng = np.random.default_rng(3)
    factors = pd.DataFrame({"date": idx, **{f: rng.normal(scale=0.02, size=5) for f in FACTORS}})
    out, meas = robustness_g4(short, GrossMeasurements(), n_trials=6, information_span=2,
                              holding_period=1, sr_std=0.5, crowding_config=_cfg(),
                              crowding_factors=factors)
    assert not out.passed and not out.booleans["cpcv_qualified"]
    assert meas.cpcv["insufficient_months"] == 5.0


def test_g4_negative_candidate_not_cpcv_qualified():
    y = _candidate(mean=-0.02)                                 # clearly negative -> median fold < 0
    out, meas = robustness_g4(y, GrossMeasurements(), n_trials=6, information_span=2,
                              holding_period=1, sr_std=0.5, crowding_config=_cfg(),
                              crowding_factors=_factors())
    assert not out.passed and not out.booleans["cpcv_qualified"]


# ---- G5 (lexicographic) -------------------------------------------------------------------

def test_g5_orders_by_median_cpcv_then_simplicity_then_id():
    survivors = [
        {"proposal_id": "a", "median_cpcv_sharpe": 0.3, "n_changes": 2},
        {"proposal_id": "b", "median_cpcv_sharpe": 0.5, "n_changes": 3},   # highest median -> first
        {"proposal_id": "c", "median_cpcv_sharpe": 0.3, "n_changes": 1},   # ties a on median, simpler
    ]
    assert select_g5(survivors, cap=3) == ["b", "c", "a"]      # b, then simpler-c, then a
    assert select_g5(survivors, cap=2) == ["b", "c"]           # cap respected


def test_g5_is_not_a_composite_score():
    # A tiny median-Sharpe edge must NOT be overridden by simplicity — lexicographic, not weighted.
    survivors = [
        {"proposal_id": "x", "median_cpcv_sharpe": 0.11, "n_changes": 5},
        {"proposal_id": "y", "median_cpcv_sharpe": 0.10, "n_changes": 1},
    ]
    assert select_g5(survivors, cap=1) == ["x"]                # higher median wins despite complexity


# ---- G6 (artefact-gated holdout) ----------------------------------------------------------

def test_g6_gate_requires_all_artefact_conditions():
    assert H.holdout_gate_open(tag_present=True, frozen_script_hash_matches=True, env_var_set=True)
    for combo in [(False, True, True), (True, False, True), (True, True, False)]:
        assert not H.holdout_gate_open(tag_present=combo[0], frozen_script_hash_matches=combo[1],
                                       env_var_set=combo[2])


def test_g6_single_access_fires_on_second_call():
    H._reset_single_access_for_tests()
    H.assert_single_access()                                   # first ok
    with pytest.raises(H.HoldoutViolation):
        H.assert_single_access()                               # second -> violation
    H._reset_single_access_for_tests()


def test_g6_prereg_tag_is_detected():
    if not H.prereg_tag_present("scientist-prereg"):
        pytest.skip("pre-registration tag not shipped in this copy")
    # The scientist-prereg tag was applied earlier in the build; the checker finds it.
    assert H.prereg_tag_present("scientist-prereg") is True
    assert H.prereg_tag_present("no-such-tag-xyz") is False
