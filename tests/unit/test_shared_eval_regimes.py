"""
test_shared_eval_regimes.py — evaluation-regime decomposition + falsifiable prediction
contrast (vs corrected parent, D-E16); hit counts only, no interval (D-E15); and the
LOOK-AHEAD GUARD on the deferred expanding past-only median (the current observation must
never enter its own threshold).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shared.evaluation.contracts import RefusalCode
from shared.evaluation.regimes import (
    evaluate_regimes,
    expanding_past_only_median,
    prediction_contrast,
)
from shared.evaluation.reporting_rules import hit_count_sentence
from shared.evaluation.thresholds import RegimesConfig

from _shared_eval_fixtures import FROZEN_MEDIAN, candidate, factor_frame, spread_series  # noqa: E402


def _cfg(median: float = FROZEN_MEDIAN) -> RegimesConfig:
    return RegimesConfig(
        evaluation_median=median,
        evaluation_median_source="test#median",
        macro_data_contract_id="baa_aaa_v1",
        min_obs_conditional=60,
    )


# ---------------------------------------------------------------------------
# look-ahead guard (must fail if the current obs enters its own median)
# ---------------------------------------------------------------------------

def test_expanding_median_excludes_current_observation() -> None:
    idx = pd.date_range("2002-01-31", periods=6, freq="ME")
    spread = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], index=idx)
    thr = expanding_past_only_median(spread, min_history_months=1, lag_months=0)
    assert np.isnan(thr.iloc[0])                       # burn-in: no past history
    # month 1 uses ONLY month 0 (strictly past) -> 10, NOT median{10,20}=15.
    assert thr.iloc[1] == pytest.approx(10.0)
    assert not np.isclose(thr.iloc[1], 15.0)           # the look-ahead value is excluded
    assert thr.iloc[2] == pytest.approx(15.0)          # months {0,1} = median{10,20}


def test_expanding_median_lagged_excludes_recent_and_current() -> None:
    # lag_months=1 (the macro-release-delay case): the threshold at month t may use
    # observations only up to t-1, so both the current AND the just-released month are out.
    idx = pd.date_range("2002-01-31", periods=6, freq="ME")
    spread = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], index=idx)
    thr = expanding_past_only_median(spread, min_history_months=1, lag_months=1)
    # positions [0, i-1): month0 and month1 have no usable history under a 1-month lag.
    assert np.isnan(thr.iloc[0]) and np.isnan(thr.iloc[1])
    # month 2 may use only position 0 -> {10}, NOT median{10,20}=15 (the release lag backs
    # the frontier off one further month than strict-past alone).
    assert thr.iloc[2] == pytest.approx(10.0)
    assert not np.isclose(thr.iloc[2], 15.0)


def test_expanding_median_burn_in() -> None:
    idx = pd.date_range("2002-01-31", periods=6, freq="ME")
    spread = pd.Series(np.arange(6, dtype=float), index=idx)
    thr = expanding_past_only_median(spread, min_history_months=3, lag_months=0)
    assert thr.iloc[:3].isna().all()                   # first 3 below the min-history floor
    assert not np.isnan(thr.iloc[3])


# ---------------------------------------------------------------------------
# prediction contrast (vs corrected parent, signed mean difference)
# ---------------------------------------------------------------------------

def _high_mask(spread: pd.Series, median: float) -> pd.Series:
    return pd.Series(spread.to_numpy() > median, index=spread.index)


def test_prediction_hit_and_miss() -> None:
    fr = factor_frame(T=200, seed=1)
    spread = spread_series(T=200)
    high = _high_mask(spread, FROZEN_MEDIAN)
    parent = candidate(fr, a0=0.001, betas={"mktb": 0.4}, noise=0.0002, seed=2)
    # extension outperforms the parent by +0.01 ONLY in the high state.
    ext = pd.Series(parent.to_numpy() + 0.01 * high.to_numpy(), index=parent.index)

    hit = prediction_contrast(ext, parent, high, predicted_sign=1, mechanically_implied=False,
                              contrast_definition="test")
    assert hit.evaluable and hit.observed_sign == 1 and hit.hit is True
    assert hit.delta_mean_vs_parent_in_target_state == pytest.approx(0.01, abs=1e-9)

    miss = prediction_contrast(ext, parent, high, predicted_sign=-1, mechanically_implied=False,
                               contrast_definition="test")
    assert miss.evaluable and miss.hit is False


def test_prediction_not_evaluable_when_a_state_empty() -> None:
    fr = factor_frame(T=120, seed=3)
    spread = pd.Series(np.full(120, 0.5), index=fr["date"])  # all below median -> no high months
    high = _high_mask(spread, FROZEN_MEDIAN)
    parent = candidate(fr, seed=4)
    ext = candidate(fr, seed=5)
    pa = prediction_contrast(ext, parent, high, predicted_sign=1, mechanically_implied=False,
                             contrast_definition="test")
    assert not pa.evaluable and pa.refusal_code is RefusalCode.PREDICTION_NOT_EVALUABLE


# ---------------------------------------------------------------------------
# evaluate_regimes decomposition + applicability
# ---------------------------------------------------------------------------

def test_decomposition_and_not_applicable_without_prediction() -> None:
    fr = factor_frame(T=200, seed=6)
    spread = spread_series(T=200)
    cand = candidate(fr, a0=0.002, betas={"mktb": 0.3}, noise=0.0004, seed=7)

    # no registered prediction -> applicable=False, but decomposition still populated.
    r = evaluate_regimes(cand, spread, config=_cfg())
    assert not r.applicable and r.prediction.refusal_code is RefusalCode.PREDICTION_NOT_EVALUABLE
    assert r.months_high + r.months_low == 200
    assert r.mean_return_high is not None and r.mean_return_low is not None

    # with a parent + prediction -> applicable.
    parent = candidate(fr, a0=0.001, betas={"mktb": 0.3}, noise=0.0004, seed=8)
    r2 = evaluate_regimes(cand, spread, parent_returns=parent, config=_cfg(), predicted_sign=1)
    assert r2.applicable and r2.prediction.evaluable


def test_mechanically_implied_excluded_from_headline_count() -> None:
    fr = factor_frame(T=200, seed=9)
    spread = spread_series(T=200)
    high = _high_mask(spread, FROZEN_MEDIAN)
    parent = candidate(fr, seed=10)
    ext = pd.Series(parent.to_numpy() + 0.01 * high.to_numpy(), index=parent.index)

    genuine = prediction_contrast(ext, parent, high, predicted_sign=1, mechanically_implied=False, contrast_definition="t")
    mech = prediction_contrast(ext, parent, high, predicted_sign=1, mechanically_implied=True, contrast_definition="t")
    sentence = hit_count_sentence({"reversal": [genuine, mech]})
    # only the genuine one counts -> 1 of 1, not 2 of 2.
    assert "1 of 1" in sentence
