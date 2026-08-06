"""WS-C (P1) — comparator tests: hand-computed rung-3 metrics, verdict tiers,
and the fail-loud thresholds loader."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from evaluation.codegen.scoring import (
    Verdict,
    inter_model_agreement,
    load_scoring_thresholds,
    rung3_series_similarity,
    score_run,
)

_TH = {
    "min_overlap_months": 3,
    "correlation_min": 0.99,
    "sign_agreement_min": 0.95,
    "mean_abs_diff_max": 0.0005,
    "tracking_error_max": 0.001,
    "exact_tier_max_abs_diff": 1.0e-8,
}


def _series(values, start="2010-01-31"):
    idx = pd.date_range(start, periods=len(values), freq="ME")
    return pd.Series(values, index=idx, name="portfolio_return")


def test_rung3_hand_computed():
    oracle = _series([0.01, -0.02, 0.03])
    cand = _series([0.02, -0.01, 0.02])
    m = rung3_series_similarity(oracle, cand)
    assert m["n_overlap"] == 3
    assert m["mean_diff"] == pytest.approx(0.01 / 3)
    assert m["tracking_error"] == pytest.approx(0.011547, abs=1e-5)
    assert m["correlation"] == pytest.approx(0.9176629, abs=1e-4)
    assert m["sign_agreement"] == 1.0
    assert m["max_abs_diff"] == pytest.approx(0.01)


def test_identical_series_scores_runs_right_exact_tier():
    oracle = _series([0.01, -0.02, 0.03, 0.005])
    result = score_run("drf", "ok", None, oracle, oracle.copy(), _TH)
    assert result.verdict is Verdict.RUNS_RIGHT and result.exact_tier
    assert result.failed_criteria == ()


def test_level_shift_fails_mean_abs_diff_only():
    oracle = _series([0.011, -0.02, 0.03, 0.006])
    cand = oracle + 0.001          # 10 bp/mo shift: corr 1, TE 0, signs unchanged
    result = score_run("drf", "ok", None, oracle, cand, _TH)
    assert result.verdict is Verdict.RUNS_WRONG
    assert result.failed_criteria == ("mean_abs_diff",)


def test_noise_fails_tracking_error():
    oracle = _series([0.01, -0.02, 0.03, 0.006, 0.015, -0.011])
    noise = [0.002, -0.002, 0.002, -0.002, 0.002, -0.002]
    cand = oracle + pd.Series(noise, index=oracle.index)
    result = score_run("mom6", "ok", None, oracle, cand, _TH)
    assert result.verdict is Verdict.RUNS_WRONG
    assert "tracking_error" in result.failed_criteria


def test_insufficient_overlap():
    oracle = _series([0.01, -0.02])
    result = score_run("str", "ok", None, oracle, oracle.copy(),
                       {**_TH, "min_overlap_months": 24})
    assert result.verdict is Verdict.RUNS_WRONG
    assert result.failed_criteria == ("insufficient_overlap",)


def test_sandbox_failure_is_wont_run():
    result = score_run("crf", "wont_run", "timeout", None, None, _TH)
    assert result.verdict is Verdict.WONT_RUN and result.reason == "timeout"
    result2 = score_run("crf", "ok", None, _series([0.01]), None, _TH)
    assert result2.verdict is Verdict.WONT_RUN and result2.reason == "no_candidate_series"


def test_month_alignment_tolerates_day_conventions():
    oracle = _series([0.01, -0.02, 0.03])                       # month-end stamps
    cand = pd.Series(oracle.values,
                     index=pd.date_range("2010-01-01", periods=3, freq="MS"))
    m = rung3_series_similarity(oracle, cand)
    assert m["n_overlap"] == 3 and m["max_abs_diff"] == 0.0


def test_degenerate_candidate_correlation_is_nan_and_fails():
    oracle = _series([0.01, -0.02, 0.03, 0.02])
    cand = _series([0.005, 0.005, 0.005, 0.005])                # constant: sd 0
    m = rung3_series_similarity(oracle, cand)
    assert math.isnan(m["correlation"])
    result = score_run("drf", "ok", None, oracle, cand, _TH)
    assert result.verdict is Verdict.RUNS_WRONG and "correlation" in result.failed_criteria


def test_inter_model_agreement_is_symmetric_in_magnitude():
    a = _series([0.01, -0.02, 0.03, 0.006])
    b = _series([0.012, -0.018, 0.028, 0.004])
    ab, ba = inter_model_agreement(a, b), inter_model_agreement(b, a)
    assert ab["correlation"] == pytest.approx(ba["correlation"])
    assert ab["tracking_error"] == pytest.approx(ba["tracking_error"])
    assert ab["mean_diff"] == pytest.approx(-ba["mean_diff"])


def test_real_thresholds_block_loads_and_is_complete():
    block = load_scoring_thresholds()
    assert set(block) >= {"min_overlap_months", "correlation_min", "sign_agreement_min",
                          "mean_abs_diff_max", "tracking_error_max",
                          "exact_tier_max_abs_diff"}


def test_thresholds_loader_fails_loud(tmp_path):
    bad = tmp_path / "thresholds.yaml"
    bad.write_text("other_block: {}\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_scoring_thresholds(bad)
