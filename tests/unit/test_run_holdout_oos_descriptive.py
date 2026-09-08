"""Unit tests for the descriptive corrected-anchor OOS runner (scripts/run_holdout_oos_descriptive).

Covers the pure statistics core (Newey-West level with a hand-computed known answer + agreement
with the sanctioned ``summarize_returns``), the descriptive record / persistence Δ, the
all-reported/no-selection assembly, the window slice, the holdout guard, and the refused real mode.
The heavy dev construction (``anchor_cells`` etc.) is exercised by the ``--rehearsal`` run, not here.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import run_holdout_oos_descriptive as R
from agents.quant.library.characteristic_sort import summarize_returns


def _months(n: int, start: str = "2016-01") -> pd.DatetimeIndex:
    return pd.period_range(start, periods=n, freq="M").to_timestamp("M")


# --------------------------------------------------------------------------- nw_level

def test_nw_level_known_answer_lag0():
    # [1,2,3,4,5], NW(0)=HC0: mean 3, S=10/5=2, var=S/T=0.4, se=sqrt(0.4), t=mean/se.
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=_months(5))
    out = R.nw_level(s, nw_lags=0)
    assert out["n_months"] == 5
    assert out["nw_lags_used"] == 0
    assert out["mean_per_month"] == pytest.approx(3.0)
    assert out["mean_pct_per_month"] == pytest.approx(300.0)
    assert out["nw_se"] == pytest.approx(math.sqrt(0.4))
    assert out["nw_t"] == pytest.approx(3.0 / math.sqrt(0.4))
    # 95% CI symmetric around the mean with half-width z*se.
    z = 1.9599639845400545
    assert out["ci_low"] == pytest.approx(3.0 - z * math.sqrt(0.4))
    assert out["ci_high"] == pytest.approx(3.0 + z * math.sqrt(0.4))
    assert (out["ci_low"] + out["ci_high"]) / 2 == pytest.approx(3.0)


def test_nw_level_matches_summarize_returns():
    rng = np.random.default_rng(20260907)
    s = pd.Series(rng.normal(0.002, 0.03, size=48), index=_months(48))
    for lags in (0, 2, None):
        got = R.nw_level(s, nw_lags=lags)["nw_t"]
        want = summarize_returns(s, R._nw_auto_lags(48) if lags is None else lags, 12)["t_stat"]
        assert got == pytest.approx(want, rel=1e-12, abs=1e-12)


def test_nw_level_ci_width_scales_with_alpha():
    s = pd.Series([0.01, -0.02, 0.03, 0.00, 0.015, -0.005], index=_months(6))
    wide = R.nw_level(s, nw_lags=1, alpha=0.10)
    narrow = R.nw_level(s, nw_lags=1, alpha=0.01)
    wide_w = wide["ci_high"] - wide["ci_low"]
    narrow_w = narrow["ci_high"] - narrow["ci_low"]
    assert narrow_w > wide_w   # 99% interval wider than 90%


def test_nw_level_degenerate_constant_series():
    s = pd.Series([0.5] * 6, index=_months(6))
    out = R.nw_level(s)
    assert out["n_months"] == 6
    assert out["mean_per_month"] == pytest.approx(0.5)
    # sd = 0 -> variance 0 -> se/t/CI undefined (None), never a laundered zero.
    assert out["nw_se"] is None and out["nw_t"] is None
    assert out["ci_low"] is None and out["ci_high"] is None


def test_nw_level_short_and_empty():
    one = R.nw_level(pd.Series([0.01], index=_months(1)))
    assert one["n_months"] == 1 and one["mean_per_month"] == pytest.approx(0.01)
    assert one["nw_t"] is None and one["ci_low"] is None
    empty = R.nw_level(pd.Series([], dtype=float))
    assert empty["n_months"] == 0 and empty["mean_per_month"] is None


def test_nw_level_drops_nan():
    s = pd.Series([0.01, np.nan, 0.03, np.nan, 0.05], index=_months(5))
    out = R.nw_level(s, nw_lags=0)
    assert out["n_months"] == 3
    assert out["mean_per_month"] == pytest.approx(0.03)


# --------------------------------------------------------------------- descriptive_record

def test_descriptive_record_persistence_delta():
    s = pd.Series([0.01, 0.02, 0.03], index=_months(3))
    rec = R.descriptive_record(s, in_sample_mean=0.015, label="x", nw_lags=0)
    assert rec["label"] == "x"
    assert rec["in_sample_mean_per_month"] == pytest.approx(0.015)
    assert rec["persistence_delta_per_month"] == pytest.approx(0.02 - 0.015)


def test_descriptive_record_delta_none_when_baseline_absent():
    s = pd.Series([0.01, 0.02, 0.03], index=_months(3))
    rec = R.descriptive_record(s, in_sample_mean=None, label="x", nw_lags=0)
    assert rec["in_sample_mean_per_month"] is None
    assert rec["persistence_delta_per_month"] is None


def test_paired_diff_record_common_months_only():
    a = pd.Series([0.05, 0.06, 0.07], index=_months(3, "2018-01"))
    b = pd.Series([0.01, 0.02], index=_months(2, "2018-01"))   # one fewer month
    rec = R._paired_diff_record(a, b, label="d", nw_lags=0)
    # differential computed only on the 2 common months: (0.05-0.01), (0.06-0.02) -> mean 0.04
    assert rec["n_months"] == 2
    assert rec["mean_per_month"] == pytest.approx(0.04)


# ------------------------------------------------------------------------- build_report

def _fake_series_by_key():
    idx_full = _months(72, "2016-01")          # 2016-01 .. 2021-12
    def s(v):
        return pd.Series(np.full(72, v), index=idx_full)
    by = {}
    for a in R.ANCHOR_ORDER:
        by[a] = {"corrected": s(0.003), "as_published": s(0.001)}
    by[R.NEG_CONTROL_KEY] = {"corrected": s(0.0001)}
    ism = {a: {"corrected": 0.003, "as_published": 0.001} for a in R.ANCHOR_ORDER}
    ism[R.NEG_CONTROL_KEY] = {"corrected": 0.0001}
    return by, ism


def test_build_report_reports_every_row_no_selection():
    by, ism = _fake_series_by_key()
    rep = R.build_report(by, ism, window=R.PSEUDO_WINDOW)
    # every anchor present with all three sub-rows; nothing dropped or ranked
    assert set(rep["anchors"]) == set(R.ANCHOR_ORDER)
    for a in R.ANCHOR_ORDER:
        assert set(rep["anchors"][a]) == {"corrected", "as_published", "corrected_minus_as_published"}
    assert rep["negative_control"] is not None
    assert "no selection" in rep["reporting_rule"]
    assert rep["window"] == {"start": R.PSEUDO_WINDOW[0], "end": R.PSEUDO_WINDOW[1]}


def test_build_report_window_slices_to_pseudo_window():
    by, ism = _fake_series_by_key()
    rep = R.build_report(by, ism, window=R.PSEUDO_WINDOW)
    # 2018-01..2021-09 inclusive = 45 months of the 72-month fake series
    assert rep["anchors"]["str"]["corrected"]["n_months"] == 45


def test_build_report_differential_is_corrected_minus_as_published():
    by, ism = _fake_series_by_key()
    rep = R.build_report(by, ism, window=R.PSEUDO_WINDOW)
    diff = rep["anchors"]["drf"]["corrected_minus_as_published"]
    assert diff["mean_per_month"] == pytest.approx(0.003 - 0.001)
    # differential's in-sample baseline = corrected_ism - as_published_ism
    assert diff["in_sample_mean_per_month"] == pytest.approx(0.003 - 0.001)


def test_build_report_survives_near_zero_row():
    # A degenerate (constant) as-published series must still be REPORTED, not dropped.
    by, ism = _fake_series_by_key()
    rep = R.build_report(by, ism, window=R.PSEUDO_WINDOW)
    pub = rep["anchors"]["mom6"]["as_published"]
    assert pub["nw_t"] is None            # constant -> undefined t
    assert pub["mean_per_month"] is not None   # but the row is present with its level


# ------------------------------------------------------------------- guards / refusal

def test_guard_no_holdout_rejects_holdout_path():
    with pytest.raises(RuntimeError, match="holdout"):
        R._guard_no_holdout([Path("data/holdout/monthly_panel.parquet")])
    # a development path passes cleanly
    R._guard_no_holdout([Path("data/development/monthly_panel_maximal.parquet")])


def test_assert_dev_only_boundary():
    # pre-holdout dates pass; a 2022 month raises; empty raises.
    R._assert_dev_only(_months(6, "2021-04"), "ok")            # ends 2021-09 < floor
    with pytest.raises(RuntimeError, match="holdout"):
        R._assert_dev_only(_months(3, "2021-12"), "leak")      # spans into 2022-02
    with pytest.raises(RuntimeError, match="no dated rows"):
        R._assert_dev_only(pd.Series([], dtype="datetime64[ns]"), "empty")


def test_main_refuses_real_run(capsys):
    rc = R.main([])
    captured = capsys.readouterr()
    assert rc == 2
    assert "REFUSED" in captured.err


def test_pinned_baselines_match_committed_artifacts():
    # The hard-coded baselines must equal the recorded corrected_parent_mean_per_month values.
    import json
    root = Path(R._REPO_ROOT)
    str_artifact = root / "results" / "scientist" / "rq4_funnel" / "rq4_funnel_reported.json"
    cap_artifacts = {a: root / "results" / "scientist" / "rq4_capability" /
                     f"rq4_capability_{a}_minilm.json" for a in ("drf", "mom6")}
    if not (str_artifact.exists() and all(p.exists() for p in cap_artifacts.values())):
        pytest.skip("recorded funnel/capability artifacts not shipped with the repository")
    with open(str_artifact) as f:
        assert json.load(f)["corrected_parent_mean_per_month"] == pytest.approx(
            R.PINNED_CORRECTED_IN_SAMPLE["str"], rel=0, abs=0)
    for a in ("drf", "mom6"):
        p = cap_artifacts[a]

        def _find(o):
            if isinstance(o, dict):
                if "corrected_parent_mean_per_month" in o:
                    return o["corrected_parent_mean_per_month"]
                for v in o.values():
                    r = _find(v)
                    if r is not None:
                        return r
            return None
        with open(p) as f:
            assert _find(json.load(f)) == pytest.approx(R.PINNED_CORRECTED_IN_SAMPLE[a], rel=0, abs=0)


def test_verify_pinned_baselines_raises_on_drift():
    good = dict(R.PINNED_CORRECTED_IN_SAMPLE)
    R._verify_pinned_baselines(good)   # exact -> no raise
    bad = dict(good)
    bad["str"] = good["str"] + 1e-6    # beyond atol
    with pytest.raises(RuntimeError, match="drifted"):
        R._verify_pinned_baselines(bad)
