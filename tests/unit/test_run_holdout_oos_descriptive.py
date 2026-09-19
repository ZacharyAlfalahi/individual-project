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


def _open_gate_but_forbid_any_read(monkeypatch):
    """Open the gate (token + env) and make every step that could read the holdout — or run the
    heavy dev precondition — fail the test if reached, so a refusal is proven to be PRE-open."""
    import holdout_inventory as H
    monkeypatch.setenv(H._GATE_ENV, "1")

    def _forbidden(*_a, **_k):
        raise AssertionError("pre-open guard bypassed: holdout / precondition step was reached")
    monkeypatch.setattr(H, "selfcheck_on_dev", _forbidden)
    monkeypatch.setattr(H, "load_holdout_inputs", _forbidden)
    return H


def test_real_refuses_before_open_on_an_unset_pin(monkeypatch, tmp_path):
    H = _open_gate_but_forbid_any_read(monkeypatch)
    monkeypatch.setattr(R, "PINNED_AS_PUBLISHED_IN_SAMPLE",
                        {**R.PINNED_AS_PUBLISHED_IN_SAMPLE, "mom6": None})
    with pytest.raises(H.HoldoutGateError, match="unset pins"):
        R.run_real(gate=H._GATE_TOKEN, out_dir=tmp_path)


def test_real_refuses_before_open_when_the_artifact_exists(monkeypatch, tmp_path):
    H = _open_gate_but_forbid_any_read(monkeypatch)
    monkeypatch.setattr(R, "_missing_pins", lambda: [])
    (tmp_path / "oos_descriptive_holdout.json").write_text("{}")
    with pytest.raises(H.HoldoutGateError, match="never overwritten"):
        R.run_real(gate=H._GATE_TOKEN, out_dir=tmp_path)
    assert (tmp_path / "oos_descriptive_holdout.json").read_text() == "{}"


def test_real_refuses_write_false_before_open(monkeypatch, tmp_path):
    # A real run that writes nothing would read the holdout and leave no record — refused pre-open.
    H = _open_gate_but_forbid_any_read(monkeypatch)
    monkeypatch.setattr(R, "_missing_pins", lambda: [])
    with pytest.raises(H.HoldoutGateError, match="write=False"):
        R.run_real(gate=H._GATE_TOKEN, out_dir=tmp_path, write=False)


def test_real_refuses_before_open_without_a_cutoff_pin_for_every_percentile_anchor(monkeypatch, tmp_path):
    # mom6 carries a percentile trim; with its dev cutoff pins removed the run must refuse before the
    # read (not raise a KeyError inside _real_substrate after the holdout is open).
    H = _open_gate_but_forbid_any_read(monkeypatch)
    monkeypatch.setattr(R, "_missing_pins", lambda: [])
    monkeypatch.setattr(R, "PINNED_TRIM_CUTOFF_DEV", {"clean": {}, "total_return": {}})
    with pytest.raises(H.HoldoutGateError, match="trim cutoff pin"):
        R.run_real(gate=H._GATE_TOKEN, out_dir=tmp_path)


def test_pin_checks_refuse_nan_on_either_side():
    with pytest.raises(RuntimeError, match="refusing to report"):
        R._check_close(0.1, float("nan"), 1e-6, "nan pin")
    with pytest.raises(RuntimeError, match="drifted"):
        R._verify_pinned_baselines({a: float("nan") for a in R.ANCHOR_ORDER},
                                   dict(R.PINNED_CORRECTED_IN_SAMPLE))


def test_pinned_baselines_match_recorded_artifacts():
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


# ------------------------------------------------------- total-return companion

def test_verify_pinned_baselines_skips_none_pins():
    # A not-yet-pinned substrate (pin value None) is SKIPPED, never silently graded; the label is
    # surfaced in the drift message for the substrate that IS pinned.
    computed = {a: 0.5 for a in R.ANCHOR_ORDER}
    R._verify_pinned_baselines(computed, {a: None for a in R.ANCHOR_ORDER}, label="total_return")
    partial = {**{a: None for a in R.ANCHOR_ORDER}, "drf": 0.5}
    R._verify_pinned_baselines(computed, partial, label="total_return")   # drf matches -> no raise
    with pytest.raises(RuntimeError, match=r"\[total_return\].*drifted"):
        R._verify_pinned_baselines({**computed, "drf": 0.6}, partial, label="total_return")


def test_every_pin_is_set_and_finite():
    # The gated real run refuses before opening while any pin is unset; guard that every
    # pin — both substrates, corrected / as-published / negative control / dev trim cutoff — is a
    # finite float, so the run self-verifies instead of degrading or refusing.
    assert R._missing_pins() == []
    for pins in (R.PINNED_CORRECTED_IN_SAMPLE, R.PINNED_AS_PUBLISHED_IN_SAMPLE,
                 R.PINNED_CORRECTED_IN_SAMPLE_TOTAL_RETURN, R.PINNED_AS_PUBLISHED_IN_SAMPLE_TOTAL_RETURN):
        for a in R.ANCHOR_ORDER:
            assert isinstance(pins[a], float) and math.isfinite(pins[a])
    for v in (R.PINNED_NEG_CONTROL_IN_SAMPLE, R.PINNED_NEG_CONTROL_IN_SAMPLE_TOTAL_RETURN,
              R.PINNED_TRIM_CUTOFF_DEV["clean"]["mom6"], R.PINNED_TRIM_CUTOFF_DEV["total_return"]["mom6"]):
        assert isinstance(v, float) and math.isfinite(v)


def test_dev_total_return_panels_are_default_flat_and_match_inventory():
    import holdout_inventory as H
    assert R._DEV_TR_PANEL == H.DEV_TR_PANEL
    assert R._DEV_TR_PROFILES == H.DEV_TR_PROFILES
    assert R._DEV_TR_PANEL.name == "monthly_panel_total_return_default_flat.parquet"
    assert R._DEV_TR_PROFILES.name == "monthly_panel_profiles_total_return_default_flat.parquet"


def test_check_close_refuses_drift_nan_and_missing_pin():
    R._check_close(0.1, 0.1 + 5e-7, 1e-6, "within atol")
    for got, want in ((0.1, 0.2), (float("nan"), 0.1), (0.1, None)):
        with pytest.raises(RuntimeError, match="refusing to report"):
            R._check_close(got, want, 1e-6, "bad")


def test_real_substrate_verifies_trimmed_as_published_at_seeded_cutoff(monkeypatch):
    """D4: mom6's as-published cutoff is resolved on the seeded panel, so its seeded dev half is
    checked against the dev-only cell rebuilt at the SEEDED cutoff — not against the dev pin (built
    at the dev-only cutoff). The persistence-Δ reference stays the dev pin. Every other cell keeps
    its strict pin check."""
    idx = pd.period_range("2021-07", periods=12, freq="M").to_timestamp("M")   # 6 dev + 6 holdout

    def ser(dev_v, hold_v):
        return pd.Series([dev_v] * 6 + [hold_v] * 6, index=idx)
    cells = {"str": {"corrected": ser(0.01, 9.0), "as_published": ser(0.02, 9.0)},
             "drf": {"corrected": ser(0.03, 9.0), "as_published": ser(0.04, 9.0)},
             "mom6": {"corrected": ser(-0.01, 9.0), "as_published": ser(0.005, 9.0)}}
    cutoff = {"trim_rule_absolute": {"method": "truncate",
                                     "bounds": {"type": "absolute", "hi": 0.13}}, "realised": 0.13}
    monkeypatch.setattr(R, "anchor_cells", lambda a, m, s, allow_holdout: cells[a])
    monkeypatch.setattr(R, "negative_control_corrected", lambda m, s, allow_holdout: ser(0.0, 9.0))
    monkeypatch.setattr(R, "as_published_trim_cutoff",
                        lambda a, m, s: cutoff if a == "mom6" else None)
    seen, loads = {}, []

    def dev_loader():
        loads.append(1)
        return "dev_maximal", "dev_signals"

    def at_cutoff(a, m, s, trim_abs):
        seen["args"] = (a, m, s, trim_abs["bounds"]["hi"])
        return pd.Series([0.005] * 6, index=idx[:6])
    monkeypatch.setattr(R, "as_published_series_at_cutoff", at_cutoff)
    kw = dict(pinned_corrected={"str": 0.01, "drf": 0.03, "mom6": -0.01},
              pinned_as_published={"str": 0.02, "drf": 0.04, "mom6": 0.0064},   # mom6 pin != 0.005
              pinned_neg_control=0.0, pinned_trim_cutoff={"mom6": 0.1298}, dev_loader=dev_loader)

    _series, ism, sv = R._real_substrate("seeded_maximal", "seeded_signals", **kw)
    assert loads == [1]
    assert seen["args"] == ("mom6", "dev_maximal", "dev_signals", 0.13)
    mom6 = sv["mom6"]["as_published"]
    assert mom6["dev_only_cell_mean_at_seeded_cutoff"] == pytest.approx(0.005)
    assert mom6["seeded_trim_cutoff"] == 0.13 and mom6["dev_trim_cutoff_pinned"] == 0.1298
    assert ism["mom6"]["as_published"] == 0.0064                       # Δ reference = dev pin
    assert ism["mom6"]["as_published_dev_mean_at_seeded_cutoff"] == pytest.approx(0.005)
    assert sv["str"]["as_published"]["committed_baseline"] == 0.02     # strict pin path

    monkeypatch.setattr(R, "as_published_series_at_cutoff",
                        lambda a, m, s, t: pd.Series([0.006] * 6, index=idx[:6]))
    with pytest.raises(RuntimeError, match="seeded cutoff"):
        R._real_substrate("seeded_maximal", "seeded_signals", **kw)
    monkeypatch.setattr(R, "as_published_series_at_cutoff", at_cutoff)
    with pytest.raises(RuntimeError, match=r"str\.corrected"):
        R._real_substrate("seeded_maximal", "seeded_signals",
                          **{**kw, "pinned_corrected": {"str": 0.5, "drf": 0.03, "mom6": -0.01}})


def test_substrate_in_sample_means_pinned_else_computed():
    idx = _months(6, "2016-01")
    by = {a: {"corrected": pd.Series(np.full(6, 0.02), index=idx),
              "as_published": pd.Series(np.full(6, 0.01), index=idx)} for a in R.ANCHOR_ORDER}
    by[R.NEG_CONTROL_KEY] = {"corrected": pd.Series(np.full(6, 0.005), index=idx)}
    out = R._substrate_in_sample_means(
        by, allow_holdout=False,
        pinned_corrected={a: 0.999 for a in R.ANCHOR_ORDER},   # pin present -> used verbatim
        pinned_as_published={}, pinned_neg_control=None)        # absent -> computed dev mean
    for a in R.ANCHOR_ORDER:
        assert out[a]["corrected"] == pytest.approx(0.999)         # pinned
        assert out[a]["as_published"] == pytest.approx(0.01)       # computed
    assert out[R.NEG_CONTROL_KEY]["corrected"] == pytest.approx(0.005)   # computed


def test_to_total_return_zero_coupon_invariance_and_schema():
    """Known-answer unit test for the shared transform: a zero-coupon bond's total return equals
    its clean return byte-for-byte (accrual must not touch it); a coupon bond's differs; schema is
    preserved plus the accrual columns. No big-data dependency."""
    import build_total_return_panel as BTR
    from agents.quant.library.accrual import THIRTY_360

    idx = _months(3, "2018-01")
    def bond(cusip, prices):
        p = np.array(prices, dtype=float)
        ret = np.r_[np.nan, p[1:] / p[:-1] - 1.0]
        return pd.DataFrame({
            "cusip": cusip, "date": idx, "maturity": pd.Timestamp("2030-01-31"),
            "rf_monthly": 0.001, "price_eom_raw": p, "price_eom_corr": p,
            "ret_raw": ret, "ret_corr": ret, "xret_raw": ret - 0.001, "xret_corr": ret - 0.001})
    maximal = pd.concat([bond("ZZZ", [100, 101, 102]),   # coupon 0 -> invariant
                         bond("CCC", [100, 100, 100])],   # coupon-bearing -> AI/coupon move it
                        ignore_index=True)
    fisd = pd.DataFrame({"cusip": ["ZZZ", "CCC"], "coupon": [0.0, 6.0],
                         "interest_frequency": [2, 2], "day_count_basis": [THIRTY_360, THIRTY_360]})

    out = BTR.to_total_return(maximal, fisd)
    # schema: original columns preserved, schedule fields dropped, accrual columns added
    assert set(maximal.columns).issubset(set(out.columns))
    assert not ({"coupon", "interest_frequency", "day_count_basis"} & set(out.columns))
    assert {"ai", "coupon_paid", "day_count_fallback"}.issubset(out.columns)
    o = out.set_index(["cusip", "date"]).sort_index()
    m = maximal.set_index(["cusip", "date"]).sort_index()
    # Z bond: total == clean, exactly (finite months only)
    z_tot, z_clean = o.loc["ZZZ", "ret_raw"].to_numpy(), m.loc["ZZZ", "ret_raw"].to_numpy()
    fin = np.isfinite(z_clean)
    assert np.allclose(z_tot[fin], z_clean[fin], atol=1e-12, rtol=0)
    # Coupon bond with flat price: clean return 0, but total return strictly positive (AI + coupon)
    c_tot = o.loc["CCC", "ret_raw"].to_numpy()
    assert c_tot[-1] > 0.0


def test_to_total_return_missing_columns_raises():
    import build_total_return_panel as BTR
    bad = pd.DataFrame({"cusip": ["A"], "date": [pd.Timestamp("2018-01-31")]})
    with pytest.raises(KeyError, match="missing required columns"):
        BTR.to_total_return(bad, pd.DataFrame({"cusip": ["A"], "coupon": [0.0],
                                               "interest_frequency": [2], "day_count_basis": ["x"]}))
