"""
Unit tests for scripts/run_dickerson_gross_error_check.py (E13, descriptive).

Covers:
  - External column resolution: exact beats contains, missing and ambiguous
    never guess and never raise.
  - Scale detection: decimal vs percent vs %-suffixed strings.
  - Printed-window fenceposts (2004-08 / 2016-12 inclusive; 149 months) and
    the drf printed-figure comparison semantics.
  - Sign-corrected str handling: the leg-convention orientation un-flips the
    README-documented -1 presentation flip.
  - Month-grain join exactness, insufficient-overlap flagging, zero-mean
    ratio guard.
  - Structural certification against README-documented facts.
  - main() exits 0 in all modes; the headline envelope; the never-wired
    tripwire on run_validation_gates.py.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_dickerson_gross_error_check as gec  # noqa: E402


def _ym_series(start: str, n: int, values) -> pd.Series:
    idx = pd.period_range(start, periods=n, freq="M").astype(str)
    return pd.Series(values, index=idx)


PRINTED_WINDOW_MONTHS = 149  # 2004-08..2016-12 inclusive


# ---------------------------------------------------------------------------
# Column resolution
# ---------------------------------------------------------------------------

def test_find_column_exact_beats_contains():
    name, status = gec.find_external_column(["mom6_1", "mom6_spread"], ["mom6_1"])
    assert (name, status) == ("mom6_1", "matched")
    name, status = gec.find_external_column(["strev", "str*"], ["str*", "str"])
    assert (name, status) == ("str*", "matched")


def test_find_column_single_contains_fallback():
    name, status = gec.find_external_column(["mom6_1x"], ["mom6_1"])
    assert (name, status) == ("mom6_1x", "matched")


def test_find_column_missing():
    name, status = gec.find_external_column(["cs", "ytm"], ["str*", "str"])
    assert (name, status) == (None, "missing")


def test_find_column_ambiguous_never_guesses():
    name, status = gec.find_external_column(["str_a", "str_b"], ["str"])
    assert (name, status) == (None, "ambiguous")


# ---------------------------------------------------------------------------
# Scale detection
# ---------------------------------------------------------------------------

def test_detect_external_scale_decimal():
    scale, label = gec.detect_external_scale(pd.Series([0.005, -0.004, 0.01]))
    assert (scale, label) == (1.0, "decimal")


def test_detect_external_scale_percent():
    scale, label = gec.detect_external_scale(pd.Series([0.5, -0.4, 1.1]))
    assert (scale, label) == (0.01, "percent")


def test_detect_external_scale_percent_strings():
    scale, label = gec.detect_external_scale(pd.Series(["0.5%", "-0.4%", "1.1%"]))
    assert (scale, label) == (0.01, "percent")


# ---------------------------------------------------------------------------
# Printed window
# ---------------------------------------------------------------------------

def test_window_slice_fenceposts():
    s = _ym_series("2004-07", 151, 1.0)  # 2004-07 .. 2017-01
    sliced = gec._window_slice(s, gec.DRR2023_T1PB_WINDOW)
    assert len(sliced) == PRINTED_WINDOW_MONTHS
    assert sliced.index.min() == "2004-08"
    assert sliced.index.max() == "2016-12"


def test_drf_printed_window_comparison_matching_mean():
    ours = _ym_series("2004-08", PRINTED_WINDOW_MONTHS, 0.00673)  # decimal
    out = gec.drf_printed_window_comparison(ours, "drf_corr")
    assert out["window_complete"] is True
    assert out["n_months"] == PRINTED_WINDOW_MONTHS
    assert out["sign_agreement"] is True
    assert out["abs_mean_ratio_ours_over_printed"] == pytest.approx(1.0, rel=1e-6)


def test_drf_printed_window_comparison_wrong_sign_reported_not_fatal():
    ours = _ym_series("2004-08", PRINTED_WINDOW_MONTHS, -0.005)
    out = gec.drf_printed_window_comparison(ours, "drf_corr")
    assert out["sign_agreement"] is False
    assert out["abs_mean_ratio_ours_over_printed"] is not None


# ---------------------------------------------------------------------------
# Series comparison
# ---------------------------------------------------------------------------

def test_compare_factor_sign_corrected_str_leg_convention():
    """The external `str*` is -1 x the raw winners-minus-losers series; the
    leg-convention orientation must un-flip it before signs are compared."""
    rng = np.random.default_rng(0)
    ours = _ym_series("2005-01", 60, rng.normal(0.01, 0.02, 60))
    theirs_published = -ours  # perfectly comoving in leg convention
    out = gec.compare_factor(ours, theirs_published, "str", "str*")
    assert out["external_sign_corrected"] is True
    assert out["corr_signed_as_published"] == pytest.approx(-1.0)
    assert out["corr_signed_leg_convention"] == pytest.approx(1.0)
    assert out["abs_corr"] == pytest.approx(1.0)
    assert out["sign_agreement_leg_convention"] is True
    assert out["abs_mean_ratio"] == pytest.approx(1.0)
    assert "note" in out


def test_compare_factor_month_grain_join():
    ours = _ym_series("2005-01", 12, 0.01)
    theirs = _ym_series("2005-07", 12, 0.02)
    out = gec.compare_factor(ours, theirs, "mom6", "mom6_1")
    assert out["n_matched_months"] == 6
    assert out["matched_first_month"] == "2005-07"
    assert out["matched_last_month"] == "2005-12"
    assert out["external_sign_corrected"] is False


def test_compare_factor_insufficient_overlap_flagged():
    ours = _ym_series("2005-01", 10, 0.01)
    theirs = _ym_series("2005-01", 10, 0.02)
    out = gec.compare_factor(ours, theirs, "mom6", "mom6_1")
    assert out["overlap_sufficient"] is False
    assert out["mean_ours_pct"] is not None


def test_compare_factor_zero_external_mean_ratio_none():
    ours = _ym_series("2005-01", 10, 0.01)
    theirs = _ym_series("2005-01", 10, [0.01, -0.01] * 5)
    out = gec.compare_factor(ours, theirs, "mom6", "mom6_1")
    assert out["abs_mean_ratio"] is None


def test_compare_factor_no_overlap():
    ours = _ym_series("2005-01", 6, 0.01)
    theirs = _ym_series("2010-01", 6, 0.02)
    out = gec.compare_factor(ours, theirs, "mom6", "mom6_1")
    assert out["n_matched_months"] == 0
    assert "mean_ours_pct" not in out


# ---------------------------------------------------------------------------
# Structural certification
# ---------------------------------------------------------------------------

def _synthetic_external(n_filler: int = 105) -> pd.DataFrame:
    months = pd.period_range("1973-02", "2021-12", freq="M").astype(str)
    df = pd.DataFrame({"year_month": months})
    cs = pd.Series(0.001, index=range(len(months)))
    gap_pos = [i for i, m in enumerate(months) if m in gec.LBFI_GAP_MONTHS]
    cs.iloc[gap_pos] = np.nan
    df["cs"] = cs.values
    df["str*"] = np.where(months >= "2002-10", 0.01, np.nan)
    df["mom6_1"] = 0.002
    for i in range(n_filler):
        df[f"filler_{i}"] = 0.0
    return df


def test_structural_certification_all_pass():
    ext = _synthetic_external()
    matched = {"str": "str*", "mom6": "mom6_1"}
    out = gec.structural_certification(ext, matched)
    assert out["checks"]["factor_column_count"]["pass"] is True
    assert out["checks"]["boundary_month"]["pass"] is True
    assert out["checks"]["lbfi_gap_months_nan"]["pass"] is True
    assert out["checks"]["units_decimal"]["pass"] is True
    assert out["all_pass"] is True
    assert out["first_non_nan_month_observational"]["str"] == "2002-10"


def test_structural_certification_wrong_column_count_fails():
    ext = _synthetic_external(n_filler=100)
    out = gec.structural_certification(ext, {"str": "str*", "mom6": "mom6_1"})
    assert out["checks"]["factor_column_count"]["pass"] is False
    assert out["all_pass"] is False


def test_structural_certification_missing_column_recorded():
    ext = _synthetic_external()
    out = gec.structural_certification(ext, {"str": None, "mom6": "mom6_1"})
    assert out["checks"]["required_columns_matched"]["pass"] is False
    assert out["all_pass"] is False


# ---------------------------------------------------------------------------
# main() — always exits 0; envelope
# ---------------------------------------------------------------------------

def _patch_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(gec, "EXTERNAL_FILE", tmp_path / "external.parquet")
    monkeypatch.setattr(gec, "FACTORS_DIR", tmp_path / "factors")
    monkeypatch.setattr(gec, "OUT", tmp_path / "headlines" / "dickerson_gross_error.json")
    (tmp_path / "factors").mkdir()


def test_main_inputs_missing_exits_zero(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as exc:
        gec.main()
    assert exc.value.code == 0
    report = json.loads((tmp_path / "headlines" / "dickerson_gross_error.json").read_text())
    assert report["status"] == "inputs_missing"
    assert report["gate"]["gating"] is False
    assert "run_timestamp" in report and "thresholds_sha256" in report


def test_main_happy_path_exits_zero(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    _synthetic_external().to_parquet(tmp_path / "external.parquet")
    dates = pd.period_range("2004-08", periods=36, freq="M").to_timestamp(how="end")
    pd.DataFrame({"date": dates, "str_raw": 0.01, "str_corr": 0.012}).to_parquet(
        tmp_path / "factors" / "str.parquet")
    pd.DataFrame({"date": dates, "mom6_raw": 0.002, "mom6_corr": 0.003}).to_parquet(
        tmp_path / "factors" / "mom6.parquet")
    pd.DataFrame({"date": dates, "drf_raw": 0.005, "drf_corr": 0.0067}).to_parquet(
        tmp_path / "factors" / "bbw_factors.parquet")

    with pytest.raises(SystemExit) as exc:
        gec.main()
    assert exc.value.code == 0

    report = json.loads((tmp_path / "headlines" / "dickerson_gross_error.json").read_text())
    assert report["gate"]["gating"] is False
    assert report["post_boundary_rows_dropped_on_load"] == 0
    assert report["structural_certification"]["all_pass"] is True
    assert report["drf_printed_figure"]["window_complete"] is False  # 36 of 149 mo
    arms = report["series_comparisons"]["str"]["arms"]
    assert arms["str_corr"]["n_matched_months"] == 36
    assert arms["str_corr"]["external_sign_corrected"] is True
    assert (tmp_path / "headlines").glob("*.tmp") is not None
    assert list((tmp_path / "headlines").glob("*.tmp")) == []


def test_main_stale_external_rows_dropped_on_load(monkeypatch, tmp_path):
    """A foreign/stale parquet with holdout-era rows is loudly recorded and
    the rows never reach a comparison."""
    _patch_paths(monkeypatch, tmp_path)
    ext = _synthetic_external()
    extra = ext.tail(1).copy()
    extra["year_month"] = "2023-06"
    pd.concat([ext, extra]).to_parquet(tmp_path / "external.parquet")
    dates = pd.period_range("2004-08", periods=30, freq="M").to_timestamp(how="end")
    for name, cols in (("str.parquet", {"str_raw": 0.01, "str_corr": 0.012}),
                       ("mom6.parquet", {"mom6_raw": 0.002, "mom6_corr": 0.003}),
                       ("bbw_factors.parquet", {"drf_raw": 0.005, "drf_corr": 0.0067})):
        pd.DataFrame({"date": dates, **cols}).to_parquet(tmp_path / "factors" / name)

    with pytest.raises(SystemExit) as exc:
        gec.main()
    assert exc.value.code == 0
    report = json.loads((tmp_path / "headlines" / "dickerson_gross_error.json").read_text())
    assert report["post_boundary_rows_dropped_on_load"] == 1
    for arm in report["series_comparisons"]["str"]["arms"].values():
        last = arm.get("matched_last_month")
        assert last is None or last <= "2021-12"


def test_never_wired_into_validation_gates():
    """TRIPWIRE — E13 is descriptive and must never enter the gate driver."""
    src = (REPO_ROOT / "scripts" / "run_validation_gates.py").read_text().lower()
    assert "dickerson_gross_error" not in src
    assert "dickerson_factor_returns" not in src


# ---------------------------------------------------------------------------
# Real-artifact smoke (skipped until the downloader has run)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not gec.EXTERNAL_FILE.exists(),
                    reason="dickerson_factor_returns.parquet not downloaded")
def test_real_external_artifact_dev_only_and_resolvable():
    df = pd.read_parquet(gec.EXTERNAL_FILE)
    assert df["year_month"].max() <= "2021-12"
    for factor, candidates in gec.EXTERNAL_COLUMNS.items():
        name, status = gec.find_external_column(list(df.columns), candidates)
        assert status == "matched", f"{factor}: {status}"
