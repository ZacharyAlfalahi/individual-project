"""Unit tests for scripts/basis_inputs.py (light: no full-panel loads)."""

from __future__ import annotations

import pandas as pd
import pytest

import scripts.basis_inputs as B


def test_bases_and_panel_paths():
    assert B.BASES == ("total_return", "clean")
    tr_panel, tr_prof = B.PANELS["total_return"]
    assert tr_panel.name == "monthly_panel_total_return_default_flat.parquet"
    assert tr_prof.name == "monthly_panel_profiles_total_return_default_flat.parquet"
    cl_panel, cl_prof = B.PANELS["clean"]
    assert cl_panel.name == "monthly_panel_maximal.parquet"
    assert cl_prof.name == "monthly_panel_profiles.parquet"
    for p in (tr_panel, tr_prof, cl_panel, cl_prof):
        assert "holdout" not in p.parts


def test_total_return_panels_match_the_holdout_runner_and_inventory():
    import holdout_inventory as H
    import run_holdout_oos_descriptive as R
    assert B.PANELS["total_return"] == (H.DEV_TR_PANEL, H.DEV_TR_PROFILES)
    assert B.PANELS["total_return"] == (R._DEV_TR_PANEL, R._DEV_TR_PROFILES)


def test_unknown_basis_refused():
    with pytest.raises(B.BasisError, match="basis must be one of"):
        B.check_basis("total")
    with pytest.raises(B.BasisError):
        B.basis_dir("tr")


def test_output_paths_live_under_one_root():
    assert B.basis_dir("clean", "audit") == B.RESULTS_ROOT / "clean" / "audit"
    assert B.factors_dir("total_return") == B.RESULTS_ROOT / "total_return" / "factors"
    assert B.shared_dir("accrual") == B.RESULTS_ROOT / "shared" / "accrual"
    assert B.RESULTS_ROOT.name == "consistent_basis"


def test_assert_dev_only_refuses_holdout_months():
    B.assert_dev_only(pd.date_range("2021-01-31", "2021-12-31", freq="ME"), "ok")
    with pytest.raises(B.BasisError, match="holdout"):
        B.assert_dev_only(pd.date_range("2021-11-30", "2022-01-31", freq="ME"), "leak")
    with pytest.raises(B.BasisError, match="no dated rows"):
        B.assert_dev_only(pd.Series([], dtype="datetime64[ns]"), "empty")


def test_total_return_loader_left_joins_profiles(monkeypatch, tmp_path):
    panel = pd.DataFrame({"cusip": ["A", "B"], "date": pd.to_datetime(["2020-01-31", "2020-01-31"]),
                          "ret_corr": [0.01, 0.02]})
    prof = pd.DataFrame({"cusip": ["A"], "date": pd.to_datetime(["2020-01-31"]), "ret_jostova_2013": [0.03]})
    p1, p2 = tmp_path / "p.parquet", tmp_path / "q.parquet"
    panel.to_parquet(p1)
    prof.to_parquet(p2)
    monkeypatch.setitem(B.PANELS, "total_return", (p1, p2))
    out = B.load_basis_maximal("total_return")
    assert list(out["cusip"]) == ["A", "B"]
    assert out.loc[out.cusip == "A", "ret_jostova_2013"].item() == 0.03
    assert pd.isna(out.loc[out.cusip == "B", "ret_jostova_2013"].item())


def test_total_return_loader_refuses_holdout_rows(monkeypatch, tmp_path):
    panel = pd.DataFrame({"cusip": ["A"], "date": pd.to_datetime(["2022-01-31"]), "ret_corr": [0.01]})
    prof = pd.DataFrame({"cusip": ["A"], "date": pd.to_datetime(["2022-01-31"])})
    p1, p2 = tmp_path / "p.parquet", tmp_path / "q.parquet"
    panel.to_parquet(p1)
    prof.to_parquet(p2)
    monkeypatch.setitem(B.PANELS, "total_return", (p1, p2))
    with pytest.raises(B.BasisError, match="holdout"):
        B.load_basis_maximal("total_return")
