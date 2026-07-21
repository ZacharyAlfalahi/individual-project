"""Inc2-G — recovery sweep (§10.5) and anchor triangulation (§10.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agents.auditor.validation.anchor_triangulation import (
    AnchorExpectation,
    run_triangulation,
    triangulate_anchor,
)
from agents.auditor.validation.recovery_sweep import (
    DEFAULT_RECOVERY_GRID,
    classify_curve,
    impute_group_b,
    run_recovery_sweep,
)


# --------------------------------------------------------------------------
# Group-B imputation (§10.5.2/§10.5.6)
# --------------------------------------------------------------------------

def _panel():
    return pd.DataFrame({
        "cusip": ["A", "B", "C"],
        "date": pd.to_datetime(["2010-01-31", "2010-01-31", "2010-01-31"]),
        "rf_monthly": [0.001, 0.001, 0.001],
        "exit_reason": ["defaulted", "defaulted", None],
        "ret_raw": [np.nan, -0.2, 0.03],   # A = Group B (NaN); B = Group A (observed)
        "ret_corr": [np.nan, -0.2, 0.03],
        "xret_raw": [np.nan, -0.201, 0.029],
        "xret_corr": [np.nan, -0.201, 0.029],
    })


def test_impute_only_touches_group_b():
    out = impute_group_b(_panel(), rho=0.4)
    # A (Group B) imputed to rho-1 = -0.6
    assert out.loc[0, "ret_corr"] == pytest.approx(-0.6)
    # B (Group A, observed) untouched
    assert out.loc[1, "ret_corr"] == pytest.approx(-0.2)
    # non-distress bond untouched
    assert out.loc[2, "ret_corr"] == pytest.approx(0.03)


def test_impute_rho_one_is_near_zero_loss():
    out = impute_group_b(_panel(), rho=1.0)
    assert out.loc[0, "ret_corr"] == pytest.approx(0.0)


def test_impute_rejects_out_of_range_rho():
    with pytest.raises(ValueError):
        impute_group_b(_panel(), rho=1.5)


# --------------------------------------------------------------------------
# Curve classification (§10.5.4)
# --------------------------------------------------------------------------

def test_classify_sign_stable_monotone():
    assert classify_curve([-1.0, -0.9, -0.8, -0.7]) == "sign_stable"


def test_classify_one_crossover():
    assert classify_curve([-0.5, -0.1, 0.2, 0.5]) == "one_crossover"


def test_classify_non_monotone_sign_stable():
    assert classify_curve([-0.5, -0.9, -0.6, -0.8]) == "non_monotone_sign_stable"


def test_classify_multiple_crossings():
    assert classify_curve([-0.1, 0.1, -0.1, 0.1]) == "multiple_crossings"


# --------------------------------------------------------------------------
# Sweep driver (§10.5.4/§10.5.5)
# --------------------------------------------------------------------------

def test_sweep_over_default_grid_sign_stable():
    # A monotone sign-stable effect fn (more recovery => less negative).
    def eff(rho):
        e = -(1.0 - rho) * 0.1
        return e, e - 0.01, e + 0.01
    res = run_recovery_sweep(eff, headline_rule="FIXED_RECOVERY", headline_rho=0.4)
    assert res.grid == DEFAULT_RECOVERY_GRID
    assert res.sign_stable
    assert res.crossover_rhos == ()
    assert res.to_dict()["headline_rho"] == 0.4


def test_drr_rule_forbids_headline_rho():
    def eff(rho):
        return -0.05, -0.06, -0.04
    # DRR_A5 is price-based: headline_rho must be None (§10.5.5).
    with pytest.raises(ValueError, match="DRR_A5"):
        run_recovery_sweep(eff, headline_rule="DRR_A5", headline_rho=0.4)
    ok = run_recovery_sweep(eff, headline_rule="DRR_A5", headline_rho=None)
    assert ok.headline_rho is None


def test_fixed_recovery_requires_rho():
    def eff(rho):
        return -0.05, -0.06, -0.04
    with pytest.raises(ValueError, match="requires a pre-registered headline_rho"):
        run_recovery_sweep(eff, headline_rule="FIXED_RECOVERY", headline_rho=None)


# --------------------------------------------------------------------------
# Anchor triangulation (§10.4)
# --------------------------------------------------------------------------

def test_locked_anchor_agreement_no_contradiction():
    exp = AnchorExpectation("mom6", "lab_trim", expected_sign=-1,
                            expected_magnitude_range=(0.001, 0.02), expected_sharpe=0.5)
    v = triangulate_anchor(exp, observed_effect=-0.005, observed_sharpe=0.52)
    assert not v.contradiction and v.sign_ok and v.magnitude_ok and v.sharpe_ok


def test_locked_anchor_sign_mismatch_is_contradiction():
    exp = AnchorExpectation("drf", "meas_err", expected_sign=-1,
                            expected_magnitude_range=(0.001, 0.02), expected_sharpe=0.5)
    v = triangulate_anchor(exp, observed_effect=+0.01, observed_sharpe=0.5)
    assert v.contradiction and not v.sign_ok


def test_pilot_anchor_never_contradicts():
    exp = AnchorExpectation("str", "lib_gap", expected_sign=-1,
                            expected_magnitude_range=(0.001, 0.02), expected_sharpe=0.5,
                            is_locked=False)
    v = triangulate_anchor(exp, observed_effect=+99.0, observed_sharpe=99.0)
    assert not v.contradiction  # pilot is descriptive only


def test_triangulated_only_if_no_locked_contradiction():
    good = triangulate_anchor(
        AnchorExpectation("mom6", "lab_trim", -1, (0.001, 0.02), 0.5),
        -0.005, 0.5)
    bad = triangulate_anchor(
        AnchorExpectation("drf", "meas_err", -1, (0.001, 0.02), 0.5),
        +0.01, 0.5)
    assert run_triangulation([good]).externally_triangulated
    assert not run_triangulation([good, bad]).externally_triangulated
