"""Unit tests for the R1 promotion-diagnostics PURE core (FL-D21e) — synthetic
daily aggregates with hand-traced answers; no raw file touched."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from run_r1_promotion_diagnostics import (
    _monthly_returns,
    compute_criteria,
    monthly_last_day_price,
)

_THRESHOLDS = {
    "diffuse_share_min": 0.005,
    "diffuse_delta_bp": 1.0,
    "max_abs_delta_r": 0.30,
    "selection_lift_min": 2.0,
    "extreme_decile_frac": 0.10,
}


def _daily():
    """Bond X trades once per month Jan-Apr 2010 at 100 (vol 10). Variant 'v'
    perturbs Feb's (only) day so the Feb month-end price becomes 130; variant
    'w' is a no-op. Hand-traced: baseline returns = [0, 0, 0] (Feb, Mar, Apr);
    under v: Feb ret = +0.30, Mar ret = 100/130-1 ≈ -0.2308 → |Δr| max = 0.30."""
    rows = []
    for m, d in [("2010-01", "2010-01-15"), ("2010-02", "2010-02-15"),
                 ("2010-03", "2010-03-15"), ("2010-04", "2010-04-15")]:
        rows.append({
            "cusip": "X", "date": d,
            "base_pxv": 1000.0, "base_vol": 10.0,
            "d_pxv_v": 300.0 if m == "2010-02" else 0.0, "d_vol_v": 0.0,
            "d_pxv_w": 0.0, "d_vol_w": 0.0,
        })
    return pd.DataFrame(rows)


def test_monthly_last_day_price_and_returns():
    daily = _daily()
    px = monthly_last_day_price(daily, "base_pxv", "base_vol")
    assert list(px["px"]) == [100.0] * 4
    ret = _monthly_returns(px)
    assert len(ret) == 3 and all(r == pytest.approx(0.0) for r in ret["ret"])


def test_monthly_returns_skip_gap_months():
    px = pd.DataFrame({
        "cusip": ["X", "X", "X"],
        "month": ["2010-01", "2010-02", "2010-04"],   # Mar missing
        "px": [100.0, 110.0, 121.0],
    })
    ret = _monthly_returns(px)
    # Feb return defined (consecutive); Apr NOT (gap over Mar).
    assert list(ret["month"]) == ["2010-02"]
    assert ret["ret"].iloc[0] == pytest.approx(0.10)


def test_compute_criteria_promotes_on_magnitude():
    daily = _daily()
    cols = {"v": ("d_pxv_v", "d_vol_v"), "w": ("d_pxv_w", "d_vol_w")}
    res = compute_criteria(daily, cols, _THRESHOLDS)
    assert res["n_baseline_bond_months"] == 3
    v = res["variants"]["v"]
    assert v["max_abs_delta_r"] == pytest.approx(0.30)
    assert v["criteria_fired"]["b_magnitude"] is True     # 0.30 >= 0.30
    assert v["criteria_fired"]["a_diffuse"] is True       # 2/3 bond-months >= 1bp
    assert v["promoted"] is True
    w = res["variants"]["w"]
    assert w["n_affected_bond_months"] == 0
    assert w["max_abs_delta_r"] == 0.0
    assert w["promoted"] is False
