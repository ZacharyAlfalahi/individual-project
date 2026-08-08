"""Unit test for the profile monthly panel's per-family adjacency return (must
match build_monthly_panel's rule: a return exists only between calendar-
consecutive months with a non-NaN price)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from build_profile_monthly_panel import _add_family_returns


def test_adjacency_return_skips_gap_months():
    panel = pd.DataFrame({
        "cusip_id": ["X"] * 4,
        "year_month": ["2010-01", "2010-02", "2010-04", "2010-05"],  # Mar missing
        "price_eom_bbw_2019": [100.0, 110.0, 121.0, 121.0],
    })
    out = _add_family_returns(panel, "bbw_2019", rf=None)
    r = dict(zip(out["year_month"], out["ret_bbw_2019"]))
    assert r["2010-01"] != r["2010-01"]                    # NaN (no prior)
    assert r["2010-02"] == pytest.approx(0.10)             # consecutive
    assert r["2010-04"] != r["2010-04"]                    # NaN (gap over Mar)
    assert r["2010-05"] == pytest.approx(0.0)              # consecutive


def test_adjacency_return_per_cusip_isolated():
    panel = pd.DataFrame({
        "cusip_id": ["X", "X", "Y", "Y"],
        "year_month": ["2010-01", "2010-02", "2010-02", "2010-03"],
        "price_eom_jostova_2013": [100.0, 105.0, 50.0, 55.0],
    })
    out = _add_family_returns(panel, "jostova_2013", rf=None)
    d = {(c, m): v for c, m, v in zip(out["cusip_id"], out["year_month"],
                                      out["ret_jostova_2013"])}
    assert d[("X", "2010-02")] == pytest.approx(0.05)      # X's own chain
    assert d[("Y", "2010-03")] == pytest.approx(0.10)      # Y's own chain
    assert d[("Y", "2010-02")] != d[("Y", "2010-02")]      # Y first month = NaN
