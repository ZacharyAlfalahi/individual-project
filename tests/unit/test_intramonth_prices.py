"""
Unit tests for the intra-month begin/end price windows
(agents/quant/library/intramonth_prices.py).

Covers:
  - Begin/end VWAP match a hand-computed answer (Jan 2010, window=5).
  - Window membership uses calendar business-day cutoffs (inclusive boundaries).
  - A month with trades only mid-month yields NaN on both sides.
  - Missing required columns raise.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from agents.quant.library.intramonth_prices import month_window_prices  # noqa: E402


def _daily(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["trd_exctn_dt"] = pd.to_datetime(df["trd_exctn_dt"])
    return df


def test_begin_end_vwap_hand_computed():
    # Jan 2010 business days: first 5 = Jan 1,4,5,6,7 (cut Jan 7);
    # last 5 = Jan 25,26,27,28,29 (cut Jan 25).
    daily = _daily([
        {"cusip_id": "X", "trd_exctn_dt": "2010-01-04", "price_vwap": 100.0, "total_vol": 10.0},
        {"cusip_id": "X", "trd_exctn_dt": "2010-01-06", "price_vwap": 102.0, "total_vol": 30.0},
        {"cusip_id": "X", "trd_exctn_dt": "2010-01-15", "price_vwap": 105.0, "total_vol": 50.0},  # mid → neither
        {"cusip_id": "X", "trd_exctn_dt": "2010-01-27", "price_vwap": 110.0, "total_vol": 20.0},
        {"cusip_id": "X", "trd_exctn_dt": "2010-01-28", "price_vwap": 108.0, "total_vol": 60.0},
    ])
    out = month_window_prices(daily, window_days=5).set_index(["cusip", "date"])
    key = ("X", pd.Timestamp("2010-01-31"))
    # begin = (100*10 + 102*30)/40 = 101.5; end = (110*20 + 108*60)/80 = 108.5
    assert out.loc[key, "price_begin"] == pytest.approx(101.5, abs=1e-9)
    assert out.loc[key, "price_end"] == pytest.approx(108.5, abs=1e-9)


def test_boundary_days_are_inclusive():
    # A trade exactly on the begin cutoff (Jan 7) counts as begin; exactly on the
    # end cutoff (Jan 25) counts as end.
    daily = _daily([
        {"cusip_id": "Y", "trd_exctn_dt": "2010-01-07", "price_vwap": 99.0, "total_vol": 10.0},
        {"cusip_id": "Y", "trd_exctn_dt": "2010-01-25", "price_vwap": 111.0, "total_vol": 10.0},
    ])
    out = month_window_prices(daily, window_days=5).set_index(["cusip", "date"])
    key = ("Y", pd.Timestamp("2010-01-31"))
    assert out.loc[key, "price_begin"] == pytest.approx(99.0, abs=1e-9)
    assert out.loc[key, "price_end"] == pytest.approx(111.0, abs=1e-9)


def test_mid_month_only_bond_month_is_absent():
    # A bond trading only mid-month has no begin- or end-window price, so it
    # contributes no row — the decomposition simply excludes that bond-month.
    daily = _daily([
        {"cusip_id": "Z", "trd_exctn_dt": "2010-01-15", "price_vwap": 105.0, "total_vol": 50.0},
    ])
    out = month_window_prices(daily, window_days=5)
    assert out.empty

    # If the same bond also trades in an end window, the row appears with a
    # valid end price but a NaN begin price.
    daily2 = _daily([
        {"cusip_id": "Z", "trd_exctn_dt": "2010-01-15", "price_vwap": 105.0, "total_vol": 50.0},
        {"cusip_id": "Z", "trd_exctn_dt": "2010-01-27", "price_vwap": 110.0, "total_vol": 10.0},
    ])
    out2 = month_window_prices(daily2, window_days=5).set_index(["cusip", "date"])
    key = ("Z", pd.Timestamp("2010-01-31"))
    assert np.isnan(out2.loc[key, "price_begin"])
    assert out2.loc[key, "price_end"] == pytest.approx(110.0, abs=1e-9)


def test_missing_column_raises():
    daily = _daily([{"cusip_id": "X", "trd_exctn_dt": "2010-01-04", "price_vwap": 100.0, "total_vol": 10.0}])
    with pytest.raises(ValueError, match="missing required columns"):
        month_window_prices(daily.drop(columns=["total_vol"]))
