"""
Unit tests for the MKTB market-factor builder (agents/quant/library/market_factor.py).

Covers:
  - Value-weighted excess return matches a hand-computed answer.
  - Eligibility, NaN-return, and non-positive-weight rows are excluded.
  - eligible_col=None weights every row (no eligibility gate).
  - The safe rate is NOT subtracted (ret_col is used verbatim).
  - Missing required columns raise; empty input returns a typed empty frame.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from agents.quant.library.market_factor import compute_market_factor  # noqa: E402


def _me(s: str) -> pd.Timestamp:
    return pd.Timestamp(s) + pd.offsets.MonthEnd(0)


def _panel(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = df["date"].map(_me)
    return df


def test_value_weighted_excess_matches_hand_computation():
    panel = _panel([
        # Month 1: VW = (100*.02 + 300*-.01 + 600*.05) / 1000 = 29/1000 = 0.029
        {"date": "2010-01", "size": 100.0, "xret": 0.02, "universe_eligible": True},
        {"date": "2010-01", "size": 300.0, "xret": -0.01, "universe_eligible": True},
        {"date": "2010-01", "size": 600.0, "xret": 0.05, "universe_eligible": True},
        # Month 2: VW = (100*.01 + 100*.03) / 200 = 0.02
        {"date": "2010-02", "size": 100.0, "xret": 0.01, "universe_eligible": True},
        {"date": "2010-02", "size": 100.0, "xret": 0.03, "universe_eligible": True},
    ])
    out = compute_market_factor(panel, ret_col="xret").set_index("date")
    assert out.loc[_me("2010-01"), "mktb"] == pytest.approx(0.029, abs=1e-12)
    assert out.loc[_me("2010-01"), "n_bonds"] == 3
    assert out.loc[_me("2010-02"), "mktb"] == pytest.approx(0.02, abs=1e-12)
    assert out.loc[_me("2010-02"), "n_bonds"] == 2


def test_excludes_ineligible_nan_and_nonpositive_weight():
    panel = _panel([
        {"date": "2010-01", "size": 100.0, "xret": 0.02, "universe_eligible": True},
        {"date": "2010-01", "size": 900.0, "xret": 0.04, "universe_eligible": True},
        # ineligible: huge weight + return that would dominate if counted
        {"date": "2010-01", "size": 5000.0, "xret": 0.99, "universe_eligible": False},
        # NaN return: dropped
        {"date": "2010-01", "size": 500.0, "xret": np.nan, "universe_eligible": True},
        # zero weight: dropped (weight must be > 0)
        {"date": "2010-01", "size": 0.0, "xret": 0.5, "universe_eligible": True},
    ])
    out = compute_market_factor(panel, ret_col="xret").set_index("date")
    # Only the two eligible, finite, positive-weight bonds count:
    # (100*.02 + 900*.04) / 1000 = (2 + 36)/1000 = 0.038
    assert out.loc[_me("2010-01"), "mktb"] == pytest.approx(0.038, abs=1e-12)
    assert out.loc[_me("2010-01"), "n_bonds"] == 2


def test_eligible_col_none_weights_all_rows():
    panel = _panel([
        {"date": "2010-01", "size": 100.0, "xret": 0.02, "universe_eligible": False},
        {"date": "2010-01", "size": 100.0, "xret": 0.06, "universe_eligible": False},
    ])
    out = compute_market_factor(panel, ret_col="xret", eligible_col=None).set_index("date")
    assert out.loc[_me("2010-01"), "mktb"] == pytest.approx(0.04, abs=1e-12)
    assert out.loc[_me("2010-01"), "n_bonds"] == 2


def test_safe_rate_not_subtracted():
    """ret_col is used verbatim — passing a raw return yields its VW mean, not
    an excess. (MKTB receives an already-excess column in production.)"""
    panel = _panel([
        {"date": "2010-01", "size": 1.0, "xret": 0.10, "universe_eligible": True},
        {"date": "2010-01", "size": 1.0, "xret": 0.20, "universe_eligible": True},
    ])
    out = compute_market_factor(panel, ret_col="xret").set_index("date")
    assert out.loc[_me("2010-01"), "mktb"] == pytest.approx(0.15, abs=1e-12)


def test_missing_column_raises():
    panel = _panel([{"date": "2010-01", "size": 1.0, "xret": 0.0, "universe_eligible": True}])
    with pytest.raises(ValueError, match="missing required columns"):
        compute_market_factor(panel.drop(columns=["size"]), ret_col="xret")


def test_empty_after_filter_returns_typed_empty_frame():
    panel = _panel([
        {"date": "2010-01", "size": 100.0, "xret": np.nan, "universe_eligible": True},
    ])
    out = compute_market_factor(panel, ret_col="xret")
    assert list(out.columns) == ["date", "mktb", "n_bonds"]
    assert len(out) == 0
    assert out["mktb"].dtype == float
    assert str(out["date"].dtype) == "datetime64[ns]"
