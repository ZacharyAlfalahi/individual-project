"""
Unit tests for the lead/lag error injector (agents/quant/library/lead_lag.py).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from agents.quant.library.lead_lag import inject_lead_lag  # noqa: E402


def _series(vals):
    dates = [pd.Timestamp("2010-01-31") + pd.offsets.MonthEnd(i) for i in range(len(vals))]
    return pd.DataFrame({"date": dates, "strategy_ret": vals})


def test_lead_error_replaces_t_with_t_plus_1():
    # Lead (+1) over Feb..Apr: Feb←Mar, Mar←Apr, Apr←May; Jan/May unchanged.
    s = _series([10.0, 20.0, 30.0, 40.0, 50.0])
    out = inject_lead_lag(s, shift_months=1, window=("2010-02", "2010-04"))
    assert list(out["strategy_ret"]) == [10.0, 30.0, 40.0, 50.0, 50.0]


def test_lag_error_replaces_t_with_t_minus_1():
    # Lag (-1) over Feb..Apr: Feb←Jan, Mar←Feb, Apr←Mar; Jan/May unchanged.
    s = _series([10.0, 20.0, 30.0, 40.0, 50.0])
    out = inject_lead_lag(s, shift_months=-1, window=("2010-02", "2010-04"))
    assert list(out["strategy_ret"]) == [10.0, 10.0, 20.0, 30.0, 50.0]


def test_lead_at_final_month_is_nan():
    # Lead window including the last month → t+1 out of range → NaN there.
    s = _series([10.0, 20.0, 30.0])
    out = inject_lead_lag(s, shift_months=1, window=("2010-01", "2010-03"))
    assert out["strategy_ret"].iloc[0] == 20.0
    assert out["strategy_ret"].iloc[1] == 30.0
    assert np.isnan(out["strategy_ret"].iloc[2])


def test_zero_shift_rejected():
    with pytest.raises(ValueError, match="non-zero"):
        inject_lead_lag(_series([1.0, 2.0]), shift_months=0, window=("2010-01", "2010-02"))
