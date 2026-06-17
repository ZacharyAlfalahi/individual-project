"""
Unit tests for ex-post vs ex-ante winsorization (agents/quant/library/winsorize.py).

Covers:
  - ex_post: one static full-sample percentile, one-sided right clip.
  - ex_ante: per-month expanding past-only thresholds (raw past, not the clipped
    values); the first month (no history) is unclipped.
  - loc right/left/both; NaNs preserved.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from agents.quant.library.winsorize import winsorize_returns  # noqa: E402


def test_ex_post_right_clips_upper_tail_only():
    # 11 values; 90th percentile = 9 exactly. Right clip: only 100 → 9.
    r = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100.0])
    out = winsorize_returns(r, level=90, loc="right", mode="ex_post")
    assert out.iloc[-1] == pytest.approx(9.0)
    assert list(out.iloc[:-1]) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # lower tail untouched


def test_ex_post_both_clips_both_tails():
    r = pd.Series([-100.0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100.0])
    out = winsorize_returns(r, level=90, loc="both", mode="ex_post")
    # 10th percentile and 90th percentile of the 12 values bound the clip.
    lo = np.percentile(r.to_numpy(), 10.0)
    hi = np.percentile(r.to_numpy(), 90.0)
    assert out.min() == pytest.approx(lo)
    assert out.max() == pytest.approx(hi)


def test_ex_ante_expanding_past_only_thresholds():
    # Month 01: no history → unclipped (the 100 survives).
    # Month 02: threshold = 90th pct of month-01 RAW = 9 → 50, 200 both → 9.
    # Month 03: threshold = 90th pct of months 01+02 RAW = 90 → 300 → 90, 3 kept.
    dates = (
        [pd.Timestamp("2010-01-31")] * 11
        + [pd.Timestamp("2010-02-28")] * 2
        + [pd.Timestamp("2010-03-31")] * 2
    )
    rets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100.0] + [50.0, 200.0] + [3.0, 300.0]
    r = pd.Series(rets)
    d = pd.Series(dates)
    out = winsorize_returns(r, dates=d, level=90, loc="right", mode="ex_ante")

    assert out.iloc[10] == pytest.approx(100.0)              # month 01 unclipped
    assert out.iloc[11] == pytest.approx(9.0)                # month 02: 50 → 9
    assert out.iloc[12] == pytest.approx(9.0)                # month 02: 200 → 9
    assert out.iloc[13] == pytest.approx(3.0)                # month 03: 3 kept
    assert out.iloc[14] == pytest.approx(90.0)               # month 03: 300 → 90


def test_nans_preserved():
    r = pd.Series([1.0, np.nan, 2.0, 3.0, 100.0])
    out = winsorize_returns(r, level=80, loc="right", mode="ex_post")
    assert np.isnan(out.iloc[1])


def test_ex_ante_requires_dates():
    with pytest.raises(ValueError, match="dates is required"):
        winsorize_returns(pd.Series([1.0, 2.0]), level=90, mode="ex_ante")
