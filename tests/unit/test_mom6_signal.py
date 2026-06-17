"""
Unit tests for the mom6 trailing-cumulative-return signal
(scripts/build_mom6_signal.py).

Covers:
  - Cumulative return over `formation_months` contiguous months matches a
    hand-computed product; the first (formation-1) rows of a run are NaN.
  - A calendar gap breaks the compounding chain (the window never spans a gap).
  - A NaN monthly return inside the window forces NaN (strict min_obs).
  - min_obs > formation_months is rejected.
  - The dual-family wrapper emits mom6_raw + mom6_corr.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import build_mom6_signal as bm  # noqa: E402


def _panel(cusip: str, months: list[str], rets: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "cusip": cusip,
        "date": [pd.Timestamp(m) + pd.offsets.MonthEnd(0) for m in months],
        "ret": rets,
    })


def test_cumulative_return_over_six_contiguous_months():
    months = [f"2010-{m:02d}" for m in range(1, 9)]  # Jan..Aug 2010
    rets = [0.01, 0.02, -0.01, 0.03, 0.00, 0.01, 0.02, -0.02]
    out = bm.compute_mom6_signal(_panel("X", months, rets), formation_months=6, min_obs=6)
    out = out.set_index("date")

    # First five rows of the run cannot fill a 6-month window → NaN.
    for m in months[:5]:
        assert pd.isna(out.loc[pd.Timestamp(m) + pd.offsets.MonthEnd(0), "mom6"])

    # June: product of (1+r) over Jan..Jun − 1.
    expected_jun = np.prod([1 + r for r in rets[:6]]) - 1
    assert out.loc[pd.Timestamp("2010-06") + pd.offsets.MonthEnd(0), "mom6"] == pytest.approx(
        expected_jun, abs=1e-12
    )
    # August: product over Mar..Aug − 1 (window slides).
    expected_aug = np.prod([1 + r for r in rets[2:8]]) - 1
    assert out.loc[pd.Timestamp("2010-08") + pd.offsets.MonthEnd(0), "mom6"] == pytest.approx(
        expected_aug, abs=1e-12
    )


def test_calendar_gap_breaks_the_window():
    # Run 1: Jan..May 2010 (5 months). Gap (no June). Run 2: Jul 2010..Jan 2011 (7).
    months = ["2010-01", "2010-02", "2010-03", "2010-04", "2010-05",
              "2010-07", "2010-08", "2010-09", "2010-10", "2010-11", "2010-12", "2011-01"]
    rets = [0.01] * len(months)
    out = bm.compute_mom6_signal(_panel("X", months, rets), formation_months=6, min_obs=6)
    out = out.set_index("date")

    # Run 1 maxes out at 5 contiguous months → never a value.
    for m in ["2010-01", "2010-05"]:
        assert pd.isna(out.loc[pd.Timestamp(m) + pd.offsets.MonthEnd(0), "mom6"])
    # Run 2: Nov 2010 is the 5th month of the run → still NaN.
    assert pd.isna(out.loc[pd.Timestamp("2010-11") + pd.offsets.MonthEnd(0), "mom6"])
    # Dec 2010 is the 6th contiguous month of run 2 (Jul..Dec) → a value.
    assert out.loc[pd.Timestamp("2010-12") + pd.offsets.MonthEnd(0), "mom6"] == pytest.approx(
        1.01 ** 6 - 1, abs=1e-12
    )


def test_nan_return_in_window_forces_nan():
    months = [f"2010-{m:02d}" for m in range(1, 8)]  # Jan..Jul
    rets = [0.01, 0.02, np.nan, 0.03, 0.01, 0.01, 0.02]  # March is NaN
    out = bm.compute_mom6_signal(_panel("X", months, rets), formation_months=6, min_obs=6)
    out = out.set_index("date")
    # June window (Jan..Jun) includes the NaN March → NaN.
    assert pd.isna(out.loc[pd.Timestamp("2010-06") + pd.offsets.MonthEnd(0), "mom6"])
    # July window (Feb..Jul) still includes March → NaN.
    assert pd.isna(out.loc[pd.Timestamp("2010-07") + pd.offsets.MonthEnd(0), "mom6"])


def test_return_at_or_below_minus_one_treated_as_invalid():
    """A ret <= -1 (non-positive price; a raw-family data error) is invalid and
    must not enter the compounding chain — any window needing it is NaN."""
    months = [f"2010-{m:02d}" for m in range(1, 8)]  # Jan..Jul
    rets = [0.01, 0.02, -1.0, 0.03, 0.01, 0.01, 0.02]  # March ret = -100% (bad price)
    out = bm.compute_mom6_signal(_panel("X", months, rets), formation_months=6, min_obs=6)
    out = out.set_index("date")
    assert pd.isna(out.loc[pd.Timestamp("2010-06") + pd.offsets.MonthEnd(0), "mom6"])
    assert pd.isna(out.loc[pd.Timestamp("2010-07") + pd.offsets.MonthEnd(0), "mom6"])


def test_min_obs_cannot_exceed_formation():
    with pytest.raises(ValueError, match="min_obs"):
        bm.compute_mom6_signal(_panel("X", ["2010-01"], [0.0]), formation_months=6, min_obs=7)


def test_dual_family_emits_both_columns():
    months = [f"2010-{m:02d}" for m in range(1, 8)]
    panel = pd.DataFrame({
        "cusip": "X",
        "date": [pd.Timestamp(m) + pd.offsets.MonthEnd(0) for m in months],
        "ret_raw": [0.01] * 7,
        "ret_corr": [0.02] * 7,
    })
    out = bm.compute_dual_family(panel, formation_months=6, min_obs=6)
    assert set(["cusip", "date", "mom6_raw", "mom6_corr"]).issubset(out.columns)
    jun = out.set_index("date").loc[pd.Timestamp("2010-06") + pd.offsets.MonthEnd(0)]
    assert jun["mom6_raw"] == pytest.approx(1.01 ** 6 - 1, abs=1e-12)
    assert jun["mom6_corr"] == pytest.approx(1.02 ** 6 - 1, abs=1e-12)
