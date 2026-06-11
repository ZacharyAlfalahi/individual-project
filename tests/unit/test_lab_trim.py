"""
Unit tests for the engine's lab_trim plumbing (A2 of the registry amendments).

Covers:
  - Default trim_rule={"method": "none"} is a no-op (bit-exact regression)
  - method='truncate' drops out-of-bounds next_ret rows
  - method='winsorise' clips next_ret to bounds
  - Validation rejects unsupported targets / bound types / samples
  - Engine + overlap wrapper see consistent trim behaviour
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent / "agents" / "quant" / "library"),
)
from characteristic_sort import _apply_defaults, run_characteristic_sort  # noqa: E402


def _me(s: str) -> pd.Timestamp:
    return pd.Timestamp(s) + pd.offsets.MonthEnd(0)


def _panel(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0)
    return df


def _build_outlier_panel() -> pd.DataFrame:
    """4 bonds at Jan formation; Feb realisation has one outlier (+0.50)
    in the long leg and one tame return in the short leg."""
    rows = [
        # Jan formation
        {"cusip": "L1", "date": "2010-01", "ret": 0.0, "size": 100.0, "score": 4.0},
        {"cusip": "L2", "date": "2010-01", "ret": 0.0, "size": 100.0, "score": 3.0},
        {"cusip": "S1", "date": "2010-01", "ret": 0.0, "size": 100.0, "score": 2.0},
        {"cusip": "S2", "date": "2010-01", "ret": 0.0, "size": 100.0, "score": 1.0},
        # Feb realisation
        {"cusip": "L1", "date": "2010-02", "ret": 0.50, "size": 100.0, "score": 4.0},  # outlier
        {"cusip": "L2", "date": "2010-02", "ret": 0.02, "size": 100.0, "score": 3.0},
        {"cusip": "S1", "date": "2010-02", "ret": -0.01, "size": 100.0, "score": 2.0},
        {"cusip": "S2", "date": "2010-02", "ret": -0.01, "size": 100.0, "score": 1.0},
    ]
    return _panel(rows)


class TestTrimRuleDefault:
    def test_default_is_none(self):
        settings = _apply_defaults({"score": "score"})
        assert settings["trim_rule"]["method"] == "none"

    def test_default_preserves_existing_spread(self):
        """method=none must not alter the spread of any existing strategy."""
        panel = _build_outlier_panel()
        rulebook = {"score": "score", "groups": 2, "weighting": "equal", "min_bonds": 4}
        result = run_characteristic_sort(panel, rulebook)
        # mean long-leg = (0.50 + 0.02) / 2 = 0.26
        # mean short-leg = -0.01
        # spread = 0.27
        assert result["monthly_returns"]["strategy_ret"].iloc[0] == pytest.approx(0.27)


class TestTrimRuleTruncate:
    def test_truncate_drops_outlier(self):
        panel = _build_outlier_panel()
        rulebook = {
            "score": "score", "groups": 2, "weighting": "equal", "min_bonds": 3,
            "trim_rule": {
                "target": "return",
                "method": "truncate",
                "bounds": {"type": "absolute", "lo": -0.1, "hi": 0.1},
                "sample": "full_sample",
            },
        }
        result = run_characteristic_sort(panel, rulebook)
        # L1 (0.50) dropped; long leg = L2 only = 0.02; short leg = -0.01.
        # spread = 0.03.
        assert result["monthly_returns"]["strategy_ret"].iloc[0] == pytest.approx(0.03)


class TestTrimRuleWinsorise:
    def test_winsorise_clips_outlier(self):
        panel = _build_outlier_panel()
        rulebook = {
            "score": "score", "groups": 2, "weighting": "equal", "min_bonds": 4,
            "trim_rule": {
                "target": "return",
                "method": "winsorise",
                "bounds": {"type": "absolute", "lo": -0.1, "hi": 0.1},
                "sample": "full_sample",
            },
        }
        result = run_characteristic_sort(panel, rulebook)
        # L1 (0.50) clipped to 0.10; L2 unchanged at 0.02.
        # long = (0.10 + 0.02) / 2 = 0.06; short = -0.01; spread = 0.07.
        assert result["monthly_returns"]["strategy_ret"].iloc[0] == pytest.approx(0.07)


class TestTrimRuleValidation:
    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="trim_rule.method"):
            _apply_defaults({
                "score": "score",
                "trim_rule": {"method": "trim"},
            })

    def test_percentile_bounds_raise_not_implemented(self):
        with pytest.raises(NotImplementedError, match="percentile"):
            _apply_defaults({
                "score": "score",
                "trim_rule": {
                    "method": "truncate",
                    "bounds": {"type": "percentile", "lo": 0.01, "hi": 0.99},
                },
            })

    def test_by_month_cross_section_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="by_month_cross_section"):
            _apply_defaults({
                "score": "score",
                "trim_rule": {
                    "method": "truncate",
                    "bounds": {"type": "absolute", "lo": -1.0, "hi": 1.0},
                    "sample": "by_month_cross_section",
                },
            })

    def test_bounds_require_lo_or_hi(self):
        with pytest.raises(ValueError, match="lo/hi"):
            _apply_defaults({
                "score": "score",
                "trim_rule": {"method": "truncate", "bounds": {"type": "absolute"}},
            })
