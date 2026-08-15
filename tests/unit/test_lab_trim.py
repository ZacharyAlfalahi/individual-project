"""
Unit tests for the engine's lab_trim plumbing (A2 of the registry amendments).

Covers:
  - Default trim_rule={"method": "none"} is a no-op (bit-exact regression)
  - method='truncate' drops out-of-bounds next_ret rows
  - method='winsorise' clips next_ret to bounds
  - Validation rejects unsupported targets / bound types / samples
  - Engine + overlap wrapper see consistent trim behaviour
"""

import pandas as pd
import pytest

from agents.quant.library.characteristic_sort import (
    _apply_defaults,
    run_characteristic_sort,
)


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


class TestTrimRulePercentile:
    """Percentile bounds resolved to absolute ONCE per cell over the full-sample
    eligible next_ret (spec E). The eligible full-sample series here is the four
    Jan-formation forward returns [0.50, 0.02, -0.01, -0.01]."""

    def _rulebook(self, trim_rule, min_bonds=3):
        return {"score": "score", "groups": 2, "weighting": "equal",
                "min_bonds": min_bonds, "trim_rule": trim_rule}

    def test_resolves_on_eligible_next_ret_one_sided_right(self):
        import numpy as np
        panel = _build_outlier_panel()
        result = run_characteristic_sort(panel, self._rulebook({
            "target": "return", "method": "truncate",
            "bounds": {"type": "percentile", "hi": 0.75, "percentile_method": "linear"},
            "sample": "full_sample",
        }))
        realised = result["bookkeeping"]["realised_trim_threshold"]
        expected_hi = float(np.quantile([0.50, 0.02, -0.01, -0.01], 0.75, method="linear"))
        assert realised["percentile_method"] == "linear"
        assert realised["n_obs"] == 4                    # condition 1: same series
        assert realised["hi"]["level"] == 0.75
        assert realised["hi"]["threshold"] == pytest.approx(expected_hi)   # condition 4
        assert "lo" not in realised                      # one-sided right (Jostova)

    def test_equivalent_to_absolute_at_realised_threshold(self):
        # Condition 5: percentile-resolved output == absolute-specified output at
        # the same realised threshold, byte-identical.
        panel = _build_outlier_panel()
        pct = run_characteristic_sort(panel, self._rulebook({
            "target": "return", "method": "truncate",
            "bounds": {"type": "percentile", "hi": 0.75, "percentile_method": "linear"},
            "sample": "full_sample",
        }))
        hi_abs = pct["bookkeeping"]["realised_trim_threshold"]["hi"]["threshold"]
        absolute = run_characteristic_sort(panel, self._rulebook({
            "target": "return", "method": "truncate",
            "bounds": {"type": "absolute", "hi": hi_abs},
            "sample": "full_sample",
        }))
        pd.testing.assert_frame_equal(pct["monthly_returns"], absolute["monthly_returns"])

    def test_winsorise_percentile_clips_at_resolved_threshold(self):
        # Workstream K: the LIVE lab_filter path is winsorise+percentile (adj=wins,
        # bounds=percentile), but only truncate+percentile and winsorise+ABSOLUTE
        # were tested. hi=0.75 on the eligible series [0.50,0.02,-0.01,-0.01] resolves
        # to 0.14; winsorise clips L1 0.50->0.14, L2 0.02 stays; long=(0.14+0.02)/2
        # =0.08; short=-0.01; spread=0.09.
        import numpy as np
        panel = _build_outlier_panel()
        result = run_characteristic_sort(panel, self._rulebook({
            "target": "return", "method": "winsorise",
            "bounds": {"type": "percentile", "hi": 0.75, "percentile_method": "linear"},
            "sample": "full_sample",
        }, min_bonds=4))
        realised = result["bookkeeping"]["realised_trim_threshold"]
        expected_hi = float(np.quantile([0.50, 0.02, -0.01, -0.01], 0.75, method="linear"))
        assert realised["hi"]["threshold"] == pytest.approx(expected_hi)   # 0.14
        assert result["monthly_returns"]["strategy_ret"].iloc[0] == pytest.approx(0.09, abs=1e-12)

    def test_absolute_trim_untouched_by_resolver(self):
        # Condition 6: absolute trims pass through the resolver with no realised
        # threshold and the original spread.
        panel = _build_outlier_panel()
        result = run_characteristic_sort(panel, self._rulebook({
            "target": "return", "method": "truncate",
            "bounds": {"type": "absolute", "lo": -0.1, "hi": 0.1},
            "sample": "full_sample",
        }))
        assert result["bookkeeping"]["realised_trim_threshold"] is None
        assert result["monthly_returns"]["strategy_ret"].iloc[0] == pytest.approx(0.03)


class TestTrimRuleValidation:
    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="trim_rule.method"):
            _apply_defaults({
                "score": "score",
                "trim_rule": {"method": "trim"},
            })

    def test_percentile_bounds_now_supported(self):
        # Percentile bounds are supported when a pre-registered percentile_method
        # is given (spec E); _apply_defaults accepts them for later resolution.
        settings = _apply_defaults({
            "score": "score",
            "trim_rule": {
                "method": "truncate",
                "bounds": {"type": "percentile", "hi": 0.995,
                           "percentile_method": "linear"},
            },
        })
        assert settings["trim_rule"]["bounds"]["type"] == "percentile"

    def test_percentile_bounds_require_method(self):
        # No library default for the interpolation method (spec E condition 2).
        with pytest.raises(ValueError, match="percentile_method"):
            _apply_defaults({
                "score": "score",
                "trim_rule": {
                    "method": "truncate",
                    "bounds": {"type": "percentile", "hi": 0.995},
                },
            })

    def test_percentile_level_out_of_range_raises(self):
        with pytest.raises(ValueError, match="percentile level"):
            _apply_defaults({
                "score": "score",
                "trim_rule": {
                    "method": "truncate",
                    "bounds": {"type": "percentile", "hi": 1.5,
                               "percentile_method": "linear"},
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
