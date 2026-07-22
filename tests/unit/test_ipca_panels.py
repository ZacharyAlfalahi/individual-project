"""
Panel-state construction (spec §2.1): leave-one-out-from-corrected, and the honest deferral of
construction-toggle IPCA propagation to the experimentalist build.
"""

from __future__ import annotations

import pandas as pd
import pytest

from agents.auditor.ipca_differential.panels import (
    CONSTRUCTION_BIASES,
    ConstructionToggleDeferred,
    config_leave_out,
    panel_states,
    to_merged,
)
from agents.quant.library.run_config import corrected


def test_config_leave_out_flips_only_the_target_axis():
    base, off = config_leave_out("meas_err")
    assert base == corrected()
    assert off.panel_view.price_family == "raw"        # flipped to as-published
    assert off.panel_view.stale_mask is True           # other panel_view axes unchanged
    assert off.panel_view.include_terminal_rows is True
    assert off.construction == base.construction        # construction untouched


def test_config_leave_out_survivorship():
    _, off = config_leave_out("survivorship")
    assert off.panel_view.include_terminal_rows is False
    assert off.panel_view.price_family == "corr"        # others still corrected


@pytest.mark.parametrize("bias", list(CONSTRUCTION_BIASES))
def test_construction_biases_are_deferred(bias):
    with pytest.raises(ConstructionToggleDeferred):
        panel_states(bias, pd.DataFrame(), None)


def test_to_merged_requires_signal_columns():
    bare = pd.DataFrame({"cusip": [], "date": [], "xret": [], "rating": [], "time_to_maturity": []})
    with pytest.raises(KeyError, match="mom6|IPCA"):
        to_merged(bare)


def test_to_merged_builds_canonical_frame():
    vp = pd.DataFrame({
        "cusip": ["A", "B"], "date": pd.to_datetime(["2010-01-31", "2010-01-31"]),
        "xret": [0.01, 0.02], "mom6": [0.1, 0.2], "var_5pct": [0.3, 0.4],
        "gamma_illiq": [0.5, 0.6], "rating": [3.0, 4.0], "time_to_maturity": [5.0, 6.0],
        "bond_vol": [0.02, 0.03],
    })
    merged = to_merged(vp)
    assert list(merged["str_reversal"]) == [0.01, 0.02]      # str_reversal = xret
    assert set(merged.columns) == {
        "cusip", "date", "str_reversal", "mom6", "var_5pct", "gamma_illiq",
        "rating", "time_to_maturity", "bond_vol",
    }
