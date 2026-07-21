"""Stage 1 — the Auditor schema contracts.

Covers ToggleFacts validation (the pre-flight input), the structural guarantee
that `is_no_op` cannot be declared in a spec, the toggle->RunConfig axis mapping
matching the verified endpoints, MetricSet round-trip, and LatticeResult
cardinality / lookup.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from agents.auditor.schemas import (
    TOGGLE_AXES,
    TOGGLE_IDS,
    CellReturns,
    LatticeResult,
    MetricSet,
    ToggleFacts,
    conditioning_statement_for,
)
from agents.auditor.schemas.toggle import ToggleFacts as _ToggleFactsClass
from agents.quant.library.run_config import corrected, uncorrected


# --------------------------------------------------------------------------
# ToggleFacts (pre-flight INPUT)
# --------------------------------------------------------------------------

def test_runnable_toggle_is_valid_with_defaults():
    tf = ToggleFacts("meas_err", runnable=True)
    assert tf.runnable and tf.fixed_state is None and tf.runnable_reason is None


def test_runnable_toggle_may_not_carry_a_fixed_state():
    with pytest.raises(ValueError, match="varied, not held"):
        ToggleFacts("meas_err", runnable=True, fixed_state="OFF")


def test_runnable_toggle_may_not_carry_a_reason():
    with pytest.raises(ValueError, match="must not carry a runnable_reason"):
        ToggleFacts("lib_gap", runnable=True, runnable_reason="ENGINE_UNSUPPORTED")


def test_non_runnable_requires_a_reason():
    with pytest.raises(ValueError, match="requires a runnable_reason"):
        ToggleFacts("survivorship", runnable=False)


def test_non_runnable_reason_must_be_in_vocabulary():
    with pytest.raises(ValueError, match="runnable_reason must be one of"):
        ToggleFacts("survivorship", runnable=False, runnable_reason="BECAUSE")


def test_non_runnable_may_hold_a_fixed_state_or_none():
    partial = ToggleFacts(
        "survivorship", runnable=False, runnable_reason="MISSING_EXIT_DATA",
        fixed_state="OFF",
    )
    refused = ToggleFacts(
        "lab_trim", runnable=False, runnable_reason="PAPER_RULE_NOT_STATED",
        fixed_state=None,
    )
    assert partial.fixed_state == "OFF"
    assert refused.fixed_state is None


def test_unknown_toggle_id_rejected():
    with pytest.raises(ValueError, match="toggle_id must be one of"):
        ToggleFacts("look_ahead", runnable=True)  # type: ignore[arg-type]


def test_is_no_op_cannot_be_declared_on_the_input_spec():
    # The structural trap (§3.3): is_no_op is a RESULT, not an input. It must not
    # be a field on ToggleFacts, so a spec file can never assert it.
    field_names = {f.name for f in dataclasses.fields(_ToggleFactsClass)}
    assert "is_no_op" not in field_names


# --------------------------------------------------------------------------
# Toggle -> RunConfig axis mapping (the single source of truth, SEAM 1)
# --------------------------------------------------------------------------

def test_every_toggle_has_an_axis():
    assert set(TOGGLE_AXES) == set(TOGGLE_IDS)


def test_axes_match_the_verified_endpoints():
    # OFF values must reproduce uncorrected(); ON values must reproduce corrected().
    off, on = uncorrected(), corrected()
    for tid, axis in TOGGLE_AXES.items():
        block_off = getattr(off, axis.block)
        block_on = getattr(on, axis.block)
        assert getattr(block_off, axis.field) == axis.off, tid
        assert getattr(block_on, axis.field) == axis.on, tid


# --------------------------------------------------------------------------
# MetricSet
# --------------------------------------------------------------------------

def _summary(**over):
    base = dict(
        n_months=120, months_per_year=12, nw_lags_used=4, average=0.01,
        annualised_average=0.12, bumpiness=0.04, sharpe=0.86, t_stat=2.1,
        first_date=None, last_date=None,
    )
    base.update(over)
    return base


def test_metricset_from_summary_and_value():
    ms = MetricSet.from_summary(_summary())
    assert ms.value("sharpe") == 0.86
    assert ms.value("average") == 0.01
    assert ms.as_metric_dict()["t_stat"] == 2.1


def test_metricset_rejects_non_attribution_metric():
    ms = MetricSet.from_summary(_summary())
    with pytest.raises(KeyError, match="not an attribution metric"):
        ms.value("n_months")


# --------------------------------------------------------------------------
# LatticeResult
# --------------------------------------------------------------------------

def _cell(on_set):
    idx = pd.date_range("2005-01-31", periods=3, freq="ME")
    return CellReturns(
        on_set=frozenset(on_set),
        run_config=None,
        returns=pd.Series([0.0, 0.0, 0.0], index=idx),
        n_bonds=pd.Series([10, 10, 10], index=idx),
        metrics_native=MetricSet.from_summary(_summary(n_months=3)),
        run_config_hash="rc",
        panel_view_hash="pv",
        return_hash="r",
        n_bonds_hash="n",
        metric_hash="m",
    )


def test_lattice_cardinality_enforced():
    # k=1 => 2 cells required; supplying 1 must raise.
    with pytest.raises(ValueError, match="expected 2"):
        LatticeResult("s", ("meas_err",), {}, (_cell(set()),))


def test_lattice_lookup_by_on_set():
    lat = LatticeResult(
        "s", ("meas_err",), {}, (_cell(set()), _cell({"meas_err"}))
    )
    assert lat.k == 1
    assert lat.cell_for(frozenset()).on_set == frozenset()
    assert lat.cell_for(frozenset({"meas_err"})).on_set == frozenset({"meas_err"})
    with pytest.raises(KeyError):
        lat.cell_for(frozenset({"stale_price"}))


# --------------------------------------------------------------------------
# Conditioning statement
# --------------------------------------------------------------------------

def test_conditioning_statement_none_for_complete():
    assert conditioning_statement_for(()) is None


def test_conditioning_statement_names_held_toggles_for_partial():
    stmt = conditioning_statement_for((("survivorship", "OFF"),))
    assert "survivorship held at OFF" in stmt
    assert "NOT the full" in stmt
