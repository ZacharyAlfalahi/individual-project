"""Stage 10 — the orchestrator (run_audit) and the numeric verifier (§11)."""

from __future__ import annotations


import pytest

from agents.auditor import AuditRefused, run_audit, verify_numbers
from agents.auditor.data.synthetic_panel import build_scenario
from agents.auditor.schemas.toggle import TOGGLE_IDS, ToggleFacts
from agents.auditor.thresholds import SupportGate

from _auditor_fixtures import all_runnable_facts

GATE = SupportGate(min_common_months=12, min_common_fraction_of_reference=0.3)


def _audit(scenario, facts=None, **kw):
    return run_audit(
        scenario.strategy,
        scenario.panel,
        facts or all_runnable_facts(),
        signals=scenario.signals,
        primary_metric="average",
        percentage_denominator_min=0.0,
        support_gate=GATE,
        **kw,
    )


# --------------------------------------------------------------------------
# run_audit end-to-end
# --------------------------------------------------------------------------

def test_run_audit_complete_scope_populates_everything():
    scenario = build_scenario(None, seed=0)
    core = _audit(scenario)
    assert core.audit_scope == "COMPLETE"
    assert core.runnable_toggles == TOGGLE_IDS
    assert core.conditioning_signature == ()
    assert core.conditioning_statement is None
    # efficiency asserted inside shapley_result; residual tiny
    assert abs(core.shapley.efficiency_residual) < 1e-9
    # every toggle has a Shapley value and a first-order DOE effect
    assert set(core.shapley.values) == set(TOGGLE_IDS)
    assert core.support.t_common > 0


def test_run_audit_recovers_injected_effect():
    scenario = build_scenario("meas_err", seed=1)
    core = _audit(scenario)
    e_meas = core.saturated.doe[frozenset({"meas_err"})]
    assert abs(e_meas) > 1e-3
    # the injected toggle's Shapley value dominates
    dominant = max(core.shapley.values, key=lambda t: abs(core.shapley.values[t]))
    assert dominant == "meas_err"


def test_run_audit_partial_scope_carries_conditioning():
    scenario = build_scenario(None, seed=2)
    facts = all_runnable_facts()
    facts = [
        ToggleFacts("survivorship", runnable=False,
                    runnable_reason="MISSING_EXIT_DATA", fixed_state="OFF")
        if f.toggle_id == "survivorship" else f
        for f in facts
    ]
    core = _audit(scenario, facts=facts)
    assert core.audit_scope == "PARTIAL"
    assert "survivorship" not in core.runnable_toggles
    assert core.conditioning_signature == (("survivorship", "OFF"),)
    assert "survivorship held at OFF" in core.conditioning_statement
    # reduced lattice => 2^4 = 16 Shapley coordinates
    assert len(core.shapley.values) == 4


def test_run_audit_refused_scope_raises():
    scenario = build_scenario(None, seed=3)
    facts = [
        ToggleFacts("lab_trim", runnable=False,
                    runnable_reason="PAPER_RULE_NOT_STATED", fixed_state=None)
        if f.toggle_id == "lab_trim" else f
        for f in all_runnable_facts()
    ]
    with pytest.raises(AuditRefused, match="REFUSED"):
        _audit(scenario, facts=facts)


def test_run_audit_reads_thresholds_fail_loud_when_absent():
    # Without explicit args, run_audit reads thresholds.yaml, which has no auditor
    # block yet => fail loud (integrity: never default a pre-registration constant).
    scenario = build_scenario(None, seed=0)
    from agents.auditor.thresholds import AuditorThresholdError
    with pytest.raises(AuditorThresholdError):
        run_audit(scenario.strategy, scenario.panel, all_runnable_facts(),
                  signals=scenario.signals)


# --------------------------------------------------------------------------
# to_dict serialisation + numeric verifier (§11)
# --------------------------------------------------------------------------

def test_audit_core_to_dict_is_serialisable():
    core = _audit(build_scenario(None, seed=0))
    d = core.to_dict()
    assert d["audit_scope"] == "COMPLETE"
    assert "shapley" in d and "saturated_bases" in d and "support" in d


def test_verifier_passes_prose_that_only_cites_report_numbers():
    core = _audit(build_scenario("meas_err", seed=1))
    d = core.to_dict()
    e_meas = d["saturated_bases"]["doe_effects"]["meas_err"]
    prose = f"The meas_err main effect is {e_meas:.4f} on the primary metric."
    result = verify_numbers(prose, d)
    assert result.ok, result.unverified
    assert result.n_checked >= 1


def test_verifier_flags_a_fabricated_number():
    core = _audit(build_scenario(None, seed=0))
    d = core.to_dict()
    prose = "The effect is 0.123456789 in Sharpe points."  # not in the report
    result = verify_numbers(prose, d)
    assert not result.ok
    assert "0.123456789" in result.unverified


def test_verifier_matches_percent_against_stored_fraction():
    # A share stored as 0.4 traces to a prose "40%".
    fake_report = {"shapley": {"shares": {"meas_err": 0.4}}}
    result = verify_numbers("look_ahead accounts for 40% of the gap", fake_report)
    assert result.ok, result.unverified
