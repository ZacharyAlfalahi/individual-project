"""Stage 3 — pre-flight executability and derived audit scope (§3.3-3.4)."""

from __future__ import annotations

import pytest

from agents.auditor.checks.preflight import (
    FOUR_RUNNABLE_CONDITIONS,
    derive_scope,
    is_runnable,
)
from agents.auditor.schemas.toggle import TOGGLE_IDS, ToggleFacts


def _all_pass():
    return {c: True for c in FOUR_RUNNABLE_CONDITIONS}


# --------------------------------------------------------------------------
# The four runnable conditions (§3.3)
# --------------------------------------------------------------------------

def test_all_four_conditions_pass_is_runnable():
    assert is_runnable(_all_pass()) is True


@pytest.mark.parametrize("fail", FOUR_RUNNABLE_CONDITIONS)
def test_any_single_condition_failing_makes_it_unrunnable(fail):
    conds = _all_pass()
    conds[fail] = False
    assert is_runnable(conds) is False


def test_missing_condition_raises():
    conds = _all_pass()
    del conds["off_state_known"]
    with pytest.raises(ValueError, match="missing"):
        is_runnable(conds)


def test_reintroducing_the_withdrawn_fifth_condition_is_rejected():
    # In-sample exposure was withdrawn (D-A30); adding it back must raise.
    conds = _all_pass()
    conds["has_in_sample_events"] = True
    with pytest.raises(ValueError, match="unknown"):
        is_runnable(conds)


# --------------------------------------------------------------------------
# derive_scope (§3.4)
# --------------------------------------------------------------------------

def _facts(**overrides):
    """Five runnable facts by default; override specific toggles."""
    facts = {t: ToggleFacts(t, runnable=True) for t in TOGGLE_IDS}
    facts.update(overrides)
    return list(facts.values())


def test_all_runnable_is_complete():
    r = derive_scope("mom6", _facts())
    assert r.audit_scope == "COMPLETE"
    assert r.runnable_toggles == TOGGLE_IDS
    assert r.conditioning_signature == ()
    assert r.conditioning_statement is None


def test_held_non_runnable_is_partial_with_conditioning():
    facts = _facts(
        survivorship=ToggleFacts(
            "survivorship", runnable=False,
            runnable_reason="MISSING_EXIT_DATA", fixed_state="OFF",
        )
    )
    r = derive_scope("bbw", facts)
    assert r.audit_scope == "PARTIAL"
    assert "survivorship" not in r.runnable_toggles
    assert r.fixed_states == {"survivorship": "OFF"}
    assert r.conditioning_signature == (("survivorship", "OFF"),)
    assert "survivorship held at OFF" in r.conditioning_statement


def test_non_runnable_without_fixed_state_is_refused():
    facts = _facts(
        lab_trim=ToggleFacts(
            "lab_trim", runnable=False,
            runnable_reason="PAPER_RULE_NOT_STATED", fixed_state=None,
        )
    )
    r = derive_scope("paperX", facts)
    assert r.audit_scope == "REFUSED"
    assert r.is_refused
    assert r.refused_toggles == ("lab_trim",)


def test_refused_dominates_partial():
    # One held toggle (PARTIAL-ish) + one refused toggle => REFUSED overall.
    facts = _facts(
        survivorship=ToggleFacts(
            "survivorship", runnable=False,
            runnable_reason="MISSING_EXIT_DATA", fixed_state="OFF",
        ),
        lab_trim=ToggleFacts(
            "lab_trim", runnable=False,
            runnable_reason="PAPER_RULE_NOT_STATED", fixed_state=None,
        ),
    )
    r = derive_scope("paperY", facts)
    assert r.audit_scope == "REFUSED"
    assert r.refused_toggles == ("lab_trim",)


def test_signature_is_canonical_order():
    facts = _facts(
        stale_price=ToggleFacts(
            "stale_price", runnable=False,
            runnable_reason="PROVENANCE_INSUFFICIENT", fixed_state="OFF",
        ),
        lib_gap=ToggleFacts(
            "lib_gap", runnable=False,
            runnable_reason="ENGINE_UNSUPPORTED", fixed_state="ON",
        ),
    )
    r = derive_scope("s", facts)
    # canonical order: stale_price precedes lib_gap in TOGGLE_IDS
    assert r.conditioning_signature == (("stale_price", "OFF"), ("lib_gap", "ON"))


def test_requires_all_five_toggles():
    with pytest.raises(ValueError, match="one fact per registered toggle"):
        derive_scope("s", [ToggleFacts("meas_err", runnable=True)])


def test_duplicate_fact_rejected():
    facts = _facts()
    facts.append(ToggleFacts("meas_err", runnable=True))
    with pytest.raises(ValueError, match="duplicate"):
        derive_scope("s", facts)
