"""Fixture-only Scientist-stage licensing (docs/reporter/reporter_spec_v0.2.md §5.3, C9)."""

from __future__ import annotations

import types

from agents.auditor.checks.fdr import run_fdr
from agents.auditor.schemas.decomposition import SaturatedBasis
from agents.reporter.bundle import StageStatus
from agents.reporter.routing import license_scientist_stage

THETA = 0.05
Q = 0.10


def _stub(*, doe, pvalues, runnable, scope="COMPLETE"):
    """A duck-typed AuditReport exposing exactly what build_scientist_case + routing read:
    core.saturated.doe, core.runnable_toggles, core.audit_scope, fdr.decisions."""
    sat = SaturatedBasis(harsanyi=dict(doe), walsh=dict(doe), doe=dict(doe))
    core = types.SimpleNamespace(
        saturated=sat, runnable_toggles=runnable, audit_scope=scope
    )
    fdr = run_fdr(pvalues, Q)
    return types.SimpleNamespace(core=core, fdr=fdr)


def test_no_failing_toggle_licenses_not_applicable():
    # Performance-INCREASING effect -> PASS -> no failing toggle -> NOT_APPLICABLE.
    rep = _stub(
        doe={frozenset({"meas_err"}): +0.30},
        pvalues={frozenset({"meas_err"}): 0.001},
        runnable=("meas_err",),
    )
    record = license_scientist_stage(rep, strategy_id="drf", theta=THETA, q=Q)
    assert record.status is StageStatus.NOT_APPLICABLE
    assert record.evidence.kind == "entry_rule"


def test_failing_toggle_gives_unobserved_entered():
    # Significant, material, performance-reducing -> FAIL -> the Scientist would enter.
    rep = _stub(
        doe={frozenset({"lib_gap"}): -0.40},
        pvalues={frozenset({"lib_gap"}): 0.001},
        runnable=("lib_gap",),
    )
    record = license_scientist_stage(rep, strategy_id="drf", theta=THETA, q=Q)
    assert record.status is StageStatus.UNOBSERVED
    assert "lib_gap" in (record.detail or "")


def test_non_complete_scope_is_not_licensed():
    rep = _stub(
        doe={frozenset({"meas_err"}): +0.30},
        pvalues={frozenset({"meas_err"}): 0.001},
        runnable=("meas_err",),
        scope="PARTIAL",
    )
    record = license_scientist_stage(rep, strategy_id="drf", theta=THETA, q=Q)
    assert record.status is StageStatus.UNOBSERVED
    assert "not licensed" in record.evidence.detail


def test_loads_theta_q_from_thresholds_when_not_injected():
    # With no injected theta/q the licensing resolves them from the committed scientist: block.
    rep = _stub(
        doe={frozenset({"meas_err"}): +0.30},
        pvalues={frozenset({"meas_err"}): 0.001},
        runnable=("meas_err",),
    )
    record = license_scientist_stage(rep, strategy_id="drf")
    assert record.status is StageStatus.NOT_APPLICABLE
