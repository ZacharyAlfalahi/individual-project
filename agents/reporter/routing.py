"""routing.py — fixture-only Scientist-stage licensing (docs/reporter/reporter_spec_v0.2.md §5.3, C9).

From persisted artefacts the Auditor's `fdr.decisions` are not on disk and no `AuditReport` can
be reconstructed (C9), so the loader defaults the Scientist stage to `UNOBSERVED`. This module
retains the spec's §5.3 licensing for FIXTURES / a future full-audit persistence: given a typed
in-memory `AuditReport`, it calls the EXISTING `build_scientist_case` (no re-authored derivation)
and licenses a single `NOT_APPLICABLE` when the audit scope is COMPLETE and no toggle FAILs.
"""

from __future__ import annotations

from shared.handoff.scientist_case import build_scientist_case, load_entry_rule_params

from agents.scientist.schemas.case import DevelopmentWindow, HoldoutStatus

from .bundle import StageEvidence, StageRecord, StageStatus


def license_scientist_stage(
    audit_report,
    *,
    strategy_id: str,
    refused_toggles: frozenset = frozenset(),
    theta: float | None = None,
    q: float | None = None,
) -> StageRecord:
    """Return the Scientist `StageRecord` for a typed in-memory `AuditReport`. Licenses
    `NOT_APPLICABLE` only when audit_scope is COMPLETE and `build_scientist_case` finds no
    failing toggle; otherwise `UNOBSERVED` (a non-COMPLETE scope is not licensed off-disk)."""
    scope = getattr(audit_report.core, "audit_scope", None)
    if scope != "COMPLETE":
        return StageRecord(
            stage="scientist",
            status=StageStatus.UNOBSERVED,
            evidence=StageEvidence(
                kind="absent",
                detail=f"audit scope {scope!r}; entry rule not licensed",
                source="routing",
            ),
        )
    if theta is None or q is None:
        params = load_entry_rule_params()
        theta = params.theta if theta is None else theta
        q = params.q if q is None else q

    case = build_scientist_case(
        audit_report,
        strategy_id=strategy_id,
        case_id=f"reporter_{strategy_id}",
        corrected_quant_config_ref="n/a",
        corrected_run_ref="n/a",
        audit_report_ref="n/a",
        development_window=DevelopmentWindow(start="2002-07", end="2021-12"),
        holdout_status=HoldoutStatus(accessible=False),
        theta=theta,
        q=q,
        refused_toggles=refused_toggles,
    )
    if case.failed_check_ids == ():
        return StageRecord(
            stage="scientist",
            status=StageStatus.NOT_APPLICABLE,
            evidence=StageEvidence(
                kind="entry_rule",
                # No numeric theta/q in the detail — it is rendered verbatim in the stage
                # summary, where a bare number would be an unbound token (INV-10).
                detail="apply_entry_rule found no failing toggle (theta/q from the "
                "pre-registered scientist block)",
                source="build_scientist_case",
            ),
        )
    return StageRecord(
        stage="scientist",
        status=StageStatus.UNOBSERVED,
        evidence=StageEvidence(
            kind="entry_rule",
            detail="audit produced failing toggle(s); the Scientist would enter",
            source="build_scientist_case",
        ),
        detail=",".join(case.failed_check_ids),
    )
