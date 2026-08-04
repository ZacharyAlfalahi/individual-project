"""Regression tests for Reporter behaviours M1, M2, M3, m6b."""

from __future__ import annotations

import pytest

from _reporter_fixtures import make_bundle
from agents.reporter.bundle import ReportabilityStatus, StageStatus
from agents.reporter.legal_states import PipelineDisposition
from agents.reporter.registry import publish
from agents.reporter.renderer import RenderedDocument, render_note
from agents.reporter.verify import verify_document
from shared.reporting.claims import ArtefactType, ClaimRecord, SourceLocator


# --- M1: partial summary must not interpolate None, and must still verify ---------------------

def test_m1_partial_summary_renders_absence_not_none():
    bundle = make_bundle(
        docs={
            ArtefactType.STRATEGY_SPEC: {"paper_facts": None},
            ArtefactType.STRATEGY_RESULT: {"summary": {"t_stat": 2.0}},  # no sharpe/average
        },
        stage_status={
            "extraction": StageStatus.SUCCEEDED,
            "compilation": StageStatus.SUCCEEDED,
            "execution": StageStatus.SUCCEEDED,
        },
    )
    doc = render_note(bundle)
    assert "None" not in doc.note_markdown
    assert "summary metrics are incomplete" in doc.note_markdown
    assert verify_document(doc, bundle).ok


# --- M2: a present-but-scopeless audit core must not crash ------------------------------------

def test_m2_audit_core_without_scope_does_not_crash():
    bundle = make_bundle(
        docs={
            ArtefactType.STRATEGY_SPEC: {"paper_facts": None},
            ArtefactType.STRATEGY_RESULT: {
                "summary": {"sharpe": 0.5, "t_stat": 2.0, "average": 0.001}
            },
            ArtefactType.AUDIT_REPORT: {"runnable_toggles": []},  # no audit_scope key
        },
        stage_status={
            "extraction": StageStatus.SUCCEEDED,
            "compilation": StageStatus.SUCCEEDED,
            "execution": StageStatus.SUCCEEDED,
            "audit": StageStatus.UNOBSERVED,  # loader's malformed-core outcome
        },
    )
    doc = render_note(bundle)  # must not raise
    assert doc.disposition is PipelineDisposition.EXECUTED_NOT_AUDITED
    assert "carries no recognised scope" in doc.note_markdown
    assert verify_document(doc, bundle).ok


# --- M3: a correctly-emitted PERCENT claim verifies (scaled backstop universe) ----------------

def test_m3_percent_claim_verifies():
    # raw 0.0123 decimal, unit percent -> displayed "1.23" (scaled x100).
    percent = ClaimRecord(
        claim_id="p",
        slot_id="p",
        source_artifact=ArtefactType.STRATEGY_RESULT,
        source_artifact_sha256="",
        source_schema_version="1",
        source_locator=SourceLocator(kind="json_pointer", pointer="/x"),
        raw_value=0.0123,
        displayed_value="1.23",
        unit="percent",
        precision=2,
        conditioning_text=None,
    )
    doc = RenderedDocument(
        reporter_run_id="r",
        disposition=PipelineDisposition.EXTRACTION_ONLY,
        reportability=ReportabilityStatus.REPORTABLE,
        note_markdown="The value is 1.23 percent.\n",
        claims=(percent,),
        evidence=(),
        legal_state_hash="x",
    )
    report = verify_document(doc)
    assert report.ok, report.errors


# --- m6b: publishing nothing must not clobber a prior publication -----------------------------

def test_m6b_publish_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        publish([], out_root=tmp_path / "out", staging_root=tmp_path / "s")
