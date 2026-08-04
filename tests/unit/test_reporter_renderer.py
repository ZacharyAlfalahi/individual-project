"""R5 replication-path renderer (docs/reporter/reporter_spec_v0.2.md §7.2, §8, C9)."""

from __future__ import annotations

import pytest

from _reporter_fixtures import audited_replication_bundle, make_bundle
from agents.reporter.bundle import StageStatus
from agents.reporter.legal_states import PipelineDisposition
from agents.reporter.renderer import render_note
from shared.reporting.claims import ArtefactType

_HAS_REAL = audited_replication_bundle  # skip helper below checks file presence


def _real_available() -> bool:
    from _reporter_fixtures import DRF_CORE, DRF_QUANT

    return DRF_CORE.exists() and DRF_QUANT.exists()


needs_real = pytest.mark.skipif(
    not _real_available(), reason="committed drf artefacts absent"
)


@needs_real
def test_audited_replication_note_renders():
    doc = render_note(audited_replication_bundle())
    assert doc.disposition is PipelineDisposition.AUDIT_COMPLETE_NO_EXTENSION
    md = doc.note_markdown
    for heading in ("## Extraction", "## Replication", "## Audit", "## Provenance"):
        assert heading in md
    # Non-reportable banner (phase D) is the first content line (INV-7).
    assert md.lstrip().startswith("> **NON-REPORTABLE**")


@needs_real
def test_every_claim_token_appears_in_the_note():
    # INV-10 precursor: each claim's displayed value is actually in the prose.
    doc = render_note(audited_replication_bundle())
    assert doc.claims  # non-empty
    for claim in doc.claims:
        assert claim.displayed_value in doc.note_markdown, claim.claim_id


@needs_real
def test_replication_and_audit_claims_present():
    doc = render_note(audited_replication_bundle())
    ids = {c.claim_id for c in doc.claims}
    assert "replication.sharpe" in ids
    assert "replication.tstat" in ids
    assert any(cid.startswith("audit.doe.") for cid in ids)
    # Audit claims are sourced from the audit core artefact.
    audit_claims = [c for c in doc.claims if c.claim_id.startswith("audit.doe.")]
    assert all(c.source_artifact is ArtefactType.AUDIT_REPORT for c in audit_claims)


@needs_real
def test_reportable_phase_f_has_no_banner():
    doc = render_note(audited_replication_bundle(phase="F"))
    assert "NON-REPORTABLE" not in doc.note_markdown


@needs_real
def test_bias_class_section_renders_three_components():
    # ADR §5.2: the bias-class decomposition section binds the three typed-leaf
    # components on the all-runnable drf core.
    doc = render_note(audited_replication_bundle(phase="F"))
    md = doc.note_markdown
    assert "## Audit — bias-class decomposition" in md
    ids = {c.claim_id for c in doc.claims}
    assert "audit.bias_class.methodological_construction_component" in ids
    assert "audit.bias_class.data_quality_component" in ids
    assert "audit.bias_class.cross_class_modulation" in ids
    # Never the banned single-total framing, never a bare "None" token.
    assert "total bias" not in md.lower()
    assert "None" not in md


# --- §5.4/§7: an absent component states its reason without collapsing the taxonomy --

def test_absence_reason_distinguishes_not_applicable_from_input_unavailable():
    from agents.auditor.schemas.toggle import data_quality_toggles
    from agents.reporter.renderer import _absence_reason

    dq = data_quality_toggles()  # ("meas_err",)
    declared_na = {
        "runnable_toggles": ["stale_price", "survivorship", "lib_gap", "lab_trim"],
        "not_applicable_toggles": ["meas_err"],
    }
    held_unavailable = {
        "runnable_toggles": ["stale_price", "survivorship", "lib_gap", "lab_trim"],
        "not_applicable_toggles": [],  # meas_err non-runnable but NOT not_applicable
    }
    assert "not applicable" in _absence_reason(declared_na, dq)
    reason = _absence_reason(held_unavailable, dq)
    assert "not opinable" in reason and "not applicable" not in reason


def test_bias_class_section_states_absence_when_block_missing():
    # A legacy/synthetic core without the partition block: stated as an absence,
    # never fabricated, no numbers emitted.
    from agents.reporter.renderer import render_bias_class_partition

    bundle = make_bundle(
        docs={
            ArtefactType.AUDIT_REPORT: {
                "audit_scope": "COMPLETE",
                "runnable_toggles": ["meas_err", "stale_price", "survivorship",
                                     "lib_gap", "lab_trim"],
            }
        },
        stage_status={"extraction": StageStatus.SUCCEEDED, "audit": StageStatus.SUCCEEDED},
    )
    frag = render_bias_class_partition(bundle)
    assert "not present in this audit core" in frag.text
    assert frag.claims == ()


def test_extraction_only_disposition_skips_downstream_sections():
    bundle = make_bundle(
        docs={ArtefactType.STRATEGY_SPEC: {"paper_facts": None}},
        stage_status={"extraction": StageStatus.SUCCEEDED},
    )
    doc = render_note(bundle)
    assert doc.disposition is PipelineDisposition.EXTRACTION_ONLY
    assert "## Replication" not in doc.note_markdown
    assert "## Audit" not in doc.note_markdown
    # Absent paper facts render an explicit statement, not silence (INV-3).
    assert "no machine-readable claimed headline metric" in doc.note_markdown


def test_note_ends_with_single_trailing_newline():
    bundle = make_bundle(
        docs={ArtefactType.STRATEGY_SPEC: {"paper_facts": None}},
        stage_status={"extraction": StageStatus.SUCCEEDED},
    )
    md = render_note(bundle).note_markdown
    assert md.endswith("\n")
    assert not md.endswith("\n\n")
