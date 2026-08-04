"""R5 extension path (docs/reporter/reporter_spec_v0.2.md §7.2, D1): proposals / diagnostics / holdout, fixtures only."""

from __future__ import annotations

import pytest

from _reporter_fixtures import DRF_CORE, DRF_QUANT, extension_bundle, make_bundle
from agents.reporter.bundle import StageStatus
from agents.reporter.legal_states import PipelineDisposition
from agents.reporter.renderer import render_note
from shared.reporting.claims import ArtefactType

needs_real = pytest.mark.skipif(
    not (DRF_CORE.exists() and DRF_QUANT.exists()), reason="committed drf artefacts absent"
)


@needs_real
def test_extension_path_disposition_and_sections():
    doc = render_note(extension_bundle())
    assert doc.disposition is PipelineDisposition.EXTENSION_PATH
    for heading in ("## Proposals", "## Diagnostics", "## Holdout"):
        assert heading in doc.note_markdown


@needs_real
def test_mechanism_claim_is_verbatim_evidence():
    doc = render_note(extension_bundle())
    mech_ev = [e for e in doc.evidence if e.evidence_id.endswith(".mechanism")]
    assert mech_ev, "expected a mechanism evidence block"
    ev = mech_ev[0]
    assert ev.source_artifact is ArtefactType.MECHANISM_LIBRARY
    assert len(ev.source_artifact_sha256) == 64  # library version_hash
    # The verbatim mechanism claim text is present in the note.
    assert ev.verbatim_text[:30] in doc.note_markdown


@needs_real
def test_crowding_diagnostic_claims_emitted():
    doc = render_note(extension_bundle())
    ids = {c.claim_id for c in doc.claims}
    assert any(cid.endswith(".crowding.alpha") for cid in ids)
    assert any(cid.endswith(".crowding.alpha_t") for cid in ids)


@needs_real
def test_holdout_has_no_success_word_and_three_statements():
    doc = render_note(extension_bundle())
    holdout_section = doc.note_markdown.split("## Holdout", 1)[1].split("## ", 1)[0]
    assert "success" not in holdout_section.lower()
    assert "Sharpe sign" in holdout_section
    assert "Sharpe interval" in holdout_section
    assert "paired difference" in holdout_section


@needs_real
def test_capacity_and_regime_render_unobserved():
    doc = render_note(extension_bundle())
    diag = doc.note_markdown.split("## Diagnostics", 1)[1].split("## ", 1)[0]
    assert "diagnostics_capacity: unobserved" in diag
    assert "diagnostics_regime: unobserved" in diag


@needs_real
def test_unknown_mechanism_ref_fails_closed():
    from agents.reporter.bundle import ProposalReport

    bundle = make_bundle(
        docs={ArtefactType.STRATEGY_SPEC: {"paper_facts": None}},
        stage_status={"extraction": StageStatus.SUCCEEDED},
        proposals=(
            ProposalReport(
                record={"proposal_id": "p9", "final_outcome": "AUDIT_FAILURE"},
                proposal={"mechanism_ref": "mech_999_nonexistent"},
            ),
        ),
    )
    with pytest.raises(KeyError):
        render_note(bundle)


@needs_real
def test_extension_claim_tokens_in_note():
    doc = render_note(extension_bundle())
    for claim in doc.claims:
        assert claim.displayed_value in doc.note_markdown, claim.claim_id
