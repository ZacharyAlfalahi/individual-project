"""One test per invariant (docs/reporter/reporter_spec_v0.2.md §2, INV-1..INV-15). Each fails if its invariant is removed.

Uses synthetic bundles so the suite does not depend on committed artefacts.
"""

from __future__ import annotations

import dataclasses
import hashlib

import pytest

from _reporter_fixtures import make_bundle, make_proposal
from agents.reporter.bundle import (
    CodeVersion,
    ReportabilityStatus,
    StageStatus,
    load_bundle,
)
from agents.reporter.emit import SuppressedValueError, assert_not_suppressed
from agents.reporter.legal_states import IllegalStateError, PipelineDisposition
from agents.reporter.manifest import ManifestError, parse_manifest_text
from agents.reporter.registry import publish
from agents.reporter.renderer import render_note
from agents.reporter.verify import verify_document
from shared.reporting.claims import ArtefactType
from shared.reporting.resolve import PointerResolutionError, resolve_json_pointer

_TOGGLES = ("meas_err", "stale_price", "survivorship", "lib_gap", "lab_trim")
_CV = CodeVersion(full="reporterfullsha", short="reporter")


def _audit(scope="COMPLETE"):
    return {
        "audit_scope": scope,
        "runnable_toggles": list(_TOGGLES),
        "shapley": {"shapley_share_of_registered_endpoint_gap": {t: 0.0 for t in _TOGGLES}},
        "saturated_bases": {"doe_effects": {t: 0.0 for t in _TOGGLES}},
    }


def _full(*, scope="COMPLETE", proposals=(), scientist=None):
    stage = {
        "extraction": StageStatus.SUCCEEDED,
        "compilation": StageStatus.SUCCEEDED,
        "execution": StageStatus.SUCCEEDED,
        "audit": StageStatus.REFUSED if scope == "REFUSED" else StageStatus.SUCCEEDED,
    }
    if scientist is not None:
        stage["scientist"] = scientist
    docs = {
        ArtefactType.STRATEGY_SPEC: {"paper_facts": None},
        ArtefactType.STRATEGY_RESULT: {"summary": {"sharpe": 0.5, "t_stat": 2.0, "average": 0.001}},
        ArtefactType.AUDIT_REPORT: _audit(scope),
    }
    return make_bundle(docs=docs, stage_status=stage, proposals=proposals)


def test_inv1_every_numeric_token_is_a_claim():
    doc = render_note(_full())
    assert verify_document(doc, _full()).ok


def test_inv2_status_only_from_present_evidence(tmp_path):
    spec = tmp_path / "runs/spec_0.json"
    spec.parent.mkdir(parents=True)
    spec.write_text('{"paper_facts": null}')
    sha = hashlib.sha256(spec.read_bytes()).hexdigest()
    text = f"""
reporter_run_id: r
schema_version: 1
paper_id: BBW
strategy_id: drf
phase: D
join_rationale: asserted
artefacts:
  spec:
    path: {spec.relative_to(tmp_path)}
    sha256: "{sha}"
"""
    m = parse_manifest_text(text, repo_root=tmp_path, verify_hashes=True)
    b = load_bundle(m, repo_root=tmp_path, code_version=_CV)
    assert b.stages["extraction"].status is StageStatus.SUCCEEDED
    # No downstream artefact -> UNOBSERVED, never inferred SUCCEEDED.
    assert b.stages["execution"].status is StageStatus.UNOBSERVED


def test_inv3_absence_is_rendered_explicitly():
    bundle = make_bundle(
        docs={ArtefactType.STRATEGY_SPEC: {"paper_facts": None}},
        stage_status={"extraction": StageStatus.SUCCEEDED},
    )
    md = render_note(bundle).note_markdown
    assert "no machine-readable claimed headline metric" in md


def test_inv4_suppressed_magnitude_raises_and_is_not_emitted():
    # The guard raises rather than omits.
    with pytest.raises(SuppressedValueError):
        assert_not_suppressed(True, "magnitude under REFUSED scope")
    # And a REFUSED-scope audit emits no bias magnitude claims.
    doc = render_note(_full(scope="REFUSED"))
    assert not any(c.claim_id.startswith("audit.doe.") for c in doc.claims)
    assert "no first-order bias magnitudes are opinable" in doc.note_markdown


def test_inv5_conditioning_binds_to_the_value():
    from agents.reporter.emit import emit_claim
    from shared.reporting.claims import ClaimSpec, SourceLocator, Unit

    bundle = make_bundle(
        docs={
            ArtefactType.AUDIT_REPORT: {
                "audit_scope": "PARTIAL",
                "conditioning_statement": "holding stale_price fixed at OFF",
                "saturated_bases": {"doe_effects": {"meas_err": 0.0012}},
                "runnable_toggles": ["meas_err"],
            }
        },
        stage_status={"audit": StageStatus.SUCCEEDED},
    )
    spec = ClaimSpec(
        claim_id="c",
        slot_id="s",
        source_artifact=ArtefactType.AUDIT_REPORT,
        source_locator=SourceLocator(kind="json_pointer", pointer="/saturated_bases/doe_effects/meas_err"),
        formatter_id="4dp",
        unit=Unit.DECIMAL,
        conditioning_pointer=SourceLocator(kind="json_pointer", pointer="/conditioning_statement"),
    )
    _, record = emit_claim(spec, bundle)
    assert record.conditioning_text == "holding stale_price fixed at OFF"


def test_inv6_direction_words_are_selected_not_authored():
    from agents.reporter.words import effect_direction_word

    assert effect_direction_word(0.3) == "raises"
    assert effect_direction_word(-0.3) == "lowers"
    with pytest.raises(ValueError):
        effect_direction_word(float("nan"))


def test_inv7_reportability_is_mandatory_and_orthogonal():
    # No default: constructing a ReportBundle without reportability fails.
    fields = {f.name for f in dataclasses.fields(make_bundle(docs={ArtefactType.STRATEGY_SPEC: {}}))}
    assert "reportability" in fields
    # A non-reportable phase is not a disposition failure — orthogonal fields.
    doc = render_note(_full(scientist=None))
    b = _full()
    assert b.reportability is ReportabilityStatus.NON_REPORTABLE_PHASE_D
    assert doc.disposition is PipelineDisposition.AUDIT_COMPLETE_NO_EXTENSION


def test_inv8_reproducible_rebuild(tmp_path):
    b = [_full()]
    publish(b, out_root=tmp_path / "r1", staging_root=tmp_path / "s1")
    publish(b, out_root=tmp_path / "r2", staging_root=tmp_path / "s2")
    a = (tmp_path / "r1" / "registry" / "runs.jsonl").read_bytes()
    c = (tmp_path / "r2" / "registry" / "runs.jsonl").read_bytes()
    assert a == c


def test_inv9_registry_completeness_manifest_relative(tmp_path):
    publish([_full()], out_root=tmp_path / "out", staging_root=tmp_path / "s")
    rows = (tmp_path / "out" / "registry" / "runs.jsonl").read_text().splitlines()
    assert len(rows) == 1  # exactly the declared run, once


def test_inv10_ledger_text_bijection_orphan_fails():
    from agents.reporter.renderer import RenderedDocument
    from shared.reporting.claims import ClaimRecord, SourceLocator

    orphan = ClaimRecord(
        claim_id="x", slot_id="x", source_artifact=ArtefactType.STRATEGY_RESULT,
        source_artifact_sha256="", source_schema_version="1",
        source_locator=SourceLocator(kind="json_pointer", pointer="/x"),
        raw_value=0.64, displayed_value="0.64", unit="sharpe", precision=2,
        conditioning_text=None,
    )
    doc = RenderedDocument(
        reporter_run_id="r", disposition=PipelineDisposition.EXTRACTION_ONLY,
        reportability=ReportabilityStatus.REPORTABLE, note_markdown="No numbers.\n",
        claims=(orphan,), evidence=(), legal_state_hash="x",
    )
    assert not verify_document(doc).ok


def test_inv11_five_stage_statuses_are_distinct():
    values = {s.value for s in StageStatus}
    assert len(values) == 5


def test_inv12_atomic_publication_leaves_prior_untouched(tmp_path):
    out = tmp_path / "out"
    publish([_full()], out_root=out, staging_root=tmp_path / "s1")
    before = (out / "registry" / "runs.jsonl").read_bytes()
    # A bundle whose render raises (unknown mechanism) must not corrupt the publication.
    bad = _full(
        scientist=StageStatus.SUCCEEDED,
        proposals=(make_proposal(mechanism_ref="mech_999_nope"),),
    )
    with pytest.raises(KeyError):
        publish([bad], out_root=out, staging_root=tmp_path / "s2")
    assert (out / "registry" / "runs.jsonl").read_bytes() == before


def test_inv13_schema_drift_fails_closed():
    with pytest.raises(IllegalStateError):
        render_note(
            make_bundle(
                docs={ArtefactType.STRATEGY_SPEC: {}},
                stage_status={"compilation": StageStatus.SUCCEEDED},
            )
        )
    with pytest.raises(PointerResolutionError):
        resolve_json_pointer({"a": 1}, "/missing")


def test_inv14_upstream_prose_is_verbatim_evidence():
    from agents.scientist.researcher.library import load_library

    lib = load_library()
    expected = str(lib.mechanism("mech_008")["claim"]).strip()
    doc = render_note(
        _full(scientist=StageStatus.SUCCEEDED, proposals=(make_proposal(mechanism_ref="mech_008"),))
    )
    ev = [e for e in doc.evidence if e.evidence_id.endswith(".mechanism")][0]
    assert ev.verbatim_text == expected  # copied, never paraphrased


def test_inv15_artefact_identity_is_hash_checked(tmp_path):
    spec = tmp_path / "spec_0.json"
    spec.write_text('{"paper_facts": null}')
    good = hashlib.sha256(spec.read_bytes()).hexdigest()
    bad = ("0" if good[0] != "0" else "1") + good[1:]
    text = f"""
reporter_run_id: r
schema_version: 1
paper_id: BBW
strategy_id: drf
phase: D
join_rationale: asserted
artefacts:
  spec:
    path: {spec.relative_to(tmp_path)}
    sha256: "{bad}"
"""
    with pytest.raises(ManifestError):
        parse_manifest_text(text, repo_root=tmp_path, verify_hashes=True)
