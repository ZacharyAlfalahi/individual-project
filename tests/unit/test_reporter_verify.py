"""R6 five-layer verifier (docs/reporter/reporter_spec_v0.2.md §9): clean pass + the adversarial cases."""

from __future__ import annotations

import pytest

from _reporter_fixtures import DRF_CORE, DRF_QUANT, audited_replication_bundle, extension_bundle
from agents.reporter.bundle import ReportabilityStatus
from agents.reporter.legal_states import PipelineDisposition
from agents.reporter.renderer import RenderedDocument, render_note
from agents.reporter.verify import (
    VerificationError,
    assert_verified,
    lint_renderer,
    lint_source,
    verify_document,
)
from shared.reporting.canonical import NanValue
from shared.reporting.claims import ArtefactType, ClaimRecord, SourceLocator

needs_real = pytest.mark.skipif(
    not (DRF_CORE.exists() and DRF_QUANT.exists()), reason="recorded drf artefacts absent (local pipeline output, not shipped with the repository)"
)


def _doc(note, claims=(), evidence=()):
    return RenderedDocument(
        reporter_run_id="r",
        disposition=PipelineDisposition.EXTRACTION_ONLY,
        reportability=ReportabilityStatus.REPORTABLE,
        note_markdown=note,
        claims=tuple(claims),
        evidence=tuple(evidence),
        legal_state_hash="x",
    )


def _claim(cid, raw, displayed, *, unit="sharpe", precision=2, pointer="/x"):
    return ClaimRecord(
        claim_id=cid,
        slot_id=cid,
        source_artifact=ArtefactType.STRATEGY_RESULT,
        source_artifact_sha256="",
        source_schema_version="1",
        source_locator=SourceLocator(kind="json_pointer", pointer=pointer),
        raw_value=raw,
        displayed_value=displayed,
        unit=unit,
        precision=precision,
        conditioning_text=None,
    )


# --- clean passes ----------------------------------------------------------------------------

@needs_real
def test_real_replication_note_verifies():
    b = audited_replication_bundle(phase="F")
    report = assert_verified(render_note(b), b)
    # 10 replication/audit-DOE claims + the 3 bias-class components (ADR §5.2:
    # methodological-construction, data-quality, cross-class modulation).
    assert report.ok and report.n_claims == 13


@needs_real
def test_real_extension_note_verifies():
    b = extension_bundle(phase="F")
    assert assert_verified(render_note(b), b).ok


def test_lint_renderer_passes_on_real_renderer():
    lint_renderer()  # must not raise


# --- Layer 1: AST lint -----------------------------------------------------------------------

def test_lint_catches_float_literal_and_difference():
    bad = "def render_x(b):\n    return 0.95 - b\n"
    violations = lint_source(bad)
    assert any("float literal" in v for v in violations)
    assert any("arithmetic" in v for v in violations)


def test_lint_allows_string_concat_and_int_index():
    ok = 'def render_x(xs):\n    return "a" + "b" + str(xs[-1])\n'
    assert lint_source(ok) == []


# --- Layer 4: ledger bijection ---------------------------------------------------------------

def test_orphan_claim_fails():
    doc = _doc("This note mentions no number.\n", claims=[_claim("c", 0.64, "0.64")])
    report = verify_document(doc)
    assert not report.ok
    assert any("orphan" in e for e in report.errors)


def test_stray_token_fails():
    doc = _doc("A stray 7.77 appears in prose.\n")
    report = verify_document(doc)
    assert not report.ok
    assert any("stray number" in e for e in report.errors)


def test_two_claims_same_value_both_covered():
    # Duplicate values are legal — multiset bijection, not "exactly one".
    doc = _doc(
        "First 0.0000 and second 0.0000.\n",
        claims=[
            _claim("a", 0.0, "0.0000", unit="decimal", precision=4),
            _claim("b", 0.0, "0.0000", unit="decimal", precision=4),
        ],
    )
    assert verify_document(doc).ok


# --- Layer 3: structural -----------------------------------------------------------------

def test_wrong_label_displayed_value_fails():
    # displayed 9.99 is not re-derivable from raw 0.64 -> structural error.
    doc = _doc("Value 9.99 here.\n", claims=[_claim("c", 0.64, "9.99")])
    report = verify_document(doc)
    assert any("structural" in e for e in report.errors)


def test_percent_decimal_mismatch_fails():
    # unit percent scales x100: displayed should be 1.23, not 0.01.
    doc = _doc("Value 0.01 here.\n", claims=[_claim("c", 0.0123, "0.01", unit="percent")])
    report = verify_document(doc)
    assert any("structural" in e for e in report.errors)


# --- NaN ------------------------------------------------------------------------------------

def test_nan_claim_renders_nan_and_verifies():
    doc = _doc(
        "The metric is nan here.\n",
        claims=[_claim("c", NanValue(reason=None), "nan", unit="sharpe", precision=2)],
    )
    assert verify_document(doc).ok


def test_assert_verified_raises_on_failure():
    doc = _doc("Stray 3.14 token.\n")
    with pytest.raises(VerificationError):
        assert_verified(doc)
