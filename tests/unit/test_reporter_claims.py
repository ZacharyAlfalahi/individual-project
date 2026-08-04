"""Ledger types (docs/reporter/reporter_spec_v0.2.md §8.1/§8.2/§8.4): construction, guards, to_dict shapes."""

from __future__ import annotations

import pytest

from shared.reporting.canonical import NanValue
from shared.reporting.claims import (
    ArtefactType,
    ClaimRecord,
    ClaimSpec,
    EvidenceBlock,
    EvidenceVerification,
    SourceLocator,
    Unit,
)


def test_json_pointer_locator_ok():
    loc = SourceLocator(kind="json_pointer", pointer="/summary/sharpe")
    assert loc.to_dict() == {"kind": "json_pointer", "pointer": "/summary/sharpe"}


def test_empty_pointer_is_allowed():
    SourceLocator(kind="json_pointer", pointer="")


def test_pointer_must_start_with_slash():
    with pytest.raises(ValueError):
        SourceLocator(kind="json_pointer", pointer="summary/sharpe")


def test_projection_pointer_requires_projection_id():
    with pytest.raises(ValueError):
        SourceLocator(kind="projection_pointer", pointer="/sharpe")
    loc = SourceLocator(
        kind="projection_pointer", pointer="/sharpe", projection_id="strategy_result_v1"
    )
    assert loc.to_dict()["projection_id"] == "strategy_result_v1"


def test_json_pointer_must_not_carry_projection_id():
    with pytest.raises(ValueError):
        SourceLocator(kind="json_pointer", pointer="/x", projection_id="oops")


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        SourceLocator(kind="xml_path", pointer="/x")  # type: ignore[arg-type]


def test_claimspec_to_dict_has_all_fields():
    spec = ClaimSpec(
        claim_id="c1",
        slot_id="replication.sharpe",
        source_artifact=ArtefactType.STRATEGY_RESULT,
        source_locator=SourceLocator(kind="json_pointer", pointer="/summary/sharpe"),
        formatter_id="decimal_4dp",
        unit=Unit.SHARPE,
        conditioning_pointer=None,
    )
    d = spec.to_dict()
    assert d["source_artifact"] == "strategy_result"
    assert d["unit"] == "sharpe"
    assert d["conditioning_pointer"] is None


def test_claimspec_rejects_wrong_types():
    loc = SourceLocator(kind="json_pointer", pointer="/x")
    with pytest.raises(TypeError):
        ClaimSpec(
            claim_id="c",
            slot_id="s",
            source_artifact="strategy_result",  # not an ArtefactType
            source_locator=loc,
            formatter_id="f",
            unit=Unit.DECIMAL,
            conditioning_pointer=None,
        )


def test_claimrecord_accepts_nanvalue_but_not_bool():
    loc = SourceLocator(kind="json_pointer", pointer="/x")
    rec = ClaimRecord(
        claim_id="c",
        slot_id="s",
        source_artifact=ArtefactType.AUDIT_REPORT,
        source_artifact_sha256="deadbeef",
        source_schema_version="1",
        source_locator=loc,
        raw_value=NanValue(reason="zero variance"),
        displayed_value="nan",
        unit="sharpe",
        precision=None,
        conditioning_text=None,
    )
    assert isinstance(rec.raw_value, NanValue)
    with pytest.raises(TypeError):
        ClaimRecord(
            claim_id="c",
            slot_id="s",
            source_artifact=ArtefactType.AUDIT_REPORT,
            source_artifact_sha256="d",
            source_schema_version="1",
            source_locator=loc,
            raw_value=True,
            displayed_value="x",
            unit="u",
            precision=None,
            conditioning_text=None,
        )


def test_evidence_block_to_dict_and_verification():
    loc = SourceLocator(kind="json_pointer", pointer="")
    ev = EvidenceBlock(
        evidence_id="e1",
        source_artifact=ArtefactType.AUDIT_REPORT,
        source_artifact_sha256="abc",
        source_locator=loc,
        verbatim_text="Audit scope: complete.",
        verification=EvidenceVerification(
            verifier="verify_numbers", ok=True, n_checked=3, unverified=()
        ),
    )
    d = ev.to_dict()
    assert d["verification"]["ok"] is True
    assert d["source_artifact"] == "audit_report"
