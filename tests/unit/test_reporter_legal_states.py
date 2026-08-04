"""R4 legal-state classification (docs/reporter/reporter_spec_v0.2.md §7.1): every legal tuple + illegal raises."""

from __future__ import annotations

import pytest

from agents.reporter.bundle import STAGES, StageEvidence, StageRecord, StageStatus
from agents.reporter.legal_states import (
    IllegalStateError,
    PipelineDisposition,
    classify_disposition,
    load_table,
)

S = StageStatus.SUCCEEDED
U = StageStatus.UNOBSERVED
R = StageStatus.REFUSED
F = StageStatus.FAILED
N = StageStatus.NOT_APPLICABLE


def _stages(**overrides):
    out = {}
    for stage in STAGES:
        status = overrides.get(stage, U)
        out[stage] = StageRecord(
            stage=stage, status=status, evidence=StageEvidence("t", "t")
        )
    return out


@pytest.mark.parametrize(
    "overrides,scope,expected",
    [
        ({"extraction": S}, None, PipelineDisposition.EXTRACTION_ONLY),
        ({"extraction": S, "compilation": R}, None, PipelineDisposition.COMPILATION_REFUSED),
        ({"extraction": S, "compilation": S}, None, PipelineDisposition.EXECUTION_UNOBSERVED),
        (
            {"extraction": S, "compilation": S, "execution": F},
            None,
            PipelineDisposition.COMPILED_NOT_EXECUTED,
        ),
        (
            {"extraction": S, "compilation": S, "execution": S},
            None,
            PipelineDisposition.EXECUTED_NOT_AUDITED,
        ),
        (
            {"extraction": S, "compilation": S, "execution": S, "audit": R},
            "REFUSED",
            PipelineDisposition.AUDIT_REFUSED,
        ),
        (
            {"extraction": S, "compilation": S, "execution": S, "audit": S},
            "PARTIAL",
            PipelineDisposition.AUDIT_PARTIAL,
        ),
        (
            {"extraction": S, "compilation": S, "execution": S, "audit": S},
            "COMPLETE",
            PipelineDisposition.AUDIT_COMPLETE_NO_EXTENSION,
        ),
        (
            {"extraction": S, "compilation": S, "execution": S, "audit": S, "scientist": N},
            "COMPLETE",
            PipelineDisposition.AUDIT_COMPLETE_NO_EXTENSION,
        ),
        (
            {"extraction": S, "compilation": S, "execution": S, "audit": S, "scientist": S},
            "COMPLETE",
            PipelineDisposition.EXTENSION_PATH,
        ),
    ],
)
def test_legal_tuples_classify(overrides, scope, expected):
    assert classify_disposition(_stages(**overrides), scope) is expected


def test_illegal_tuple_raises_naming_it():
    # extraction UNOBSERVED is impossible (spec is always present) -> not a legal state.
    with pytest.raises(IllegalStateError):
        classify_disposition(_stages(compilation=S), None)


def test_audit_succeeded_without_scope_is_illegal():
    with pytest.raises(IllegalStateError):
        classify_disposition(
            _stages(extraction=S, compilation=S, execution=S, audit=S), None
        )


def test_table_is_content_hashed():
    t = load_table()
    assert len(t.content_hash) == 64
    assert t.schema_version == 1
