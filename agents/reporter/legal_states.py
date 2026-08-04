"""legal_states.py — R4 state classification (docs/reporter/reporter_spec_v0.2.md §7.1).

Classifies a run's stage-completeness state into a `PipelineDisposition` via an exhaustive,
content-hashed legal-tuple table (`legal_states.yaml`). This is NOT the Scientist's ordered
failure-gate logic — combinations here can be illegal rather than merely later, so a tuple
absent from the table raises `IllegalStateError` naming the tuple (INV-13) instead of being
silently coerced to a nearby disposition.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping

import yaml

from shared.reporting.canonical import canonical_hash

from .bundle import StageRecord

_TABLE_FILE = Path(__file__).resolve().parent / "legal_states.yaml"

# The stages whose statuses (plus audit_scope) determine the disposition, in key order.
_KEY_STAGES = ("extraction", "compilation", "execution", "audit", "scientist")


class PipelineDisposition(str, Enum):
    EXTRACTION_ONLY = "extraction_only"
    COMPILATION_REFUSED = "compilation_refused"
    COMPILED_NOT_EXECUTED = "compiled_not_executed"
    EXECUTION_UNOBSERVED = "execution_unobserved"
    EXECUTED_NOT_AUDITED = "executed_not_audited"
    AUDIT_REFUSED = "audit_refused"
    AUDIT_PARTIAL = "audit_partial"
    AUDIT_COMPLETE_NO_EXTENSION = "audit_complete_no_extension"
    EXTENSION_PATH = "extension_path"


class IllegalStateError(ValueError):
    """A stage tuple is not in the legal-state table — an impossible artefact combination."""


@dataclass(frozen=True)
class LegalStateTable:
    rows: dict[tuple, str]
    schema_version: int
    content_hash: str


def load_table(path: str | Path | None = None) -> LegalStateTable:
    p = Path(path) if path is not None else _TABLE_FILE
    data = yaml.safe_load(p.read_text())
    if not isinstance(data, dict) or "rows" not in data:
        raise IllegalStateError("legal_states.yaml is malformed")
    rows: dict[tuple, str] = {}
    for row in data["rows"]:
        key = tuple(row["key"])
        if len(key) != len(_KEY_STAGES) + 1:
            raise IllegalStateError(f"legal-state key has wrong arity: {key}")
        if key in rows:
            raise IllegalStateError(f"duplicate legal-state key: {key}")
        rows[key] = row["disposition"]
    return LegalStateTable(
        rows=rows,
        schema_version=int(data["schema_version"]),
        content_hash=canonical_hash(data),
    )


def classify_disposition(
    stages: Mapping[str, StageRecord],
    audit_scope: str | None,
    *,
    table: LegalStateTable | None = None,
) -> PipelineDisposition:
    """Classify the run's disposition from its stage records and audit scope. Raises
    `IllegalStateError` if the tuple is not a declared legal state."""
    tbl = table if table is not None else load_table()
    key = tuple(stages[s].status.value for s in _KEY_STAGES) + (audit_scope or "none",)
    disposition = tbl.rows.get(key)
    if disposition is None:
        raise IllegalStateError(f"stage tuple {key} is not a legal state")
    return PipelineDisposition(disposition)
