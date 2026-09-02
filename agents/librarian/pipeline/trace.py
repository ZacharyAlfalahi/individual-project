"""
ExtractionTrace sidecar (build brief §5.5, D12).

The published ``StrategySpec`` carries only merged, provenance-tagged facts. The
per-model reasoning that produced each fact lives here, in a separate
``ExtractionTrace`` -- hash-referenced from the spec header (``trace_sha256``).
The trace is the RQ1 dataset (D12): both models' raw values, quotes, and
locate results; the normalised values agreement was judged on; the agreement
bit; the final (tag, reason); and which quote shipped.

Shape (per build brief §5.5):

    ExtractionTrace
      +- header:  TraceRunHeader   -- run/registry/model/timestamp provenance
      +- records: (FieldTraceRecord, ...)   -- one per (paper, strategy, field)

    FieldTraceRecord
      +- field
      +- model_a: ModelTrace   -- raw, quote, locate_result (page/span or None)
      +- model_b: ModelTrace
      +- normalised_a / normalised_b   -- the values agreement was judged on
      +- agreement: bool
      +- final_tag / final_reason      -- the merged Inherited's tag + reason
      +- ship_choice   -- "model_a" | "model_b" | None (which quote shipped)

``sha256()`` hashes the canonically-serialised ``to_dict()`` (sorted-key JSON) so
the header's ``trace_sha256`` is reproducible and byte-comparable across loads.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from agents.quant.config import Locator

from ..errors import LibrarianSchemaError

# The legal ship choices (which model's quote was shipped for a STATED value).
SHIP_CHOICES: frozenset[str] = frozenset(("model_a", "model_b"))


def _locator_to_dict(loc: Locator | None) -> dict | None:
    return loc.to_dict() if loc is not None else None


@dataclass(frozen=True)
class ModelTrace:
    """One model's contribution to a field: its raw (pre-normalisation) answer,
    the quote it claimed, and whether that quote located (a ``Locator`` or
    ``None``). ``answered=False`` records a model that reported the paper
    silent. ``parse_failed=True`` (B2, additive) marks a silence that was a
    FORMAT/SCHEMA failure rather than the model reporting genuine paper silence
    -- the distinct trace signal the §3.6 gate reads to keep the quote/format
    bucket out of the missed-evidence denominator."""

    answered: bool
    raw: object = None
    quote: str | None = None
    locate_result: Locator | None = None
    model_id: str | None = None
    parse_failed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.answered, bool):
            raise LibrarianSchemaError("ModelTrace.answered must be a bool")
        if self.locate_result is not None and not isinstance(self.locate_result, Locator):
            raise LibrarianSchemaError("ModelTrace.locate_result must be a Locator or None")

    def to_dict(self) -> dict:
        return {
            "answered": self.answered,
            "raw": self.raw,
            "quote": self.quote,
            "locate_result": _locator_to_dict(self.locate_result),
            "model_id": self.model_id,
            "parse_failed": self.parse_failed,
        }


@dataclass(frozen=True)
class FieldTraceRecord:
    """The full dual-model trace for one field of one strategy (D12)."""

    field: str
    model_a: ModelTrace
    model_b: ModelTrace
    normalised_a: object
    normalised_b: object
    agreement: bool
    final_tag: str
    final_reason: str
    ship_choice: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or self.field.strip() == "":
            raise LibrarianSchemaError("FieldTraceRecord.field must be a non-empty string")
        for name, val in (("model_a", self.model_a), ("model_b", self.model_b)):
            if not isinstance(val, ModelTrace):
                raise LibrarianSchemaError(f"FieldTraceRecord.{name} must be a ModelTrace")
        if not isinstance(self.agreement, bool):
            raise LibrarianSchemaError("FieldTraceRecord.agreement must be a bool")
        for name, val in (("final_tag", self.final_tag), ("final_reason", self.final_reason)):
            if not isinstance(val, str) or val.strip() == "":
                raise LibrarianSchemaError(f"FieldTraceRecord.{name} must be a non-empty string")
        if self.ship_choice is not None and self.ship_choice not in SHIP_CHOICES:
            raise LibrarianSchemaError(
                f"FieldTraceRecord.ship_choice must be one of {sorted(SHIP_CHOICES)} or None; "
                f"got {self.ship_choice!r}"
            )

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "model_a": self.model_a.to_dict(),
            "model_b": self.model_b.to_dict(),
            "normalised_a": self.normalised_a,
            "normalised_b": self.normalised_b,
            "agreement": self.agreement,
            "final_tag": self.final_tag,
            "final_reason": self.final_reason,
            "ship_choice": self.ship_choice,
        }


@dataclass(frozen=True)
class TraceRunHeader:
    """Run provenance for a trace (build brief §2): the ids + hashes that pin
    exactly which registries/models/text produced this run's records."""

    paper_id: str
    strategy_label: str
    registry_version: str
    registry_hash: str
    silence_table_version: str
    canonical_text_hash: str
    model_a_id: str
    model_b_id: str
    run_id: str | None = None
    timestamp: str | None = None
    prompt_template_hashes: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "paper_id",
            "strategy_label",
            "registry_version",
            "registry_hash",
            "silence_table_version",
            "canonical_text_hash",
            "model_a_id",
            "model_b_id",
        ):
            val = getattr(self, name)
            if not isinstance(val, str) or val.strip() == "":
                raise LibrarianSchemaError(f"TraceRunHeader.{name} must be a non-empty string")

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "strategy_label": self.strategy_label,
            "registry_version": self.registry_version,
            "registry_hash": self.registry_hash,
            "silence_table_version": self.silence_table_version,
            "canonical_text_hash": self.canonical_text_hash,
            "model_a_id": self.model_a_id,
            "model_b_id": self.model_b_id,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "prompt_template_hashes": self.prompt_template_hashes,
        }


@dataclass(frozen=True)
class ExtractionTrace:
    """The dual-model trace sidecar for one (paper, strategy): a run header + one
    ``FieldTraceRecord`` per field (D12). ``sha256`` hashes the canonically
    serialised contents so the spec header can reference it reproducibly."""

    header: TraceRunHeader
    records: tuple[FieldTraceRecord, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.header, TraceRunHeader):
            raise LibrarianSchemaError("ExtractionTrace.header must be a TraceRunHeader")
        if isinstance(self.records, list):
            object.__setattr__(self, "records", tuple(self.records))
        if not isinstance(self.records, tuple):
            raise LibrarianSchemaError("ExtractionTrace.records must be a tuple of FieldTraceRecord")
        for i, r in enumerate(self.records):
            if not isinstance(r, FieldTraceRecord):
                raise LibrarianSchemaError(
                    f"ExtractionTrace.records[{i}] must be a FieldTraceRecord; got {type(r).__name__}"
                )

    def to_dict(self) -> dict:
        return {
            "header": self.header.to_dict(),
            "records": [r.to_dict() for r in self.records],
        }

    def _canonical_json(self) -> str:
        """Sorted-key, ASCII, separator-tight JSON of ``to_dict`` -- the hashed
        view (reproducible across loads, independent of dict insertion order)."""
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=True, separators=(",", ":"))

    def sha256(self) -> str:
        """sha256 over the canonically serialised trace. Stamped into the spec
        header (``trace_sha256``) so the spec and its trace cannot silently
        desync (D12)."""
        return hashlib.sha256(self._canonical_json().encode("utf-8")).hexdigest()

    def record_for(self, field_name: str) -> FieldTraceRecord | None:
        for r in self.records:
            if r.field == field_name:
                return r
        return None
