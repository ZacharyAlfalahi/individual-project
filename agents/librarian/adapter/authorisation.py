"""
Authorisation-record loader (D27) -- format + loader only.

The human-authored channel for recording a *deliberate divergence from a correctly
extracted fact*. Two entry types:

  * ``binding_substitution`` -- override the concept->column resolution for one
    strategy (a concept the panel doesn't carry, bound by hand to a chosen column).
  * ``field_override``       -- override one Part 2 *paper-language* field's value
    (never an engine field) with a deliberate project value.

Either match sets the strategy's ``variant`` flag (D23/D27: variants are excluded
from replication-fidelity aggregates).

**The bright line (D27), enforced structurally.** An override may only ever divert
a *correctly extracted* fact -- it may NEVER overwrite a fact believed *wrongly*
extracted (those go to the manual-review lane; patching them here would launder
RQ1). The loader cannot read intent, so it enforces the two structural halves:

  1. a ``field_override.field`` must be a Part 2 field (in ``ALREADY_FINAL_PART2``),
     never an engine coordinate -- checked here at load;
  2. an override applies only where the spec carries the field as STATED/INFERRED
     (a real extracted fact), never UNKNOWN -- checked at apply time (``adapt.py``),
     which holds the spec.

The records FILE is human-authored (a template ships empty); this module only
loads + validates its shape. An absent file loads as *no records*.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ..errors import LibrarianSchemaError
from ..schema.fields import ALREADY_FINAL_PART2

_DEFAULT_RECORDS_PATH = Path(__file__).resolve().parent / "data" / "authorisation_records.yaml"

_ENTRY_TYPES: frozenset[str] = frozenset(("binding_substitution", "field_override"))


@dataclass(frozen=True)
class BindingSubstitution:
    """Bind a concept to a hand-chosen column for one strategy (DESIGN, variant)."""

    paper_id: str
    strategy_label: str
    concept_id: str
    column: str
    note: str


@dataclass(frozen=True)
class FieldOverride:
    """Override one Part 2 paper-language field's value for one strategy (DESIGN,
    variant). ``field`` is validated to be a Part 2 field, never an engine field."""

    paper_id: str
    strategy_label: str
    field: str
    value: object
    note: str

    def __post_init__(self) -> None:
        if self.field not in ALREADY_FINAL_PART2:
            raise LibrarianSchemaError(
                f"field_override.field {self.field!r} is not a Part 2 field -- an override "
                "may target only a paper-language field, never an engine coordinate (D27 bright line)"
            )


@dataclass(frozen=True)
class AuthorisationRecords:
    """The loaded authorisation records: two typed lists + per-(strategy) lookups."""

    version: str
    bindings: tuple[BindingSubstitution, ...] = ()
    overrides: tuple[FieldOverride, ...] = ()

    def lookup_binding(
        self, paper_id: str, strategy_label: str, concept_id: str
    ) -> BindingSubstitution | None:
        for b in self.bindings:
            if (b.paper_id, b.strategy_label, b.concept_id) == (paper_id, strategy_label, concept_id):
                return b
        return None

    def lookup_override(
        self, paper_id: str, strategy_label: str, field: str
    ) -> FieldOverride | None:
        for o in self.overrides:
            if (o.paper_id, o.strategy_label, o.field) == (paper_id, strategy_label, field):
                return o
        return None

    @property
    def is_empty(self) -> bool:
        return not self.bindings and not self.overrides


# A shared "no authorisations" singleton for the common (empty) case.
NO_AUTHORISATIONS = AuthorisationRecords(version="v1")


def _require(rec: dict, key: str, i: int) -> object:
    if key not in rec:
        raise LibrarianSchemaError(f"authorisation record {i} missing required key {key!r}")
    return rec[key]


def load_authorisation_records(path: str | Path | None = None) -> AuthorisationRecords:
    """Load + validate the authorisation-records file. An absent file (the common
    case -- there are no authorisations) loads as ``NO_AUTHORISATIONS``."""
    p = Path(path) if path is not None else _DEFAULT_RECORDS_PATH
    if not p.exists():
        return NO_AUTHORISATIONS
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        return NO_AUTHORISATIONS
    if not isinstance(raw, dict):
        raise LibrarianSchemaError("authorisation records must be a mapping at top level")

    version = raw.get("version")
    if not isinstance(version, str) or version.strip() == "":
        raise LibrarianSchemaError("authorisation records must declare a non-empty 'version'")

    records = raw.get("records", []) or []
    if not isinstance(records, list):
        raise LibrarianSchemaError("authorisation 'records' must be a list")

    bindings: list[BindingSubstitution] = []
    overrides: list[FieldOverride] = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise LibrarianSchemaError(f"authorisation record {i} must be a mapping")
        etype = rec.get("type")
        if etype not in _ENTRY_TYPES:
            raise LibrarianSchemaError(
                f"authorisation record {i} 'type' must be one of {sorted(_ENTRY_TYPES)}; got {etype!r}"
            )
        if etype == "binding_substitution":
            bindings.append(
                BindingSubstitution(
                    paper_id=str(_require(rec, "paper_id", i)),
                    strategy_label=str(_require(rec, "strategy_label", i)),
                    concept_id=str(_require(rec, "concept_id", i)),
                    column=str(_require(rec, "column", i)),
                    note=str(_require(rec, "note", i)),
                )
            )
        else:  # field_override
            overrides.append(
                FieldOverride(
                    paper_id=str(_require(rec, "paper_id", i)),
                    strategy_label=str(_require(rec, "strategy_label", i)),
                    field=str(_require(rec, "field", i)),
                    value=_require(rec, "value", i),
                    note=str(_require(rec, "note", i)),
                )
            )
    return AuthorisationRecords(
        version=str(version), bindings=tuple(bindings), overrides=tuple(overrides)
    )
