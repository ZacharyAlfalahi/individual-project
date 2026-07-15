"""
Standing-substitutions table loader (evaluation contract §6).

Loads ``config/standing_substitutions_v1.yaml`` -- the project-wide, pre-registered
execution conventions -- into a typed ``StandingSubstitutionTable``. A *standing
substitution* replaces a STATED paper-language field value with a DESIGN value by a
fixed predicate, WITHOUT flagging the strategy a variant (contrast the per-strategy
``field_override`` / ``binding_substitution`` channel in ``adapter/authorisation.py``,
which DOES set ``variant=True`` and is excluded from replication-fidelity aggregates,
D23). The canonical example is par-weighting: the panel's authoritative size field is
par (offering_amt), so a paper that value-weights by market value is compiled with the
par proxy -- a project-wide convention, not a per-paper divergence.

**Contract §6 properties, enforced structurally here:** ``provenance == "DESIGN"`` and
``changes_variant_status is False`` (a standing sub is INCLUDED in fidelity aggregates).
A per-strategy divergence that should quarantine a run must NOT be encoded here.

**Versioning + hashing (mirrors the silence policy).** ``version`` (from the file) +
``content_hash`` = sha256 of the *file bytes*, computed before parse (an artifact-freeze
check, not a semantic hash). ``verify_hash(expected)`` lets a caller assert the recorded
hash -- and lets the G2 harness enforce the contract §6 temporal rule (the standing
file's hash must predate, and match, the run it authorises). The recorded canonical
hash is ``STANDING_SUBS_V1_SHA256`` below (kept OUTSIDE the hashed file).

This module is data-only: it never applies a substitution (that is adapter work); it
answers "is there a standing substitution for (field, stated_value)?" and hands back a
typed record. An absent file loads as ``NO_STANDING_SUBS`` (the empty case) used by
ordinary adapter calls; the G2 harness passes an explicit path and asserts ``verify_hash``
so absence/tamper is loud.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..errors import LibrarianSchemaError

_DEFAULT_STANDING_SUBS_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "config" / "standing_substitutions_v1.yaml"
)

# The recorded canonical sha256 of the frozen ``config/standing_substitutions_v1.yaml``
# (the file bytes). Editing the table changes this hash -- update it here deliberately;
# the diff is the freeze record. The G2 harness asserts ``verify_hash`` against this.
STANDING_SUBS_V1_SHA256 = "5f2bb9e52a849cedd5e68d3833f45cf2d1881236ef450201a98a3c8d124835a1"


@dataclass(frozen=True)
class Substitution:
    """One standing substitution (contract §6): replace ``field``'s STATED value --
    when it is one of ``when_stated_in`` -- with ``replacement``, tagged DESIGN, and
    NOT variant-flagging. ``__post_init__`` enforces the two §6 properties."""

    id: str
    field: str
    when_stated_in: tuple[str, ...]
    replacement: object
    provenance: str
    changes_variant_status: object
    note: str
    source_decision: str

    def __post_init__(self) -> None:
        if self.provenance != "DESIGN":
            raise LibrarianSchemaError(
                f"standing substitution {self.id!r} must be provenance=DESIGN (contract §6); "
                f"got {self.provenance!r}"
            )
        if self.changes_variant_status is not False:
            raise LibrarianSchemaError(
                f"standing substitution {self.id!r} must set changes_variant_status=false "
                "(a standing convention stays IN fidelity aggregates; a variant-flagging "
                "divergence belongs in the per-strategy override channel, not here)"
            )
        if not self.note or not str(self.note).strip():
            raise LibrarianSchemaError(
                f"standing substitution {self.id!r} requires a non-empty note (DESIGN needs a reason)"
            )


@dataclass(frozen=True)
class StandingSubstitutionTable:
    """The loaded standing-substitutions table: a ``version`` + the typed
    ``Substitution`` rows, plus the byte ``content_hash`` of the source file."""

    version: str
    content_hash: str
    substitutions: tuple[Substitution, ...] = ()

    def substitution_for(self, field: str, stated_value: object) -> Substitution | None:
        """The standing substitution for a (field, STATED value), or None. Exact match
        on field + membership in ``when_stated_in`` (no fuzzy matching)."""
        for sub in self.substitutions:
            if sub.field == field and stated_value in sub.when_stated_in:
                return sub
        return None

    def verify_hash(self, expected_sha256: str) -> bool:
        """True iff the loaded file's byte-hash equals ``expected_sha256``. The G2
        harness asserts this (contract §6): the standing file's hash must predate --
        and match -- the run it authorises, so an absent/edited file is caught (the
        empty ``NO_STANDING_SUBS`` hash never matches a recorded sha256)."""
        return self.content_hash == expected_sha256


# The empty "no standing substitutions" table -- the default for ordinary adapter calls
# (so behaviour is byte-identical to pre-standing-subs). Its empty hash never matches a
# recorded sha256, so a harness that requires a real file fails loud via ``verify_hash``.
NO_STANDING_SUBS = StandingSubstitutionTable(version="none", content_hash="", substitutions=())


def _require(rec: dict, key: str, sub_id: object) -> object:
    if key not in rec:
        raise LibrarianSchemaError(f"standing substitution {sub_id!r} missing required key {key!r}")
    return rec[key]


def load_standing_substitutions(path: str | Path | None = None) -> StandingSubstitutionTable:
    """Load + validate the standing-substitutions file into a typed table.

    ``content_hash`` is the sha256 of the file bytes, computed before parse. An absent
    file loads as ``NO_STANDING_SUBS`` (the empty case) -- callers that require a real,
    hash-verified file (the G2 harness) assert ``verify_hash`` against the recorded sha."""
    p = Path(path) if path is not None else _DEFAULT_STANDING_SUBS_PATH
    if not p.exists():
        return NO_STANDING_SUBS
    raw_bytes = p.read_bytes()
    content_hash = hashlib.sha256(raw_bytes).hexdigest()
    raw = yaml.safe_load(raw_bytes)
    if not isinstance(raw, dict):
        raise LibrarianSchemaError("standing-substitutions table must be a mapping at top level")

    version = raw.get("version")
    if not isinstance(version, str) or version.strip() == "":
        raise LibrarianSchemaError("standing-substitutions table must declare a non-empty 'version'")

    rows = raw.get("substitutions", []) or []
    if not isinstance(rows, list):
        raise LibrarianSchemaError("standing-substitutions 'substitutions' must be a list")

    subs: list[Substitution] = []
    seen_ids: set[str] = set()
    for rec in rows:
        if not isinstance(rec, dict):
            raise LibrarianSchemaError("each standing substitution must be a mapping")
        sub_id = str(_require(rec, "id", "<unknown>"))
        if sub_id in seen_ids:
            raise LibrarianSchemaError(f"duplicate standing substitution id {sub_id!r}")
        seen_ids.add(sub_id)
        when = _require(rec, "when_stated_in", sub_id)
        if not isinstance(when, list) or not when:
            raise LibrarianSchemaError(
                f"standing substitution {sub_id!r} 'when_stated_in' must be a non-empty list"
            )
        subs.append(
            Substitution(
                id=sub_id,
                field=str(_require(rec, "field", sub_id)),
                when_stated_in=tuple(str(v) for v in when),
                replacement=_require(rec, "replacement", sub_id),
                provenance=str(_require(rec, "provenance", sub_id)),
                changes_variant_status=_require(rec, "changes_variant_status", sub_id),
                note=str(_require(rec, "note", sub_id)),
                source_decision=str(_require(rec, "source_decision", sub_id)),
            )
        )
    return StandingSubstitutionTable(
        version=str(version), content_hash=content_hash, substitutions=tuple(subs)
    )
