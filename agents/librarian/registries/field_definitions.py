"""
Per-field definitions loader (RQ1 close-out, the frozen ``{definition}`` slot).

Loads ``data/prompts/definitions.yaml`` -- one concise, factual gloss per field
routed to a ``{definition}``-carrying prompt template (the enum / int / date /
paper_metric field-types) -- into a typed ``FieldDefinitions``. The prompt
builder (``real_client.py``) consumes this to fill the ``{definition}`` slot,
replacing the earlier ``_humanise(field)`` light gloss. Rendered prompt content
affects extraction, so the gloss is frozen + byte-hashed here before reportable
Phase-F runs (the same freeze discipline as the silence-policy table).

**Versioning + hashing.** The table exposes ``version`` (from the file) and
``content_hash`` = sha256 of the *file bytes* (the recorded canonical hash in
``docs/librarian/specs/part2_schema_and_silence_policy_v1_1.md`` is the byte hash
of exactly this file). ``verify_hash(expected)`` lets a caller/test assert against
it. Like the silence-policy hash, the byte-hash covers formatting + comments -- it
is the artifact-freeze check, not a semantic hash.

This module is data-only: it never decides which fields are routed to a
``{definition}`` template (that is the manifest's/builder's job); it answers "what
is the frozen definition for this field?" and raises on an unknown field -- a
routed field with no gloss is a build error, never a silent fallback.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..errors import LibrarianSchemaError

# The one file this loader consumes (frozen, hash-stamped in the schema doc).
_DEFAULT_DEFINITIONS_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "prompts" / "definitions.yaml"
)


@dataclass(frozen=True)
class FieldDefinitions:
    """The loaded per-field definitions: a ``version`` + a ``{field -> gloss}``
    mapping, plus the byte ``content_hash`` of the source file."""

    version: str
    content_hash: str
    definitions: dict  # {field: str}

    def definition_for(self, field: str) -> str:
        """The frozen gloss for ``field``. Raises if the field has no definition --
        a routed ``{definition}``-template field with no gloss is a build error,
        never a silent fallback (fail-closed, matching the repo)."""
        if field not in self.definitions:
            raise LibrarianSchemaError(
                f"no frozen definition for field {field!r} in definitions.yaml"
            )
        return self.definitions[field]

    def __contains__(self, field: object) -> bool:
        return field in self.definitions

    def __len__(self) -> int:
        return len(self.definitions)

    def verify_hash(self, expected_sha256: str) -> bool:
        """True iff the loaded file's byte-hash equals ``expected_sha256``. A
        caller/test asserts against the hash recorded in the schema doc so a
        silent edit to the frozen definitions is caught."""
        return self.content_hash == expected_sha256


def load_field_definitions(path: str | Path | None = None) -> FieldDefinitions:
    """Load and validate ``data/prompts/definitions.yaml`` into a typed table.

    ``content_hash`` is the sha256 of the file bytes (the artifact-freeze check),
    computed before parsing so it is independent of the parse."""
    p = Path(path) if path is not None else _DEFAULT_DEFINITIONS_PATH
    if not p.exists():
        raise LibrarianSchemaError(f"field-definitions file not found at {p}")
    raw_bytes = p.read_bytes()
    content_hash = hashlib.sha256(raw_bytes).hexdigest()
    raw = yaml.safe_load(raw_bytes)
    if not isinstance(raw, dict):
        raise LibrarianSchemaError("field-definitions file must be a mapping at top level")

    version = raw.get("version")
    if not isinstance(version, str) or version.strip() == "":
        raise LibrarianSchemaError("field-definitions file must declare a non-empty 'version'")

    defs_raw = raw.get("definitions")
    if not isinstance(defs_raw, dict) or len(defs_raw) == 0:
        raise LibrarianSchemaError("field-definitions 'definitions' must be a non-empty mapping")

    definitions: dict[str, str] = {}
    for field, gloss in defs_raw.items():
        if not isinstance(gloss, str) or gloss.strip() == "":
            raise LibrarianSchemaError(
                f"definition for field {field!r} must be a non-empty string"
            )
        definitions[str(field)] = gloss.strip()

    return FieldDefinitions(
        version=str(version), content_hash=content_hash, definitions=definitions
    )
