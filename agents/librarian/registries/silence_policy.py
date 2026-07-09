"""
Silence-policy table loader (D26 / D32c, v1).

Loads ``config/silence_policy_v1.yaml`` -- the per-field routing table for
UNKNOWN(not_stated) fields at adapter intake -- into a typed
``SilencePolicyTable``. The adapter (D26) consumes this to decide, per field,
what a paper's *silence* means:

  * ``tag_and_proceed`` -- paper silent -> the adapter omits the field and the
                           factory fills + tags its documented default. The
                           default *value* keeps exactly one home (P5): it lives
                           in the factory, NOT here; the table only records the
                           policy + a default *label* for reporting.
  * ``refuse``          -- paper silent on a load-bearing field -> typed refusal
                           naming the field.
  * ``flag``            -- proceed on silence, but a STATED value the engine
                           cannot honour raises a review flag.
  * ``conditional_refuse`` -- the policy depends on structural context; v1's only
                           instance is ``sort_kind`` keyed on whether a control
                           axis is present (D32c).

Two override keys ride on ``tag_and_proceed`` rows (D23 refuse-on-conflict, not
refuse-on-silence): ``refuse_on_stated`` (a STATED non-representable value flips
to a refusal) and ``flag_on_stated`` (a STATED value the engine can't honour
flips to a flag).

**Versioning + hashing.** The table exposes ``version`` (from the file) and
``content_hash`` = sha256 of the *file bytes* (the recorded canonical hash in
``docs/part2_schema_and_silence_policy_v1.md`` is the byte hash of exactly this
file). ``verify_hash(expected)`` lets a caller/test assert against it. Note the
byte-hash covers formatting + comments -- it is the artifact-freeze check, not a
semantic hash (contrast the signal registry's canonical-content hash).

This module is data-only: it never routes a field itself (that is adapter work);
it answers "what is the policy for (block, field)?" and hands back a typed record.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..errors import LibrarianSchemaError

# The one file this loader consumes (v1, hash-stamped in the schema doc).
_DEFAULT_SILENCE_POLICY_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "config" / "silence_policy_v1.yaml"
)

# The closed policy vocabulary (D26).
POLICIES: frozenset[str] = frozenset(
    ("tag_and_proceed", "refuse", "flag", "conditional_refuse")
)

# The two top-level blocks in the table.
_BLOCKS: tuple[str, ...] = ("sort_block", "common")


@dataclass(frozen=True)
class FieldPolicy:
    """The resolved silence policy for one (block, field).

    ``policy`` is the top-level routing; the remaining fields carry the row's
    extra structure and are populated only where relevant:

      * ``default``           -- the default *label* for ``tag_and_proceed`` rows
                                 (reporting only; the value's home is the factory).
      * ``refuse_on_stated``  -- STATED values that flip to a refusal (D23).
      * ``flag_on_stated``    -- STATED values that flip to a flag.
      * ``sort_kind``         -- the ``conditional_refuse`` branch table:
                                 ``{context -> FieldPolicy}`` (v1: ``when_no_control``
                                 / ``when_control_present`` for ``sort_kind``).
      * ``reason`` / ``note`` -- human strings carried verbatim from the table.
    """

    block: str
    field: str
    policy: str
    default: object | None = None
    refuse_on_stated: tuple[str, ...] = ()
    flag_on_stated: tuple[str, ...] = ()
    sort_kind: dict | None = None  # {context: FieldPolicy} for conditional_refuse
    reason: str | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise LibrarianSchemaError(
                f"silence policy {self.policy!r} for {self.block}.{self.field} "
                f"is not one of {sorted(POLICIES)}"
            )
        if self.policy == "conditional_refuse" and not self.sort_kind:
            raise LibrarianSchemaError(
                f"conditional_refuse policy for {self.block}.{self.field} requires a "
                "branch table (e.g. when_no_control / when_control_present)"
            )

    @property
    def is_conditional(self) -> bool:
        return self.policy == "conditional_refuse"

    def resolve(self, context: str) -> "FieldPolicy":
        """For a ``conditional_refuse`` field, return the branch policy for a
        structural ``context`` (e.g. ``"when_no_control"``); for any other field,
        return self (context ignored)."""
        if not self.is_conditional:
            return self
        table = self.sort_kind or {}
        if context not in table:
            raise LibrarianSchemaError(
                f"conditional_refuse field {self.block}.{self.field} has no branch for "
                f"context {context!r}; known: {sorted(table)}"
            )
        return table[context]


@dataclass(frozen=True)
class SilencePolicyTable:
    """The loaded silence-policy table: a ``version`` + per-(block, field)
    ``FieldPolicy`` records, plus the byte ``content_hash`` of the source file."""

    version: str
    content_hash: str
    policies: dict  # {block: {field: FieldPolicy}}

    def policy_for(self, block: str, field: str) -> FieldPolicy:
        """The ``FieldPolicy`` for a (block, field). Raises if the block or field
        is unknown -- an unrecognised field is a build error, never a silent
        default."""
        if block not in self.policies:
            raise LibrarianSchemaError(
                f"unknown silence-policy block {block!r}; known: {sorted(self.policies)}"
            )
        fields = self.policies[block]
        if field not in fields:
            raise LibrarianSchemaError(
                f"no silence policy for field {field!r} in block {block!r}"
            )
        return fields[field]

    def verify_hash(self, expected_sha256: str) -> bool:
        """True iff the loaded file's byte-hash equals ``expected_sha256``. A
        caller/test asserts against the hash recorded in the schema doc so a
        silent edit to the frozen table is caught."""
        return self.content_hash == expected_sha256


def _parse_field_policy(block: str, field: str, spec: object) -> FieldPolicy:
    if not isinstance(spec, dict):
        raise LibrarianSchemaError(
            f"silence policy for {block}.{field} must be a mapping; got {type(spec).__name__}"
        )
    policy = spec.get("policy")
    if not isinstance(policy, str):
        raise LibrarianSchemaError(f"silence policy for {block}.{field} must declare a 'policy'")

    if policy == "conditional_refuse":
        # v1 shape: nested when_* branches, each a {policy, default?, reason?} row.
        branches: dict[str, FieldPolicy] = {}
        for key, val in spec.items():
            if key.startswith("when_"):
                branches[key] = _parse_field_policy(block, f"{field}.{key}", val)
        if not branches:
            raise LibrarianSchemaError(
                f"conditional_refuse field {block}.{field} declares no when_* branches"
            )
        return FieldPolicy(
            block=block,
            field=field,
            policy=policy,
            sort_kind=branches,
            reason=spec.get("reason"),
            note=spec.get("note"),
        )

    refuse_on_stated = spec.get("refuse_on_stated", [])
    flag_on_stated = spec.get("flag_on_stated", [])
    if not isinstance(refuse_on_stated, list) or not isinstance(flag_on_stated, list):
        raise LibrarianSchemaError(
            f"refuse_on_stated / flag_on_stated for {block}.{field} must be lists"
        )
    return FieldPolicy(
        block=block,
        field=field,
        policy=policy,
        default=spec.get("default"),
        refuse_on_stated=tuple(str(v) for v in refuse_on_stated),
        flag_on_stated=tuple(str(v) for v in flag_on_stated),
        reason=spec.get("reason"),
        note=spec.get("note"),
    )


def load_silence_policy_table(path: str | Path | None = None) -> SilencePolicyTable:
    """Load and validate ``config/silence_policy_v1.yaml`` into a typed table.

    ``content_hash`` is the sha256 of the file bytes (the artifact-freeze check),
    computed before parsing so it is independent of the parse."""
    p = Path(path) if path is not None else _DEFAULT_SILENCE_POLICY_PATH
    if not p.exists():
        raise LibrarianSchemaError(f"silence-policy table not found at {p}")
    raw_bytes = p.read_bytes()
    content_hash = hashlib.sha256(raw_bytes).hexdigest()
    raw = yaml.safe_load(raw_bytes)
    if not isinstance(raw, dict):
        raise LibrarianSchemaError("silence-policy table must be a mapping at top level")

    version = raw.get("version")
    if not isinstance(version, str) or version.strip() == "":
        raise LibrarianSchemaError("silence-policy table must declare a non-empty 'version'")

    policies: dict[str, dict[str, FieldPolicy]] = {}
    for block in _BLOCKS:
        block_raw = raw.get(block)
        if not isinstance(block_raw, dict) or len(block_raw) == 0:
            raise LibrarianSchemaError(
                f"silence-policy block {block!r} must be a non-empty mapping"
            )
        policies[block] = {
            field: _parse_field_policy(block, field, spec) for field, spec in block_raw.items()
        }
    return SilencePolicyTable(version=str(version), content_hash=content_hash, policies=policies)
