"""
Tag-reason registry loader + guard (D24 part 1).

A tag may only ever be applied for a reason that appears as a row in
``data/tag_reason_registry.yaml``. Each row fixes the reason code, the required
evidence, the sole legitimate producer, and the downstream handling. The
registry ships as a validator-enforced file: **a tag applied outside its row is
a build error** (``assert_tag_in_registry`` raises ``LibrarianSchemaError``).

The ``adapter/``, ``inference/`` and ``default/`` namespaces are reported
separately in every results table (D24); ``namespaces()`` exposes the grouping
so a reporting layer can partition rows without re-parsing reason strings.

Refusals are *events* on the run record, never tags -- so there is no refusal
row and this module never sees one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ..errors import LibrarianSchemaError

# Default registry location (co-located with the schema code).
_DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "data" / "tag_reason_registry.yaml"

# The reporting namespaces reported separately (D24). "" is the default (no
# separate namespace); the other three each get their own results-table column.
SEPARATE_NAMESPACES: tuple[str, ...] = ("inference", "adapter", "default")
_KNOWN_NAMESPACES: frozenset[str] = frozenset(("",) + SEPARATE_NAMESPACES)


@dataclass(frozen=True)
class TagReasonRow:
    """One authoritative (tag, reason) row. ``required_evidence`` is the tuple of
    ``Evidence`` subfield names a value of this kind must carry; ``sole_producer``
    is the one component allowed to emit it; ``downstream`` names the consumer's
    routing."""

    tag: str
    reason: str
    namespace: str
    required_evidence: tuple[str, ...]
    sole_producer: str
    downstream: str

    @property
    def key(self) -> tuple[str, str]:
        """The (tag, reason) identity -- the registry's primary key."""
        return (self.tag, self.reason)

    def to_dict(self) -> dict:
        return {
            "tag": self.tag,
            "reason": self.reason,
            "namespace": self.namespace,
            "required_evidence": list(self.required_evidence),
            "sole_producer": self.sole_producer,
            "downstream": self.downstream,
        }


@dataclass(frozen=True)
class TagReasonRegistry:
    """The loaded registry: a version + the rows, keyed by (tag, reason)."""

    version: str
    rows: tuple[TagReasonRow, ...]

    def __post_init__(self) -> None:
        seen: set[tuple[str, str]] = set()
        for row in self.rows:
            if row.key in seen:
                raise LibrarianSchemaError(
                    f"duplicate tag-reason row for {row.key} -- one row per (tag, reason)"
                )
            seen.add(row.key)

    def has(self, tag: str, reason: str) -> bool:
        return (tag, reason) in {r.key for r in self.rows}

    def get(self, tag: str, reason: str) -> TagReasonRow | None:
        for row in self.rows:
            if row.key == (tag, reason):
                return row
        return None

    def namespaces(self) -> dict[str, tuple[TagReasonRow, ...]]:
        """Rows grouped by reporting namespace (D24: adapter / inference /
        default reported separately). Keys are the namespace strings; the empty
        string groups the default-namespace rows."""
        out: dict[str, list[TagReasonRow]] = {ns: [] for ns in _KNOWN_NAMESPACES}
        for row in self.rows:
            out.setdefault(row.namespace, []).append(row)
        return {ns: tuple(rows) for ns, rows in out.items()}

    def rows_in_namespace(self, namespace: str) -> tuple[TagReasonRow, ...]:
        return tuple(r for r in self.rows if r.namespace == namespace)


def load_tag_reason_registry(path: str | Path | None = None) -> TagReasonRegistry:
    """Load and validate the tag-reason registry file into typed rows."""
    p = Path(path) if path is not None else _DEFAULT_REGISTRY_PATH
    if not p.exists():
        raise LibrarianSchemaError(f"tag-reason registry not found at {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise LibrarianSchemaError("tag-reason registry must be a mapping at top level")

    version = raw.get("version")
    if not isinstance(version, str) or version.strip() == "":
        raise LibrarianSchemaError("tag-reason registry must declare a non-empty 'version'")

    raw_rows = raw.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) == 0:
        raise LibrarianSchemaError("tag-reason registry 'rows' must be a non-empty list")

    rows: list[TagReasonRow] = []
    for i, rr in enumerate(raw_rows):
        if not isinstance(rr, dict):
            raise LibrarianSchemaError(f"tag-reason row {i} must be a mapping")
        try:
            tag = rr["tag"]
            reason = rr["reason"]
            namespace = rr.get("namespace", "")
            required_evidence = rr["required_evidence"]
            sole_producer = rr["sole_producer"]
            downstream = rr["downstream"]
        except KeyError as exc:
            raise LibrarianSchemaError(
                f"tag-reason row {i} missing required key {exc}"
            ) from exc
        if not isinstance(required_evidence, list):
            raise LibrarianSchemaError(
                f"tag-reason row {i} required_evidence must be a list"
            )
        if namespace not in _KNOWN_NAMESPACES:
            raise LibrarianSchemaError(
                f"tag-reason row {i} namespace {namespace!r} not one of {sorted(_KNOWN_NAMESPACES)}"
            )
        rows.append(
            TagReasonRow(
                tag=str(tag),
                reason=str(reason),
                namespace=str(namespace),
                required_evidence=tuple(str(e) for e in required_evidence),
                sole_producer=str(sole_producer),
                downstream=str(downstream),
            )
        )
    return TagReasonRegistry(version=version, rows=tuple(rows))


def assert_tag_in_registry(
    tag: str,
    reason: str,
    namespace: str = "",
    registry: TagReasonRegistry | None = None,
) -> TagReasonRow:
    """Raise ``LibrarianSchemaError`` if ``(tag, reason)`` is not a registered
    row, or if ``namespace`` is given and disagrees with the row's namespace.
    Returns the matching row on success.

    D24: a tag applied outside its row is a build error."""
    reg = registry if registry is not None else load_tag_reason_registry()
    row = reg.get(tag, reason)
    if row is None:
        raise LibrarianSchemaError(
            f"tag {tag!r} applied for reason {reason!r} is not a registered tag-reason "
            "row (D24: a tag applied outside its row is a build error)"
        )
    if namespace and namespace != row.namespace:
        raise LibrarianSchemaError(
            f"tag-reason ({tag!r}, {reason!r}) is registered under namespace "
            f"{row.namespace!r}, not {namespace!r}"
        )
    return row
