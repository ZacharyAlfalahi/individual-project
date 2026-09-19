"""
Concept -> Column table loader (D27) -- the Quant side of the D22/D27 wall.

Turns a Signal Concept Registry concept id (+ its canonical parameter tuple) into
the panel column name the engine sorts on. The Librarian never imports this (it
speaks concept ids only, D3); the *adapter* reads it to resolve a ``SignalRef``
into a ``Binding``.

  * ``lookup(concept_id, params) -> str | None`` -- the bound column, or ``None``
    when the table has no row for that (concept, params). A ``None`` becomes a
    ``Binding(MISSING, ...)`` at the adapter, which the factory refuses with
    ``MISSING_BINDING`` (the resolver is its only emitter, P4).

**Mirror rule (D27).** Exactly one column per (concept_id, canonical parameter
tuple): the table is a *function*, so a bind is never ``AMBIGUOUS``. A duplicate
(concept_id, params) key is a build error -- caught here at load (mirrors
``SignalConceptRegistry``'s duplicate-id guard), so ambiguity is structurally
impossible downstream.

**Registry handshake (D27(1)).** The table declares the ``registry_version`` it
mirrors; the adapter asserts a spec's stamped ``registry_version`` equals it
before any lookup (a drift refuses the run upfront). Exposed as an attribute here;
the assertion itself lives at the adapter (which holds the spec).

**Hashing.** ``content_hash`` = sha256 over the canonically serialised rows
(sorted-key JSON), reproducible across loads and independent of YAML formatting --
the same content-hash discipline the signal registry uses.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from ..library.characteristic_sort import _RESERVED_COLUMNS

_DEFAULT_TABLE_PATH = Path(__file__).resolve().parent / "data" / "concept_column_table.yaml"

# The canonical parameter tuple: sorted (name, value) pairs, so two loads of the
# same params (dict order aside) key identically. Empty for all v1 concepts.
ParamTuple = tuple[tuple[str, object], ...]


class ConceptColumnError(ValueError):
    """The concept->column table is malformed (a duplicate key, a reserved or
    empty column, a bad shape) -- a build error, not a refusal."""


def _canonical_params(params: Mapping[str, object] | None) -> ParamTuple:
    """A dict of params -> the canonical (sorted-by-name) tuple used as the lookup
    key, so param order never changes identity."""
    if not params:
        return ()
    return tuple(sorted((str(k), v) for k, v in params.items()))


@dataclass(frozen=True)
class ConceptColumnRow:
    """One (concept_id, canonical params) -> column mapping."""

    concept_id: str
    params: ParamTuple
    column: str

    def __post_init__(self) -> None:
        if not isinstance(self.concept_id, str) or self.concept_id.strip() == "":
            raise ConceptColumnError("ConceptColumnRow.concept_id must be a non-empty string")
        if not isinstance(self.column, str) or self.column.strip() == "":
            raise ConceptColumnError(
                f"ConceptColumnRow.column must be a non-empty string (concept {self.concept_id!r})"
            )
        # A bind to an engine-internal reserved column would explode late in the
        # engine (or silently mis-merge); refuse it here, at the table (Binding
        # guards it too -- defence in depth).
        if self.column in _RESERVED_COLUMNS:
            raise ConceptColumnError(
                f"concept {self.concept_id!r} maps to reserved engine column {self.column!r} "
                f"(reserved: {_RESERVED_COLUMNS})"
            )

    def _canonical(self) -> dict:
        return {
            "concept_id": self.concept_id,
            "params": [list(p) for p in self.params],
            "column": self.column,
        }


@dataclass(frozen=True)
class ConceptColumnTable:
    """The loaded concept->column table: a ``version``, the ``registry_version`` it
    mirrors, and the rows. ``content_hash`` = sha256 over the canonically
    serialised rows (sorted by (concept_id, params))."""

    version: str
    registry_version: str
    rows: tuple[ConceptColumnRow, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or self.version.strip() == "":
            raise ConceptColumnError("ConceptColumnTable.version must be a non-empty string")
        if not isinstance(self.registry_version, str) or self.registry_version.strip() == "":
            raise ConceptColumnError("ConceptColumnTable.registry_version must be a non-empty string")
        seen: set[tuple[str, ParamTuple]] = set()
        index: dict[tuple[str, ParamTuple], str] = {}
        for r in self.rows:
            if not isinstance(r, ConceptColumnRow):
                raise ConceptColumnError(
                    f"ConceptColumnTable.rows entries must be ConceptColumnRow; got {type(r).__name__}"
                )
            key = (r.concept_id, r.params)
            if key in seen:
                # Mirror rule (D27): one column per (concept, params) -- a duplicate
                # would make a bind AMBIGUOUS, which the adapter forbids structurally.
                raise ConceptColumnError(
                    f"duplicate (concept_id, params) key {key!r} -- the table is a function "
                    "(D27 mirror rule): exactly one column per (concept, params)"
                )
            seen.add(key)
            index[key] = r.column
        object.__setattr__(self, "_index", index)

    def lookup(self, concept_id: str, params: Mapping[str, object] | None = None) -> str | None:
        """The bound column for ``(concept_id, params)``, or ``None`` when the
        table has no row (which the adapter turns into a MISSING binding)."""
        return self._index.get((concept_id, _canonical_params(params)))  # type: ignore[attr-defined]

    def has_row(self, concept_id: str, params: Mapping[str, object] | None = None) -> bool:
        return (concept_id, _canonical_params(params)) in self._index  # type: ignore[attr-defined]

    @property
    def content_hash(self) -> str:
        """sha256 over the canonically serialised rows (sorted by concept_id then
        params). Reproducible across loads; independent of YAML formatting."""
        canonical = [
            r._canonical()
            for r in sorted(self.rows, key=lambda r: (r.concept_id, r.params))
        ]
        payload = json.dumps(canonical, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def concept_ids(self) -> tuple[str, ...]:
        return tuple(sorted({r.concept_id for r in self.rows}))


def load_concept_column_table(path: str | Path | None = None) -> ConceptColumnTable:
    """Load and validate the concept->column table file into a typed table.

    The file declares ``version``, ``registry_version``, and a ``rows`` list; each
    row carries ``concept_id``, ``params`` (a mapping, empty for v1 concepts), and
    ``column``."""
    p = Path(path) if path is not None else _DEFAULT_TABLE_PATH
    if not p.exists():
        raise ConceptColumnError(f"concept->column table not found at {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConceptColumnError("concept->column table must be a mapping at top level")

    version = raw.get("version")
    if not isinstance(version, str) or version.strip() == "":
        raise ConceptColumnError("concept->column table must declare a non-empty 'version'")
    registry_version = raw.get("registry_version")
    if not isinstance(registry_version, str) or registry_version.strip() == "":
        raise ConceptColumnError("concept->column table must declare a non-empty 'registry_version'")

    raw_rows = raw.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) == 0:
        raise ConceptColumnError("concept->column table 'rows' must be a non-empty list")

    rows: list[ConceptColumnRow] = []
    for i, rr in enumerate(raw_rows):
        if not isinstance(rr, dict):
            raise ConceptColumnError(f"concept->column row {i} must be a mapping")
        try:
            cid = rr["concept_id"]
            column = rr["column"]
        except KeyError as exc:
            raise ConceptColumnError(f"concept->column row {i} missing required key {exc}") from exc
        raw_params = rr.get("params", {})
        if not isinstance(raw_params, dict):
            raise ConceptColumnError(f"concept->column row {i} params must be a mapping")
        rows.append(
            ConceptColumnRow(
                concept_id=str(cid),
                params=_canonical_params(raw_params),
                column=str(column),
            )
        )
    return ConceptColumnTable(
        version=str(version), registry_version=str(registry_version), rows=tuple(rows)
    )
