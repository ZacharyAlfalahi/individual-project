"""
Ledger check -- the deterministic assumptions-mismatch screen (D29).

The frozen engine (``agents/quant/library``) applies exactly 40 baked-in
assumptions (``assumptions_ledger_v2.md``). Some the adapter configures to match
the paper; some the engine applies *silently* regardless of what the paper says.
This module catches the second class: it reads a strategy's Part 2 fields and,
where the paper *explicitly states* something the engine cannot honour, emits an
``ASSUMPTION_MISMATCH`` refusal instead of letting the engine run a different
assumption and calling it a replication (the plausible-but-wrong number).

Two pieces:

  * ``LedgerCheckTable`` / ``load_ledger_check_table`` -- a versioned, hash-stamped
    data artifact (``data/ledger_check_table.yaml``). One row per checkable STRUCT
    item: the Part 2 field, the value the engine always applies (``engine_fixed``),
    and the stated values that contradict it (``incompatible_stated``). Mirrors the
    ``concept_column`` content-hash discipline (sha256 over canonically-serialised
    rows). A malformed row is a build error, never a silent default.
  * ``check_assumptions(spec, table)`` -- deterministic. For each row, if the
    strategy's field is STATED and its value is in ``incompatible_stated``, emit a
    ``ConfigRefusal(code=ASSUMPTION_MISMATCH, ...)``. Silent / INFERRED / UNKNOWN
    fields never mismatch (the engine's assumption stands and is tagged by the
    silence policy). Run-to-completion: collect ALL mismatches.

**Scope (which rows exist) is committed in the YAML, not here.** The table is the
A/C re-sort of the STRUCT ledger items (``part2_field_inventory.md``); LIMIT items
are already factory/adapter refusals (not double-handled), and SEM items are out
of deterministic scope. ``check_assumptions`` is scope-agnostic: it runs whatever
rows the finalised table carries.

**Wiring (D28/D29).** ``check_assumptions`` is a *hard gate feeding*
``run_strategy``: a non-empty tuple means refuse -- do not execute. An
``ASSUMPTION_MISMATCH`` is itself an RQ2 outcome (it lands in the coverage
denominator like every other refusal); you count mismatches by counting refusals,
not by running broken backtests.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from .refusal import ConfigRefusal, RefusalCode

if TYPE_CHECKING:  # runtime import would cycle: librarian.schema -> quant.config
    from agents.librarian.schema.strategy_spec import StrategySpec

_DEFAULT_TABLE_PATH = Path(__file__).resolve().parent / "data" / "ledger_check_table.yaml"

# Where each checkable field lives on the spec: common fields hang off Part2;
# sort fields hang off each Leg (checked per-leg).
_BLOCKS: frozenset[str] = frozenset(("common", "sort"))
# The A/C re-sort tags (part2_field_inventory.md). A = clean STRUCT (strong check);
# C = STRUCT-but-usually-UNKNOWN (rarely fires; the silence policy carries the load).
_CATEGORIES: frozenset[str] = frozenset(("A", "C"))


class LedgerCheckError(ValueError):
    """The ledger check table is malformed (a bad block/category, an empty
    field or incompatible set, a duplicate row) -- a build error, not a refusal."""


@dataclass(frozen=True)
class LedgerCheckRow:
    """One checkable STRUCT item: a Part 2 field the engine fixes silently, and
    the stated values that contradict the fixed assumption."""

    ledger_item: int
    block: str
    part2_field: str
    engine_fixed: str
    incompatible_stated: tuple[str, ...]
    category: str
    detail: str
    cite: str | None = None
    borderline: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.ledger_item, int) or isinstance(self.ledger_item, bool):
            raise LedgerCheckError("LedgerCheckRow.ledger_item must be an int")
        if self.block not in _BLOCKS:
            raise LedgerCheckError(
                f"LedgerCheckRow.block {self.block!r} must be one of {sorted(_BLOCKS)} "
                f"(item {self.ledger_item})"
            )
        for name, val in (("part2_field", self.part2_field),
                          ("engine_fixed", self.engine_fixed),
                          ("detail", self.detail)):
            if not isinstance(val, str) or val.strip() == "":
                raise LedgerCheckError(
                    f"LedgerCheckRow.{name} must be a non-empty string (item {self.ledger_item})"
                )
        if self.category not in _CATEGORIES:
            raise LedgerCheckError(
                f"LedgerCheckRow.category {self.category!r} must be one of {sorted(_CATEGORIES)} "
                f"(item {self.ledger_item})"
            )
        # An empty incompatible set is a row with nothing to test -- a build error.
        # This structurally bars int-only fields (e.g. realisation_min_survivors),
        # which cannot be an enum-membership check.
        if not isinstance(self.incompatible_stated, tuple) or len(self.incompatible_stated) == 0:
            raise LedgerCheckError(
                f"LedgerCheckRow.incompatible_stated must be a non-empty tuple "
                f"(item {self.ledger_item}, field {self.part2_field!r})"
            )
        for v in self.incompatible_stated:
            if not isinstance(v, str) or v.strip() == "":
                raise LedgerCheckError(
                    f"LedgerCheckRow.incompatible_stated entries must be non-empty strings "
                    f"(item {self.ledger_item}, field {self.part2_field!r})"
                )
        if not isinstance(self.borderline, bool):
            raise LedgerCheckError(
                f"LedgerCheckRow.borderline must be a bool (item {self.ledger_item})"
            )
        if self.cite is not None and not isinstance(self.cite, str):
            raise LedgerCheckError(
                f"LedgerCheckRow.cite must be a str or None (item {self.ledger_item})"
            )

    def _canonical(self) -> dict:
        return {
            "ledger_item": self.ledger_item,
            "block": self.block,
            "part2_field": self.part2_field,
            "engine_fixed": self.engine_fixed,
            "incompatible_stated": list(self.incompatible_stated),
            "category": self.category,
            "detail": self.detail,
            "cite": self.cite,
            "borderline": self.borderline,
        }


@dataclass(frozen=True)
class LedgerCheckTable:
    """The loaded ledger check table: a ``version`` + the rows. ``content_hash`` =
    sha256 over the canonically serialised rows (sorted by (ledger_item, block,
    part2_field)); reproducible across loads, independent of YAML formatting."""

    version: str
    rows: tuple[LedgerCheckRow, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or self.version.strip() == "":
            raise LedgerCheckError("LedgerCheckTable.version must be a non-empty string")
        # One row per (block, part2_field): the checker navigates by field, so a
        # duplicate would double-fire. Keep the table a function over fields.
        seen: set[tuple[str, str]] = set()
        for r in self.rows:
            if not isinstance(r, LedgerCheckRow):
                raise LedgerCheckError(
                    f"LedgerCheckTable.rows entries must be LedgerCheckRow; got {type(r).__name__}"
                )
            key = (r.block, r.part2_field)
            if key in seen:
                raise LedgerCheckError(
                    f"duplicate (block, part2_field) row {key!r} -- the table is a function "
                    "over fields (the checker navigates by field)"
                )
            seen.add(key)

    @property
    def content_hash(self) -> str:
        canonical = [
            r._canonical()
            for r in sorted(self.rows, key=lambda r: (r.ledger_item, r.block, r.part2_field))
        ]
        payload = json.dumps(canonical, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def verify_hash(self, expected_sha256: str) -> bool:
        """True iff ``content_hash`` equals ``expected_sha256`` -- a caller/test
        asserts against the hash recorded when the table was frozen so a silent edit is caught."""
        return self.content_hash == expected_sha256


def load_ledger_check_table(path: str | Path | None = None) -> LedgerCheckTable:
    """Load and validate the ledger check table file into a typed table.

    The file declares ``version`` and a ``rows`` list; each row carries
    ``ledger_item``, ``block``, ``part2_field``, ``engine_fixed``,
    ``incompatible_stated`` (a non-empty list), ``category``, ``detail``, and the
    optional ``cite`` / ``borderline``. A malformed row is a build error."""
    p = Path(path) if path is not None else _DEFAULT_TABLE_PATH
    if not p.exists():
        raise LedgerCheckError(f"ledger check table not found at {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise LedgerCheckError("ledger check table must be a mapping at top level")

    version = raw.get("version")
    if not isinstance(version, str) or version.strip() == "":
        raise LedgerCheckError("ledger check table must declare a non-empty 'version'")

    raw_rows = raw.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) == 0:
        raise LedgerCheckError("ledger check table 'rows' must be a non-empty list")

    rows: list[LedgerCheckRow] = []
    for i, rr in enumerate(raw_rows):
        if not isinstance(rr, dict):
            raise LedgerCheckError(f"ledger check row {i} must be a mapping")
        try:
            item = rr["ledger_item"]
            block = rr["block"]
            field = rr["part2_field"]
            engine_fixed = rr["engine_fixed"]
            incompatible = rr["incompatible_stated"]
            category = rr["category"]
            detail = rr["detail"]
        except KeyError as exc:
            raise LedgerCheckError(f"ledger check row {i} missing required key {exc}") from exc
        if not isinstance(incompatible, list):
            raise LedgerCheckError(
                f"ledger check row {i} 'incompatible_stated' must be a list"
            )
        rows.append(
            LedgerCheckRow(
                ledger_item=item,
                block=str(block),
                part2_field=str(field),
                engine_fixed=str(engine_fixed),
                incompatible_stated=tuple(str(v) for v in incompatible),
                category=str(category),
                detail=str(detail),
                cite=rr.get("cite"),
                borderline=bool(rr.get("borderline", False)),
            )
        )
    return LedgerCheckTable(version=str(version), rows=tuple(rows))


# ---------------------------------------------------------------------------
# The checker.
# ---------------------------------------------------------------------------


def _field_for(container: object, field: str, where: str):
    """The Inherited field off a Part2 or Leg container. A table row that names a
    non-existent field is a build error, surfaced loudly (not a silent no-fire)."""
    if not hasattr(container, field):
        raise LedgerCheckError(
            f"ledger check references unknown field {field!r} on {where}"
        )
    return getattr(container, field)


def check_assumptions(
    spec: "StrategySpec", table: LedgerCheckTable
) -> tuple[ConfigRefusal, ...]:
    """Deterministic assumptions-mismatch screen. For each table row: if the
    strategy's field is STATED and its value is in ``incompatible_stated``, emit a
    ``ConfigRefusal(code=ASSUMPTION_MISMATCH, ...)`` carrying that field's evidence.
    INFERRED / UNKNOWN / DESIGN fields never mismatch (the silence policy owns the
    engine-default case). Run-to-completion: ALL mismatches are collected, rows in
    file order and legs in index order (a stable, byte-reproducible tuple).

    The returned tuple gates ``run_strategy``: non-empty means refuse."""
    sid = spec.header.strategy_label.value or spec.header.paper_id
    out: list[ConfigRefusal] = []
    for row in table.rows:
        if row.block == "common":
            fld = _field_for(spec.part2, row.part2_field, "Part2")
            if fld.tag == "STATED" and fld.value in row.incompatible_stated:
                out.append(
                    ConfigRefusal(
                        sid, RefusalCode.ASSUMPTION_MISMATCH, row.part2_field,
                        row.detail, fld.evidence,
                    )
                )
        else:  # "sort": a per-leg field -- fire once per offending leg
            for i, leg in enumerate(spec.part2.legs):
                fld = _field_for(leg, row.part2_field, f"legs[{i}]")
                if fld.tag == "STATED" and fld.value in row.incompatible_stated:
                    out.append(
                        ConfigRefusal(
                            sid, RefusalCode.ASSUMPTION_MISMATCH,
                            f"legs[{i}].{row.part2_field}", row.detail, fld.evidence,
                        )
                    )
    return tuple(out)
