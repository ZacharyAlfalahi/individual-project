"""
Transform table loader + appliers (D25, build brief §6).

Loads ``data/transform_table.yaml`` -- the adapter's vocabulary/structure rulebook
-- into a typed ``TransformTable`` and applies the value_changing enum transforms
(``long_leg``, ``weighting``, ``expost_trim``). The enum lookup rows live in the
YAML (data, not code): an applier never hard-codes a mapping, it reads the row.

Appliers are PURE value functions returning a small outcome
(``Produced`` / ``Omit`` / ``Review``); the provenance wrapping (identity
pass-through vs INFERRED(adapter, rule_id)) is ``legs.py``'s job, so this module
never imports the provenance layer -- it owns the *rulebook*, not the *tags*.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from ..errors import LibrarianSchemaError

_DEFAULT_TABLE_PATH = Path(__file__).resolve().parent / "data" / "transform_table.yaml"

_VALID_KINDS: frozenset[str] = frozenset(("identity", "value_changing", "signal"))


# --- transform outcomes (the applier vocabulary) ---------------------------------

@dataclass(frozen=True)
class Produced:
    """The transform produced an engine value (``legs.py`` wraps it). ``value`` is
    the engine value; for ``long_leg`` it is a ``{long_group, short_group}`` dict."""

    value: object


@dataclass(frozen=True)
class Omit:
    """Pass ``None`` to the factory (it fills + tags its documented default)."""


@dataclass(frozen=True)
class Review:
    """The transform cannot faithfully produce a value -> a REVIEW_REQUIRED
    refusal naming ``detail`` (never a fabricated value)."""

    detail: str


TransformOutcome = Produced | Omit | Review


@dataclass(frozen=True)
class TransformRow:
    """One transform: the field it keys on, its id, inputs (single-hop: spec
    fields only), primary input, output kwarg(s), kind, enum lookup, and note."""

    field: str
    id: str
    inputs: tuple[str, ...]
    primary_input: str
    output: tuple[str, ...]      # one or two factory kwargs
    kind: str                    # identity | value_changing | signal
    lookup: Mapping              # enum map (empty for identity/signal/trim)
    note: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise LibrarianSchemaError(
                f"transform {self.id!r} kind {self.kind!r} not one of {sorted(_VALID_KINDS)}"
            )

    def _canonical(self) -> dict:
        return {
            "field": self.field,
            "id": self.id,
            "inputs": list(self.inputs),
            "primary_input": self.primary_input,
            "output": list(self.output),
            "kind": self.kind,
            "lookup": _canonicalise(self.lookup),
            "note": self.note,
        }


def _canonicalise(obj: object) -> object:
    """Recursively sort mapping keys so the content hash is formatting-independent."""
    if isinstance(obj, Mapping):
        return {k: _canonicalise(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonicalise(v) for v in obj]
    return obj


@dataclass(frozen=True)
class TransformTable:
    """The loaded transform table: ``version``, the transform rows keyed by field,
    the combiner map, and a reproducible ``content_hash``."""

    version: str
    transforms: Mapping[str, TransformRow]
    combiner: Mapping[str, Mapping]

    # --- lookup helpers ----------------------------------------------------------

    def row_for(self, field: str) -> TransformRow | None:
        return self.transforms.get(field)

    def engine_fields(self) -> tuple[str, ...]:
        """The Part 2 fields that have an engine hook (a transform row) -- keyed by
        primary input."""
        return tuple(self.transforms.keys())

    def all_inputs(self) -> frozenset[str]:
        """Every spec field any transform reads (primary + secondary), so a
        completeness check can tell a consumed field from a check-only one."""
        out: set[str] = set()
        for row in self.transforms.values():
            out.update(row.inputs)
        return frozenset(out)

    # --- value_changing appliers (read the YAML lookup rows) ---------------------

    def apply_long_leg(self, direction: object, n_groups: object) -> TransformOutcome:
        """``long_leg`` direction + ``n_groups`` -> ``{long_group, short_group}``.
        Needs a concrete ``n_groups`` (top group = g-1): a silent/None n_groups
        cannot place the endpoint without learning the factory default (P5) -> Review.
        An off-menu ``other`` has no engine direction -> Review."""
        if not isinstance(n_groups, int) or isinstance(n_groups, bool):
            return Review(
                "long_leg direction needs n_groups' value (top group = g-1); "
                "n_groups is silent/absent, so the direction cannot be placed without "
                "the factory default (P5)"
            )
        lookup = self.transforms["long_leg"].lookup
        if direction not in lookup:
            return Review(
                f"long_leg={direction!r} has no engine direction "
                f"(known: {sorted(lookup)}); cannot set long/short groups"
            )
        endpoints = lookup[direction]
        g = n_groups
        pos = {"bottom": 0, "top": g - 1}
        return Produced({"long_group": pos[endpoints["long"]], "short_group": pos[endpoints["short"]]})

    def apply_weighting(self, scheme: object, base: object | None) -> TransformOutcome:
        """``weighting_scheme`` (+ ``weighting_base``) -> engine ``weighting``.
        ``base`` is ``None`` when weighting_base is silent. ``equal`` ignores base;
        ``value`` reads base (par -> size is the faithful match; market_value/other
        are forwarded as-is so the FACTORY refuses OUT_OF_ENUM_WEIGHTING). A silent
        scheme, or value with a silent base, omits (factory par default)."""
        lookup = self.transforms["weighting_scheme"].lookup
        if scheme not in lookup:
            # scheme == 'other' (or unknown): forward the raw value so the factory
            # refuses OUT_OF_ENUM_WEIGHTING (representability is the factory's, D25).
            return Produced(scheme)
        mapped = lookup[scheme]
        if isinstance(mapped, str):
            # scheme == 'equal' -> "equal" (base irrelevant).
            return Produced(mapped)
        # scheme == 'value' -> a base map.
        if base is None:
            # base silent -> omit; the factory fills its par ("size") default.
            return Omit()
        if base in mapped:
            return Produced(mapped[base])
        # base == 'other' (or unknown) -> forward so the factory refuses.
        return Produced(base)

    def apply_trim(self, method: object) -> TransformOutcome:
        """``expost_trim`` method -> engine trim. v1: ``none`` omits (factory
        default no-trim); any real method needs lo/hi bounds the v1 StrategySpec
        does not carry -> Review (a documented schema seam, never an unbounded trim)."""
        if method == "none":
            return Omit()
        return Review(
            f"expost_trim={method!r} STATED, but the v1 StrategySpec carries the trim "
            "method only (no lo/hi bounds); a faithful trim cannot be built -- resolve "
            "via the bias-toggle registry / a schema extension (v1 seam)"
        )

    # --- hashing -----------------------------------------------------------------

    @property
    def content_hash(self) -> str:
        canonical = {
            "version": self.version,
            "transforms": [self.transforms[k]._canonical() for k in sorted(self.transforms)],
            "combiner": _canonicalise(self.combiner),
        }
        payload = json.dumps(canonical, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _as_output_tuple(output: object) -> tuple[str, ...]:
    if isinstance(output, str):
        return (output,)
    if isinstance(output, list):
        return tuple(str(o) for o in output)
    raise LibrarianSchemaError(f"transform 'output' must be a str or list; got {output!r}")


def load_transform_table(path: str | Path | None = None) -> TransformTable:
    """Load and validate ``transform_table.yaml`` into a typed table."""
    p = Path(path) if path is not None else _DEFAULT_TABLE_PATH
    if not p.exists():
        raise LibrarianSchemaError(f"transform table not found at {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise LibrarianSchemaError("transform table must be a mapping at top level")

    version = raw.get("version")
    if not isinstance(version, str) or version.strip() == "":
        raise LibrarianSchemaError("transform table must declare a non-empty 'version'")

    raw_transforms = raw.get("transforms")
    if not isinstance(raw_transforms, dict) or len(raw_transforms) == 0:
        raise LibrarianSchemaError("transform table 'transforms' must be a non-empty mapping")

    transforms: dict[str, TransformRow] = {}
    for field, spec in raw_transforms.items():
        if not isinstance(spec, dict):
            raise LibrarianSchemaError(f"transform {field!r} must be a mapping")
        try:
            tid = spec["id"]
            inputs = spec["inputs"]
            primary = spec["primary_input"]
            output = spec["output"]
            kind = spec["kind"]
        except KeyError as exc:
            raise LibrarianSchemaError(f"transform {field!r} missing required key {exc}") from exc
        if not isinstance(inputs, list) or not inputs:
            raise LibrarianSchemaError(f"transform {field!r} 'inputs' must be a non-empty list")
        transforms[str(field)] = TransformRow(
            field=str(field),
            id=str(tid),
            inputs=tuple(str(i) for i in inputs),
            primary_input=str(primary),
            output=_as_output_tuple(output),
            kind=str(kind),
            lookup=spec.get("lookup", {}) or {},
            note=spec.get("note"),
        )

    combiner = raw.get("combiner", {})
    if not isinstance(combiner, dict):
        raise LibrarianSchemaError("transform table 'combiner' must be a mapping")

    return TransformTable(version=str(version), transforms=transforms, combiner=combiner)
