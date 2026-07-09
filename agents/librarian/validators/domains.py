"""
Closed-domain declarations + validator (D24 part 2).

A ``Domain`` is one of two shapes:

  * value-set -- a frozenset of allowed enum values. Every Part 2 enum domain
                 includes ``"other"`` (P3: menus over prose, with a first-class
                 escape).
  * numeric range -- an int (or float) with inclusive ``[min, max]`` bounds,
                 either of which may be unbounded (``None``). A parametric upper
                 bound (``max_field`` + ``max_exclusive``) expresses engine
                 ranges like ``long_group in [0, groups)``; such a domain cannot
                 be validated without the referenced field's value, so
                 ``validate_value`` reports it as un-checkable standalone (the
                 adapter/engine-config gate supplies the reference).

``load_domains`` reads ``data/domains.yaml`` into ``{family -> {field ->
Domain}}``. ``validate_value(field, value)`` checks a value against the Part 2
family (the Librarian-side gate). ``iter_domain_cases()`` yields representative
values per field -- a seam for the future G1 exhaustive-enumeration generator
(D24 requires a test over each transform rule's entire input domain); kept
deliberately simple here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import yaml

from ..errors import LibrarianSchemaError

_DEFAULT_DOMAINS_PATH = Path(__file__).resolve().parent.parent / "data" / "domains.yaml"

# How many representative int values iter_domain_cases samples above the min.
_INT_SAMPLE_SPAN = 3


@dataclass(frozen=True)
class Domain:
    """A closed domain for one field. Exactly one of ``values`` (enum) or the
    numeric bounds is meaningful, selected by ``kind``."""

    field: str
    kind: str                      # "enum" | "int" | "float"
    values: frozenset[str] | None = None
    min: int | float | None = None
    max: int | float | None = None
    max_field: str | None = None   # parametric upper bound (engine ranges)
    max_exclusive: bool = False

    @property
    def is_enum(self) -> bool:
        return self.kind == "enum"

    @property
    def is_numeric(self) -> bool:
        return self.kind in ("int", "float")

    @property
    def is_parametric(self) -> bool:
        """True when the upper bound references another field (cannot be checked
        standalone)."""
        return self.max_field is not None

    def contains(self, value: object) -> bool:
        """Is ``value`` in this domain? Parametric-max domains ignore the upper
        bound (the reference isn't available here) and check only the lower
        bound + type -- the adapter/engine gate does the full parametric check."""
        if self.is_enum:
            return isinstance(value, str) and value in (self.values or frozenset())
        if self.kind == "int":
            if not isinstance(value, int) or isinstance(value, bool):
                return False
        elif self.kind == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False
        else:  # pragma: no cover -- guarded at load time
            raise LibrarianSchemaError(f"unknown domain kind {self.kind!r}")
        if self.min is not None and value < self.min:
            return False
        if not self.is_parametric and self.max is not None and value > self.max:
            return False
        return True

    def representatives(self) -> tuple[object, ...]:
        """A small set of in-domain representative values (a G1 seam)."""
        if self.is_enum:
            return tuple(sorted(self.values or frozenset()))
        if self.kind == "int":
            lo = self.min if self.min is not None else 0
            return tuple(lo + k for k in range(_INT_SAMPLE_SPAN))
        # float
        lo = self.min if self.min is not None else 0.0
        return (float(lo), float(lo) + 1.0)


def _parse_domain(field: str, spec: dict) -> Domain:
    kind = spec.get("kind")
    if kind == "enum":
        values = spec.get("values")
        if not isinstance(values, list) or len(values) == 0:
            raise LibrarianSchemaError(f"enum domain {field!r} must list non-empty 'values'")
        return Domain(field=field, kind="enum", values=frozenset(str(v) for v in values))
    if kind in ("int", "float"):
        return Domain(
            field=field,
            kind=kind,
            min=spec.get("min"),
            max=spec.get("max"),
            max_field=spec.get("max_field"),
            max_exclusive=bool(spec.get("max_exclusive", False)),
        )
    raise LibrarianSchemaError(f"domain {field!r} has unknown kind {kind!r}")


def load_domains(path: str | Path | None = None) -> dict[str, dict[str, Domain]]:
    """Load ``domains.yaml`` into ``{family -> {field -> Domain}}`` (families:
    ``part2``, ``engine``)."""
    p = Path(path) if path is not None else _DEFAULT_DOMAINS_PATH
    if not p.exists():
        raise LibrarianSchemaError(f"domains file not found at {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise LibrarianSchemaError("domains file must be a mapping at top level")

    out: dict[str, dict[str, Domain]] = {}
    for family in ("part2", "engine"):
        fam = raw.get(family, {})
        if not isinstance(fam, dict):
            raise LibrarianSchemaError(f"domains family {family!r} must be a mapping")
        out[family] = {name: _parse_domain(name, spec) for name, spec in fam.items()}
    return out


def validate_value(
    field: str,
    value: object,
    domains: dict[str, dict[str, Domain]] | None = None,
    family: str = "part2",
) -> bool:
    """True iff ``value`` is in ``field``'s domain (default family ``part2``).
    Raises ``LibrarianSchemaError`` if the field has no declared domain (an
    unknown field is a build error, not a silent pass)."""
    doms = domains if domains is not None else load_domains()
    fam = doms.get(family, {})
    if field not in fam:
        raise LibrarianSchemaError(
            f"no domain declared for field {field!r} in family {family!r}"
        )
    return fam[field].contains(value)


def iter_domain_cases(
    domains: dict[str, dict[str, Domain]] | None = None,
    family: str = "part2",
) -> Iterator[tuple[str, tuple[object, ...]]]:
    """Yield ``(field, representative_values)`` for each field in ``family`` -- a
    seam for the future G1 exhaustive-enumeration generator (D24). Simple by
    design: enum fields yield their whole value set, int/float fields a short
    sample above the min."""
    doms = domains if domains is not None else load_domains()
    for field, dom in doms.get(family, {}).items():
        yield field, dom.representatives()
