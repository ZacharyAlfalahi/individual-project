"""
Provenance-carrying value wrappers for QuantConfig.

Two "flavours" share the ``{value, tag, evidence}`` shape but use distinct tag
vocabularies, enforced as separate frozen dataclasses so flavour separation is
*structural* -- an ``Inherited`` cannot carry a ``BOUND`` tag, and a ``Binding``
cannot carry ``STATED``:

  Inherited[T] -- a fact carried forward from the paper / upstream spec.
      tags: STATED | INFERRED | DESIGN | UNKNOWN
  Binding      -- a mapping from an abstract concept to a real panel column.
      tags: BOUND | AMBIGUOUS | MISSING

``DESIGN`` is included beyond the brief's {STATED, INFERRED, UNKNOWN} to match
the repo's established provenance vocabulary (``configs/ipca_instruments.yaml``,
``docs/quant/specs/BBW_anchor_implementation_spec.md``). It is load-bearing: the
par-weighting substitution (BBW spec §2.4 / §5 -- the paper prints
value-weighting, the pipeline uses par) is a deliberate project decision,
honestly ``DESIGN``; it is not ``STATED`` and not ``INFERRED``-with-a-rule (the
inference-rule registry does not exist yet).

No self-reported confidence scores. ``UNKNOWN`` / ``MISSING`` are
valid values, not errors.

Named ``Inherited`` / ``Binding`` rather than ``Fact[T]`` so as not to pre-empt
the upstream ``StrategySpec`` schema (designed later); the inherited flavour can
be unified with it then.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from ..library.characteristic_sort import _RESERVED_COLUMNS

T = TypeVar("T")

InheritedTag = Literal["STATED", "INFERRED", "DESIGN", "UNKNOWN"]
BindingTag = Literal["BOUND", "AMBIGUOUS", "MISSING"]

_INHERITED_TAGS: tuple[str, ...] = ("STATED", "INFERRED", "DESIGN", "UNKNOWN")
_BINDING_TAGS: tuple[str, ...] = ("BOUND", "AMBIGUOUS", "MISSING")


class ProvenanceError(ValueError):
    """A provenance record is internally inconsistent -- a tag outside its
    flavour's vocabulary, or evidence missing the payload its tag requires."""


def _nonempty(s: object) -> bool:
    return isinstance(s, str) and s.strip() != ""


@dataclass(frozen=True)
class Locator:
    """A page + character span into the canonical parsed text, locating a STATED
    quote (D6/D7).

    D7 makes the locator globally mandatory for STATED (not Librarian-only), so the
    requirement is enforced in ``Inherited.__post_init__``, not here. Added additively
    to the frozen provenance layer per the D6 amendment (2026-07-09): additive fields
    + one strengthened guard, no change to existing values, tags, or ``to_rulebook``.
    """

    page: int
    char_start: int
    char_end: int

    def __post_init__(self) -> None:
        for name, v in (
            ("page", self.page),
            ("char_start", self.char_start),
            ("char_end", self.char_end),
        ):
            # bool is an int subclass; reject it (mirrors the config layer's int guards).
            if not isinstance(v, int) or isinstance(v, bool):
                raise ProvenanceError(f"Locator.{name} must be an int; got {v!r}")
        if self.page < 0:
            raise ProvenanceError(f"Locator.page must be >= 0; got {self.page}")
        if self.char_start < 0 or self.char_end < self.char_start:
            raise ProvenanceError(
                "Locator span invalid: 0 <= char_start <= char_end required; "
                f"got char_start={self.char_start}, char_end={self.char_end}"
            )

    def to_dict(self) -> dict:
        return {"page": self.page, "char_start": self.char_start, "char_end": self.char_end}


@dataclass(frozen=True)
class Evidence:
    """
    The justification attached to a provenance record. Which subfields are
    required depends on the wrapping record's tag (validated there):

      quote       -- verbatim source quote              (Inherited STATED)
      rule_id     -- inference-rule id                  (Inherited INFERRED)
      rule_text   -- inference-rule text                (Inherited INFERRED, opt.)
      note        -- free text: the recorded decision (DESIGN), what was
                     searched (UNKNOWN / MISSING), or why a column was chosen
                     (BOUND)
      candidates  -- the columns considered              (Binding AMBIGUOUS)
      chosen      -- the column selected                 (Binding AMBIGUOUS)
      column      -- the bound column                    (Binding BOUND)
      locator     -- page + char span into canonical text (Inherited STATED, D7)
      unknown_reason -- structured UNKNOWN reason code    (Inherited UNKNOWN, D11)

    ``candidates`` is a tuple so ``Evidence`` stays hashable/frozen;
    ``to_dict`` converts it to a list (PyYAML/JSON cannot represent tuples).

    ``locator`` and ``unknown_reason`` are additive (D6 amendment, 2026-07-09).
    ``unknown_reason`` is a bare string whose value domain (D11's three codes plus
    D24's ``input_unknown``) is enforced by the tag-reason registry validator (D24, P5),
    **not** here -- a hardcoded check would reject legitimate adapter ``input_unknown``.
    """

    quote: str | None = None
    rule_id: str | None = None
    rule_text: str | None = None
    note: str | None = None
    candidates: tuple[str, ...] | None = None
    chosen: str | None = None
    column: str | None = None
    locator: Locator | None = None
    unknown_reason: str | None = None

    def __post_init__(self) -> None:
        # Coerce a list of candidates to a tuple so Evidence stays hashable and
        # genuinely immutable (a list field would silently break both).
        if isinstance(self.candidates, list):
            object.__setattr__(self, "candidates", tuple(self.candidates))

    def to_dict(self) -> dict:
        """JSON/YAML-safe dict: drops ``None`` fields, tuples -> lists."""
        out: dict = {}
        for key, val in (
            ("quote", self.quote),
            ("rule_id", self.rule_id),
            ("rule_text", self.rule_text),
            ("note", self.note),
            (
                "candidates",
                list(self.candidates) if self.candidates is not None else None,
            ),
            ("chosen", self.chosen),
            ("column", self.column),
            ("unknown_reason", self.unknown_reason),
        ):
            if val is not None:
                out[key] = val
        if self.locator is not None:
            out["locator"] = self.locator.to_dict()
        return out


@dataclass(frozen=True)
class Inherited(Generic[T]):
    """A value carried forward from the paper / upstream spec, with provenance.

    It is structurally impossible to hold a value without a ``tag`` + ``evidence``
    record: all three fields are required by the constructor.
    """

    value: T | None
    tag: InheritedTag
    evidence: Evidence

    def __post_init__(self) -> None:
        if self.tag not in _INHERITED_TAGS:
            raise ProvenanceError(
                f"Inherited.tag must be one of {_INHERITED_TAGS}; got {self.tag!r} "
                "(binding tags BOUND/AMBIGUOUS/MISSING are not valid here)"
            )
        if not isinstance(self.evidence, Evidence):
            raise ProvenanceError("Inherited.evidence must be an Evidence")
        if self.tag == "STATED" and not _nonempty(self.evidence.quote):
            raise ProvenanceError("STATED requires a non-empty verbatim quote")
        if self.tag == "STATED" and self.evidence.locator is None:
            raise ProvenanceError(
                "STATED requires a locator (page + char span into the canonical "
                "text) per D7 -- locator required for every STATED, globally"
            )
        if self.tag == "INFERRED" and not _nonempty(self.evidence.rule_id):
            raise ProvenanceError("INFERRED requires a rule_id")
        if self.tag == "DESIGN" and not _nonempty(self.evidence.note):
            raise ProvenanceError("DESIGN requires a note recording the decision")
        if self.tag == "UNKNOWN" and not _nonempty(self.evidence.note):
            raise ProvenanceError("UNKNOWN requires a note on what was searched")


@dataclass(frozen=True)
class Binding:
    """A mapping from an abstract concept (e.g. the strategy signal) to a real
    panel column name, with provenance. ``value`` is the bound column name, or
    ``None`` when ``MISSING``."""

    value: str | None
    tag: BindingTag
    evidence: Evidence

    def __post_init__(self) -> None:
        if self.tag not in _BINDING_TAGS:
            raise ProvenanceError(
                f"Binding.tag must be one of {_BINDING_TAGS}; got {self.tag!r} "
                "(inherited tags STATED/INFERRED/DESIGN/UNKNOWN are not valid here)"
            )
        if not isinstance(self.evidence, Evidence):
            raise ProvenanceError("Binding.evidence must be an Evidence")

        if self.tag == "MISSING":
            if self.value is not None:
                raise ProvenanceError("MISSING binding must have value=None")
            if not _nonempty(self.evidence.note):
                raise ProvenanceError("MISSING requires a note on what was searched")
            return

        # BOUND / AMBIGUOUS: a usable column name that must not collide with an
        # engine-internal reserved name (else the engine would raise late, or
        # silently mis-merge).
        if not _nonempty(self.value):
            raise ProvenanceError(f"{self.tag} binding requires a non-empty column name")
        if self.value in _RESERVED_COLUMNS:
            raise ProvenanceError(
                f"binding to reserved engine-internal column {self.value!r} is not "
                f"allowed (reserved: {_RESERVED_COLUMNS})"
            )
        if self.tag == "BOUND" and not _nonempty(self.evidence.column):
            raise ProvenanceError("BOUND requires evidence.column (the chosen column)")
        if self.tag == "AMBIGUOUS":
            if not self.evidence.candidates:
                raise ProvenanceError("AMBIGUOUS requires non-empty evidence.candidates")
            if self.value != self.evidence.chosen:
                raise ProvenanceError("AMBIGUOUS requires value == evidence.chosen")

    @property
    def is_usable(self) -> bool:
        """True when the binding resolved to a column (BOUND or AMBIGUOUS)."""
        return self.tag in ("BOUND", "AMBIGUOUS")
