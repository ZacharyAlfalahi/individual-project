"""
SignalRef -- the single type every characteristic reference in a spec takes
(the sort signal, the control axis of a double sort). D22: signal identity is a
*menu pick* from the Signal Concept Registry, not a computation and not free
text.

A SignalRef carries three things:

  * ``concept_id``  -- an ``Inherited[str]``: the registry id the paper's signal
                       matches (exact + binary matching, no similarity scores),
                       or the first-class escape ``"unrecognised"`` ("not one of
                       ours"). Provenance-wrapped like every fact.
  * ``parameters``  -- a mapping ``name -> Inherited`` of the concept's dials
                       (window, months, min-obs...). Each value is
                       provenance-wrapped; the *set* of legal names/types is the
                       concept's parameter schema, checked later with a loaded
                       registry (``validate_librarian_spec``), NOT here.
  * ``as_described``-- the paper's own words: a label + located quotes,
                       audit-only, never machine-consumed.

This module is a *pure value object*: it imports the frozen provenance layer
(``Inherited``) but NOT the registry. Registry *membership* (is ``concept_id``
a known concept? do the parameter names/types match its schema?) is a
registry-aware check performed in ``validators/spec_validators.py`` with a
registry passed in -- so a SignalRef never needs to load, hash, or version a
registry to be constructed. The only cross-registry rule enforced here is the
purely structural escape invariant: an ``unrecognised`` concept MUST carry at
least one located quote in ``as_described`` (you cannot claim "not one of ours"
without showing the paper's words that failed to match).

The D24 "wall-split" is honoured structurally: nothing here carries a column
name. ``concept_id`` is a registry id; columns live on the Quant side.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from agents.quant.config import Inherited

from ..errors import LibrarianSchemaError

# The first-class escape value (D22): the paper's signal is "not one of ours".
UNRECOGNISED: str = "unrecognised"


@dataclass(frozen=True)
class LocatedQuote:
    """A verbatim quote from the paper plus its page + character span into the
    canonical parsed text (the same locator discipline STATED evidence uses,
    D7). Audit-only -- carried in ``as_described`` so a human can check the
    paper's own words behind a SignalRef, never machine-consumed."""

    text: str
    page: int
    char_start: int
    char_end: int

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or self.text.strip() == "":
            raise LibrarianSchemaError("LocatedQuote.text must be a non-empty string")
        for name, v in (
            ("page", self.page),
            ("char_start", self.char_start),
            ("char_end", self.char_end),
        ):
            # bool is an int subclass; reject it (mirrors the provenance layer).
            if not isinstance(v, int) or isinstance(v, bool):
                raise LibrarianSchemaError(f"LocatedQuote.{name} must be an int; got {v!r}")
        if self.page < 0:
            raise LibrarianSchemaError(f"LocatedQuote.page must be >= 0; got {self.page}")
        if self.char_start < 0 or self.char_end < self.char_start:
            raise LibrarianSchemaError(
                "LocatedQuote span invalid: 0 <= char_start <= char_end required; "
                f"got char_start={self.char_start}, char_end={self.char_end}"
            )

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "page": self.page,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


@dataclass(frozen=True)
class DescribedSignal:
    """The paper's own words for a signal: a short label plus the located quotes
    that justify the SignalRef. Audit-only (D22)."""

    label: str
    quotes: tuple[LocatedQuote, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or self.label.strip() == "":
            raise LibrarianSchemaError("DescribedSignal.label must be a non-empty string")
        # Coerce a list of quotes to a tuple so the object stays frozen + hashable.
        if isinstance(self.quotes, list):
            object.__setattr__(self, "quotes", tuple(self.quotes))
        if not isinstance(self.quotes, tuple):
            raise LibrarianSchemaError("DescribedSignal.quotes must be a tuple of LocatedQuote")
        for q in self.quotes:
            if not isinstance(q, LocatedQuote):
                raise LibrarianSchemaError(
                    f"DescribedSignal.quotes entries must be LocatedQuote; got {type(q).__name__}"
                )

    def to_dict(self) -> dict:
        return {"label": self.label, "quotes": [q.to_dict() for q in self.quotes]}


@dataclass(frozen=True)
class SignalRef:
    """A characteristic reference: a registry concept id (or the ``unrecognised``
    escape), the concept's parameters, and the paper's own words.

    Pure value object -- no registry import. Registry membership and parameter
    -schema conformance are checked in ``validate_librarian_spec``; here only
    the structural shape and the escape invariant are enforced."""

    concept_id: Inherited  # Inherited[str]
    as_described: DescribedSignal
    parameters: Mapping[str, Inherited] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.concept_id, Inherited):
            raise LibrarianSchemaError(
                "SignalRef.concept_id must be an Inherited[str]; "
                f"got {type(self.concept_id).__name__}"
            )
        cid = self.concept_id.value
        if cid is not None and not isinstance(cid, str):
            raise LibrarianSchemaError(
                f"SignalRef.concept_id.value must be a str or None; got {cid!r}"
            )
        if not isinstance(self.as_described, DescribedSignal):
            raise LibrarianSchemaError(
                "SignalRef.as_described must be a DescribedSignal; "
                f"got {type(self.as_described).__name__}"
            )

        # Parameters: a mapping name(str) -> Inherited; freeze it (MappingProxy)
        # so the SignalRef stays genuinely immutable.
        if not isinstance(self.parameters, Mapping):
            raise LibrarianSchemaError("SignalRef.parameters must be a Mapping")
        for name, val in self.parameters.items():
            if not isinstance(name, str) or name.strip() == "":
                raise LibrarianSchemaError(
                    f"SignalRef.parameters keys must be non-empty strings; got {name!r}"
                )
            if not isinstance(val, Inherited):
                raise LibrarianSchemaError(
                    f"SignalRef.parameters[{name!r}] must be an Inherited; "
                    f"got {type(val).__name__}"
                )
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))

        # Escape invariant (structural, registry-free): claiming "not one of
        # ours" requires the paper's words that failed to match.
        if cid == UNRECOGNISED and len(self.as_described.quotes) == 0:
            raise LibrarianSchemaError(
                f"SignalRef with concept_id == {UNRECOGNISED!r} requires a non-empty "
                "as_described.quotes (the paper's words behind the unrecognised signal)"
            )

    @property
    def is_unrecognised(self) -> bool:
        return self.concept_id.value == UNRECOGNISED

    def to_dict(self) -> dict:
        return {
            "concept_id": {
                "value": self.concept_id.value,
                "tag": self.concept_id.tag,
                "evidence": self.concept_id.evidence.to_dict(),
            },
            "parameters": {
                name: {
                    "value": val.value,
                    "tag": val.tag,
                    "evidence": val.evidence.to_dict(),
                }
                for name, val in self.parameters.items()
            },
            "as_described": self.as_described.to_dict(),
        }
