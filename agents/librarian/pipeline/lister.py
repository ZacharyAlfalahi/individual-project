"""
Enumeration stage -- the lister (build brief §5.2, D20).

Per paper, the Librarian first enumerates every *construction* the paper
describes, before extracting any spec. Each construction carries:

  * ``name``  -- the paper's own name for the construction (+ a located quote);
  * ``cls``   -- ``"strategy"`` (spawns a spec) or ``"auxiliary"`` (recorded, not
                 specced -- e.g. a market factor MKT used only as a control);
  * ``grid``  -- ``GridInfo(is_grid, headline_cell, cells_noted)``: the D20 grid
                 rule. A parameter sweep of ONE signal concept is one grid -> one
                 strategy at the paper's headline cell, the other cells recorded
                 as metadata (a J x K momentum table -> one strategy, not J*K).

Enumeration is dual-model checked (D20): both models return a construction list;
they must agree on the SET of construction names AND each name's class. Any
disagreement -> an ``EnumerationDisagreement`` event (the paper goes to review,
never auto-reconciled). A per-paper hand ``gold`` list (same schema) loads via
``load_gold_list``.

This module operates on ANY ``CanonicalText`` (stub or frozen) via ``.locate()``;
the frozen-status gate is the orchestrator's job (``run_paper``), not the
lister's.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from agents.quant.config import Locator

from ..config.canonical_text import CanonicalText
from ..errors import LibrarianSchemaError
from .failures import EnumerationDisagreement

# The construction classes (D20).
STRATEGY: str = "strategy"
AUXILIARY: str = "auxiliary"
CONSTRUCTION_CLASSES: frozenset[str] = frozenset((STRATEGY, AUXILIARY))


@dataclass(frozen=True)
class GridInfo:
    """The D20 grid rule for one construction. ``is_grid`` True means the paper
    reports a parameter sweep of a single concept; ``headline_cell`` names the
    cell the spec is built at; ``cells_noted`` records the other cells as
    metadata (never spawned as separate strategies)."""

    is_grid: bool = False
    headline_cell: str | None = None
    cells_noted: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.is_grid, bool):
            raise LibrarianSchemaError("GridInfo.is_grid must be a bool")
        if isinstance(self.cells_noted, list):
            object.__setattr__(self, "cells_noted", tuple(self.cells_noted))
        if not isinstance(self.cells_noted, tuple) or any(
            not isinstance(c, str) for c in self.cells_noted
        ):
            raise LibrarianSchemaError("GridInfo.cells_noted must be a tuple of strings")
        if self.headline_cell is not None and (
            not isinstance(self.headline_cell, str) or self.headline_cell.strip() == ""
        ):
            raise LibrarianSchemaError("GridInfo.headline_cell must be a non-empty string or None")
        if self.is_grid and self.headline_cell is None:
            raise LibrarianSchemaError(
                "a grid construction must name its headline_cell (D20: the spec is built at "
                "the paper's headline cell)"
            )

    def to_dict(self) -> dict:
        return {
            "is_grid": self.is_grid,
            "headline_cell": self.headline_cell,
            "cells_noted": list(self.cells_noted),
        }


@dataclass(frozen=True)
class Construction:
    """One construction the paper describes (D20): its name (+ located quote),
    its class (strategy | auxiliary), and its grid info. ``locator`` is the
    ``.locate()`` result for ``quote`` (``None`` when the quote did not locate --
    a gold list may omit it)."""

    name: str
    quote: str
    cls: str
    grid: GridInfo = GridInfo()
    locator: Locator | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or self.name.strip() == "":
            raise LibrarianSchemaError("Construction.name must be a non-empty string")
        if not isinstance(self.quote, str) or self.quote.strip() == "":
            raise LibrarianSchemaError("Construction.quote must be a non-empty string")
        if self.cls not in CONSTRUCTION_CLASSES:
            raise LibrarianSchemaError(
                f"Construction.cls must be one of {sorted(CONSTRUCTION_CLASSES)}; got {self.cls!r}"
            )
        if not isinstance(self.grid, GridInfo):
            raise LibrarianSchemaError("Construction.grid must be a GridInfo")
        if self.locator is not None and not isinstance(self.locator, Locator):
            raise LibrarianSchemaError("Construction.locator must be a Locator or None")

    @property
    def is_strategy(self) -> bool:
        return self.cls == STRATEGY

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "quote": self.quote,
            "cls": self.cls,
            "grid": self.grid.to_dict(),
            "locator": self.locator.to_dict() if self.locator is not None else None,
        }


@dataclass(frozen=True)
class EnumerationResult:
    """The outcome of the enumeration stage for one paper. Exactly one of the two
    outcomes is populated:

      * agreement: ``constructions`` is the agreed list, ``disagreement`` is
        ``None``. ``strategies`` / ``auxiliaries`` partition it.
      * disagreement: ``disagreement`` is an ``EnumerationDisagreement`` event
        (routes to review) and ``constructions`` is empty."""

    paper_id: str
    constructions: tuple[Construction, ...] = ()
    disagreement: EnumerationDisagreement | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.paper_id, str) or self.paper_id.strip() == "":
            raise LibrarianSchemaError("EnumerationResult.paper_id must be a non-empty string")
        if isinstance(self.constructions, list):
            object.__setattr__(self, "constructions", tuple(self.constructions))
        for c in self.constructions:
            if not isinstance(c, Construction):
                raise LibrarianSchemaError("EnumerationResult.constructions must be Constructions")
        if self.disagreement is not None and not isinstance(
            self.disagreement, EnumerationDisagreement
        ):
            raise LibrarianSchemaError(
                "EnumerationResult.disagreement must be an EnumerationDisagreement or None"
            )
        if self.disagreement is not None and len(self.constructions) > 0:
            raise LibrarianSchemaError(
                "an EnumerationResult in disagreement carries no constructions (the paper goes "
                "to review, never a partial list)"
            )

    @property
    def agreed(self) -> bool:
        return self.disagreement is None

    @property
    def strategies(self) -> tuple[Construction, ...]:
        return tuple(c for c in self.constructions if c.is_strategy)

    @property
    def auxiliaries(self) -> tuple[Construction, ...]:
        return tuple(c for c in self.constructions if not c.is_strategy)

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "constructions": [c.to_dict() for c in self.constructions],
            "disagreement": self.disagreement.to_dict() if self.disagreement is not None else None,
        }


# The dual-model comparison key for one construction (D20: agree on the set of
# names AND each name's class; grid metadata is not part of the agreement key --
# it is recorded, not adjudicated).
def _cls_by_name(constructions: tuple[Construction, ...]) -> dict[str, str]:
    out: dict[str, str] = {}
    for c in constructions:
        if c.name in out and out[c.name] != c.cls:
            raise LibrarianSchemaError(
                f"one model listed construction {c.name!r} with two classes -- a self-inconsistent "
                "enumeration is a build error, not a cross-model disagreement"
            )
        out[c.name] = c.cls
    return out


def enumerate_constructions(
    canonical_text: CanonicalText,
    model_a: tuple[Construction, ...],
    model_b: tuple[Construction, ...],
    paper_id: str,
    relocate: bool = True,
) -> EnumerationResult:
    """Dual-model enumeration (D20). ``model_a`` / ``model_b`` are the two models'
    construction lists (produced upstream). Agreement is judged on the SET of
    construction names AND each name's class. On agreement the model_a list is
    shipped (both models produced the same names/classes); when ``relocate`` each
    shipped construction's ``quote`` is re-located against ``canonical_text`` so
    its ``locator`` is filled from the same text the spec indexes into.

    Any name-set or class disagreement -> an ``EnumerationDisagreement`` (routes
    to review). NEVER auto-reconciled."""
    if not isinstance(canonical_text, CanonicalText):
        raise LibrarianSchemaError("enumerate_constructions requires a CanonicalText")
    a_by_name = _cls_by_name(tuple(model_a))
    b_by_name = _cls_by_name(tuple(model_b))

    a_names = set(a_by_name)
    b_names = set(b_by_name)

    only_a = tuple(sorted(a_names - b_names))
    only_b = tuple(sorted(b_names - a_names))
    class_conflicts = tuple(
        sorted(n for n in (a_names & b_names) if a_by_name[n] != b_by_name[n])
    )

    if only_a or only_b or class_conflicts:
        detail_bits = []
        if only_a:
            detail_bits.append(f"only model_a: {list(only_a)}")
        if only_b:
            detail_bits.append(f"only model_b: {list(only_b)}")
        if class_conflicts:
            detail_bits.append(f"class conflict on: {list(class_conflicts)}")
        return EnumerationResult(
            paper_id=paper_id,
            disagreement=EnumerationDisagreement(
                paper_id=paper_id,
                detail="; ".join(detail_bits),
                only_model_a=only_a,
                only_model_b=only_b,
            ),
        )

    # Agreement: ship model_a's list (names + classes identical to model_b's),
    # re-locating each quote against the canonical text if requested.
    shipped: list[Construction] = []
    for c in model_a:
        loc = canonical_text.locate(c.quote) if relocate else c.locator
        shipped.append(
            Construction(name=c.name, quote=c.quote, cls=c.cls, grid=c.grid, locator=loc)
        )
    return EnumerationResult(paper_id=paper_id, constructions=tuple(shipped))


def load_gold_list(path: str | Path) -> EnumerationResult:
    """Load a per-paper hand-authored gold enumeration list (build brief §5.2:
    same schema, ``gold: true``) into an ``EnumerationResult``. The gold file
    carries ``paper_id`` and a ``constructions`` list of
    ``{name, quote, class, grid?}`` rows. Gold rows are trusted as the agreed
    list -- no dual-model step, no relocation (a gold list is authored, not
    extracted)."""
    p = Path(path)
    if not p.exists():
        raise LibrarianSchemaError(f"gold enumeration list not found at {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise LibrarianSchemaError("gold enumeration list must be a mapping at top level")
    if not raw.get("gold", False):
        raise LibrarianSchemaError("gold enumeration list must carry 'gold: true'")
    paper_id = raw.get("paper_id")
    if not isinstance(paper_id, str) or paper_id.strip() == "":
        raise LibrarianSchemaError("gold enumeration list must declare a non-empty 'paper_id'")
    raw_cons = raw.get("constructions")
    if not isinstance(raw_cons, list):
        raise LibrarianSchemaError("gold enumeration list 'constructions' must be a list")

    constructions: list[Construction] = []
    for i, rc in enumerate(raw_cons):
        if not isinstance(rc, dict):
            raise LibrarianSchemaError(f"gold construction {i} must be a mapping")
        raw_grid = rc.get("grid", {}) or {}
        grid = GridInfo(
            is_grid=bool(raw_grid.get("is_grid", False)),
            headline_cell=raw_grid.get("headline_cell"),
            cells_noted=tuple(str(c) for c in raw_grid.get("cells_noted", [])),
        )
        try:
            constructions.append(
                Construction(
                    name=str(rc["name"]),
                    quote=str(rc["quote"]),
                    cls=str(rc["class"]),
                    grid=grid,
                )
            )
        except KeyError as exc:
            raise LibrarianSchemaError(
                f"gold construction {i} missing required key {exc}"
            ) from exc
    return EnumerationResult(paper_id=paper_id, constructions=tuple(constructions))
