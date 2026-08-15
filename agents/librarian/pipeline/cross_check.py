"""
Cross-check -- deterministic Part 1 <-> Part 2 reconciliation (build brief §5.4,
D14/D18).

Part 1 (what the paper *says* it is -- an enum) and Part 2 (what facts it
*contains* -- a pattern of block-filledness) are extracted independently (D14).
This module compares them and raises REVIEW FLAGS on inconsistency. It NEVER
auto-repairs (D14/D18): a flag is a signal for a human, never a silent edit.

Two rules:

  * **Marker rule (D18/D32b).** The sort block's markers are the sort signal
    (a SignalRef concept) and ``n_groups``. If the declared family is a sort
    (``formation_structure == sorted_portfolios``) but ALL its markers are
    UNKNOWN, the paper cannot really be the sort it claims to be -> review flag.
    Markers present with otherwise-sparse details is a normal paper (no alarm) --
    D18 deliberately replaced the brittle "substantially filled" threshold for
    the *declared* family.

  * **Non-declared-block rule (D14).** If the paper declares it is NOT a sort
    (``formation_structure != sorted_portfolios``) yet the sort block is
    substantially filled (its markers are STATED), the passes diverge -> review
    flag. (v1 has one block, so "non-declared block filled" = "sort block filled
    on a non-sort paper".)

Both rules read only STATED/UNKNOWN tags -- structurally different evidence from
the enum, so a shared misread fails differently (D14).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import LibrarianSchemaError
from ..schema import fields as F
from ..schema.strategy_spec import Part1, Part2

# The Part-1 enum value that declares a sort-family paper.
SORTED_PORTFOLIOS: str = "sorted_portfolios"

# (v1.2) Fitted-model families that construct in the estimation block, NOT the
# sort block. A paper declaring one of these expects an EMPTY (stub) sort block,
# so it must never trip the non-declared-block rule. The KPP gold's stub Part2 is
# all-UNKNOWN and would not trip it anyway, but suppressing explicitly is robust
# against a fitted-model spec that happens to fill a sort field.
FITTED_FAMILIES: frozenset[str] = frozenset({"estimated_factor_model", "trained_predictor"})

# Review-flag kinds (typed, deterministic; never a tag -- D24).
MARKERS_ALL_UNKNOWN: str = "markers_all_unknown"
NON_DECLARED_BLOCK_FILLED: str = "non_declared_block_filled"


@dataclass(frozen=True)
class ReviewFlag:
    """A deterministic cross-check finding routed to the review lane. NOT a tag
    and NOT a mutation -- a human adjudicates. ``kind`` is one of the two rule
    ids above; ``detail`` explains the mismatch."""

    kind: str
    detail: str

    def __post_init__(self) -> None:
        if self.kind not in (MARKERS_ALL_UNKNOWN, NON_DECLARED_BLOCK_FILLED):
            raise LibrarianSchemaError(
                f"ReviewFlag.kind must be a known cross-check rule id; got {self.kind!r}"
            )
        if not isinstance(self.detail, str) or self.detail.strip() == "":
            raise LibrarianSchemaError("ReviewFlag.detail must be a non-empty string")

    @property
    def routing(self) -> str:
        return "review"

    def to_dict(self) -> dict:
        return {"kind": self.kind, "detail": self.detail, "routing": self.routing}


def _marker_is_unknown_sort_signal(part2: Part2) -> bool:
    """True iff every leg's sort_signal concept_id is UNKNOWN (tag UNKNOWN or a
    None concept value). A leg whose signal is stated marks the block present."""
    for leg in part2.legs:
        cid = leg.sort_signal.concept_id
        if cid.tag != "UNKNOWN" and cid.value is not None:
            return False
    return True


def _marker_is_unknown_n_groups(part2: Part2) -> bool:
    """True iff every leg's n_groups is UNKNOWN."""
    for leg in part2.legs:
        if leg.n_groups.tag != "UNKNOWN":
            return False
    return True


def _sort_block_markers_all_unknown(part2: Part2) -> bool:
    """The D18 test for the sort family: BOTH markers (sort_signal + n_groups)
    all-UNKNOWN across every leg."""
    return _marker_is_unknown_sort_signal(part2) and _marker_is_unknown_n_groups(part2)


def _sort_block_substantially_filled(part2: Part2) -> bool:
    """The non-declared-block test: at least one leg has BOTH markers STATED --
    the paper contains a real sort construction regardless of what it declared."""
    for leg in part2.legs:
        cid = leg.sort_signal.concept_id
        signal_stated = cid.tag == "STATED" and cid.value is not None
        groups_stated = leg.n_groups.tag == "STATED"
        if signal_stated and groups_stated:
            return True
    return False


def cross_check(part1: Part1, part2: Part2) -> list[ReviewFlag]:
    """Reconcile Part 1's declared family against Part 2's block-filledness (D14,
    D18). Returns a (possibly empty) list of ``ReviewFlag``s. Pure + read-only:
    it inspects the two spec parts and mutates NOTHING (D14/D18: never
    auto-repairs)."""
    if not isinstance(part1, Part1):
        raise LibrarianSchemaError("cross_check expects a Part1")
    if not isinstance(part2, Part2):
        raise LibrarianSchemaError("cross_check expects a Part2")

    flags: list[ReviewFlag] = []
    declared = part1.formation_structure.value
    declares_sort = (
        part1.formation_structure.tag == "STATED" and declared == SORTED_PORTFOLIOS
    )

    # Marker rule (D18): declared a sort, but its markers are all UNKNOWN.
    if declares_sort and _sort_block_markers_all_unknown(part2):
        flags.append(
            ReviewFlag(
                MARKERS_ALL_UNKNOWN,
                f"formation_structure declares {SORTED_PORTFOLIOS!r} but both sort-block markers "
                f"({', '.join(sorted(F.MARKERS))}) are UNKNOWN across all legs (D18)",
            )
        )

    # Non-declared-block rule (D14): declared NOT a sort, yet the sort block is
    # substantially filled. Only fires when Part 1 makes a positive non-sort
    # declaration (a STATED enum other than sorted_portfolios) that is NOT a
    # fitted-model family (which constructs in the estimation block, v1.2).
    declares_non_sort = (
        part1.formation_structure.tag == "STATED"
        and declared != SORTED_PORTFOLIOS
        and declared not in FITTED_FAMILIES
    )
    if declares_non_sort and _sort_block_substantially_filled(part2):
        flags.append(
            ReviewFlag(
                NON_DECLARED_BLOCK_FILLED,
                f"formation_structure declares {declared!r} (not a sort) yet the sort block is "
                "substantially filled (a leg has both markers STATED) (D14)",
            )
        )

    return flags
