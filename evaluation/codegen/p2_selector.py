"""P2 coverage-boundary arm selector (WS-C).

Turns a ``CensusResult`` + a frozen zoo-list (the ordered scale-layer paper
names) into the two P2 arms, with ZERO selection discretion:

  * **Arm A** = the WHOLE refusal set. Its size is *discovered* by the census,
    never chosen — the coverage boundary is however wide the census says it is.
  * **Arm B** = the first ``min(|Arm A|, arm_b_max)`` COMPILABLE members in
    frozen zoo-list order — a size-matched (capped) control group of things the
    deterministic compiler CAN build, drawn deterministically from the frozen
    order so the choice precedes any result.

Both arms are ordered by the frozen zoo-list, so the selection is fully
reproducible from (census, zoo_list, thresholds). Every routed member (refusal
or compilable) MUST appear in the frozen zoo-list — a member absent from it means
the frozen list is stale, which is a build error, not a silent drop.

The ``below_floor`` flag (``|Arm A| < below_floor_min_arm_a``) is carried here
and honoured downstream: below floor, P2 emits only the census fate table +
taxonomy (agreement/divergence are suppressed — see ``p2_metrics``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from evaluation.codegen.census import CensusResult


class P2SelectionError(ValueError):
    """The zoo-list is inconsistent with the census (a routed member is absent
    from the frozen order) — a build error, surfaced loudly."""


@dataclass(frozen=True)
class ArmSelection:
    """The two P2 arms, both in frozen zoo-list order, plus the discovered Arm-A
    size and the below-floor flag."""

    arm_a: tuple[str, ...]        # every refused paper (the whole refusal set)
    arm_b: tuple[str, ...]        # first min(|A|, arm_b_max) compilable, zoo order
    arm_a_size: int
    below_floor: bool

    def members(self) -> tuple[str, ...]:
        """Arm A ∪ Arm B, Arm A first — the driver's generation loop order."""
        return self.arm_a + self.arm_b


def _zoo_index(zoo_list: Sequence[str]) -> dict[str, int]:
    index: dict[str, int] = {}
    for pos, name in enumerate(zoo_list):
        if not isinstance(name, str) or name.strip() == "":
            raise P2SelectionError("zoo_list entries must be non-empty strings")
        if name in index:
            raise P2SelectionError(f"duplicate paper {name!r} in the frozen zoo-list")
        index[name] = pos
    return index


def select_arms(
    census: CensusResult,
    zoo_list: Sequence[str],
    thresholds: dict,
) -> ArmSelection:
    """Deterministically select Arm A (whole refusal set) and Arm B (first
    ``min(|A|, arm_b_max)`` compilable), both in frozen zoo-list order.

    ``thresholds`` is the ``p2_codegen`` block (``arms.arm_b_max`` /
    ``arms.below_floor_min_arm_a``). Raises ``P2SelectionError`` if any routed
    member is missing from the frozen zoo-list."""
    try:
        arm_b_max = int(thresholds["arms"]["arm_b_max"])
        below_floor_min = int(thresholds["arms"]["below_floor_min_arm_a"])
    except (KeyError, TypeError) as exc:
        raise KeyError(
            "p2_codegen thresholds missing arms.arm_b_max / arms.below_floor_min_arm_a"
        ) from exc

    index = _zoo_index(zoo_list)

    def zpos(paper_id: str) -> int:
        if paper_id not in index:
            raise P2SelectionError(
                f"census member {paper_id!r} is absent from the frozen zoo-list — "
                "the frozen order is stale relative to the census"
            )
        return index[paper_id]

    arm_a = tuple(sorted((m.paper_id for m in census.refusal_set()), key=zpos))
    arm_a_size = len(arm_a)

    b_target = min(arm_a_size, arm_b_max)
    compilable_in_zoo = sorted((m.paper_id for m in census.compilable_set()), key=zpos)
    arm_b = tuple(compilable_in_zoo[:b_target])

    below_floor = arm_a_size < below_floor_min
    return ArmSelection(
        arm_a=arm_a, arm_b=arm_b, arm_a_size=arm_a_size, below_floor=below_floor
    )
