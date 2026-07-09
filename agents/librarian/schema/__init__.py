"""
Librarian schema -- the frozen ``StrategySpec`` and its parts (D6/D13/D17/D19)
plus the ``SignalRef`` value object (D22). Paper language only (D3).

Public re-exports so callers write ``from agents.librarian.schema import
StrategySpec`` rather than reaching into the submodules.
"""

from __future__ import annotations

from .fields import (
    ALREADY_FINAL_PART2,
    COMMON_FIELDS,
    INT_FIELDS,
    LEG_FIELDS,
    MARKERS,
    PART1_FIELDS,
    SORT_BLOCK_FIELDS,
)
from .signal_ref import (
    UNRECOGNISED,
    DescribedSignal,
    LocatedQuote,
    SignalRef,
)
from .strategy_spec import (
    Combiner,
    Leg,
    MethodSummary,
    Part1,
    Part2,
    SpecHeader,
    StrategySpec,
)

__all__ = [
    # value objects
    "LocatedQuote",
    "DescribedSignal",
    "SignalRef",
    "UNRECOGNISED",
    # spec dataclasses
    "SpecHeader",
    "MethodSummary",
    "Part1",
    "Leg",
    "Combiner",
    "Part2",
    "StrategySpec",
    # field-name / menu constants
    "PART1_FIELDS",
    "SORT_BLOCK_FIELDS",
    "LEG_FIELDS",
    "COMMON_FIELDS",
    "ALREADY_FINAL_PART2",
    "MARKERS",
    "INT_FIELDS",
]
