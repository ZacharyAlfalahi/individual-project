"""
Unit tests for the deterministic cross-check (build brief §5.4, D14/D18).

Marker rule: declared sort family + markers all-UNKNOWN -> review flag.
Non-declared-block rule: declared NOT a sort + sort block filled -> review flag.
Never mutates the spec.
"""

from agents.librarian.pipeline.cross_check import (
    MARKERS_ALL_UNKNOWN,
    NON_DECLARED_BLOCK_FILLED,
    ReviewFlag,
    cross_check,
)

from _librarian_fixtures import (
    build_leg,
    build_part1,
    build_part2,
    signal_ref,
    stated,
    unknown,
)


def _unknown_signal_ref():
    """A SignalRef whose concept_id is UNKNOWN (marker absent)."""
    sr = signal_ref("mom6")
    # replace the concept_id Inherited with an UNKNOWN one.
    from agents.librarian.schema.signal_ref import DescribedSignal, SignalRef

    return SignalRef(
        concept_id=unknown(),
        as_described=DescribedSignal(label="mystery", quotes=sr.as_described.quotes),
    )


# --- marker rule (D18) ------------------------------------------------------

def test_markers_all_unknown_flags_review():
    part1 = build_part1(formation_structure=stated("sorted_portfolios"))
    # both markers UNKNOWN across the single leg.
    leg = build_leg(sort_signal=_unknown_signal_ref(), n_groups=unknown())
    part2 = build_part2(legs=(leg,))
    flags = cross_check(part1, part2)
    assert any(f.kind == MARKERS_ALL_UNKNOWN for f in flags)


def test_markers_present_is_no_alarm_even_if_sparse():
    # D18: markers STATED but the rest of the block sparse -> NO flag.
    part1 = build_part1(formation_structure=stated("sorted_portfolios"))
    leg = build_leg(
        sort_signal=signal_ref("mom6"),
        n_groups=stated(5),
        bucketing_method=unknown(),
        stripe_aggregation=unknown(),
        control_missing_policy=unknown(),
    )
    part2 = build_part2(legs=(leg,))
    assert cross_check(part1, part2) == []


def test_one_marker_unknown_is_not_all_unknown():
    # only fires when BOTH markers are UNKNOWN (D18 / D32b: two markers).
    part1 = build_part1(formation_structure=stated("sorted_portfolios"))
    leg = build_leg(sort_signal=signal_ref("mom6"), n_groups=unknown())
    part2 = build_part2(legs=(leg,))
    assert cross_check(part1, part2) == []


# --- non-declared-block rule (D14) ------------------------------------------

def test_non_declared_block_filled_flags_review():
    # declares a non-sort family, but the sort block is substantially filled.
    part1 = build_part1(formation_structure=stated("estimated_factor_model"))
    part2 = build_part2()  # default legs have both markers STATED
    flags = cross_check(part1, part2)
    assert any(f.kind == NON_DECLARED_BLOCK_FILLED for f in flags)


def test_declared_sort_with_filled_block_is_clean():
    part1 = build_part1(formation_structure=stated("sorted_portfolios"))
    part2 = build_part2()
    assert cross_check(part1, part2) == []


def test_unknown_formation_structure_fires_neither_rule():
    # no positive declaration -> neither rule fires (both need a STATED enum).
    part1 = build_part1(formation_structure=unknown())
    part2 = build_part2()
    assert cross_check(part1, part2) == []


# --- never mutates ----------------------------------------------------------

def test_cross_check_never_mutates_the_spec():
    part1 = build_part1(formation_structure=stated("estimated_factor_model"))
    part2 = build_part2()
    before = part2.to_dict()
    _ = cross_check(part1, part2)
    assert part2.to_dict() == before  # D14/D18: never auto-repairs


def test_review_flag_routes_to_review():
    f = ReviewFlag(MARKERS_ALL_UNKNOWN, "detail")
    assert f.routing == "review"
