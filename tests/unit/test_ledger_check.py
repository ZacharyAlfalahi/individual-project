"""
Unit tests for check_assumptions (D29): the deterministic assumptions-mismatch
screen. A field STATED with a value the frozen engine cannot honour refuses
(ASSUMPTION_MISMATCH); silent / INFERRED / UNKNOWN fields never mismatch. The
check runs to completion (collects ALL mismatches).
"""

import pytest

from _librarian_fixtures import build_leg, build_part2, build_spec, stated, unknown

from agents.quant.config import (
    Evidence,
    Inherited,
    RefusalCode,
    check_assumptions,
    load_ledger_check_table,
)


@pytest.fixture(scope="module")
def table():
    return load_ledger_check_table()


def _inferred(value, rule_id="TEST-RULE-1"):
    return Inherited(value, "INFERRED", Evidence(rule_id=rule_id))


# --- no false positives on a clean spec -------------------------------------

def test_clean_synthetic_spec_no_mismatch(table):
    """The default synthetic spec states every field at the engine's own
    assumption, so nothing mismatches. If the check fires here it over-fires."""
    assert check_assumptions(build_spec(), table) == ()


@pytest.mark.skip(
    reason="decisive anchor test -- blocked on the 3 anchor gold specs (str, drf, "
    "mom6) per brief section 3. Unskip and load the golds when they land: each "
    "anchor's methodology matches every engine assumption, so check_assumptions "
    "must return zero mismatches (over-firing on a known-good anchor = broken)."
)
def test_anchor_specs_no_false_positives(table):  # pragma: no cover
    for anchor_id in ("str", "drf", "mom6"):
        spec = _load_anchor_gold(anchor_id)  # noqa: F821 -- lands with the golds
        assert check_assumptions(spec, table) == (), anchor_id


# --- a STATED incompatible value fires --------------------------------------

def test_stated_incompatible_sort_field_fires(table):
    spec = build_spec(part2=build_part2(legs=(build_leg(sort_kind=stated("conditional")),)))
    refusals = check_assumptions(spec, table)

    assert len(refusals) == 1
    r = refusals[0]
    assert r.code == RefusalCode.ASSUMPTION_MISMATCH
    assert r.field == "legs[0].sort_kind"          # per-leg field is disambiguated
    assert r.strategy_id == "Synthetic Momentum"   # spec.header.strategy_label.value
    assert r.evidence is not None and r.evidence.quote  # carries the field's own quote
    assert "conditional" in r.detail or "independent" in r.detail


def test_stated_incompatible_common_field_fires(table):
    spec = build_spec(part2=build_part2(return_label=stated("formation")))
    refusals = check_assumptions(spec, table)

    assert len(refusals) == 1
    assert refusals[0].field == "return_label"
    assert refusals[0].code == RefusalCode.ASSUMPTION_MISMATCH


def test_final_common_rows_fire(table):
    # The three rows added when the table was frozen (items 5, 6, 21). Item 21 (non_overlapping
    # at H>1) is the most material -- a genuinely different construction.
    spec = build_spec(part2=build_part2(
        min_bonds_granularity=stated("per_group"),
        lag_convention=stated("business_day"),
        overlap_convention=stated("non_overlapping"),
    ))
    fields = sorted(r.field for r in check_assumptions(spec, table))
    assert fields == ["lag_convention", "min_bonds_granularity", "overlap_convention"]


# --- silent / INFERRED / UNKNOWN never mismatch -----------------------------

def test_unknown_field_no_mismatch(table):
    # sort_kind UNKNOWN: the engine's assumption stands, tagged by the silence policy.
    spec = build_spec(part2=build_part2(legs=(build_leg(sort_kind=unknown()),)))
    assert check_assumptions(spec, table) == ()


def test_inferred_field_no_mismatch(table):
    # A value that WOULD be incompatible if STATED, but is INFERRED -> no mismatch.
    spec = build_spec(part2=build_part2(legs=(build_leg(sort_kind=_inferred("conditional")),)))
    assert check_assumptions(spec, table) == ()


# --- run-to-completion: all mismatches collected ----------------------------

def test_multiple_incompatible_all_collected(table):
    legs = (
        build_leg(sort_kind=stated("conditional")),      # legs[0].sort_kind
        build_leg(bucketing_method=stated("breakpoint")),  # legs[1].bucketing_method
    )
    spec = build_spec(part2=build_part2(legs=legs, return_label=stated("formation")))
    refusals = check_assumptions(spec, table)

    fields = sorted(r.field for r in refusals)
    assert fields == ["legs[0].sort_kind", "legs[1].bucketing_method", "return_label"]
    assert all(r.code == RefusalCode.ASSUMPTION_MISMATCH for r in refusals)


def test_same_incompatible_fires_once_per_leg(table):
    # Two legs both STATE breakpoint bucketing -> two distinct refusals, each with
    # its own leg-disambiguated field (per-leg run-to-completion).
    legs = (
        build_leg(bucketing_method=stated("breakpoint")),
        build_leg(bucketing_method=stated("breakpoint")),
    )
    spec = build_spec(part2=build_part2(legs=legs))
    refusals = check_assumptions(spec, table)

    assert sorted(r.field for r in refusals) == [
        "legs[0].bucketing_method",
        "legs[1].bucketing_method",
    ]
