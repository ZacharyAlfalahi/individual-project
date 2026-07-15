"""
Gold loader (G2 Piece B): parses each anchor gold markdown into a full, validating
``StrategySpec`` with real D7 locators on every STATED field.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (evaluation/*)

import pytest  # noqa: E402

from evaluation.gold_specs.gold_loader import (  # noqa: E402
    GoldParseError,
    is_binding,
    load_gold_spec,
)
from agents.librarian.schema.strategy_spec import (  # noqa: E402
    _COMMON_INHERITED_FIELDS,
    _LEG_INHERITED_FIELDS,
    _PAPER_FACTS_INHERITED_FIELDS,
)
from agents.librarian.validators import validate_librarian_spec  # noqa: E402

ANCHORS = ("str", "drf", "mom6")


@pytest.fixture(scope="module")
def specs():
    return {a: load_gold_spec(a) for a in ANCHORS}


def test_all_three_load(specs):
    assert set(specs) == set(ANCHORS)


def test_full_field_coverage(specs):
    # Every schema field is present (the loader hard-errors on a missing one).
    for a, s in specs.items():
        leg = s.part2.legs[0]
        for f in _LEG_INHERITED_FIELDS:
            assert getattr(leg, f) is not None, (a, f)
        for f in _COMMON_INHERITED_FIELDS:
            assert getattr(s.part2, f) is not None, (a, f)
        assert s.paper_facts is not None, a
        for f in _PAPER_FACTS_INHERITED_FIELDS:
            assert getattr(s.paper_facts, f) is not None, (a, f)
        assert s.part1.formation_structure is not None
        assert s.part1.asset_class is not None
        assert s.part1.method_summary is not None


def test_every_stated_field_has_a_real_locator(specs):
    # D7: every STATED Inherited carries a page+char-span locator (the loader would
    # have raised otherwise; assert it explicitly across the whole spec).
    for a, s in specs.items():
        leg = s.part2.legs[0]
        checkables = (
            [getattr(leg, f) for f in _LEG_INHERITED_FIELDS]
            + [getattr(s.part2, f) for f in _COMMON_INHERITED_FIELDS]
            + [getattr(s.paper_facts, f) for f in _PAPER_FACTS_INHERITED_FIELDS]
            + [s.part1.formation_structure, s.part1.asset_class, s.part2.combiner.kind,
               leg.sort_signal.concept_id, s.header.strategy_label]
        )
        if leg.control_axis is not None:
            checkables.append(leg.control_axis.concept_id)
        for inh in checkables:
            if inh.tag == "STATED":
                loc = inh.evidence.locator
                assert loc is not None, (a, inh.value)
                assert isinstance(loc.char_start, int) and isinstance(loc.char_end, int)


def test_specs_validate_fail_closed(specs):
    for a, s in specs.items():
        assert validate_librarian_spec(s) == [], a


def test_str_spot_values(specs):
    s = specs["str"]
    leg = s.part2.legs[0]
    assert leg.n_groups.value == 10
    assert leg.sort_kind.value == "single"
    assert leg.long_leg.value == "highest_signal"
    assert leg.sort_signal.concept_id.value == "prior_1m_excess_return"
    assert leg.control_axis is None
    assert s.part2.weighting_base.value == "market_value"
    assert s.part2.weighting_base.tag == "STATED"
    assert s.part2.combiner.kind.value == "single_leg"


def test_drf_double_sort(specs):
    s = specs["drf"]
    leg = s.part2.legs[0]
    assert leg.sort_kind.value == "independent"
    assert leg.n_groups.value == 5
    assert leg.control_axis is not None
    assert leg.control_axis.concept_id.value == "credit_rating"
    assert leg.control_n_groups.value == 5
    assert s.part2.weighting_base.value == "par"


def test_mom6_spot_values(specs):
    s = specs["mom6"]
    leg = s.part2.legs[0]
    assert leg.n_groups.value == 10
    assert leg.sort_signal.concept_id.value == "past_6m_cumulative_return"
    assert s.part2.overlap_convention.value == "overlapping"
    assert s.part2.cohort_weighting.value == "equal"


def test_claimed_headline_metric_parses_as_dict(specs):
    m = specs["str"].paper_facts.claimed_headline_metric.value
    assert isinstance(m, dict)
    assert set(m) == {"mean", "t_stat", "unit"}
    assert m["mean"] == -0.99


def test_is_binding_flags():
    assert is_binding("str") is True
    assert is_binding("drf") is True
    assert is_binding("mom6") is False  # pending JNPS canonical-text freeze


def test_unknown_anchor_raises():
    with pytest.raises(GoldParseError):
        load_gold_spec("nope")
