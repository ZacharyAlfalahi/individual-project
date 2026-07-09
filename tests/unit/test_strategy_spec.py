"""Unit tests for the Librarian StrategySpec dataclasses (D6/D13/D17/D19)."""

import pytest

from agents.librarian.errors import LibrarianSchemaError
from agents.librarian.schema import Combiner, StrategySpec

from _librarian_fixtures import (
    build_leg,
    build_part1,
    build_part2,
    build_spec,
    signal_ref,
    stated,
)


# --- a valid spec constructs -----------------------------------------------

def test_valid_strategy_spec_constructs():
    spec = build_spec()
    assert isinstance(spec, StrategySpec)
    assert spec.header.paper_id == "SYNTH-0001"
    assert len(spec.part2.legs) == 1
    assert spec.part2.combiner.kind.value == "single_leg"


# --- raw (unwrapped) fact-bearing fields are rejected ----------------------

def test_raw_common_field_rejected():
    # signal_lag passed as a bare int (not Inherited) -> schema error.
    with pytest.raises(LibrarianSchemaError):
        build_part2(signal_lag=0)


def test_raw_leg_field_rejected():
    with pytest.raises(LibrarianSchemaError):
        build_leg(n_groups=5)  # bare int, not Inherited


def test_raw_combiner_kind_rejected():
    with pytest.raises(LibrarianSchemaError):
        Combiner(kind="single_leg")  # bare str, not Inherited


def test_leg_sort_signal_must_be_signal_ref():
    with pytest.raises(LibrarianSchemaError):
        build_leg(sort_signal=stated("mom6"))  # Inherited, not a SignalRef


# --- leg / combiner shape --------------------------------------------------

def test_part2_requires_nonempty_legs():
    with pytest.raises(LibrarianSchemaError):
        build_part2(legs=())


def test_part2_legs_list_coerced_to_tuple():
    p2 = build_part2(legs=[build_leg()])
    assert isinstance(p2.legs, tuple)


def test_leg_control_axis_optional_and_typed():
    leg = build_leg(control_axis=signal_ref("var5pct"))
    assert leg.control_axis is not None
    with pytest.raises(LibrarianSchemaError):
        build_leg(control_axis=stated("var5pct"))  # Inherited, not SignalRef


def test_multi_leg_spec_constructs():
    p2 = build_part2(
        legs=[build_leg(), build_leg(sort_signal=signal_ref("var5pct"))],
        combiner=Combiner(kind=stated("equal_average")),
    )
    assert len(p2.legs) == 2
    assert p2.combiner.kind.value == "equal_average"


# --- to_dict round-trips ---------------------------------------------------

def test_to_dict_round_trips():
    spec = build_spec()
    d = spec.to_dict()
    assert set(d.keys()) == {"header", "part1", "part2"}
    # header carries the provenance-wrapped strategy_label
    assert d["header"]["strategy_label"]["value"] == "Synthetic Momentum"
    assert d["header"]["strategy_label"]["tag"] == "STATED"
    # part1
    assert d["part1"]["formation_structure"]["value"] == "sorted_portfolios"
    assert d["part1"]["method_summary"]["summary"]["value"].startswith("ranks bonds")
    # part2 common fields (28) + legs + combiner
    p2 = d["part2"]
    assert p2["signal_lag"]["value"] == 0
    assert len(p2["legs"]) == 1
    leg = p2["legs"][0]
    assert leg["sort_signal"]["concept_id"]["value"] == "mom6"
    assert leg["n_groups"]["value"] == 5
    assert p2["combiner"]["kind"]["value"] == "single_leg"


def test_bad_top_level_parts_rejected():
    with pytest.raises(LibrarianSchemaError):
        StrategySpec(header="nope", part1=build_part1(), part2=build_part2())
    with pytest.raises(LibrarianSchemaError):
        StrategySpec(header=None, part1=build_part1(), part2=build_part2())
