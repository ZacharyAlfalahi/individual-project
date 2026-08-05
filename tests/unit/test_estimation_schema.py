"""
Unit tests for the fitted-factor-model schema (v1.2): EstimationBlock +
InstrumentRef + InstrumentSet, the optional StrategySpec siblings, and the
isolated ``validate_estimation_block`` pass.

Isolation is the headline property under test: a sort spec that leaves
``estimation``/``instruments`` == None must serialise BYTE-IDENTICALLY to a
pre-v1.2 spec, and the shared sort validator/scorer must never reach an
estimation field.
"""

import pytest

from agents.quant.config import Evidence, Inherited

from agents.librarian.errors import LibrarianSchemaError
from agents.librarian.schema import (
    ESTIMATION_FIELD_TYPES,
    ESTIMATION_FIELDS,
    InstrumentRef,
    InstrumentSet,
    DescribedSignal,
    LocatedQuote,
)
from agents.librarian.schema.estimation_fields import (
    ESTIMATION_ENUM_FIELDS,
    ESTIMATION_INT_FIELDS,
    ESTIMATION_PROSE_FIELDS,
    ESTIMATION_SET_FIELDS,
)
from agents.librarian.validators import validate_estimation_block, validate_librarian_spec

from _librarian_fixtures import (
    FakeInstrumentRegistry,
    build_estimation_block,
    build_instrument_set,
    build_kpp_spec,
    build_spec,
    instrument_ref,
    stated,
    unknown,
)


# --- field-name / partition version signature -------------------------------

def test_estimation_block_has_eleven_fields():
    assert len(ESTIMATION_FIELDS) == 11


def test_field_type_partition_covers_all_fields_disjointly():
    assert set(ESTIMATION_FIELD_TYPES) == set(ESTIMATION_FIELDS)
    partitions = [
        ESTIMATION_ENUM_FIELDS, ESTIMATION_INT_FIELDS,
        ESTIMATION_SET_FIELDS, ESTIMATION_PROSE_FIELDS,
    ]
    union = set().union(*partitions)
    assert union == set(ESTIMATION_FIELDS)
    total = sum(len(p) for p in partitions)
    assert total == 11  # disjoint => sizes sum to the total


# --- construction + serialisation -------------------------------------------

def test_estimation_block_constructs_and_round_trips():
    est = build_estimation_block()
    d = est.to_dict()
    assert set(d.keys()) == set(ESTIMATION_FIELDS)
    assert d["n_factors_preferred"]["value"] == 5


def test_instrument_set_constructs_and_round_trips():
    iset = build_instrument_set()
    d = iset.to_dict()
    assert [i["concept_id"]["value"] for i in d["instruments"]] == [
        "past_6m_cumulative_return", "credit_rating",
    ]


def test_estimation_block_rejects_raw_value():
    # a fact-bearing field passed a raw value (not an Inherited) is a schema bug.
    with pytest.raises(LibrarianSchemaError):
        build_estimation_block(n_factors_preferred=5)  # 5, not stated(5)


def test_instrument_ref_rejects_raw_concept_id():
    with pytest.raises(LibrarianSchemaError):
        InstrumentRef(
            concept_id="past_6m_cumulative_return",  # raw str, not Inherited
            source_class=stated("bond"), transform=unknown(), lag=unknown(),
            as_described=DescribedSignal(label="x", quotes=(LocatedQuote("x", 1, 0, 1),)),
        )


def test_instrument_set_rejects_empty():
    with pytest.raises(LibrarianSchemaError):
        InstrumentSet(instruments=())


# --- isolation: a sort spec is byte-identical to pre-v1.2 -------------------

def test_sort_spec_serialises_without_v12_keys():
    spec = build_spec()  # no estimation/instruments
    d = spec.to_dict()
    assert "estimation" not in d
    assert "instruments" not in d
    assert set(d.keys()) == {"header", "part1", "part2", "paper_facts"}


def test_kpp_spec_emits_v12_keys():
    spec = build_kpp_spec()
    d = spec.to_dict()
    assert "estimation" in d and "instruments" in d


# --- validation: clean KPP spec, and the negatives it catches ---------------

def test_kpp_spec_validates_clean():
    spec = build_kpp_spec()
    # base validator (registry=None): the stub Part2 carries no real signal.
    assert validate_librarian_spec(spec) == []
    # estimation validator: instrument membership + D8 negatives.
    assert validate_estimation_block(spec, FakeInstrumentRegistry()) == []


def test_estimation_validator_flags_unknown_instrument():
    spec = build_kpp_spec(
        instruments=build_instrument_set(
            instruments=(instrument_ref("not_a_real_instrument"),)
        )
    )
    errors = validate_estimation_block(spec, FakeInstrumentRegistry())
    assert len(errors) == 1
    assert "not in the Instrument Concept Registry" in errors[0].reason


def test_estimation_validator_flags_design_tag():
    design = Inherited("instrumented_pca", "DESIGN", Evidence(note="project choice"))
    spec = build_kpp_spec(estimation=build_estimation_block(model_family=design))
    errors = validate_estimation_block(spec, FakeInstrumentRegistry())
    assert any("DESIGN" in e.reason for e in errors)


def test_estimation_validator_flags_unrecognised_without_quotes():
    from agents.librarian.schema.signal_ref import UNRECOGNISED

    bad = InstrumentRef(
        concept_id=stated(UNRECOGNISED), source_class=stated("bond"),
        transform=unknown(), lag=unknown(),
        as_described=DescribedSignal(label="mystery"),  # no quotes
    )
    spec = build_kpp_spec(instruments=build_instrument_set(instruments=(bad,)))
    errors = validate_estimation_block(spec, FakeInstrumentRegistry())
    assert any("unrecognised" in e.reason for e in errors)


def test_estimation_validator_empty_when_no_fitted_blocks():
    spec = build_spec()  # a plain sort spec: no estimation/instruments
    assert validate_estimation_block(spec, FakeInstrumentRegistry()) == []
