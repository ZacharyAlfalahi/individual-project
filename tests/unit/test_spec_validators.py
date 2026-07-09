"""
Unit tests for validate_librarian_spec -- the four mandatory D8 negatives
(DESIGN present; STATED without locator; unknown concept_id / unrecognised;
AMBIGUOUS in output).
"""

import pytest

from agents.quant.config import Evidence, Inherited
from agents.quant.config.provenance import Binding

from agents.librarian.errors import LibrarianSchemaError
from agents.librarian.schema import DescribedSignal, SignalRef
from agents.librarian.validators import validate_librarian_spec
from agents.librarian.validators.spec_validators import _check_no_ambiguous

from _librarian_fixtures import (
    FakeSignalRegistry,
    build_leg,
    build_part2,
    build_spec,
    located_quote,
    signal_ref,
    stated,
)


# --- clean baseline ---------------------------------------------------------

def test_clean_spec_has_no_errors():
    errs = validate_librarian_spec(build_spec(), registry=FakeSignalRegistry())
    assert errs == []


# --- (1) D8: no DESIGN anywhere --------------------------------------------

def test_design_tag_is_caught():
    design = Inherited("value", "DESIGN", Evidence(note="a project decision"))
    p2 = build_part2(weighting_base=design)
    errs = validate_librarian_spec(build_spec(part2=p2), registry=FakeSignalRegistry())
    assert any("DESIGN" in e.reason for e in errs)
    assert any(e.field == "part2.weighting_base" for e in errs)


# --- (2) D7: STATED without a locator --------------------------------------

def test_stated_without_locator_is_caught():
    # Construction blocks this (D7), so build a defective Inherited that skips
    # __post_init__ via object.__new__ to simulate a value slipping past.
    bad = object.__new__(Inherited)
    object.__setattr__(bad, "value", "x")
    object.__setattr__(bad, "tag", "STATED")
    object.__setattr__(bad, "evidence", Evidence(quote="q"))  # no locator
    p2 = build_part2(lag_convention=bad)
    errs = validate_librarian_spec(build_spec(part2=p2), registry=FakeSignalRegistry())
    assert any("locator" in e.reason for e in errs)
    assert any(e.field == "part2.lag_convention" for e in errs)


def test_construction_already_blocks_stated_without_locator():
    # Defence-in-depth: the type itself rejects it too.
    with pytest.raises(Exception):
        Inherited("x", "STATED", Evidence(quote="q"))  # no locator


# --- (3) D22: registry-aware SignalRef -------------------------------------

def test_unknown_concept_id_rejected():
    leg = build_leg(sort_signal=signal_ref("not_a_real_concept"))
    p2 = build_part2(legs=[leg])
    errs = validate_librarian_spec(build_spec(part2=p2), registry=FakeSignalRegistry())
    assert any("not in the Signal Concept Registry" in e.reason for e in errs)


def test_unrecognised_without_as_described_rejected_at_construction():
    # The pure SignalRef guards this structurally (build error, not a validator
    # finding) -- the correct answer for out-of-registry is 'unrecognised' WITH
    # the paper's words.
    from agents.librarian.schema import UNRECOGNISED
    with pytest.raises(LibrarianSchemaError):
        SignalRef(
            concept_id=stated(UNRECOGNISED),
            as_described=DescribedSignal(label="mystery", quotes=()),
        )


def test_unrecognised_with_as_described_passes_registry_check():
    from agents.librarian.schema import UNRECOGNISED
    leg = build_leg(
        sort_signal=signal_ref(UNRECOGNISED, quotes=(located_quote(),), label="mystery")
    )
    p2 = build_part2(legs=[leg])
    errs = validate_librarian_spec(build_spec(part2=p2), registry=FakeSignalRegistry())
    # unrecognised is deliberately not looked up in the registry -> no error.
    assert errs == []


def test_parameter_not_in_schema_rejected():
    leg = build_leg(sort_signal=signal_ref("mom6", params={"bogus": stated(3)}))
    p2 = build_part2(legs=[leg])
    errs = validate_librarian_spec(build_spec(part2=p2), registry=FakeSignalRegistry())
    assert any("not in the schema" in e.reason for e in errs)


def test_parameter_wrong_type_rejected():
    # mom6.months expects int; give it a str.
    leg = build_leg(sort_signal=signal_ref("mom6", params={"months": stated("six")}))
    p2 = build_part2(legs=[leg])
    errs = validate_librarian_spec(build_spec(part2=p2), registry=FakeSignalRegistry())
    assert any("must be int" in e.reason for e in errs)


# --- (4) adapter-never-AMBIGUOUS -------------------------------------------

def test_binding_cannot_occupy_a_fact_bearing_slot():
    # AMBIGUOUS lives only on Binding; the schema rejects a Binding in an
    # Inherited slot -> the invariant holds by construction (structural).
    amb = Binding("a", "AMBIGUOUS", Evidence(candidates=("a", "b"), chosen="a", note="x"))
    with pytest.raises(LibrarianSchemaError):
        build_part2(weighting_scheme=amb)


def test_no_ambiguous_check_clean_on_valid_spec():
    # The assertion pass returns no errors on a structurally valid spec.
    assert _check_no_ambiguous(build_spec()) == []


# --- caller-contract guard -------------------------------------------------

def test_validate_rejects_non_spec():
    with pytest.raises(LibrarianSchemaError):
        validate_librarian_spec("not a spec")


def test_registry_optional_skips_membership_checks():
    # With no registry, an unknown concept_id is NOT flagged (membership skipped)
    # but the unrecognised-escape structural rule still applies via SignalRef.
    leg = build_leg(sort_signal=signal_ref("not_a_real_concept"))
    p2 = build_part2(legs=[leg])
    errs = validate_librarian_spec(build_spec(part2=p2), registry=None)
    assert errs == []
