"""Unit tests for the Librarian SignalRef value object (D22)."""

import pytest

from agents.quant.config import Evidence, Inherited, Locator

from agents.librarian.errors import LibrarianSchemaError
from agents.librarian.schema.signal_ref import (
    UNRECOGNISED,
    DescribedSignal,
    LocatedQuote,
    SignalRef,
)


def _stated(value):
    return Inherited(value, "STATED", Evidence(quote="q", locator=Locator(1, 0, 1)))


def _quote():
    return LocatedQuote(text="past six-month return", page=3, char_start=10, char_end=32)


# --- well-formed construction ----------------------------------------------

def test_well_formed_signal_ref_constructs():
    sig = SignalRef(
        concept_id=_stated("mom6"),
        as_described=DescribedSignal(label="6-month momentum", quotes=(_quote(),)),
        parameters={"months": _stated(6)},
    )
    assert sig.concept_id.value == "mom6"
    assert not sig.is_unrecognised
    assert sig.parameters["months"].value == 6
    # parameters frozen (MappingProxy) -- cannot mutate.
    with pytest.raises(TypeError):
        sig.parameters["months"] = _stated(7)


def test_signal_ref_to_dict_round_trips_shape():
    sig = SignalRef(
        concept_id=_stated("mom6"),
        as_described=DescribedSignal(label="mom", quotes=(_quote(),)),
        parameters={"months": _stated(6)},
    )
    d = sig.to_dict()
    assert d["concept_id"]["value"] == "mom6"
    assert d["parameters"]["months"]["value"] == 6
    assert d["as_described"]["label"] == "mom"
    assert d["as_described"]["quotes"][0]["text"] == "past six-month return"


# --- the escape rule (structural, registry-free) ---------------------------

def test_unrecognised_requires_nonempty_as_described():
    # unrecognised with no quotes -> build error.
    with pytest.raises(LibrarianSchemaError):
        SignalRef(
            concept_id=_stated(UNRECOGNISED),
            as_described=DescribedSignal(label="mystery signal", quotes=()),
        )
    # unrecognised WITH the paper's words -> ok.
    sig = SignalRef(
        concept_id=_stated(UNRECOGNISED),
        as_described=DescribedSignal(label="mystery signal", quotes=(_quote(),)),
    )
    assert sig.is_unrecognised


# --- shape guards ----------------------------------------------------------

def test_raw_concept_id_rejected():
    with pytest.raises(LibrarianSchemaError):
        SignalRef(
            concept_id="mom6",  # not an Inherited
            as_described=DescribedSignal(label="mom", quotes=(_quote(),)),
        )


def test_raw_parameter_value_rejected():
    with pytest.raises(LibrarianSchemaError):
        SignalRef(
            concept_id=_stated("mom6"),
            as_described=DescribedSignal(label="mom", quotes=(_quote(),)),
            parameters={"months": 6},  # not an Inherited
        )


def test_bad_as_described_type_rejected():
    with pytest.raises(LibrarianSchemaError):
        SignalRef(concept_id=_stated("mom6"), as_described="not a DescribedSignal")


def test_located_quote_validates_span_and_rejects_bool():
    LocatedQuote("t", 0, 0, 0)  # empty span, page 0 ok
    with pytest.raises(LibrarianSchemaError):
        LocatedQuote("", 1, 0, 1)  # empty text
    with pytest.raises(LibrarianSchemaError):
        LocatedQuote("t", -1, 0, 1)  # page < 0
    with pytest.raises(LibrarianSchemaError):
        LocatedQuote("t", 1, 5, 2)  # end < start
    with pytest.raises(LibrarianSchemaError):
        LocatedQuote("t", True, 0, 1)  # bool page rejected


def test_described_signal_coerces_list_quotes_to_tuple():
    ds = DescribedSignal(label="mom", quotes=[_quote()])
    assert isinstance(ds.quotes, tuple)
