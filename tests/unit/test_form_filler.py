"""
Unit tests for the form-filler (build brief §5.3, D9/D11/D16).

The D9 merge, offline against the Cluster-2 stub:
  * agree + both quotes locate -> STATED, earliest-span quote shipped, both in trace;
  * disagree on normalised value -> UNKNOWN(disagreement);
  * a quote that does not locate -> UNKNOWN(quote_match_failure);
  * neither model answers -> UNKNOWN(not_stated);
  * method_summary 3-slot with 1-3 quotes.
"""

import pytest

from agents.librarian.errors import LibrarianSchemaError
from agents.librarian.pipeline.form_filler import (
    DISAGREEMENT,
    NOT_STATED,
    QUOTE_MATCH_FAILURE,
    SINGLE_RESPONSE,
    fill_field,
    fill_method_summary,
    normalise,
)
from agents.librarian.pipeline.model_client import FakeModelClient

from _librarian_pipeline_fixtures import (
    Q_HIGHEST,
    Q_MONTHLY,
    Q_NOT_IN_STUB,
    Q_QUINTILES,
    Q_RANKED,
    Q_VALUE_WEIGHTED,
    answer,
    load_stub,
    silent,
)


@pytest.fixture
def stub():
    return load_stub()


def _pair(field, ans_a, ans_b):
    return (
        FakeModelClient("fake-a", {field: ans_a}),
        FakeModelClient("fake-b", {field: ans_b}),
    )


# --- normalise --------------------------------------------------------------

def test_normalise_ints_fold_surface_forms():
    assert normalise("n_groups", 5) == 5
    assert normalise("n_groups", "5 groups") == 5
    assert normalise("n_groups", 5.0) == 5


def test_normalise_int_rejects_non_integer():
    with pytest.raises(LibrarianSchemaError):
        normalise("n_groups", "quarterly")
    with pytest.raises(LibrarianSchemaError):
        normalise("n_groups", 5.5)


def test_normalise_enum_tokens_fold():
    assert normalise("weighting_scheme", "Value Weighted") == "value_weighted"
    assert normalise("weighting_scheme", "value") == "value"
    assert normalise("weighting_scheme", None) is None


# --- (4) agree + locate -> STATED, earliest span shipped --------------------

def test_agree_and_locate_ships_earliest_span(stub):
    field = "n_groups"
    # model_a's quote is on page 0 (earlier span); model_b's is also page 0 but
    # its char_start is later -> model_a ships. Both normalise to 5.
    a = answer(field, 5, Q_QUINTILES)          # "sorted into quintiles" @ page0
    b = answer(field, "5 groups", Q_HIGHEST)   # "highest-signal bonds" @ page0, later
    fa, fb = _pair(field, a, b)
    out = fill_field(field, fa, fb, stub)
    assert out.value.tag == "STATED" and out.value.value == 5
    # STATED carries a locator (satisfies D7 frozen guard).
    assert out.value.evidence.locator is not None
    # earliest span shipped; both raw answers land in the trace.
    assert out.trace.ship_choice == "model_a"
    assert out.trace.model_a.raw == 5 and out.trace.model_b.raw == "5 groups"
    assert out.trace.agreement is True


def test_earliest_span_can_ship_model_b(stub):
    field = "weighting_scheme"
    # model_a cites a page-1 quote; model_b cites a page-0 quote (earlier span).
    a = answer(field, "value", Q_MONTHLY)          # page 1
    b = answer(field, "value", Q_QUINTILES)        # page 0 -> earlier -> ships
    fa, fb = _pair(field, a, b)
    out = fill_field(field, fa, fb, stub)
    assert out.value.tag == "STATED"
    assert out.trace.ship_choice == "model_b"
    assert out.value.evidence.locator.page == 0


# --- (3) disagree on normalised value -> UNKNOWN(disagreement) --------------

def test_disagreement_on_value(stub):
    field = "n_groups"
    a = answer(field, 5, Q_QUINTILES)
    b = answer(field, 10, Q_HIGHEST)
    fa, fb = _pair(field, a, b)
    out = fill_field(field, fa, fb, stub)
    assert out.value.tag == "UNKNOWN"
    assert out.value.evidence.unknown_reason == DISAGREEMENT
    assert out.trace.ship_choice is None


def test_lone_answer_is_single_response(stub):
    # only one model answers, the other content-silent -> no two-model agreement.
    # This is single_response (a value-vs-silence event), NOT disagreement (both
    # answered, values conflict) -- D11 amendment 2026-07-09. Routes to review.
    field = "n_groups"
    a = answer(field, 5, Q_QUINTILES)
    fa, fb = _pair(field, a, silent(field))
    out = fill_field(field, fa, fb, stub)
    assert out.value.tag == "UNKNOWN"
    assert out.value.evidence.unknown_reason == SINGLE_RESPONSE
    assert out.value.evidence.unknown_reason != DISAGREEMENT  # the honest-label guard


# --- (2) quote that does not locate -> UNKNOWN(quote_match_failure) ---------

def test_quote_that_does_not_locate(stub):
    field = "n_groups"
    a = answer(field, 5, Q_QUINTILES)
    b = answer(field, 5, Q_NOT_IN_STUB)  # agrees on value, but quote is absent
    fa, fb = _pair(field, a, b)
    out = fill_field(field, fa, fb, stub)
    assert out.value.tag == "UNKNOWN"
    assert out.value.evidence.unknown_reason == QUOTE_MATCH_FAILURE


# --- (1) neither model answers -> UNKNOWN(not_stated) -----------------------

def test_paper_silent(stub):
    field = "n_groups"
    fa, fb = _pair(field, silent(field), silent(field))
    out = fill_field(field, fa, fb, stub)
    assert out.value.tag == "UNKNOWN"
    assert out.value.evidence.unknown_reason == NOT_STATED
    assert out.trace.model_a.answered is False and out.trace.model_b.answered is False


# --- k-ladder seam ----------------------------------------------------------

def test_k_ladder_is_named_but_unbuilt(stub):
    field = "n_groups"
    fa, fb = _pair(field, silent(field), silent(field))
    with pytest.raises(LibrarianSchemaError):
        fill_field(field, fa, fb, stub, k=3)


# --- method_summary (D16) ---------------------------------------------------

def test_method_summary_three_slot_with_quotes(stub):
    a = answer(
        "method_summary",
        "Ranks bonds by past return. Sorts into quintile legs. Reports a factor.",
        quote=None,
        quotes=(Q_RANKED, Q_QUINTILES, Q_VALUE_WEIGHTED),
    )
    b = answer(
        "method_summary",
        "Ranks by return; forms long-short legs; outputs a series.",
        quote=None,
        quotes=(Q_MONTHLY,),
    )
    fa = FakeModelClient("fake-a", {"method_summary": a})
    fb = FakeModelClient("fake-b", {"method_summary": b})
    summary, trace = fill_method_summary(fa, fb, stub)
    assert summary is not None
    # 1-3 located quotes (D16 cap); model_a's first quote (page 0) is earliest.
    assert 1 <= len(summary.quotes) <= 3
    assert summary.summary.tag == "STATED"
    assert summary.summary.evidence.locator is not None
    assert trace.ship_choice == "model_a"


def test_method_summary_none_when_silent(stub):
    fa = FakeModelClient("fake-a")   # no script -> silent
    fb = FakeModelClient("fake-b")
    summary, trace = fill_method_summary(fa, fb, stub)
    assert summary is None
    assert trace.final_tag == "UNKNOWN"
    assert trace.final_reason == NOT_STATED
