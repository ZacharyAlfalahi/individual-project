"""
Unit tests for the signal-ref filler (D22, D9/D11), offline via FakeModelClient
against the Cluster-2 stub + the REAL Signal Concept Registry.

``fill_signal_ref`` reuses the D9 merge for the concept_id (and each registry
parameter), so these tests assert:
  * agree on a known registry concept + quotes locate -> STATED concept_id with a
    locator; the assembled SignalRef validates against the registry;
  * agree on 'unrecognised' + quotes locate -> escape invariant satisfied; valid;
  * disagree on the concept id -> UNKNOWN(disagreement), SignalRef None;
  * exactly one model answers -> UNKNOWN(single_response), SignalRef None
    (asserted NOT disagreement -- the honest-label guard);
  * a param-bearing fake registry: agree on concept + param value -> the param is
    a STATED Inherited; a silent param -> UNKNOWN(not_stated);
  * the assembled SignalRefs drive validate_librarian_spec with zero errors.

No real quote fixtures are authored: every quote is a verbatim substring of the
synthetic stub (so its L0 locate succeeds), never lifted from a real paper.
"""

import pytest

from agents.librarian.pipeline.model_client import FakeModelClient, ModelAnswer
from agents.librarian.pipeline.signal_filler import fill_signal_ref
from agents.librarian.registries.signal_concept_registry import (
    load_signal_concept_registry,
)
from agents.librarian.schema.signal_ref import UNRECOGNISED
from agents.librarian.validators import validate_librarian_spec

from _librarian_fixtures import (
    FakeSignalRegistry,
    build_leg,
    build_part2,
    build_spec,
)
from _librarian_pipeline_fixtures import (
    Q_HIGHEST,
    Q_QUINTILES,
    load_stub,
)

FIELD = "sort_signal"


@pytest.fixture
def stub():
    return load_stub()


@pytest.fixture
def registry():
    # The REAL v1 registry (concepts: size, maturity, credit_rating, ...).
    return load_signal_concept_registry()


def _ans(raw, quote, field=FIELD):
    return ModelAnswer(field=field, answered=True, raw=raw, quote=quote)


def _silent(field=FIELD):
    return ModelAnswer(field=field, answered=False)


def _pair(field, ans_a, ans_b):
    return (
        FakeModelClient("fake-a", {field: ans_a}),
        FakeModelClient("fake-b", {field: ans_b}),
    )


class _ParamRegistry:
    """A fake registry that DOES declare a parameter, to exercise the (usually
    empty) parameter loop -- the real v1 concepts have empty schemas."""

    def __init__(self, schemas=None):
        self._schemas = schemas or {"mom6": {"months": int}}

    def has_concept(self, concept_id: str) -> bool:
        return concept_id in self._schemas

    def parameter_schema(self, concept_id: str):
        return self._schemas[concept_id]


# --- known registry concept, agreement -------------------------------------

def test_known_concept_agreement_is_stated_and_validates(stub, registry):
    # Both models name the known concept 'size'; both justifying quotes locate.
    a = _ans("size", Q_QUINTILES)     # page 0
    b = _ans("size", Q_HIGHEST)       # page 0, later span
    fa, fb = _pair(FIELD, a, b)
    out = fill_signal_ref(FIELD, fa, fb, stub, registry)

    assert out.concept_id.tag == "STATED"
    assert out.concept_id.value == "size"
    assert out.concept_id.evidence.locator is not None  # D7 locator present
    assert out.signal_ref is not None
    # as_described carries the paper's words (audit-only for a known concept).
    assert len(out.signal_ref.as_described.quotes) >= 1
    assert out.trace.agreement is True

    # The assembled SignalRef validates against a registry that knows 'size'.
    reg = FakeSignalRegistry(schemas={"size": {}})
    leg = build_leg(sort_signal=out.signal_ref)
    spec = build_spec(part2=build_part2(legs=[leg]))
    assert validate_librarian_spec(spec, registry=reg) == []


# --- unrecognised escape ----------------------------------------------------

def test_unrecognised_agreement_satisfies_escape_and_validates(stub, registry):
    a = _ans(UNRECOGNISED, Q_QUINTILES)
    b = _ans(UNRECOGNISED, Q_HIGHEST)
    fa, fb = _pair(FIELD, a, b)
    out = fill_signal_ref(FIELD, fa, fb, stub, registry)

    assert out.concept_id.tag == "STATED"
    assert out.concept_id.value == UNRECOGNISED
    assert out.signal_ref is not None
    # Escape invariant: unrecognised requires non-empty as_described.quotes.
    assert len(out.signal_ref.as_described.quotes) >= 1
    assert out.signal_ref.is_unrecognised

    leg = build_leg(sort_signal=out.signal_ref)
    spec = build_spec(part2=build_part2(legs=[leg]))
    # unrecognised is deliberately out-of-registry; still validates clean.
    assert validate_librarian_spec(spec, registry=FakeSignalRegistry()) == []


# --- concept-id disagreement -> UNKNOWN(disagreement), no SignalRef ---------

def test_concept_id_disagreement_yields_no_signal_ref(stub, registry):
    a = _ans("size", Q_QUINTILES)
    b = _ans("maturity", Q_HIGHEST)   # both locate, but concepts differ
    fa, fb = _pair(FIELD, a, b)
    out = fill_signal_ref(FIELD, fa, fb, stub, registry)

    assert out.concept_id.tag == "UNKNOWN"
    assert out.concept_id.evidence.unknown_reason == "disagreement"
    assert out.signal_ref is None
    assert out.trace.ship_choice is None


# --- single response -> UNKNOWN(single_response), no SignalRef --------------

def test_lone_answer_is_single_response_not_disagreement(stub, registry):
    # Only model_a answers; model_b is content-silent. D9 needs both to agree, so
    # a lone answer is single_response -- NOT disagreement (both answered, differ).
    a = _ans("size", Q_QUINTILES)
    fa, fb = _pair(FIELD, a, _silent())
    out = fill_signal_ref(FIELD, fa, fb, stub, registry)

    assert out.concept_id.tag == "UNKNOWN"
    assert out.concept_id.evidence.unknown_reason == "single_response"
    assert out.concept_id.evidence.unknown_reason != "disagreement"  # honest-label
    assert out.signal_ref is None


# --- paper silent -> UNKNOWN(not_stated), no SignalRef ----------------------

def test_paper_silent_yields_no_signal_ref(stub, registry):
    fa, fb = _pair(FIELD, _silent(), _silent())
    out = fill_signal_ref(FIELD, fa, fb, stub, registry)

    assert out.concept_id.tag == "UNKNOWN"
    assert out.concept_id.evidence.unknown_reason == "not_stated"
    assert out.signal_ref is None


# --- quote-gate failure -> UNKNOWN(quote_match_failure), no SignalRef -------

def test_quote_gate_failure_yields_no_signal_ref(stub, registry):
    a = _ans("size", Q_QUINTILES)
    b = _ans("size", "a phrase that never appears anywhere in the stub text")
    fa, fb = _pair(FIELD, a, b)
    out = fill_signal_ref(FIELD, fa, fb, stub, registry)

    assert out.concept_id.tag == "UNKNOWN"
    assert out.concept_id.evidence.unknown_reason == "quote_match_failure"
    assert out.signal_ref is None


# --- param-bearing fake registry: filled param + silent param ---------------

def test_param_bearing_registry_fills_and_validates(stub):
    param_reg = _ParamRegistry({"mom6": {"months": int}})
    # Agree on the concept AND on the parameter value (6 vs "6 months" -> fold).
    fa = FakeModelClient(
        "fake-a",
        {FIELD: _ans("mom6", Q_QUINTILES), "months": _ans(6, Q_QUINTILES, field="months")},
    )
    fb = FakeModelClient(
        "fake-b",
        {FIELD: _ans("mom6", Q_HIGHEST), "months": _ans("6 months", Q_HIGHEST, field="months")},
    )
    out = fill_signal_ref(FIELD, fa, fb, stub, param_reg)

    assert out.concept_id.value == "mom6"
    assert out.signal_ref is not None
    months = out.signal_ref.parameters["months"]
    assert months.tag == "STATED"
    assert months.value == 6
    assert len(out.param_traces) == 1

    # Validates against a registry that knows mom6 with a 'months' int param.
    reg = FakeSignalRegistry(schemas={"mom6": {"months": int}})
    leg = build_leg(sort_signal=out.signal_ref)
    spec = build_spec(part2=build_part2(legs=[leg]))
    assert validate_librarian_spec(spec, registry=reg) == []


def test_param_bearing_registry_silent_param_is_unknown(stub):
    param_reg = _ParamRegistry({"mom6": {"months": int}})
    # Concept agrees; the parameter is paper-silent (no script for "months").
    fa = FakeModelClient("fake-a", {FIELD: _ans("mom6", Q_QUINTILES)})
    fb = FakeModelClient("fake-b", {FIELD: _ans("mom6", Q_HIGHEST)})
    out = fill_signal_ref(FIELD, fa, fb, stub, param_reg)

    assert out.signal_ref is not None
    months = out.signal_ref.parameters["months"]
    assert months.tag == "UNKNOWN"
    assert months.evidence.unknown_reason == "not_stated"

    # A UNKNOWN(None) param is legal -- validates against the same schema.
    reg = FakeSignalRegistry(schemas={"mom6": {"months": int}})
    leg = build_leg(sort_signal=out.signal_ref)
    spec = build_spec(part2=build_part2(legs=[leg]))
    assert validate_librarian_spec(spec, registry=reg) == []
