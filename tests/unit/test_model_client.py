"""
Unit tests for the model-client boundary (build brief §5.3, D9/D10/D33).

Offline only: ``FakeModelClient`` returns scripted answers and satisfies the
``ModelClient`` Protocol. No network, no SDK, no keys.
"""

import pytest

from agents.librarian.config.canonical_text import load_canonical_text
from agents.librarian.errors import LibrarianSchemaError
from agents.librarian.pipeline.model_client import (
    FieldQuery,
    FakeModelClient,
    ModelAnswer,
    ModelClient,
)

from _librarian_pipeline_fixtures import STUB_PATH


@pytest.fixture
def stub():
    return load_canonical_text(STUB_PATH)


# --- FieldQuery -------------------------------------------------------------

def test_field_query_well_formed():
    q = FieldQuery(field="weighting_scheme", kind="enum", template_hash="abc")
    assert q.field == "weighting_scheme"
    assert q.kind == "enum"
    assert q.k == 1  # k=1 primary (D10)


def test_field_query_rejects_bad_kind():
    with pytest.raises(LibrarianSchemaError):
        FieldQuery(field="x", kind="not_a_kind")


def test_field_query_rejects_bad_k():
    with pytest.raises(LibrarianSchemaError):
        FieldQuery(field="x", kind="enum", k=0)
    with pytest.raises(LibrarianSchemaError):
        FieldQuery(field="x", kind="enum", k=True)


# --- ModelAnswer ------------------------------------------------------------

def test_answered_answer_requires_a_quote():
    # answered=True with no quote / quotes -> build error (nothing to locate).
    with pytest.raises(LibrarianSchemaError):
        ModelAnswer(field="x", answered=True, raw="value")


def test_silent_answer_drops_value_and_quote():
    a = ModelAnswer(field="x", answered=False, raw="ignored", quote="ignored")
    assert a.raw is None
    assert a.quote is None
    assert a.quotes == ()


def test_answer_to_dict_round_trips_shape():
    a = ModelAnswer(field="x", answered=True, raw=5, quote="q", model_id="m")
    d = a.to_dict()
    assert d["field"] == "x" and d["answered"] is True and d["raw"] == 5
    assert d["quote"] == "q" and d["model_id"] == "m"


# --- FakeModelClient / Protocol --------------------------------------------

def test_fake_model_client_satisfies_protocol():
    client = FakeModelClient(model_id="fake-a")
    assert isinstance(client, ModelClient)


def test_fake_returns_scripted_answer(stub):
    client = FakeModelClient(
        model_id="fake-a",
        answers={"n_groups": ModelAnswer("n_groups", True, raw=5, quote="sorted into quintiles")},
    )
    q = FieldQuery(field="n_groups", kind="int")
    ans = client.answer(q, stub)
    assert ans.answered and ans.raw == 5
    # the client stamps its id when the script left it unset.
    assert ans.model_id == "fake-a"


def test_fake_unscripted_field_is_silent(stub):
    client = FakeModelClient(model_id="fake-a")
    ans = client.answer(FieldQuery(field="unmapped", kind="enum"), stub)
    assert ans.answered is False


def test_fake_supports_callable_script(stub):
    def script(query, canonical_text):
        # a script that reads the text (the rare case).
        assert canonical_text is stub
        return ModelAnswer(query.field, True, raw="value", quote="value-weighted")

    client = FakeModelClient(model_id="fake-b", answers={"weighting_scheme": script})
    ans = client.answer(FieldQuery(field="weighting_scheme", kind="enum"), stub)
    assert ans.answered and ans.raw == "value"


def test_fake_rejects_non_answer_script(stub):
    client = FakeModelClient(model_id="fake-a", answers={"x": "not an answer"})
    with pytest.raises(LibrarianSchemaError):
        client.answer(FieldQuery(field="x", kind="enum"), stub)
