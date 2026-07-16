"""
Unit tests for the live-extraction client (agents/librarian/pipeline/real_client.py).

Offline only: the vendor backend is stubbed (registered under a "stub" vendor), so
the whole response -> ModelAnswer mapping, the per-(field, kind, text) cache, the
format-failure counter, and the pure helpers are exercised with NO network/SDK.
Locks the mapping table the live smoke depends on into the regression suite.
"""

from __future__ import annotations

import json

import pytest

from agents.librarian.config import CanonicalText
from agents.librarian.pipeline import real_client as rc
from agents.librarian.pipeline.model_client import FieldQuery
from agents.librarian.registries import load_signal_concept_registry


# ---------------------------------------------------------------------------
# Pure helpers.
# ---------------------------------------------------------------------------

def test_parse_json_plain_fenced_and_sliced():
    assert rc._parse_json('{"a": 1}') == {"a": 1}
    assert rc._parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert rc._parse_json('here you go: {"a": 1} thanks') == {"a": 1}
    assert rc._parse_json("not json at all") is None
    assert rc._parse_json("[1, 2, 3]") is None  # top-level must be an object


def test_coerce_int():
    assert rc._coerce_int(5) == 5
    assert rc._coerce_int("5") == 5
    assert rc._coerce_int(5.0) == 5
    assert rc._coerce_int(5.5) is None
    assert rc._coerce_int(True) is None  # bool is not an int value here
    assert rc._coerce_int("five") is None


@pytest.mark.parametrize(
    "kind,parsed,exp_answered,exp_raw",
    [
        ("enum", {"answered": True, "value": "value", "quote": "q"}, True, "value"),
        ("enum", {"answered": True, "value": "", "quote": "q"}, False, None),      # empty value -> silent
        ("enum", {"answered": True, "value": "value"}, False, None),               # no quote -> silent
        ("enum", {"answered": False}, False, None),
        ("part1_enum", {"answered": True, "value": "corporate_bonds", "quote": "q"}, True, "corporate_bonds"),
        ("int", {"answered": True, "value": 5, "quote": "quintiles"}, True, 5),
        ("int", {"answered": True, "value": "5", "quote": "q"}, True, 5),
        ("int", {"answered": True, "value": "nope", "quote": "q"}, False, None),    # unparseable int -> silent
        ("signal_ref", {"answered": True, "concept_id": "var_5pct", "quote": "q"}, True, "var_5pct"),
        ("signal_ref", {"answered": True, "concept_id": None, "quote": "q"}, False, None),
    ],
)
def test_answer_from_parsed_mapping(kind, parsed, exp_answered, exp_raw):
    ans = rc._answer_from_parsed("f", kind, parsed, "model-x")
    assert ans.answered is exp_answered
    assert ans.model_id == "model-x"
    if exp_answered:
        assert ans.raw == exp_raw


def test_answer_from_parsed_method_summary():
    ok = rc._answer_from_parsed(
        "method_summary", "method_summary",
        {"answered": True, "summary": "three slots", "quotes": ["x", "y"]}, "m",
    )
    assert ok.answered and ok.raw == "three slots" and ok.quotes == ("x", "y")
    # answered but no quotes -> silent (ModelAnswer would otherwise reject it)
    empty = rc._answer_from_parsed(
        "method_summary", "method_summary",
        {"answered": True, "summary": "s", "quotes": []}, "m",
    )
    assert empty.answered is False


def test_is_non_retryable():
    class _Status(Exception):
        def __init__(self, sc):
            self.status_code = sc

    assert rc._is_non_retryable(_Status(401)) is True
    assert rc._is_non_retryable(_Status(403)) is True
    assert rc._is_non_retryable(_Status(400)) is True
    assert rc._is_non_retryable(_Status(429)) is False   # rate limit IS retryable
    assert rc._is_non_retryable(_Status(500)) is False   # server error IS retryable

    class AuthenticationError(Exception):
        pass

    assert rc._is_non_retryable(AuthenticationError()) is True
    assert rc._is_non_retryable(ValueError("boom")) is False


# ---------------------------------------------------------------------------
# Client with a stubbed backend.
# ---------------------------------------------------------------------------

class _StubBackend:
    """Returns a fixed (text, version) and counts calls (for the cache test)."""

    def __init__(self, text: str, version: str = "stub-v9"):
        self._text = text
        self._version = version
        self.calls = 0

    def generate(self, prompt: str, max_output_tokens: int):
        self.calls += 1
        return self._text, self._version


@pytest.fixture
def builder():
    return rc.PromptBuilder.load(load_signal_concept_registry())


def _stub_ct() -> CanonicalText:
    return CanonicalText(
        source_pdf="stub.pdf",
        source_sha256="deadbeef",
        parser={"name": "stub", "version": "1"},
        normalisation={"ladder_level": "L0", "rules": []},
        pages=("the 5% VaR appears on this page",),
        status="stub",
    )


def _client_with(monkeypatch, builder, backend) -> rc.RealModelClient:
    monkeypatch.setitem(rc._BACKENDS, "stub", lambda model_id, api_key, temperature: backend)
    return rc.RealModelClient(model_id="stub-1", vendor="stub", api_key="k", builder=builder)


def test_client_maps_valid_reply_and_stamps_version(monkeypatch, builder):
    text = json.dumps({"field": "weighting_scheme", "answered": True, "value": "value", "quote": "q"})
    backend = _StubBackend(text, version="stub-v42")
    client = _client_with(monkeypatch, builder, backend)

    ans = client.answer(FieldQuery("weighting_scheme", "enum"), _stub_ct())
    assert ans.answered is True
    assert ans.raw == "value"
    assert ans.model_id == "stub-v42"       # returned version, not the configured id


def test_client_caches_per_field(monkeypatch, builder):
    text = json.dumps({"field": "weighting_scheme", "answered": True, "value": "value", "quote": "q"})
    backend = _StubBackend(text)
    client = _client_with(monkeypatch, builder, backend)
    q = FieldQuery("weighting_scheme", "enum")
    ct = _stub_ct()

    a1 = client.answer(q, ct)
    a2 = client.answer(q, ct)              # signal_filler re-asks; cache must serve it
    assert a1 is a2
    assert backend.calls == 1              # exactly one live call


def test_client_format_failure_is_soft(monkeypatch, builder):
    backend = _StubBackend("this is not json", version="stub-v1")
    client = _client_with(monkeypatch, builder, backend)

    ans = client.answer(FieldQuery("weighting_scheme", "enum"), _stub_ct())
    assert ans.answered is False           # unparseable -> silent, run continues
    assert client.format_failures == 1


def test_client_fails_loud_on_non_retryable(monkeypatch, builder):
    class _AuthErr(Exception):
        status_code = 401

    class _DeadBackend:
        def generate(self, prompt, max_output_tokens):
            raise _AuthErr("bad key")

    client = _client_with(monkeypatch, builder, _DeadBackend())
    with pytest.raises(rc.RealClientError):
        client.answer(FieldQuery("weighting_scheme", "enum"), _stub_ct())
