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


def test_retry_after_seconds_parses_google_body():
    # Gemini puts the delay in the error BODY, not a Retry-After header.
    assert rc._retry_after_seconds(Exception("429 ... Please retry in 57.13s")) == pytest.approx(57.13)
    assert rc._retry_after_seconds(Exception("... 'retryDelay': '30s' ...")) == 30.0
    assert rc._retry_after_seconds(Exception("no delay mentioned")) is None


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


# --- paper_facts field types (v1.1): date + paper_metric ---------------------

def test_coerce_paper_metric_is_all_or_nothing():
    full = {"mean": 0.7, "t_stat": 3.6, "unit": "pct_per_month"}
    assert rc._coerce_paper_metric(full) == {"mean": 0.7, "t_stat": 3.6, "unit": "pct_per_month"}
    # numeric strings are accepted (the model quoted the printed figure)
    assert rc._coerce_paper_metric({"mean": "-0.99", "t_stat": "-4.46", "unit": "pct_per_month"}) == {
        "mean": -0.99, "t_stat": -4.46, "unit": "pct_per_month"
    }
    # a mean without its t-statistic is NOT a headline claim -> silent
    assert rc._coerce_paper_metric({"mean": 0.7, "unit": "pct_per_month"}) is None
    assert rc._coerce_paper_metric({"mean": 0.7, "t_stat": 3.6}) is None
    assert rc._coerce_paper_metric({"mean": 0.7, "t_stat": 3.6, "unit": "  "}) is None
    assert rc._coerce_paper_metric({"mean": True, "t_stat": 3.6, "unit": "x"}) is None
    assert rc._coerce_paper_metric({"mean": "n/a", "t_stat": 3.6, "unit": "x"}) is None
    assert rc._coerce_paper_metric("0.7 (t=3.6)") is None
    assert rc._coerce_paper_metric(None) is None


def test_answer_from_parsed_paper_metric(builder):
    parsed = {
        "field": "claimed_headline_metric", "answered": True,
        "value": {"mean": 0.7, "t_stat": 3.6, "unit": "pct_per_month"},
        "quote": "0.70% per month (t = 3.60)",
    }
    ans = rc._answer_from_parsed("claimed_headline_metric", "paper_metric", parsed, "m")
    assert ans.answered is True
    assert ans.raw == {"mean": 0.7, "t_stat": 3.6, "unit": "pct_per_month"}

    # partial value -> silent, exactly like an unparseable int
    partial = dict(parsed, value={"mean": 0.7, "unit": "pct_per_month"})
    assert rc._answer_from_parsed("claimed_headline_metric", "paper_metric", partial, "m").answered is False
    # complete value but no quote -> silent (evidence rule, unchanged)
    no_quote = {k: v for k, v in parsed.items() if k != "quote"}
    assert rc._answer_from_parsed("claimed_headline_metric", "paper_metric", no_quote, "m").answered is False


def test_render_date_and_paper_metric_kinds(builder):
    date_out = builder.render(FieldQuery("sample_start", "date"), "Downside Risk Factor (DRF)")
    assert "sample_start" in date_out and "YYYY-MM" in date_out

    metric_out = builder.render(FieldQuery("claimed_headline_metric", "paper_metric"), "Momentum (6m)")
    # the strategy label is load-bearing here: a paper reports many numbers, and
    # the field is "the headline figure THIS strategy claims" (D20).
    assert "Momentum (6m)" in metric_out
    assert "pct_per_month" in metric_out


def test_paper_facts_kinds_never_route_to_the_enum_menu():
    """The crash-guard. paper_facts fields have no domains.yaml menu, so if
    _field_kind ever defaulted them to `enum` the renderer would fail loud in
    _menu_for and abort a whole live run at the first paper_facts field."""
    from agents.librarian.pipeline.form_filler import _field_kind

    assert _field_kind("sample_start") == "date"
    assert _field_kind("sample_end") == "date"
    assert _field_kind("claimed_headline_metric") == "paper_metric"


def test_coerce_date_enforces_the_schema_pattern():
    """The JSON Schema declares ^\\d{4}-\\d{2}$ but is inlined into the prompt, not
    vendor-enforced -- this is the only enforcement point. Without it two models
    both answering "July 2004" would agree, locate, and ship STATED with a value
    no gold can match."""
    assert rc._coerce_date("2004-07") == "2004-07"
    assert rc._coerce_date("  2016-12  ") == "2016-12"
    for bad in ("2004", "July 2004", "2004-7", "2004-13", "2004-00", "", None, 200407):
        assert rc._coerce_date(bad) is None


def test_coerce_paper_metric_rejects_an_off_menu_unit():
    # pct_per_year vs pct_per_month is a 12x scaling error, so an unrecognised
    # unit must degrade to silent rather than ship.
    ok = {"mean": 0.7, "t_stat": 3.6, "unit": "pct_per_year"}
    assert rc._coerce_paper_metric(ok)["unit"] == "pct_per_year"
    for bad_unit in ("% per month", "bps", "BOGUS", "pct per month", ""):
        assert rc._coerce_paper_metric({"mean": 0.7, "t_stat": 3.6, "unit": bad_unit}) is None


def test_paper_metric_unit_menu_matches_the_schema():
    """Tripwire: the runtime menu and the schema's enum are two copies of one
    fact. If they drift, the schema shown to the model stops describing what the
    client will actually accept."""
    import json
    from pathlib import Path

    schema = json.loads(
        (Path(rc.__file__).resolve().parents[1] / "data" / "schemas" / "paper_metric.schema.json")
        .read_text(encoding="utf-8")
    )
    assert set(schema["properties"]["value"]["properties"]["unit"]["enum"]) == set(
        rc._PAPER_METRIC_UNITS
    )
