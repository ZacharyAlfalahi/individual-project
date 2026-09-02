"""A-lever (2026-09-02): paper-text-first prompt assembly + provider prefix caching.

Pins the four properties the cost architecture rests on:
  1. assembly order -- the paper-text block is the PREFIX, instruction+schema the
     suffix, and the joined bytes are what the disk cache / raw archive record;
  2. backend dispatch -- a backend exposing ``generate_split`` receives the parts,
     any other backend receives the joined string (stubs/fakes untouched);
  3. the Anthropic backend marks the prefix block ``cache_control: ephemeral`` and
     sends byte-identical text as two blocks;
  4. cache-token accounting -- provider cache reads/writes accumulate on the client
     and surface in operational_usage() and the run manifest (prompt tokens are the
     UNCACHED remainder on cache-aware vendors).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from agents.librarian.config.canonical_text import CanonicalText            # noqa: E402
from agents.librarian.pipeline import real_client as rc                     # noqa: E402
from agents.librarian.pipeline.model_client import FieldQuery               # noqa: E402
from agents.librarian.registries import load_signal_concept_registry        # noqa: E402
from shared.reporting.run_manifest import build_operational_profile        # noqa: E402


@pytest.fixture
def builder():
    return rc.PromptBuilder.load(load_signal_concept_registry())


def _ct() -> CanonicalText:
    return CanonicalText(
        source_pdf="stub.pdf", source_sha256="deadbeef",
        parser={"name": "stub", "version": "1"},
        normalisation={"ladder_level": "L0", "rules": []},
        pages=("the 5% VaR appears on this page",),
        status="stub",
    )


# ---------------------------------------------------------------------------
# 1. Assembly order + joined bytes.
# ---------------------------------------------------------------------------

def test_field_prompt_prefix_is_the_paper_text_block(builder):
    q = FieldQuery("weighting_scheme", "enum")
    prefix, suffix = rc.assemble_field_prompt(builder, q, "Strategy X", _ct())

    assert prefix.startswith("PAPER TEXT (your ONLY source")
    assert "the 5% VaR appears on this page" in prefix
    assert "PAPER TEXT" not in suffix                       # paper text ONLY in the prefix
    assert "Return ONLY a single JSON object" in suffix     # schema instruction in suffix
    assert "Strategy X" in suffix or "weighting" in suffix  # rendered instruction in suffix

    joined = prefix + suffix
    assert joined.count("PAPER TEXT") == 1                  # exactly one paper block


def test_enumeration_prompt_same_split_semantics(builder):
    prefix, suffix = rc.assemble_enumeration_prompt(builder, _ct())
    assert prefix.startswith("PAPER TEXT (your ONLY source")
    assert "Return ONLY a single JSON object" in suffix
    assert "PAPER TEXT" not in suffix


def test_same_paper_prefix_is_byte_stable_across_fields(builder):
    """The whole point: every field call on one paper shares an identical prefix."""
    p1, _ = rc.assemble_field_prompt(builder, FieldQuery("weighting_scheme", "enum"),
                                     "A", _ct())
    p2, _ = rc.assemble_field_prompt(builder, FieldQuery("holding_period", "int"),
                                     "B", _ct())
    p3, _ = rc.assemble_enumeration_prompt(builder, _ct())
    assert p1 == p2 == p3


# ---------------------------------------------------------------------------
# 2. Dispatch: split-capable backend gets parts; plain backend gets joined bytes.
# ---------------------------------------------------------------------------

class _SplitBackend:
    def __init__(self, text: str):
        self._text = text
        self.split_calls: list[tuple[str, str]] = []
        self.plain_calls = 0
        self.last_usage = None

    def generate(self, prompt, max_output_tokens):
        self.plain_calls += 1
        return self._text, "v1"

    def generate_split(self, prefix, suffix, max_output_tokens):
        self.split_calls.append((prefix, suffix))
        self.last_usage = {"prompt": 10, "completion": 5,
                           "cache_creation": 100, "cache_read": 900}
        return self._text, "v1"


def _client(monkeypatch, builder, backend, **kw):
    monkeypatch.setitem(rc._BACKENDS, "stub", lambda model_id, api_key, temperature: backend)
    return rc.RealModelClient(model_id="stub-1", vendor="stub", api_key="k",
                              builder=builder, **kw)


def test_split_backend_receives_parts_and_cache_tokens_accumulate(monkeypatch, builder):
    text = json.dumps({"field": "weighting_scheme", "answered": True,
                       "value": "value", "quote": "q"})
    backend = _SplitBackend(text)
    client = _client(monkeypatch, builder, backend)

    client.answer(FieldQuery("weighting_scheme", "enum"), _ct())
    assert backend.plain_calls == 0 and len(backend.split_calls) == 1
    prefix, suffix = backend.split_calls[0]
    assert prefix.startswith("PAPER TEXT")
    assert "Return ONLY" in suffix

    # A-lever accounting: cache split accumulates and surfaces.
    assert client.total_cache_creation_tokens == 100
    assert client.total_cache_read_tokens == 900
    usage = client.operational_usage()
    assert usage["cache_creation_tokens"] == 100
    assert usage["cache_read_tokens"] == 900


def test_plain_backend_still_gets_joined_bytes(monkeypatch, builder):
    """Backends without generate_split (every stub/fake, Gemini, Mistral) receive
    the identical joined string -- proven end-to-end by the existing disk-cache
    suite; pinned here explicitly."""
    text = json.dumps({"field": "weighting_scheme", "answered": True,
                       "value": "value", "quote": "q"})

    class _Plain:
        def __init__(self):
            self.prompts = []

        def generate(self, prompt, max_output_tokens):
            self.prompts.append(prompt)
            return text, "v1"

    backend = _Plain()
    client = _client(monkeypatch, builder, backend)
    client.answer(FieldQuery("weighting_scheme", "enum"), _ct())
    assert len(backend.prompts) == 1
    assert backend.prompts[0].startswith("PAPER TEXT")          # prefix first
    assert "Return ONLY" in backend.prompts[0]                  # suffix joined after
    assert client.total_cache_creation_tokens == 0              # cache-unaware -> zero


# ---------------------------------------------------------------------------
# 3. The Anthropic backend's block shape.
# ---------------------------------------------------------------------------

def test_anthropic_generate_split_marks_prefix_cacheable(monkeypatch):
    captured = {}

    class _FakeUsage:
        input_tokens = 7
        output_tokens = 3
        cache_creation_input_tokens = 111
        cache_read_input_tokens = 999

    class _FakeBlock:
        type = "text"
        text = "{}"

    class _FakeResp:
        content = [_FakeBlock()]
        usage = _FakeUsage()
        model = "stub-v1"

    class _FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeResp()

    backend = object.__new__(rc._AnthropicBackend)   # skip __post_init__ (no SDK client)
    backend.model_id = "claude-test"
    backend.temperature = 0.0
    backend._client = type("C", (), {"messages": _FakeMessages()})()

    text, version = backend.generate_split("PREFIX-BYTES", "SUFFIX-BYTES", 512)

    blocks = captured["messages"][0]["content"]
    assert blocks[0]["text"] == "PREFIX-BYTES"
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert blocks[1]["text"] == "SUFFIX-BYTES"
    assert "cache_control" not in blocks[1]
    assert backend.last_usage == {"prompt": 7, "completion": 3,
                                  "cache_creation": 111, "cache_read": 999}
    assert version == "stub-v1"


# ---------------------------------------------------------------------------
# 4. Manifest carries the cache split (additive; absent when not provided).
# ---------------------------------------------------------------------------

def test_operational_profile_cache_fields_additive():
    with_cache = build_operational_profile(
        phase="report", model_calls=2, prompt_tokens=10, completion_tokens=5,
        cache_creation_tokens=100, cache_read_tokens=900,
    )
    assert with_cache["tokens"] == {"prompt": 10, "completion": 5,
                                    "cache_creation": 100, "cache_read": 900}

    without = build_operational_profile(
        phase="report", model_calls=2, prompt_tokens=10, completion_tokens=5,
    )
    assert without["tokens"] == {"prompt": 10, "completion": 5}   # pre-A shape unchanged


def test_reorder_is_a_byte_multiset_permutation_of_the_historical_template(builder):
    """Review-recommended tripwire: the new assembly must contain EXACTLY the
    historical prompt's bytes, reordered -- freezing the 'modulo ordering'
    guarantee against future template edits silently changing content."""
    q = FieldQuery("weighting_scheme", "enum")
    ct = _ct()
    prefix, suffix = rc.assemble_field_prompt(builder, q, "Strategy X", ct)

    instruction = builder.render(q, "Strategy X")
    schema = builder.schema_for(q.kind)
    paper_text = "\n\n".join(ct.pages)
    historical = (
        f"{instruction}\n\n"
        "PAPER TEXT (your ONLY source -- quote verbatim, character-for-character):\n"
        f"<<<\n{paper_text}\n>>>\n\n"
        "Return ONLY a single JSON object (no prose, no code fence) matching "
        f"this JSON Schema:\n{json.dumps(schema)}"
    )
    assert sorted(prefix + suffix) == sorted(historical)   # same bytes...
    assert (prefix + suffix) != historical                 # ...genuinely reordered
