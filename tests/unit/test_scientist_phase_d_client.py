"""Rung-3 phase_d wiring (SC-SCI-9): the PhaseDModelClient adapter (protocol conformance, pacing
skip, retry-vs-reraise classification), the fail-loud model_stack loader, and the fail-loud factory.
All offline — a FAKE backend is injected, so no SDK, no network, no Librarian import, no keys."""

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.scientist.researcher import phase_d_client as P  # noqa: E402
from agents.scientist.researcher.llm_source import LLMResearcherSource  # noqa: E402


class _FakeBackend:
    """Scripted stand-in for a Librarian vendor backend: generate(prompt, max_output_tokens) ->
    (text, version). Each script step is ("return", text) or ("raise", exc)."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.max_tokens_seen = []

    def generate(self, prompt, max_output_tokens):
        self.calls += 1
        self.max_tokens_seen.append(max_output_tokens)
        action, payload = self.script.pop(0)
        if action == "raise":
            raise payload
        return payload, "v-fake"


def _client(script, **kw):
    return P.PhaseDModelClient(vendor="gemini", model_id="gemini-3.1-flash-lite",
                               max_output_tokens=2048, backend=_FakeBackend(script), **kw)


# ---- adapter conformance ------------------------------------------------------------------

def test_client_conforms_and_returns_text():
    c = _client([("return", "[]")])
    assert c.name == "gemini-3.1-flash-lite"
    assert c.generate("prompt", seed=0) == "[]"
    assert c._backend.max_tokens_seen == [2048]                 # budget threaded to the vendor call


def test_client_is_a_valid_llm_researcher_client():
    # A PhaseDModelClient drops straight into the rung-3 source (satisfies its ModelClient protocol).
    src = LLMResearcherSource(_client([("return", "[]")]))
    assert src.source_name == "llm_researcher" and src.client.name == "gemini-3.1-flash-lite"


# ---- retry vs re-raise --------------------------------------------------------------------

def test_retryable_error_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr(P.time, "sleep", lambda s: None)        # no real backoff sleeps
    c = _client([("raise", RuntimeError("rate limited")), ("return", "ok")], max_retries=4)
    assert c.generate("p", seed=0) == "ok"
    assert c._backend.calls == 2                                # retried once


def test_non_retryable_error_reraises_immediately(monkeypatch):
    monkeypatch.setattr(P.time, "sleep", lambda s: None)
    boom = RuntimeError("unauthorized")
    boom.status_code = 401                                      # a 4xx a retry can't fix
    c = _client([("raise", boom)], max_retries=5)
    with pytest.raises(RuntimeError):
        c.generate("p", seed=0)
    assert c._backend.calls == 1                                # NOT retried


def test_exhausted_retries_raise(monkeypatch):
    monkeypatch.setattr(P.time, "sleep", lambda s: None)
    c = _client([("raise", RuntimeError("x")), ("raise", RuntimeError("x"))], max_retries=2)
    with pytest.raises(RuntimeError, match="exhausted"):
        c.generate("p", seed=0)


# ---- fail-loud config loader --------------------------------------------------------------

def test_load_model_stack_reads_the_real_thresholds():
    ms = P.load_model_stack()
    assert ms["phase_d"]["model_a"]["model_id"] == "gemini-3.1-flash-lite"
    assert ms["phase_d"]["model_b"]["vendor"] == "mistral"
    assert ms["temperature"] > 0                                # generation diversity (SC-SCI-9)


def test_load_model_stack_hard_raises_when_absent(tmp_path):
    bad = tmp_path / "t.yaml"
    bad.write_text(yaml.safe_dump({"scientist": {"entry_rule": {}}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="fail loud"):
        P.load_model_stack(bad)


# ---- fail-loud factory (no key -> actionable error, never a silent stub) ------------------

def test_build_phase_d_clients_fails_loud_without_a_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        P.build_phase_d_clients()
