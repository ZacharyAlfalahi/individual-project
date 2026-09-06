"""WS-C (P1) — the live codegen client + budget guard. All OFFLINE: a FAKE backend is injected,
so no SDK, no network, no Librarian import, no API keys. Mirrors test_scientist_phase_d_client.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation.codegen import live_client as L  # noqa: E402
from evaluation.codegen.live_client import (  # noqa: E402
    BudgetExceededError,
    BudgetGuard,
    LiveCodegenClient,
    build_codegen_factory,
    load_budget,
)

_PRICES = {
    "m-cheap": {"input": 1.0, "output": 2.0},      # $/1M
    "m-free": {"input": 0.0, "output": 0.0},
}


def _guard(cap=30.0, per_call=16000, prices=None):
    return BudgetGuard(usd_cap=cap, per_call_output_token_cap=per_call,
                       prices_usd_per_1m=prices or _PRICES)


class _FakeBackend:
    """Scripted Librarian-style backend: generate(prompt, max_output_tokens) -> (text, version),
    and a last_usage side channel. Each step is ("return", text) or ("raise", exc)."""

    def __init__(self, script, usage=None, version="v-fake"):
        self.script = list(script)
        self.calls = 0
        self.max_tokens_seen = []
        self._usage = usage
        self._version = version
        self.last_usage = None

    def generate(self, prompt, max_output_tokens):
        self.calls += 1
        self.max_tokens_seen.append(max_output_tokens)
        self.last_usage = dict(self._usage) if self._usage else None
        action, payload = self.script.pop(0)
        if action == "raise":
            raise payload
        return payload, self._version


def _client(script, *, budget=None, usage=None, version="v-fake", **kw):
    return LiveCodegenClient(vendor="anthropic", model_id="m-cheap", max_output_tokens=16000,
                             budget=budget, backend=_FakeBackend(script, usage, version), **kw)


# ---- BudgetGuard -------------------------------------------------------------------------

def test_estimate_usd_is_per_million():
    g = _guard()
    # 1M input @ $1 + 0.5M output @ $2 = $1 + $1 = $2
    assert g.estimate_usd("m-cheap", 1_000_000, 500_000) == pytest.approx(2.0)


def test_output_cap_is_the_min_of_configured_and_cap():
    g = _guard(per_call=16000)
    assert g.output_cap_for(100000) == 16000
    assert g.output_cap_for(2048) == 2048


def test_charge_accumulates_and_logs():
    g = _guard()
    g.charge("m-cheap", 1000, 1000)
    g.charge("m-cheap", 1000, 1000)
    assert len(g.records) == 2
    assert g.spent_usd == pytest.approx(2 * (0.001 * 1.0 + 0.001 * 2.0))


def test_charge_raises_when_cap_crossed():
    g = _guard(cap=0.001)                              # $0.001 cap
    with pytest.raises(BudgetExceededError, match="sub-cap"):
        g.charge("m-cheap", 1_000_000, 1_000_000)     # ~$3, way over
    assert g.spent_usd > g.usd_cap                     # the record is truthful about the spend


def test_unpriced_model_fails_loud():
    g = _guard()
    with pytest.raises(BudgetExceededError, match="no price registered"):
        g.charge("unknown-model", 10, 10)


# ---- LiveCodegenClient -------------------------------------------------------------------

def test_client_conforms_and_threads_the_capped_ceiling():
    g = _guard(per_call=16000)
    c = _client([("return", "```python\nx=1\n```")], budget=g,
                usage={"prompt": 100, "completion": 50})
    assert hasattr(c, "generate") and hasattr(c, "name")          # ModelClient protocol shape
    assert c.name == "m-cheap"
    assert c.generate("prompt", seed=0) == "```python\nx=1\n```"
    assert c._backend.max_tokens_seen == [16000]                  # capped ceiling threaded
    assert c.last_model_version == "v-fake"
    assert g.spent_usd == pytest.approx(0.0001 * 1.0 + 0.00005 * 2.0)   # usage charged


def test_generate_charges_and_can_trip_the_cap(monkeypatch):
    g = _guard(cap=0.0)                                # any spend trips it
    c = _client([("return", "ok")], budget=g, usage={"prompt": 10, "completion": 10})
    with pytest.raises(BudgetExceededError):
        c.generate("p", seed=0)


def test_retryable_error_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    c = _client([("raise", RuntimeError("rate limited")), ("return", "ok")],
                usage={"prompt": 1, "completion": 1}, max_retries=4)
    assert c.generate("p", seed=0) == "ok"
    assert c._backend.calls == 2


def test_non_retryable_error_reraises_immediately(monkeypatch):
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    boom = RuntimeError("unauthorized")
    boom.status_code = 401
    c = _client([("raise", boom)], max_retries=5)
    with pytest.raises(RuntimeError):
        c.generate("p", seed=0)
    assert c._backend.calls == 1


def test_budget_error_is_not_retried(monkeypatch):
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    g = _guard(cap=0.0)
    c = _client([("return", "ok"), ("return", "ok")], budget=g,
                usage={"prompt": 10, "completion": 10}, max_retries=5)
    with pytest.raises(BudgetExceededError):
        c.generate("p", seed=0)
    assert c._backend.calls == 1                       # the cap halts, never retries


# ---- factory + loader --------------------------------------------------------------------

def test_factory_fails_loud_without_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    factory = build_codegen_factory(budget=_guard())
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        factory({"vendor": "anthropic", "model_id": "claude-sonnet-4-6",
                 "api_key_env": "ANTHROPIC_API_KEY"})


def test_load_budget_reads_the_real_thresholds():
    g = load_budget()
    assert g.usd_cap == 30.0
    assert g.per_call_output_token_cap == 16000
    assert g.estimate_usd("gemini-3.1-flash-lite", 1_000_000, 1_000_000) == 0.0   # free dev pair


def test_load_budget_hard_raises_when_absent(tmp_path):
    bad = tmp_path / "t.yaml"
    bad.write_text("p1_codegen:\n  scoring: {}\n", encoding="utf-8")
    with pytest.raises(KeyError, match="p1_codegen.budget"):
        load_budget(bad)
