"""Part C — the LLM explainer (§11): verify → retry → deterministic fallback.

All offline (FakeExplainerClient / stubbed backend) — no network. The explainer's
guarantee: the returned prose is ALWAYS number-faithful, because a model that never
produces verifiable prose falls back to the deterministic renderer.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from agents.auditor import (
    ExplainerResponse,
    FakeExplainerClient,
    LiveExplainerClient,
    explain,
    render_report,
    verify_numbers,
)
from agents.auditor.checks.report import AuditorConfig, run_full_audit
from agents.auditor.data.synthetic_panel import build_scenario
from agents.auditor.thresholds import (
    AuditorThresholdError,
    BayesParams,
    EconomicGapBands,
    SupportGate,
    load_explainer_config,
)

from _auditor_fixtures import all_runnable_facts

_CFG = AuditorConfig(
    primary_metric="average",
    percentage_denominator_min=0.0,
    support_gate=SupportGate(min_common_months=12, min_common_fraction_of_reference=0.3),
    n_replicates=120,
    block_length_months=6,
    min_effective_blocks=3,
    vartheta=0.002,
    d_max=0.2,
    fdr_q=0.1,
    bayes=BayesParams(prior_scale=0.1, epsilon=1e-8),
    gap_bands=EconomicGapBands(small=0.005, moderate=0.02, large=0.05),
)


@pytest.fixture(scope="module")
def report():
    scenario = build_scenario("meas_err", seed=1)
    return run_full_audit(
        scenario.strategy, scenario.panel, all_runnable_facts(), _CFG,
        signals=scenario.signals, n_trials=20, sr_std=0.1, seed=1,
    )


# --------------------------------------------------------------------------
# Happy path: a compliant model produces verifiable prose
# --------------------------------------------------------------------------

def test_compliant_model_prose_is_accepted(report):
    # The model returns the deterministic findings verbatim as the explanation —
    # every number traces, so it verifies with no fallback.
    good = render_report(report)
    client = FakeExplainerClient(json.dumps({"explanation": good}), model_version="fake-v1")
    out = explain(report, client, configured_model_id="test-model")
    assert out.verified and not out.used_fallback
    assert out.prose == good
    assert out.n_attempts == 1
    assert out.model_version == "fake-v1"
    assert out.configured_model_id == "test-model"
    assert len(out.prompt_hash) == 64


def test_prose_wrapped_in_code_fence_is_parsed(report):
    good = render_report(report)
    fenced = "```json\n" + json.dumps({"explanation": good}) + "\n```"
    out = explain(report, FakeExplainerClient(fenced))
    assert out.verified and not out.used_fallback


# --------------------------------------------------------------------------
# Fabrication and malformed output both fall back to the deterministic renderer
# --------------------------------------------------------------------------

def test_fabricated_number_falls_back_to_deterministic_renderer(report):
    bad = json.dumps({"explanation": "The effect is 0.123456789 in Sharpe points."})
    client = FakeExplainerClient(bad)
    out = explain(report, client, max_attempts=3)
    assert out.used_fallback
    assert out.verified                      # the fallback renderer is verifier-safe
    assert out.prose == render_report(report)
    assert out.n_attempts == 3
    assert client.calls == 3                 # retried before falling back


def test_malformed_json_falls_back(report):
    out = explain(report, FakeExplainerClient("this is not json at all"), max_attempts=2)
    assert out.used_fallback and out.verified
    assert out.prose == render_report(report)


def test_retry_then_succeed(report):
    good = render_report(report)

    def scripted(prompt, attempt):
        # first attempt fabricates, second returns the clean findings
        if attempt == 0:
            return json.dumps({"explanation": "bogus 9.8765 number"})
        return json.dumps({"explanation": good})

    out = explain(report, FakeExplainerClient(scripted), max_attempts=3)
    assert out.verified and not out.used_fallback
    assert out.n_attempts == 2


def test_returned_prose_always_passes_the_verifier(report):
    # Whatever the model does, the returned prose must verify against the report.
    for text in ("garbage", json.dumps({"explanation": "42000 fake"}), "{}"):
        out = explain(report, FakeExplainerClient(text), max_attempts=1)
        assert verify_numbers(out.prose, report.to_dict()).ok


# --------------------------------------------------------------------------
# Config loader (fail-loud) + live-client wiring (stubbed backend, no network)
# --------------------------------------------------------------------------

def test_load_explainer_config_reads_both_phases():
    dev = load_explainer_config("dev")
    reported = load_explainer_config("reported")
    assert dev.vendor == "gemini" and "flash-lite" in dev.model_id
    assert reported.vendor == "anthropic" and reported.model_id.startswith("claude")
    assert dev.temperature == 0 and reported.max_retries >= 1


def test_load_explainer_config_rejects_bad_phase():
    with pytest.raises(ValueError, match="phase must be"):
        load_explainer_config("prod")


def test_load_explainer_config_fail_loud_when_absent(tmp_path):
    p = tmp_path / "thresholds.yaml"
    p.write_text(textwrap.dedent("auditor:\n  primary_metric: average\n"))
    with pytest.raises(AuditorThresholdError, match="explainer"):
        load_explainer_config("dev", p)


def test_live_client_uses_the_librarian_backend(monkeypatch):
    # Stub the reused make_backend so no SDK / network is touched.
    class _StubBackend:
        def generate(self, prompt, max_output_tokens):
            return json.dumps({"explanation": "ok"}), "stub-v1"

    import agents.librarian.pipeline.real_client as rc
    monkeypatch.setattr(rc, "make_backend", lambda *a, **k: _StubBackend())

    client = LiveExplainerClient("anthropic", "claude-sonnet-4-6", "key", max_output_tokens=256)
    resp = client.generate("prompt")
    assert isinstance(resp, ExplainerResponse)
    assert resp.model_version == "stub-v1"
    assert json.loads(resp.text)["explanation"] == "ok"


# --------------------------------------------------------------------------
# Deterministic renderer: a coordinate without a t statistic
# --------------------------------------------------------------------------

def test_renderer_handles_a_coordinate_with_no_t_statistic(report):
    """A bootstrap-routed or inert coordinate carries ``t_stat``/``p_value`` = None.

    The deterministic renderer is the fallback that keeps a report renderable when the model
    prose fails verification, so it must not raise on that shape: formatting None with ``:.4f``
    would make the fallback itself the failure. Tripwire — it fires if the guard is removed.
    """
    from agents.auditor.explainer.renderer import render_report_dict

    d = report.to_dict()
    label = next(iter(d["inference"]))
    d["inference"][label] = {**d["inference"][label], "t_stat": None, "p_value": None}

    text = render_report_dict(d)

    assert f"Effect {label}: " in text and "t n/a" in text and "p-value n/a" in text
