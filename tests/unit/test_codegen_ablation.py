"""WS-C (P1) — the scored ablation pipeline (generate -> sandbox -> score -> archive -> tabulate).

OFFLINE: no vendor calls. A FAKE client returns fixed code; the code runs in the REAL WS-C sandbox
against the exported corr-family panel and is scored against the oracle. Exercises the three
branches (valid RUN, malformed -> WONT_RUN, runtime error -> WONT_RUN) and the result shape. Skips
when the dev panel/oracles are absent."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation.codegen.ablation import run_scored_ablation  # noqa: E402
from evaluation.codegen.live_client import BudgetGuard  # noqa: E402
from evaluation.codegen.panel_export import build_codegen_panel  # noqa: E402

# A valid strategy: equal-weighted mean monthly return. Honours the output contract, will RUN
# (verdict RUNS_WRONG vs the drf oracle — it is not drf — which is exactly a scored-pipeline datum).
_VALID = (
    "```python\n"
    "import os\n"
    "import pandas as pd\n"
    "p = pd.read_parquet(os.environ['PANEL_PATH'])\n"
    "g = p.dropna(subset=['ret']).groupby('date')['ret'].mean().reset_index()\n"
    "g.columns = ['date', 'portfolio_return']\n"
    "g.to_csv(os.environ['OUTPUT_PATH'], index=False)\n"
    "```\n"
)
_MALFORMED = "no fenced code block here — just prose"
_ERRORS = "```python\nraise RuntimeError('strategy blew up')\n```\n"


class _FakeClient:
    def __init__(self, code: str):
        self.name = "fake-model"
        self._code = code
        self.last_model_version = "fake-v1"

    def generate(self, prompt: str, *, seed: int) -> str:
        return self._code

    def operational_usage(self) -> dict:
        return {"model_calls": 1, "prompt_tokens": 0, "completion_tokens": 0, "retries": 0}


def _factory(code):
    return lambda model: _FakeClient(code)


def _budget():
    return BudgetGuard(usd_cap=30.0, per_call_output_token_cap=16000,
                       prices_usd_per_1m={"m": {"input": 0.0, "output": 0.0}})


@pytest.fixture(scope="module")
def panel_path(tmp_path_factory):
    try:
        panel = build_codegen_panel()
    except FileNotFoundError as exc:
        pytest.skip(f"dev panel/signals absent: {exc}")
    from evaluation.codegen.oracles import load_oracle_series
    try:
        load_oracle_series("drf")
    except FileNotFoundError as exc:
        pytest.skip(f"oracle factors absent: {exc}")
    out = tmp_path_factory.mktemp("panel") / "engine_panel_corr.parquet"
    panel.to_parquet(out, index=False)
    return out


def _run(code, panel_path, tmp_path, strategies=("drf",)):
    return run_scored_ablation(
        strategies, phase="dev", client_factory=_factory(code), budget=_budget(),
        panel_path=panel_path, cache_root=tmp_path / "cache", sandbox_root=tmp_path / "sandbox",
        ensure_panel=False)


def test_valid_strategy_runs_and_is_scored(panel_path, tmp_path):
    out = _run(_VALID, panel_path, tmp_path)
    assert out["phase"] == "dev" and out["reportable"] is False   # dev is never reportable
    assert len(out["runs"]) == 2                                  # 2 dev models x 1 strategy
    row = out["raw_metrics_table"][0]
    assert row["strategy"] == "drf" and row["verdict"] in {"RUNS_RIGHT", "RUNS_WRONG"}
    assert row["n_overlap"] and row["n_overlap"] > 24             # scored on the overlap
    assert sum(out["verdict_counts"].values()) == 2
    assert "budget" in out and out["budget"]["usd_cap"] == 30.0


def test_malformed_response_is_wont_run(panel_path, tmp_path):
    out = _run(_MALFORMED, panel_path, tmp_path)
    assert all(r["verdict"] == "WONT_RUN" for r in out["runs"])
    assert all(r["sandbox_reason"] == "malformed_response" for r in out["runs"])
    assert all(r["code_extracted"] is False for r in out["runs"])


def test_runtime_error_is_wont_run(panel_path, tmp_path):
    out = _run(_ERRORS, panel_path, tmp_path)
    assert all(r["verdict"] == "WONT_RUN" for r in out["runs"])
    assert all(r["sandbox_status"] == "wont_run" for r in out["runs"])
    assert all(r["code_extracted"] is True for r in out["runs"])   # code extracted, then errored


def test_generation_error_is_recorded_and_does_not_abort(panel_path, tmp_path):
    # A vendor failure (rate-limit exhaustion) is a typed generation_error, not a codegen datum:
    # the run is recorded, the loop continues, and the result is non-reportable.
    class _Boom:
        name = "boom-model"
        last_model_version = None

        def generate(self, prompt, *, seed):
            raise RuntimeError("codegen client 'x' exhausted 6 retries")

        def operational_usage(self):
            return {"model_calls": 0}

    out = run_scored_ablation(
        ("drf",), phase="dev", client_factory=lambda m: _Boom(), budget=_budget(),
        panel_path=panel_path, cache_root=tmp_path / "c", sandbox_root=tmp_path / "s",
        ensure_panel=False)
    assert len(out["generation_errors"]) == 2                     # both dev models failed
    assert all(r["sandbox_status"] == "generation_error" for r in out["runs"])
    assert out["reportable"] is False


def test_budget_breach_still_halts(panel_path, tmp_path):
    from evaluation.codegen.live_client import BudgetExceededError

    class _Overspend:
        name = "spendy"
        last_model_version = None

        def generate(self, prompt, *, seed):
            raise BudgetExceededError("cap crossed")

    with pytest.raises(BudgetExceededError):
        run_scored_ablation(("drf",), phase="dev", client_factory=lambda m: _Overspend(),
                            budget=_budget(), panel_path=panel_path, cache_root=tmp_path / "c",
                            sandbox_root=tmp_path / "s", ensure_panel=False)


def test_cache_replay_is_zero_regeneration(panel_path, tmp_path):
    # Second identical run hits the ResponseCache — a spend-free replay (the artefact-log invariant).
    class _Counting(_FakeClient):
        calls = 0

        def generate(self, prompt, *, seed):
            type(self).calls += 1
            return _VALID

    cache = tmp_path / "cache"
    run_scored_ablation(("drf",), phase="dev", client_factory=lambda m: _Counting(_VALID),
                        budget=_budget(), panel_path=panel_path, cache_root=cache,
                        sandbox_root=tmp_path / "sb1", ensure_panel=False)
    first = _Counting.calls
    run_scored_ablation(("drf",), phase="dev", client_factory=lambda m: _Counting(_VALID),
                        budget=_budget(), panel_path=panel_path, cache_root=cache,
                        sandbox_root=tmp_path / "sb2", ensure_panel=False)
    assert _Counting.calls == first                               # no new generation calls
