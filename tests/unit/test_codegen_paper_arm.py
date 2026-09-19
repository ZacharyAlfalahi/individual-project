"""P1 paper arm — paper → code with no specification in between.

Covers the paper prompt, its freeze, the ``prompt_fn`` / ``deny_read_paths`` plumbing through the
scored ablation and the sandbox, the three-state reportability, the arm comparison and the CLI's
refusals. No vendor call is made. Tests that need the frozen canonical texts or the development panel
(not shipped with the repository) skip when those are absent.
"""

from __future__ import annotations

import hashlib
import json
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import run_p1_paper_codegen as CLI  # noqa: E402
from evaluation.codegen import paper_arm as PA  # noqa: E402
from evaluation.codegen.ablation import run_scored_ablation  # noqa: E402
from evaluation.codegen.live_client import BudgetGuard  # noqa: E402
from evaluation.codegen.runner import STRATEGIES, contract_freeze_ok, prompt_asset_hashes  # noqa: E402
from evaluation.codegen.sandbox import (  # noqa: E402
    SandboxSpec,
    detect_mechanism,
    run_sandboxed,
    seatbelt_profile,
)

_TEXTS_PRESENT = all(t.canonical_text.exists() for t in PA.PAPER_TARGETS.values())
needs_texts = pytest.mark.skipif(
    not _TEXTS_PRESENT,
    reason="requires the frozen canonical texts (evaluation/canonical_texts/) — not shipped; see README")


# --------------------------------------------------------------------------------------------
# prompt
# --------------------------------------------------------------------------------------------

@needs_texts
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_prompt_carries_label_paper_schema_and_contract_and_no_spec(strategy):
    prompt = PA.build_paper_prompt(strategy)
    assert f"Strategy to implement: {PA.PAPER_TARGETS[strategy].label}\n" in prompt
    assert PA.paper_text(strategy) in prompt
    for asset in PA._SHARED_ASSETS:
        assert asset.read_text(encoding="utf-8") in prompt
    assert "{{" not in prompt and "}}" not in prompt
    # nothing from the field-key arm's serialised specification
    for marker in ('"part2"', '"evidence"', '"STATED"', "claimed_headline_metric",
                   "Strategy specification (typed extraction"):
        assert marker not in prompt


@needs_texts
def test_prompt_is_deterministic_and_the_three_bbw_targets_differ():
    assert PA.build_paper_prompt("drf") == PA.build_paper_prompt("drf")
    bbw = {PA.build_paper_prompt(s) for s in ("drf", "crf", "lrf")}
    assert len(bbw) == 3, "shared-paper prompts must differ, else they share one cache key"


def test_every_oracle_strategy_has_a_target_and_the_paper_text_joins_like_the_librarian():
    assert set(PA.PAPER_TARGETS) == set(STRATEGIES)
    source = (REPO_ROOT / "agents/librarian/pipeline/real_client.py").read_text(encoding="utf-8")
    assert '"\\n\\n".join(canonical_text.pages)' in source
    assert '"\\n\\n".join(ct.pages)' in Path(PA.__file__).read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------------
# freeze
# --------------------------------------------------------------------------------------------

@needs_texts
def test_freeze_passes_when_every_hash_is_recorded_and_fails_on_a_missing_one(tmp_path):
    hashes = PA.paper_arm_asset_hashes()
    contract = tmp_path / "contract.md"
    contract.write_text("\n".join(f"{k}: {v}" for k, v in hashes.items()), encoding="utf-8")
    assert PA.paper_arm_freeze_ok(contract)[0]
    first = next(iter(hashes.values()))
    contract.write_text(contract.read_text(encoding="utf-8").replace(first, "0" * 64), encoding="utf-8")
    ok, msg = PA.paper_arm_freeze_ok(contract)
    assert not ok and "not recorded" in msg
    assert not PA.paper_arm_freeze_ok(tmp_path / "absent.md")[0]


@needs_texts
def test_real_contract_freeze_verifies_when_present():
    if not PA.CONTRACT.exists():
        pytest.skip("paper-arm contract not shipped with the repository")
    assert PA.paper_arm_freeze_ok() == (True, "paper-arm contract freeze verified")


def test_paper_template_lives_outside_the_field_key_freeze_glob():
    assert "template_paper.md" not in prompt_asset_hashes()
    from evaluation.codegen import runner as _runner
    if not _runner._CONTRACT.exists():
        pytest.skip("P1 field-key mini-contract not shipped with the repository")
    assert contract_freeze_ok()[0], "the paper arm must not alter the P1 freeze"


# --------------------------------------------------------------------------------------------
# sandbox read denial
# --------------------------------------------------------------------------------------------

def test_empty_deny_list_leaves_the_seatbelt_profile_unchanged(tmp_path):
    from evaluation.codegen.sandbox import _SEATBELT_PROFILE
    assert seatbelt_profile(tmp_path) == _SEATBELT_PROFILE.format(jail=tmp_path)
    denied = seatbelt_profile(tmp_path, (tmp_path / "secret",))
    assert denied.endswith(f'(deny file-read* (subpath "{(tmp_path / "secret").resolve()}"))\n')


def test_shim_with_a_deny_list_refuses_rather_than_running_unconfined(tmp_path):
    spec = SandboxSpec(python_bin=Path(sys.executable), jail_dir=tmp_path / "jail",
                       mechanism="shim", deny_read_paths=(tmp_path,))
    with pytest.raises(ValueError, match="seatbelt"):
        run_sandboxed("print(1)", spec)


@pytest.mark.skipif(detect_mechanism() != "seatbelt", reason="read denial is seatbelt-only")
def test_seatbelt_denies_reading_a_listed_tree(tmp_path):
    secret_dir = tmp_path / "oracle_factors"
    secret_dir.mkdir()
    (secret_dir / "series.csv").write_text("date,portfolio_return\n2020-01-31,0.01\n")
    code = textwrap.dedent(f"""
        import os, shutil
        shutil.copy({str(secret_dir / "series.csv")!r}, os.environ["OUTPUT_PATH"])
    """)
    jail = tmp_path / "jail"
    allowed = run_sandboxed(code, SandboxSpec(python_bin=Path(sys.executable), jail_dir=jail))
    assert allowed.status == "ok"
    denied = run_sandboxed(code, SandboxSpec(python_bin=Path(sys.executable),
                                             jail_dir=tmp_path / "jail2",
                                             deny_read_paths=(secret_dir,)))
    assert denied.status == "wont_run" and denied.reason == "nonzero_exit"


# --------------------------------------------------------------------------------------------
# prompt_fn plumbing through the scored ablation (real sandbox, fake client)
# --------------------------------------------------------------------------------------------

_VALID = (
    "```python\nimport os\nimport pandas as pd\n"
    "p = pd.read_parquet(os.environ['PANEL_PATH'])\n"
    "g = p.dropna(subset=['ret']).groupby('date')['ret'].mean().reset_index()\n"
    "g.columns = ['date', 'portfolio_return']\n"
    "g.to_csv(os.environ['OUTPUT_PATH'], index=False)\n```\n"
)


class _RecordingClient:
    def __init__(self, seen: list[str]):
        self.name = "fake-model"
        self.last_model_version = None
        self._seen = seen

    def generate(self, prompt: str, *, seed: int) -> str:
        self._seen.append(prompt)
        return _VALID


@pytest.fixture(scope="module")
def panel_path(tmp_path_factory):
    from evaluation.codegen.oracles import load_oracle_series
    from evaluation.codegen.panel_export import build_codegen_panel
    try:
        panel = build_codegen_panel()
        load_oracle_series("drf")
    except FileNotFoundError as exc:
        pytest.skip(f"dev panel/oracles absent: {exc}")
    out = tmp_path_factory.mktemp("panel") / "engine_panel_corr.parquet"
    panel.to_parquet(out, index=False)
    return out


def test_prompt_fn_reaches_the_client_the_cache_and_the_record(panel_path, tmp_path):
    seen: list[str] = []
    budget = BudgetGuard(usd_cap=30.0, per_call_output_token_cap=16000,
                         prices_usd_per_1m={"m": {"input": 0.0, "output": 0.0}})

    def run(root):
        return run_scored_ablation(
            ("drf",), phase="dev", client_factory=lambda m: _RecordingClient(seen), budget=budget,
            panel_path=panel_path, cache_root=tmp_path / "cache", sandbox_root=root,
            ensure_panel=False, prompt_fn=lambda s: f"PAPER PROMPT FOR {s}",
            deny_read_paths=((tmp_path / "denied"),) if detect_mechanism() == "seatbelt" else ())

    record = run(tmp_path / "sandbox")
    assert set(seen) == {"PAPER PROMPT FOR drf"} and len(seen) == 2   # one call per model
    expected = hashlib.sha256(b"PAPER PROMPT FOR drf").hexdigest()
    assert record["prompt_sha256"] == {"drf": expected}
    assert all(r["sandbox_status"] == "ok" for r in record["runs"])
    run(tmp_path / "sandbox2")
    assert len(seen) == 2, "a second run must replay the cache keyed on the paper prompt"


# --------------------------------------------------------------------------------------------
# reportability
# --------------------------------------------------------------------------------------------

def _record(versions, *, phase="reported", errors=(), prompts=None):
    runs = [{"model_id": m, "returned_model_version": v} for m, v in versions]
    return {"phase": phase, "runs": runs, "generation_errors": list(errors),
            "prompt_sha256": prompts or {"drf": "a"}}


def test_live_run_with_matching_skus_is_reportable():
    rep = PA.reportability(_record([("claude-sonnet-4-6", "claude-sonnet-4-6"),
                                    ("gemini-3.5-flash", "gemini-3.5-flash-001")]))
    assert rep["sku_match"] is True and rep["reportable"] and rep["reportable_blockers"] == []


def test_replay_is_unverified_unless_a_matching_live_record_vouches():
    replay = _record([("claude-sonnet-4-6", None), ("gemini-3.5-flash", None)])
    rep = PA.reportability(replay)
    assert rep["sku_match"] is None and not rep["reportable"]
    live = {**_record([("claude-sonnet-4-6", "claude-sonnet-4-6")]), "sku_match": True}
    assert PA.reportability(replay, live_record=live)["reportable"]
    stale = {**live, "prompt_sha256": {"drf": "different"}}
    assert not PA.reportability(replay, live_record=stale)["reportable"]


def test_mismatch_partial_dev_and_errors_each_block():
    assert PA.reportability(_record([("claude-sonnet-4-6", "claude-opus-4")]))["sku_match"] is False
    partial = PA.reportability(_record([("claude-sonnet-4-6", "claude-sonnet-4-6"),
                                        ("gemini-3.5-flash", None)]))
    assert partial["sku_match"] is None and not partial["reportable"]
    assert not PA.reportability(_record([("m", "m")], phase="dev"))["reportable"]
    errored = PA.reportability(_record([("claude-sonnet-4-6", "claude-sonnet-4-6")], errors=[{}]))
    assert any("infrastructure" in b for b in errored["reportable_blockers"])


# --------------------------------------------------------------------------------------------
# arm comparison
# --------------------------------------------------------------------------------------------

def _scored(verdicts: dict[tuple[str, str], str]) -> dict:
    return {"models": ["a", "b"], "prompt_sha256": {}, "panel_sha256": "p",
            "raw_metrics_table": [
                {"strategy": s, "model_id": m, "verdict": v, "reason": None, "failed_criteria": [],
                 "n_overlap": 100, "correlation": 0.5, "sign_agreement": 0.9, "mean_diff": 0.0001,
                 "tracking_error": 0.001, "max_abs_diff": 0.01}
                for (s, m), v in verdicts.items()]}


def test_compare_arms_counts_transitions_and_keeps_missing_cells_as_no_datum():
    fk = _scored({(s, m): "RUNS_RIGHT" for s in STRATEGIES for m in ("a", "b")})
    paper_cells = {(s, m): "RUNS_WRONG" for s in STRATEGIES for m in ("a", "b")}
    del paper_cells[("lrf", "b")]                                  # a generation error
    comp = PA.compare_arms(fk, _scored(paper_cells))
    assert comp["counts_all_10"]["field_key"]["RUNS_RIGHT"] == 10
    assert comp["counts_all_10"]["paper"] == {"RUNS_RIGHT": 0, "RUNS_WRONG": 9, "WONT_RUN": 0,
                                              "NO_DATUM": 1}
    assert comp["counts_byte_equality_8"]["paper"]["RUNS_WRONG"] == 8
    assert comp["transitions"] == {"RUNS_RIGHT→NO_DATUM": 1, "RUNS_RIGHT→RUNS_WRONG": 9}
    md = PA.render_comparison_md(comp, basis="total_return")
    assert "Panels byte-identical: True" in md and "no datum" in md


# --------------------------------------------------------------------------------------------
# CLI refusals (hermetic: panels are tmp files, no vendor client is ever built)
# --------------------------------------------------------------------------------------------

def _paths(tmp_path, *, sha_matches=True, max_date="2021-12-31", holdout=False):
    import pandas as pd
    if holdout:
        # Refused on the path alone, before anything is opened (tests/conftest.py forbids even
        # creating a file under a holdout path).
        base = tmp_path / "holdout"
        return CLI.PaperArmPaths(out=tmp_path / "out", panel=base / "engine_panel_corr.parquet",
                                 field_key_record=base / "p1_results_reported.json",
                                 factors_dir=tmp_path / "factors",
                                 sandbox_root=tmp_path / "out" / "sandbox")
    base = tmp_path / "dev"
    base.mkdir()
    panel = base / "engine_panel_corr.parquet"
    pd.DataFrame({"date": pd.to_datetime(["2005-01-31", max_date])}).to_parquet(panel)
    record = base / "p1_results_reported.json"
    sha = hashlib.sha256(panel.read_bytes()).hexdigest()
    record.write_text(json.dumps({"panel_sha256": sha if sha_matches else "0" * 64}))
    return CLI.PaperArmPaths(out=tmp_path / "out", panel=panel, field_key_record=record,
                             factors_dir=tmp_path / "factors", sandbox_root=tmp_path / "out" / "sandbox")


def test_verify_panel_accepts_the_field_key_arms_dev_panel(tmp_path):
    paths = _paths(tmp_path)
    assert CLI.verify_panel(paths) == hashlib.sha256(paths.panel.read_bytes()).hexdigest()


@pytest.mark.parametrize("kwargs, match", [
    ({"sha_matches": False}, "differs"),
    ({"max_date": "2022-01-31"}, "holdout"),
    ({"holdout": True}, "holdout"),
])
def test_verify_panel_refuses(tmp_path, kwargs, match):
    with pytest.raises(CLI.PaperArmRefusal, match=match):
        CLI.verify_panel(_paths(tmp_path, **kwargs))


def test_deny_list_covers_holdout_oracles_specs_and_earlier_generations():
    deny = {str(p) for p in CLI.deny_read_paths("total_return")}
    for fragment in ("data/holdout", "data/development/factors", "total_return/factors",
                     "evaluation/gold_specs", "/runs", "codegen/archive", "codegen/sandbox"):
        assert any(fragment in p for p in deny), fragment


def test_execute_refuses_an_existing_sandbox_root_before_any_generation(monkeypatch, tmp_path, capsys):
    paths = _paths(tmp_path)
    paths.sandbox_root.mkdir(parents=True)
    monkeypatch.setattr(CLI, "resolve_paths", lambda basis, out_root: paths)
    monkeypatch.setattr(CLI, "build_paper_prompt", lambda s: f"prompt {s}")
    monkeypatch.setattr(CLI, "paper_arm_asset_hashes", lambda: {})
    monkeypatch.setattr(CLI, "paper_arm_freeze_ok", lambda: (True, "ok"))
    monkeypatch.setattr(CLI, "_oracle_sanity", lambda *a, **k: [])
    monkeypatch.setattr(CLI, "CACHE_ROOT", tmp_path / "cache")

    def _boom(*a, **k):
        raise AssertionError("no generation may start")

    monkeypatch.setattr(CLI, "run_scored_ablation", _boom)
    monkeypatch.setattr(CLI, "_load_dotenv", _boom)
    assert CLI.main(["--execute", "--phase", "reported", "--basis", "total_return"]) == 2
    assert "already exists" in capsys.readouterr().err


def test_execute_refuses_without_the_freeze(monkeypatch, tmp_path, capsys):
    paths = _paths(tmp_path)
    monkeypatch.setattr(CLI, "resolve_paths", lambda basis, out_root: paths)
    monkeypatch.setattr(CLI, "build_paper_prompt", lambda s: f"prompt {s}")
    monkeypatch.setattr(CLI, "paper_arm_asset_hashes", lambda: {})
    monkeypatch.setattr(CLI, "paper_arm_freeze_ok", lambda: (False, "not frozen"))
    monkeypatch.setattr(CLI, "_oracle_sanity", lambda *a, **k: [])
    monkeypatch.setattr(CLI, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(CLI, "run_scored_ablation",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("generated")))
    assert CLI.main(["--execute", "--phase", "reported", "--basis", "total_return"]) == 2
    assert "not frozen" in capsys.readouterr().err
