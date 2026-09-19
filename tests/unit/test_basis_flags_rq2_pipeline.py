"""Basis flags for the RQ2 + pipeline scripts.

Covers scripts/run_quant.py and scripts/run_p1_codegen.py (+ evaluation/codegen/preflight.py): flag
parsing, path resolution with and without --basis (no flag must
resolve to the default paths exactly), and the P1 LLM pre-flight (thresholds prices, $10 vendor caps,
pure cache replay constructs no live client).

HERMETIC: no panel is read, no experiment runs, no LLM/API call is made — every loader, the panel
export, the scored ablation and the vendor client factory are monkeypatched; caches are tmp dirs.
"""

from __future__ import annotations

import hashlib
import json
import math
import types
from pathlib import Path

import pytest
import yaml

import run_p1_codegen as CLI
import run_quant
from agents.quant.library.run_config import corrected
from agents.scientist.researcher.cache import ResponseCache
from evaluation.codegen import live_client as L
from evaluation.codegen import preflight as P
from evaluation.codegen.panel_export import CODEGEN_PANEL
from evaluation.codegen.runner import load_models

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BI = run_quant.basis_inputs          # the basis_inputs module object the scripts actually use
PRICES = yaml.safe_load((REPO_ROOT / "docs" / "thresholds.yaml").read_text())["p1_codegen"]["budget"]
CLAUDE, GEMINI_F = "claude-sonnet-4-6", "gemini-3.5-flash"


class _Stop(Exception):
    """Sentinel to stop a driver right after the call under test."""


# ==========================================================================
# run_quant
# ==========================================================================

def test_run_quant_parser_defaults_and_basis_choices():
    assert run_quant.build_parser().parse_args([]).basis is None
    for basis in BI.BASES:
        assert run_quant.build_parser().parse_args(["--basis", basis]).basis == basis
    with pytest.raises(SystemExit):
        run_quant.build_parser().parse_args(["--basis", "dirty"])


def test_run_quant_base_panel_resolution():
    # No basis = the default panel (P4 reads run_quant.BASE_PANEL directly).
    assert run_quant.resolve_base_panel(None) == run_quant.BASE_PANEL
    assert run_quant.BASE_PANEL == REPO_ROOT / "data/development/monthly_panel_total_return.parquet"
    assert run_quant.resolve_base_panel("total_return") == (
        REPO_ROOT / "data/development/monthly_panel_total_return_default_flat.parquet")
    assert run_quant.resolve_base_panel("clean") == REPO_ROOT / "data/development/monthly_panel_maximal.parquet"
    with pytest.raises(BI.BasisError):
        run_quant.resolve_base_panel("dirty")


def test_run_quant_default_out_dir():
    assert run_quant.default_out_dir(None) == REPO_ROOT / "results/quant/run"
    assert run_quant.default_out_dir("clean") == (
        REPO_ROOT / "results/consistent_basis/clean/quant/run")


def test_run_quant_run_log_default_has_no_basis_keys_and_basis_records_sha(monkeypatch, tmp_path):
    monkeypatch.setattr(run_quant, "_thresholds_sha256", lambda: "th")
    subs = types.SimpleNamespace(version="v1")

    default = run_quant.build_run_log(subs, corrected(), ["str"], [])
    assert default["base_panel"] == "data/development/monthly_panel_total_return.parquet"
    assert "basis" not in default and "base_panel_sha256" not in default     # no basis keys

    fake_panel = tmp_path / "panel.parquet"
    fake_panel.write_bytes(b"synthetic panel bytes")
    monkeypatch.setitem(BI.PANELS, "clean", (fake_panel, tmp_path / "profiles.parquet"))
    log = run_quant.build_run_log(subs, corrected(), ["str"], [], basis="clean")
    assert log["basis"] == "clean"
    assert log["base_panel"] == str(fake_panel)
    assert log["base_panel_sha256"] == hashlib.sha256(b"synthetic panel bytes").hexdigest()
    assert set(log) - set(default) == {"basis", "base_panel_sha256"}


@pytest.mark.parametrize("basis", [None, "total_return", "clean"])
def test_run_quant_run_all_reads_the_resolved_panel(monkeypatch, basis):
    seen = {}

    def fake_load_inputs(run_config, *, base_panel=None):
        seen["base_panel"] = base_panel
        raise _Stop

    monkeypatch.setattr(run_quant, "load_inputs", fake_load_inputs)
    with pytest.raises(_Stop):
        run_quant.run_all(("str",), basis=basis)
    assert seen["base_panel"] == run_quant.resolve_base_panel(basis)


def test_run_quant_main_threads_basis(monkeypatch):
    seen = {}
    monkeypatch.setattr(run_quant, "run_all", lambda anchors, **kw: seen.update(kw, anchors=anchors) or 0)
    assert run_quant.main(["--anchor", "mom6", "--basis", "total_return"]) == 0
    assert seen == {"anchors": ("mom6",), "out_dir": None, "basis": "total_return"}
    run_quant.main([])
    assert seen["basis"] is None and seen["anchors"] == run_quant.ANCHORS


# ==========================================================================
# run_p1_codegen — flags + paths
# ==========================================================================

def test_p1_parser_defaults():
    a = CLI.build_parser().parse_args([])
    assert a.dry_run is True and a.phase == "dev"
    assert a.basis is None and a.factors_dir is None and a.scratch is None
    assert a.max_usd_anthropic == 10.0 and a.max_usd_gemini == 10.0
    b = CLI.build_parser().parse_args(["--execute", "--basis", "clean", "--factors-dir", "f",
                                       "--max-usd-anthropic", "2.5", "--max-usd-gemini", "1"])
    assert (b.dry_run, b.basis, b.factors_dir, b.max_usd_anthropic, b.max_usd_gemini) == (
        False, "clean", "f", 2.5, 1.0)


def test_p1_paths_without_basis_are_defaults():
    p = CLI.resolve_paths(None, None, None)
    assert p.scratch == REPO_ROOT / "data/development/codegen"
    assert p.panel_out == CODEGEN_PANEL == REPO_ROOT / "data/development/codegen/engine_panel_corr.parquet"
    assert p.source_panel is None and p.factors_dir is None      # panel_export / ORACLES own resolution
    assert p.sandbox_root == REPO_ROOT / "runs/p1_codegen/sandbox"


@pytest.mark.parametrize("basis", ["total_return", "clean"])
def test_p1_paths_with_basis_live_under_the_results_tree(basis):
    root = REPO_ROOT / "results/consistent_basis" / basis / "codegen"
    p = CLI.resolve_paths(basis, None, None)
    assert p.scratch == root
    assert p.panel_out == root / "engine_panel_corr.parquet"
    assert p.source_panel == BI.PANELS[basis][0]
    assert p.factors_dir == REPO_ROOT / "results/consistent_basis" / basis / "factors"
    assert p.sandbox_root == root / "sandbox"
    explicit = CLI.resolve_paths(basis, "/tmp/s", "/tmp/f")
    assert explicit.scratch == Path("/tmp/s") and explicit.factors_dir == Path("/tmp/f")
    assert explicit.panel_out == Path("/tmp/s/engine_panel_corr.parquet")     # --scratch isolates all outputs
    assert explicit.sandbox_root == Path("/tmp/s/sandbox")
    default = CLI.resolve_paths(None, "/tmp/s", None)
    assert default.panel_out == Path("/tmp/s/engine_panel_corr.parquet")
    assert default.sandbox_root == Path("/tmp/s/sandbox")


def test_oracle_path_relocates_by_file_name(tmp_path):
    from evaluation.codegen.oracles import oracle_path
    assert oracle_path("drf") == REPO_ROOT / "data/development/factors/bbw_factors.parquet"
    assert oracle_path("drf", tmp_path) == tmp_path / "bbw_factors.parquet"
    assert oracle_path("mom6", tmp_path) == tmp_path / "mom6.parquet"


# ==========================================================================
# preflight — estimate, caps, replay client
# ==========================================================================

def _prompt(strategy: str) -> str:
    return f"fake prompt for {strategy}"


def _usd(model_id: str, n_in: int, n_out: int) -> float:
    p = PRICES["prices_usd_per_1m"][model_id]
    return n_in / 1e6 * p["input"] + n_out / 1e6 * p["output"]


def _reported_models():
    return load_models("reported")


def test_census_splits_hits_and_misses_on_the_generate_once_key(tmp_path):
    cache = ResponseCache(tmp_path)
    cache.put(_prompt("s0"), CLAUDE, 0, "cached")
    cache.put(_prompt("s0"), CLAUDE, 1, "other seed — not a hit at seed 0")
    hits, misses = P.cache_census(["s0", "s1"], _reported_models(), cache, prompt_fn=_prompt)
    assert [(h["strategy"], h["model_id"]) for h in hits] == [("s0", CLAUDE)]
    assert sorted((m["strategy"], m["model_id"]) for m in misses) == [
        ("s0", GEMINI_F), ("s1", CLAUDE), ("s1", GEMINI_F)]


def test_estimate_bounds_input_from_the_prompt_and_output_at_the_enforced_cap(tmp_path):
    cache = ResponseCache(tmp_path)
    cache.put("unrelated a", CLAUDE, 0, "x" * 400)             # short cached responses never lower the bound
    long_prompt = "z" * 40_000                                  # 40,000 bytes / 2 -> 20,000 tokens > the floor

    def prompt_fn(strategy):
        return long_prompt if strategy == "long" else _prompt(strategy)

    pre = P.run_preflight(["s0", "long"], _reported_models(), cache_root=tmp_path, budget=L.load_budget(),
                          caps_usd={"anthropic": 10.0, "gemini": 10.0}, prompt_fn=prompt_fn)
    cap = PRICES["per_call_output_token_cap"]
    assert cap == 16000
    for model in (CLAUDE, GEMINI_F):
        rec = pre["estimate"]["per_model"][model]
        assert rec["calls"] == 2 and rec["input_tokens"] == 8000 + 20000
        assert rec["output_tokens_per_call"] == cap
        assert rec["usd"] == pytest.approx(_usd(model, 8000, cap) + _usd(model, 20000, cap), abs=1e-6)
    claude, gem = (pre["estimate"]["per_model"][m]["usd"] for m in (CLAUDE, GEMINI_F))
    assert pre["estimate"]["per_vendor_usd"] == {"anthropic": claude, "gemini": gem}
    assert pre["cache"] == {"hits": 0, "misses": 4, "hit_rows": [], "miss_rows": pre["cache"]["miss_rows"]}
    assert pre["refused"] is False and pre["live_models"] == sorted([CLAUDE, GEMINI_F])


def test_input_token_bound_never_below_the_floor():
    assert P.input_token_bound("short") == P.INPUT_TOKENS_UPPER_BOUND == 8000
    assert P.input_token_bound("é" * 10_000) == 10_000          # 20,000 UTF-8 bytes / 2


def test_preflight_refuses_just_above_10_usd_and_proceeds_just_below(tmp_path):
    per_call = _usd(CLAUDE, 8000, 16000)                       # short prompt -> input floor; output at the cap
    n_under = math.floor(10.0 / per_call)
    caps = {"anthropic": CLI.DEFAULT_MAX_USD, "gemini": CLI.DEFAULT_MAX_USD}
    claude_only = [m for m in _reported_models() if m["model_id"] == CLAUDE]

    def pre(n):
        return P.run_preflight([f"s{i}" for i in range(n)], claude_only, cache_root=tmp_path,
                               budget=L.load_budget(), caps_usd=caps, prompt_fn=_prompt)

    under, over = pre(n_under), pre(n_under + 1)
    assert under["estimate"]["per_vendor_usd"]["anthropic"] <= 10.0
    assert under["refused"] is False and under["violations"] == []
    assert over["estimate"]["per_vendor_usd"]["anthropic"] > 10.0
    assert over["refused"] is True and "anthropic" in over["violations"][0]


def test_gemini_cap_is_independent_of_anthropic(tmp_path):
    gem_only = [m for m in _reported_models() if m["model_id"] == GEMINI_F]
    pre = P.run_preflight(["s0", "s1"], gem_only, cache_root=tmp_path, budget=L.load_budget(),
                          caps_usd={"anthropic": 10.0, "gemini": 0.01}, prompt_fn=_prompt)
    assert pre["refused"] is True and pre["violations"][0].startswith("gemini")


def test_uncapped_priced_vendor_is_a_violation_and_free_vendor_is_not():
    assert P.cap_violations({"mistral": 0.0}, {"anthropic": 10.0}) == []
    assert P.cap_violations({"mistral": 0.5}, {"anthropic": 10.0}) != []
    assert P.cap_violations({"anthropic": 10.0}, {"anthropic": 10.0}) == []


def test_replay_client_halts_instead_of_generating():
    client = P.CacheReplayClient(CLAUDE)
    with pytest.raises(L.BudgetExceededError):                # the ablation re-raises this family
        client.generate("p", seed=0)


def test_metered_by_model_sums_budget_records():
    budget = L.load_budget()
    budget.charge(CLAUDE, 100, 10)
    budget.charge(CLAUDE, 200, 20)
    m = P.metered_by_model(budget)[CLAUDE]
    assert (m["calls"], m["prompt_tokens"], m["completion_tokens"]) == (2, 300, 30)
    assert m["usd_estimate"] == pytest.approx(_usd(CLAUDE, 300, 30), abs=1e-6)


# ==========================================================================
# run_p1_codegen — execute path, fully stubbed
# ==========================================================================

def _stub_cli(monkeypatch, tmp_path, strategies):
    monkeypatch.setattr(CLI, "detect_mechanism", lambda: "stub")
    monkeypatch.setattr(CLI, "prompt_asset_hashes", lambda: {})
    monkeypatch.setattr(CLI, "run_ablation", lambda *a, **k: {"prompt_sha256": {}, "models": []})
    monkeypatch.setattr(CLI, "_oracle_sanity", lambda *a, **k: [])
    monkeypatch.setattr(CLI, "contract_freeze_ok", lambda: (True, "stubbed freeze"))
    monkeypatch.setattr(CLI, "build_prompt", _prompt)
    monkeypatch.setattr(CLI, "STRATEGIES", tuple(strategies))
    monkeypatch.setattr(CLI, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(BI, "sha256", lambda p: "stub-sha")   # never hash a real panel
    for env in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "MISTRAL_API_KEY"):
        monkeypatch.delenv(env, raising=False)


def _boom(what):
    def f(*a, **k):
        raise AssertionError(f"{what} must not be called")
    return f


def _fake_ablation(record):
    """Stands in for run_scored_ablation: builds each client via the CLI's factory and replays the
    cache exactly like generate_once (hit -> no call)."""
    def fake(strategies, *, phase, client_factory, budget, panel_path, cache_root,
             sandbox_root, factors_dir):
        record.update(panel_path=panel_path, cache_root=cache_root, sandbox_root=sandbox_root,
                      factors_dir=factors_dir, clients=[])
        cache = ResponseCache(cache_root)
        for s in strategies:
            for model in load_models(phase):
                client = client_factory(model)
                record["clients"].append(type(client).__name__)
                prompt = CLI.build_prompt(s)
                if cache.get(prompt, model["model_id"], 0) is None:
                    cache.put(prompt, model["model_id"], 0, client.generate(prompt, seed=0))
        return {"verdict_counts": {}, "reportable": False, "sku_match": True,
                "budget": budget.summary()}
    return fake


def test_zero_miss_replay_constructs_no_live_client_and_reads_no_credentials(monkeypatch, tmp_path):
    strategies = ("drf", "str")
    _stub_cli(monkeypatch, tmp_path, strategies)
    cache = ResponseCache(tmp_path / "cache")
    for s in strategies:
        for m in _reported_models():
            cache.put(_prompt(s), m["model_id"], 0, "```python\npass\n```")

    monkeypatch.setattr(P, "build_codegen_factory", _boom("build_codegen_factory"))
    monkeypatch.setattr(L.LiveCodegenClient, "__init__", _boom("LiveCodegenClient"))
    monkeypatch.setattr(CLI, "_load_dotenv", _boom("_load_dotenv"))
    exported = {}
    monkeypatch.setattr(CLI, "export_codegen_panel", lambda out, *, source_panel: (
        exported.update(out=out, source_panel=source_panel) or (out, "panelsha")))
    seen = {}
    monkeypatch.setattr(CLI, "run_scored_ablation", _fake_ablation(seen))

    scratch = tmp_path / "scratch"
    rc = CLI.main(["--execute", "--phase", "reported", "--basis", "total_return",
                   "--scratch", str(scratch)])
    assert rc == 0
    assert seen["clients"] == ["CacheReplayClient"] * 4
    # an explicit --scratch isolates every output: the panel export and the sandbox go under it, not the tree
    assert exported == {"out": scratch / "engine_panel_corr.parquet",
                        "source_panel": BI.PANELS["total_return"][0]}
    assert seen["factors_dir"] == BI.factors_dir("total_return")
    assert seen["sandbox_root"] == scratch / "sandbox"

    out = json.loads((scratch / "p1_results_reported.json").read_text())
    pre = out["llm_policy"]["preflight"]
    assert (pre["cache"]["hits"], pre["cache"]["misses"]) == (4, 0)
    assert pre["live_models"] == [] and pre["refused"] is False
    assert pre["estimate"]["total_usd"] == 0.0
    assert out["llm_policy"]["metered_by_model"] == {}
    assert out["inputs"]["basis"] == "total_return"
    assert out["inputs"]["source_panel_sha256"] == "stub-sha"


def test_cli_refuses_over_cap_before_any_client_or_export(monkeypatch, tmp_path, capsys):
    _stub_cli(monkeypatch, tmp_path, ("drf", "str"))           # empty cache -> 4 misses
    monkeypatch.setattr(P, "build_codegen_factory", _boom("build_codegen_factory"))   # the factory is resolved in preflight
    for name in ("_load_dotenv", "export_codegen_panel", "run_scored_ablation"):
        monkeypatch.setattr(CLI, name, _boom(name))
    monkeypatch.setattr(L.LiveCodegenClient, "__init__", _boom("LiveCodegenClient"))
    assert 2 * _usd(CLAUDE, 8000, 16000) > 0.5                  # precondition for the cap below
    rc = CLI.main(["--execute", "--phase", "reported", "--scratch", str(tmp_path / "s"),
                   "--max-usd-anthropic", "0.5"])
    assert rc == 2
    assert "REFUSED (pre-flight, no call made)" in capsys.readouterr().err


def test_cli_with_misses_under_cap_builds_live_clients_only_for_missing_models(monkeypatch, tmp_path):
    _stub_cli(monkeypatch, tmp_path, ("drf",))
    cache = ResponseCache(tmp_path / "cache")
    cache.put(_prompt("drf"), "mistral-small-latest", 0, "cached")   # mistral hit, gemini miss
    monkeypatch.setattr(CLI, "_load_dotenv", lambda path: None)
    monkeypatch.setenv("GEMINI_API_KEY", "test-placeholder-not-a-key")

    class _FakeLive:
        def __init__(self, model):
            self.name = model["model_id"]
            self.last_model_version = None

        def generate(self, prompt, *, seed):
            return "```python\npass\n```"

    built = []
    monkeypatch.setattr(P, "build_codegen_factory",
                        lambda *, budget, temperature: built.append(budget) or _FakeLive)
    monkeypatch.setattr(CLI, "export_codegen_panel", lambda out, *, source_panel: (out, "sha"))
    seen = {}
    monkeypatch.setattr(CLI, "run_scored_ablation", _fake_ablation(seen))

    rc = CLI.main(["--execute", "--phase", "dev", "--scratch", str(tmp_path / "s")])
    assert rc == 0 and len(built) == 1
    assert sorted(seen["clients"]) == ["CacheReplayClient", "_FakeLive"]
    assert seen["factors_dir"] is None                                   # no basis: the default oracles
    assert seen["sandbox_root"] == tmp_path / "s" / "sandbox"             # --scratch isolates the sandbox
    pre = json.loads((tmp_path / "s" / "p1_results_dev.json").read_text())["llm_policy"]["preflight"]
    assert (pre["cache"]["hits"], pre["cache"]["misses"]) == (1, 1)
    assert pre["live_models"] == ["gemini-3.1-flash-lite"]


def test_cli_missing_key_for_a_live_model_refuses(monkeypatch, tmp_path):
    _stub_cli(monkeypatch, tmp_path, ("drf",))
    monkeypatch.setattr(CLI, "_load_dotenv", lambda path: None)
    monkeypatch.setattr(P, "build_codegen_factory", _boom("build_codegen_factory"))
    for name in ("export_codegen_panel", "run_scored_ablation"):
        monkeypatch.setattr(CLI, name, _boom(name))
    assert CLI.main(["--execute", "--phase", "dev", "--scratch", str(tmp_path / "s")]) == 2
