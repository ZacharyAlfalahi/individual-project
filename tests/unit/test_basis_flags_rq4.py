"""Basis / input / LLM-budget flags for the RQ4 Scientist drivers.

Synthetic or monkeypatched ONLY: no development panel is loaded and no experiment runs; every live
client factory here is a mock that FAILS the test if it is called when it must not be; no API key is
read (.env loading is monkeypatched wherever the live path could be reached); /data/holdout/ is never
touched. The recorded AuditReport JSONs (tests skip when they are absent — local pipeline outputs,
not shipped with the repository) and the mechanism library are read, with a synthetic
conditioning-variable availability set in place of the dev-panel column scan.

Load-bearing guarantees:
  (1) no flag = the recorded inputs, paths and output bytes (no extra JSON keys, no extra file in
      the output folder — a recorded-inputs run's LLM budget goes to a gitignored runs/ sidecar);
  (2) --basis / --audit-report / --audit-dir / --factors-dir / --out resolve with flag > basis >
      recorded, and a basis audit default resolves under <basis>/audit/full_run;
  (3) BBW-4 and every crowding bundle read from the given factors dir (thresholds.yaml untouched), and
      every driver threads factors_dir + basis into its loaders;
  (4) the LLM pre-flight prices each miss at a per-call upper bound (input max(8000, ceil(prompt
      UTF-8 bytes/2)), output = the live client's enforced ceiling) with the thresholds.yaml prices,
      refuses above a cap BEFORE any client is built, rejects invalid caps, builds a live client only
      for a model with a miss, forwards only planned misses (each at most once), halts the run on an
      unplanned call, and a zero-miss run never constructs a live client;
  (5) the planned prompt IS the prompt the wall-bound source sends (the pre-flight sees no statistic).
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import scripts.basis_inputs as BI  # noqa: E402
import run_rq4_capability_benchmark as C  # noqa: E402
import run_rq4_capability_generative as G  # noqa: E402
import run_rq4_exhaustive_benchmark as B  # noqa: E402
import run_rq4_funnel as F  # noqa: E402
import run_rq4_spend_sidecar as SPEND  # noqa: E402
from agents.scientist.researcher.cache import ResponseCache  # noqa: E402
from agents.scientist.researcher.context_builder import ALLOWED_RESEARCHER_FIELDS  # noqa: E402
from agents.scientist.researcher.llm_source import LLMResearcherSource  # noqa: E402
from agents.scientist.researcher.phase_d_client import load_model_stack  # noqa: E402
from shared.evaluation.crowding import load_crowding_factor_bundle  # noqa: E402
from shared.evaluation.thresholds import load_crowding_config  # noqa: E402


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------

def _no_live_factory(*_args, **_kwargs):
    raise AssertionError("a live client factory was called — it must not be")


def _stack(stack_key: str) -> dict:
    """{vendor: model_id} for a registered model_stack pair."""
    ms = load_model_stack()
    return {ms[stack_key][s]["vendor"]: ms[stack_key][s]["model_id"] for s in ("model_a", "model_b")}


def _output_ceiling() -> int:
    """The output ceiling every live Scientist client is built with."""
    return int(load_model_stack()["max_output_tokens"])


def _put(root: Path, prompt: str, model: str, seeds, response: str = "[]") -> None:
    cache = ResponseCache(root)
    for s in seeds:
        cache.put(prompt, model, s, response)


def _expected_usd(model: str, n_calls: int, input_tokens_per_call: int = 8000) -> float:
    price = SPEND.load_prices()[model]
    return (n_calls * input_tokens_per_call * price["input"]
            + n_calls * _output_ceiling() * price["output"]) / 1_000_000


def _synthetic_available(library) -> set:
    return {v for vs in library.variable_families.values() for v in vs}


def _elig(library, available, holding_period=1):
    return [F.evaluate(mm, strategy_family=F._STRATEGY_FAMILY, holding_period=holding_period,
                       templates=library.templates, variable_families=library.variable_families,
                       available_variables=available) for mm in library.mechanisms]


def _write_factors(d: Path, names=("bbw_factors", "mktb", "str", "mom6")) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2010-01-31", periods=3, freq="ME")
    frames = {
        "bbw_factors": {"drf_corr": [0.01, 0.02, 0.03], "crf_corr": [0.04, 0.05, 0.06],
                        "lrf_corr": [0.07, 0.08, 0.09], "drf_raw": [9.0, 9.0, 9.0]},
        "mktb": {"mktb_corr": [0.001, 0.002, 0.003], "mktb_raw": [8.0, 8.0, 8.0]},
        "str": {"str_corr": [0.1, 0.2, 0.3]},
        "mom6": {"mom6_corr": [0.4, 0.5, 0.6]},
    }
    for name in names:
        pd.DataFrame({"date": dates, **frames[name]}).to_parquet(d / f"{name}.parquet")
    return d


class _FakeLive:
    def __init__(self, name):
        self.name, self.calls = name, []

    def generate(self, prompt, *, seed):
        self.calls.append((prompt, seed))
        return "[]"

    def operational_usage(self):
        n = len(self.calls)
        return {"model_calls": n, "prompt_tokens": 100 * n, "completion_tokens": 50 * n,
                "retries": 0}


# ---------------------------------------------------------------------------------------------
# (1)/(2) flag parsing + path resolution
# ---------------------------------------------------------------------------------------------

def test_funnel_parser_defaults_are_the_recorded_run():
    a = F.build_arg_parser().parse_args([])
    assert (a.basis, a.audit_report, a.factors_dir, a.out) == (None, None, None, None)
    assert (a.max_usd_anthropic, a.max_usd_gemini) == (10.0, 10.0)
    a = F.build_arg_parser().parse_args(
        ["--basis", "total_return", "--audit-report", "r.json", "--factors-dir", "fd",
         "--out", "o", "--max-usd-anthropic", "2.5", "--max-usd-gemini", "1"])
    assert (a.basis, a.audit_report, a.factors_dir, a.out) == ("total_return", "r.json", "fd", "o")
    assert (a.max_usd_anthropic, a.max_usd_gemini) == (2.5, 1.0)
    a = F.build_arg_parser().parse_args(["--max-usd-anthropic", "0", "--max-usd-gemini", "10"])
    assert (a.max_usd_anthropic, a.max_usd_gemini) == (0.0, 10.0)   # both policy bounds inclusive
    with pytest.raises(SystemExit):
        F.build_arg_parser().parse_args(["--basis", "total"])


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "-0.5", "10.01", "1e3", "ten"])
def test_usd_cap_flags_reject_non_finite_negative_and_above_the_policy_ceiling(bad, capsys):
    # --flag=value so every value (including '-inf', which argparse would read as an option name
    # when passed as its own token) reaches the cap validator
    for flag in ("--max-usd-anthropic", "--max-usd-gemini"):
        with pytest.raises(SystemExit):
            F.build_arg_parser().parse_args([f"{flag}={bad}"])
        err = capsys.readouterr().err
        assert flag in err and ("spend cap" in err or "could not convert" in err)
        with pytest.raises(SystemExit):                          # parsed before any other work
            G.main([f"{flag}={bad}"])
        err = capsys.readouterr().err
        assert flag in err and ("spend cap" in err or "could not convert" in err)


def test_capability_and_exhaustive_parsers():
    a = C.build_arg_parser().parse_args([])
    assert (a.basis, a.audit_dir, a.factors_dir, a.out) == (None, None, None, None)
    assert a.anchors == ["drf", "mom6"] and a.embedder == "minilm"
    a = C.build_arg_parser().parse_args(["--basis", "clean", "--audit-dir", "ad",
                                         "--factors-dir", "fd", "--out", "o"])
    assert (a.basis, a.audit_dir, a.factors_dir, a.out) == ("clean", "ad", "fd", "o")
    b = B.build_arg_parser().parse_args([])
    assert (b.basis, b.audit_report, b.factors_dir, b.out) == (None,) * 4
    b = B.build_arg_parser().parse_args(["--basis", "total_return"])
    assert b.basis == "total_return"
    for parser in (C.build_arg_parser(), B.build_arg_parser()):
        with pytest.raises(SystemExit):
            parser.parse_args(["--basis", "tr"])


def test_funnel_paths_without_flags_are_recorded():
    assert F.resolve_funnel_paths() == {"basis": None, "audit": F._STR_AUDIT,
                                        "factors_dir": None, "out": F._DEFAULT_OUT}
    assert F._STR_AUDIT == _REPO_ROOT / "results/auditor/str_corrected/str_report.json"
    assert F._DEFAULT_OUT == _REPO_ROOT / "results/scientist/rq4_funnel"
    assert F.is_recorded_inputs(None, None, None) is True
    assert F.is_recorded_inputs("clean", None, None) is False


@pytest.mark.parametrize("basis", BI.BASES)
def test_basis_defaults_live_under_the_consistent_basis_root(basis):
    root = BI.RESULTS_ROOT / basis
    audit = root / "audit" / "full_run"                  # where the basis full audit writes
    f = F.resolve_funnel_paths(basis)
    assert f["audit"] == audit / "str_report.json"
    assert f["factors_dir"] == root / "factors" == BI.factors_dir(basis)
    assert f["out"] == root / "rq4" / "funnel"
    c = C.resolve_capability_paths(basis)
    assert (c["audit"], c["factors_dir"], c["out"]) == (audit, root / "factors",
                                                       root / "rq4" / "capability")
    g = G.resolve_generative_paths(basis)
    assert (g["audit"], g["out"]) == (audit, root / "rq4" / "capability_generative")
    e = B.resolve_exhaustive_paths(basis)
    assert e["audit"] == audit / "str_report.json"
    assert e["out_json"] == root / "rq4" / "exhaustive_benchmark" / "rq4_exhaustive_benchmark.json"
    assert e["out_md"] == root / "rq4" / "exhaustive_benchmark" / "rq4_exhaustive_benchmark.md"
    # basis audit outputs present locally: every default names a real report
    if (root / "audit").is_dir():
        assert f["audit"].is_file() and e["audit"].is_file()
        assert all((c["audit"] / f"{a}_report.json").is_file() for a in ("drf", "mom6"))


def test_benchmark_paths_without_flags_are_recorded():
    c = C.resolve_capability_paths()
    assert (c["audit"], c["factors_dir"], c["out"]) == (       # recorded audit resolves per anchor
        None, None, _REPO_ROOT / "results/scientist/rq4_capability")
    g = G.resolve_generative_paths()
    assert (g["audit"], g["factors_dir"], g["out"]) == (
        None, None, _REPO_ROOT / "results/scientist/rq4_capability_generative")
    e = B.resolve_exhaustive_paths()
    assert e["audit"] == F._STR_AUDIT
    assert e["factors_dir"] is None
    assert e["out_json"] == _REPO_ROOT / "results/scientist/rq4_exhaustive_benchmark.json"
    assert e["out_md"] == _REPO_ROOT / "results/rq4_exhaustive_benchmark.md"


def test_explicit_flags_override_basis_and_recorded_defaults(tmp_path):
    f = F.resolve_funnel_paths("clean", audit_report=tmp_path / "a.json",
                               factors_dir=tmp_path / "fd", out=tmp_path / "o")
    assert (f["audit"], f["factors_dir"], f["out"]) == (tmp_path / "a.json", tmp_path / "fd",
                                                       tmp_path / "o")
    f = F.resolve_funnel_paths(None, factors_dir=tmp_path)          # one flag, no basis
    assert (f["audit"], f["factors_dir"], f["out"]) == (F._STR_AUDIT, tmp_path, F._DEFAULT_OUT)
    e = B.resolve_exhaustive_paths("total_return", out=tmp_path / "o")
    assert e["out_json"].parent == e["out_md"].parent == tmp_path / "o"
    with pytest.raises(BI.BasisError):
        F.resolve_funnel_paths("total")


def test_recorded_run_budget_sidecars_live_in_a_gitignored_runs_dir():
    if not (_REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    for d in (F._BUDGET_SIDECAR_DIR, G._BUDGET_SIDECAR_DIR):
        assert d.relative_to(_REPO_ROOT).parts[0] == "runs"
        try:
            ignored = subprocess.run(["git", "check-ignore", "-q", str(d / "x.llm_budget.json")],
                                     cwd=_REPO_ROOT, check=False).returncode == 0
        except FileNotFoundError:
            pytest.skip("git not available")
        assert ignored, f"{d} is not gitignored"


# ---------------------------------------------------------------------------------------------
# (3) factors dir: BBW-4 + crowding bundles
# ---------------------------------------------------------------------------------------------

def test_bbw4_frame_reads_the_given_factors_dir(tmp_path):
    _write_factors(tmp_path)
    fr = F.bbw4_frame(tmp_path)
    assert list(fr.columns) == ["date", "mktb", "drf", "crf", "lrf"]
    assert fr["drf"].tolist() == [0.01, 0.02, 0.03]
    assert fr["mktb"].tolist() == [0.001, 0.002, 0.003]
    assert fr["lrf"].tolist() == [0.07, 0.08, 0.09]


def test_bbw4_frame_default_reads_the_recorded_dir(monkeypatch):
    seen = []

    def fake_read(path, *a, **k):
        seen.append(Path(path))
        if Path(path).name == "mktb.parquet":
            return pd.DataFrame({"date": [1], "mktb_corr": [0.0]})
        return pd.DataFrame({"date": [1], "drf_corr": [0.0], "crf_corr": [0.0], "lrf_corr": [0.0]})
    monkeypatch.setattr(F.pd, "read_parquet", fake_read)
    monkeypatch.setattr(F, "require_licensed_input", lambda path, what="": path)
    F.bbw4_frame()
    assert seen == [_REPO_ROOT / "data/development/factors/bbw_factors.parquet",
                    _REPO_ROOT / "data/development/factors/mktb.parquet"]


def test_crowding_bundles_repointed_in_code_to_the_factors_dir(tmp_path):
    _write_factors(tmp_path)
    base = F.crowding_config_for(None)
    assert base == load_crowding_config()                      # no dir = the registered config
    cfg = F.crowding_config_for(tmp_path)
    assert (cfg.factor_set, cfg.hac_lag_rule, cfg.min_obs) == (base.factor_set, base.hac_lag_rule,
                                                               base.min_obs)
    for name, (path, column) in cfg.bundles.items():
        assert Path(path) == tmp_path / Path(base.bundles[name][0]).name
        assert column == base.bundles[name][1]
    frame = load_crowding_factor_bundle(cfg)
    assert list(frame.columns) == ["date", *base.factor_set] and len(frame) == 3
    assert frame["str"].tolist() == [0.1, 0.2, 0.3]
    assert load_crowding_config() == base                      # thresholds.yaml untouched


def test_crowding_bundle_missing_from_factors_dir_fails_loud(tmp_path):
    _write_factors(tmp_path, names=("bbw_factors", "mktb"))
    with pytest.raises(FileNotFoundError, match="str.parquet"):
        F.crowding_config_for(tmp_path)


def test_corrected_parents_route_to_the_basis_loader(monkeypatch):
    import agents.auditor.ipca_differential.runner as runner

    class Loaded(Exception):
        pass

    def basis_loader(basis):
        raise Loaded(("basis", basis))

    def dev_loader():
        raise Loaded(("dev", None))
    monkeypatch.setattr(F.BI, "load_basis_inputs", basis_loader)
    monkeypatch.setattr(runner, "load_dev_inputs", dev_loader)
    for build in (F.corrected_str_parent, lambda b: C.corrected_parent("drf", b)):
        with pytest.raises(Loaded) as ei:
            build("total_return")
        assert ei.value.args[0] == ("basis", "total_return")
        with pytest.raises(Loaded) as ei:
            build(None)
        assert ei.value.args[0] == ("dev", None)


def test_capability_routes_audit_dir_and_basis(monkeypatch):
    seen = {}

    class Stop(Exception):
        pass

    def fake_case(path, **kw):
        seen["report"], seen["case_kw"] = Path(path), kw
        return types.SimpleNamespace(failed_check_ids=(), case_id="x"), None

    def fake_parent(anchor_id, basis=None):
        seen["parent"] = (anchor_id, basis)
        raise Stop
    monkeypatch.setattr(F, "build_case", fake_case)
    monkeypatch.setattr(C, "corrected_parent", fake_parent)
    with pytest.raises(Stop):
        C.run_capability("drf", basis="total_return")
    assert seen["report"] == BI.basis_dir("total_return", "audit", "full_run", "drf_report.json")
    assert seen["parent"] == ("drf", "total_return")
    assert seen["case_kw"]["corrected_run_ref"] == F.repo_relative(
        BI.basis_dir("total_return", "audit", "full_run"))
    seen.clear()
    with pytest.raises(Stop):
        C.run_capability("mom6")                                  # recorded
    assert seen["report"] == C._audit_report("mom6")
    assert seen["parent"] == ("mom6", None) and seen["case_kw"]["corrected_run_ref"] == "mom6_corrected"
    seen.clear()
    with pytest.raises(Stop):                                     # generative sibling, clients injected
        G.run_capability_generative("drf", llm_clients=[], audit_dir="/tmp/ad", basis="clean")
    assert seen["report"] == Path("/tmp/ad/drf_report.json") and seen["parent"] == ("drf", "clean")


def test_benchmarks_thread_factors_dir_and_basis_into_their_loaders(monkeypatch, tmp_path):
    """Spies on the parent loader, bbw4_frame and crowding_config_for; the heavy stages after the
    spies are never reached (Stop)."""
    library = F.load_library()
    idx = pd.DatetimeIndex(["2010-01-31"])
    panel, parent = pd.DataFrame({"date": idx}), pd.Series([0.001], index=idx)
    seen: dict = {}

    class Stop(Exception):
        pass

    def fake_case(path=None, **kw):
        return (types.SimpleNamespace(failed_check_ids=("lib_gap",), case_id="c"),
                types.SimpleNamespace(q=0.10))

    def capability_parent(anchor_id, basis=None):
        seen["parent"] = (anchor_id, basis)
        return panel, {"signal_lag": 1}, parent, 1, C._ANCHORS[anchor_id]["holding_period"]

    def str_parent(basis=None):
        seen["parent"] = ("str", basis)
        return panel, {"signal_lag": 1}, parent, 1

    def bbw4(fd=None):
        seen["bbw4"] = fd

    def crowding(fd=None):
        seen["crowding"] = fd
        raise Stop

    def delays():                                    # exhaustive: the stage right after bbw4
        raise Stop
    monkeypatch.setattr(F, "build_case", fake_case)
    monkeypatch.setattr(F, "available_conditioning_variables",
                        lambda: _synthetic_available(library))
    monkeypatch.setattr(F, "load_macro_series", lambda lo, hi: {})
    monkeypatch.setattr(F, "bbw4_frame", bbw4)
    monkeypatch.setattr(F, "crowding_config_for", crowding)
    monkeypatch.setattr(F, "corrected_str_parent", str_parent)
    monkeypatch.setattr(C, "corrected_parent", capability_parent)
    monkeypatch.setattr(B, "load_reporting_delays", delays)

    fd = tmp_path / "fd"
    runs = [
        (lambda: C.run_capability("drf", basis="total_return", factors_dir=fd),
         ("drf", "total_return"), fd),
        (lambda: C.run_capability("mom6", basis="clean"), ("mom6", "clean"), BI.factors_dir("clean")),
        (lambda: C.run_capability("drf"), ("drf", None), None),                       # recorded
        (lambda: G.run_capability_generative("drf", llm_clients=[], basis="clean", factors_dir=fd),
         ("drf", "clean"), fd),
        (lambda: G.run_capability_generative("mom6", llm_clients=[], basis="total_return"),
         ("mom6", "total_return"), BI.factors_dir("total_return")),
    ]
    for run, parent_call, factors in runs:
        seen.clear()
        with pytest.raises(Stop):
            run()
        assert seen == {"parent": parent_call, "bbw4": factors, "crowding": factors}

    seen.clear()
    with pytest.raises(Stop):
        B.main(["--basis", "total_return", "--factors-dir", str(fd)])
    assert seen == {"parent": ("str", "total_return"), "bbw4": fd}


# ---------------------------------------------------------------------------------------------
# (4) LLM budget pre-flight
# ---------------------------------------------------------------------------------------------

def test_per_call_bound_is_the_codegen_preflight_rule_and_the_live_output_ceiling(monkeypatch):
    import agents.scientist.researcher.phase_d_client as pdc
    from evaluation.codegen import preflight
    assert F.input_token_bound is preflight.input_token_bound         # one rule, one source
    assert F.input_token_bound("x" * 16_000) == 8000                 # the floor
    assert F.input_token_bound("x" * 16_001) == 8001                 # ceil(bytes / 2)
    assert F.input_token_bound("é" * 10_000) == 10_000              # UTF-8 bytes, not characters

    built = {}

    class Recorder:                                                   # no SDK, no vendor
        def __init__(self, **kw):
            built[kw["model_id"]] = kw["max_output_tokens"]
    monkeypatch.setattr(pdc, "PhaseDModelClient", Recorder)
    ms = load_model_stack()
    for stack_key in ("phase_d", "phase_f"):
        for slot in ("model_a", "model_b"):
            spec = ms[stack_key][slot]
            monkeypatch.setenv(spec["api_key_env"], "placeholder-not-a-key")
            pdc._client_from(spec, ms)
    assert set(built.values()) == {_output_ceiling()}               # the ceiling the estimate prices


def test_preflight_refuses_above_default_cap_before_any_client(tmp_path):
    pf = _stack("phase_f")
    with pytest.raises(F.LLMBudgetExceeded) as ei:
        F.prepare_budgeted_clients(["synthetic prompt"], stack_key="phase_f", k=1000,
                                   cache_root=tmp_path / "cache", client_factory=_no_live_factory)
    rec = ei.value.record
    assert rec["decision"] == "refused" and rec["live_clients_constructed"] is False
    assert rec["cache"][pf["anthropic"]] == {"hits": 0, "misses": 1000}
    row = rec["preflight_estimate"]["models"][pf["anthropic"]]
    assert row["input_tokens_upper_bound"] == 1000 * 8000
    assert row["output_tokens_per_call"] == _output_ceiling()
    assert "enforced output ceiling" in row["output_tokens_basis"]
    usd = rec["preflight_estimate"]["by_vendor_usd"]
    assert usd["anthropic"] == pytest.approx(_expected_usd(pf["anthropic"], 1000))
    assert usd["gemini"] == pytest.approx(_expected_usd(pf["gemini"], 1000))
    assert usd["anthropic"] > 10.0 > usd["gemini"]
    assert [v.split(":")[0] for v in rec["violations"]] == ["anthropic"]
    assert "outside the bound" in rec["policy"] and "reasoning tokens" in rec["policy"]
    assert not (tmp_path / "cache").exists()          # the pre-flight never creates/writes the cache


def test_preflight_input_bound_follows_each_planned_prompt(tmp_path):
    pf = _stack("phase_f")
    long_prompt = "x" * 40_000                          # a 20,000-token bound, above the floor
    with pytest.raises(F.LLMBudgetExceeded) as ei:
        F.prepare_budgeted_clients([long_prompt, "short"], stack_key="phase_f", k=3,
                                   cache_root=tmp_path / "c", client_factory=_no_live_factory,
                                   max_usd={"anthropic": 0.0})
    row = ei.value.record["preflight_estimate"]["models"][pf["anthropic"]]
    assert row["misses"] == 6 and row["input_tokens_upper_bound"] == 3 * 20_000 + 3 * 8000
    price = SPEND.load_prices()[pf["anthropic"]]
    assert row["usd"] == pytest.approx(
        ((3 * 20_000 + 3 * 8000) * price["input"] + 6 * _output_ceiling() * price["output"])
        / 1_000_000)


def test_preflight_custom_caps_refuse(tmp_path):
    with pytest.raises(F.LLMBudgetExceeded, match="gemini"):
        F.prepare_budgeted_clients(["p"], stack_key="phase_f", k=5, cache_root=tmp_path / "c",
                                   client_factory=_no_live_factory, max_usd={"gemini": 0.0})


def test_preflight_rejects_invalid_caps_before_any_client(tmp_path):
    for bad in (float("nan"), float("inf"), -1.0, 10.5):
        with pytest.raises(ValueError, match="spend cap"):
            F.prepare_budgeted_clients(["p"], stack_key="phase_f", k=1, cache_root=tmp_path / "c",
                                       client_factory=_no_live_factory, max_usd={"anthropic": bad})
    assert F.check_usd_cap("10") == 10.0 and F.check_usd_cap(0) == 0.0


def test_preflight_proceeds_below_cap_and_builds_live_clients_only_for_misses(tmp_path):
    pf = _stack("phase_f")
    cache_root = tmp_path / "cache"
    _put(cache_root, "p", pf["gemini"], range(5))                 # gemini fully cached
    requested = []

    def factory(models):
        requested.append(list(models))
        return [_FakeLive(n) for n in models]
    clients, rec = F.prepare_budgeted_clients(["p"], stack_key="phase_f", k=5,
                                              cache_root=cache_root, client_factory=factory)
    assert requested == [[pf["anthropic"]]]                       # the hit-only model is never built
    assert rec["decision"] == "live_calls_for_misses"
    assert [c.name for c in clients] == [pf["anthropic"], pf["gemini"]]
    assert rec["cache"] == {pf["anthropic"]: {"hits": 0, "misses": 5},
                            pf["gemini"]: {"hits": 5, "misses": 0}}
    usd = rec["preflight_estimate"]["by_vendor_usd"]
    assert usd["anthropic"] == pytest.approx(_expected_usd(pf["anthropic"], 5))
    assert usd["anthropic"] <= 10.0
    assert usd["gemini"] == 0.0

    anthropic, gemini = clients
    assert anthropic.generate("p", seed=0) == "[]"
    with pytest.raises(F.UnplannedCallRefused, match="unplanned"):
        anthropic.generate("p", seed=0)                           # each planned key at most once
    with pytest.raises(F.UnplannedCallRefused, match="unplanned"):
        anthropic.generate("a prompt the pre-flight never planned", seed=1)
    with pytest.raises(F.UnplannedCallRefused, match="unplanned"):
        gemini.generate("p", seed=0)                              # fully cached -> replay-only
    usage = F.llm_usage(clients)
    assert usage[pf["anthropic"]]["live_calls"] == 1
    price = SPEND.load_prices()[pf["anthropic"]]
    assert usage[pf["anthropic"]]["metered_usd"] == pytest.approx(
        (100 * price["input"] + 50 * price["output"]) / 1_000_000)
    assert usage[pf["gemini"]] == {"live_client_constructed": False, "live_calls": 0}

    with pytest.raises(RuntimeError, match="unrequested"):        # a factory must build exactly those
        F.prepare_budgeted_clients(["p"], stack_key="phase_f", k=5, cache_root=cache_root,
                                   client_factory=lambda models: [_FakeLive(n) for n in pf.values()])


def test_build_live_pair_constructs_only_the_requested_models(monkeypatch):
    import agents.scientist.researcher.phase_d_client as pdc
    built = []
    monkeypatch.setattr(F, "_load_dotenv", lambda path: None)     # never read .env in a test
    monkeypatch.setattr(pdc, "_client_from",
                        lambda spec, ms: built.append(spec["model_id"]) or
                        types.SimpleNamespace(name=spec["model_id"]))
    pf, pd_ = _stack("phase_f"), _stack("phase_d")
    assert [c.name for c in F.build_live_pair("reported", [pf["gemini"]])] == [pf["gemini"]]
    assert built == [pf["gemini"]]
    built.clear()
    assert [c.name for c in F.build_live_pair("dev", list(pd_.values()))] == list(pd_.values())
    built.clear()
    assert [c.name for c in G.build_generative_clients([pf["anthropic"]])] == [pf["anthropic"]]
    assert built == [pf["anthropic"]]


def test_zero_miss_replay_never_constructs_a_live_client(tmp_path):
    pf = _stack("phase_f")
    cache_root = tmp_path / "cache"
    for model in pf.values():
        _put(cache_root, "p", model, range(5))
    clients, rec = F.prepare_budgeted_clients(["p"], stack_key="phase_f", k=5,
                                              cache_root=cache_root,
                                              client_factory=_no_live_factory)
    assert rec["decision"] == "replay_only" and rec["live_clients_constructed"] is False
    assert all(v == {"hits": 5, "misses": 0} for v in rec["cache"].values())
    assert rec["preflight_estimate"]["by_vendor_usd"] == {"anthropic": 0.0, "gemini": 0.0}
    for c in clients:
        with pytest.raises(F.UnplannedCallRefused, match="unplanned"):
            c.generate("p", seed=0)


def test_unpriced_model_with_misses_is_refused(tmp_path):
    thresholds = tmp_path / "thresholds.yaml"
    thresholds.write_text(
        "scientist:\n  model_stack:\n    max_output_tokens: 100\n"
        "    phase_d: {model_a: {vendor: gemini, model_id: g-free}, "
        "model_b: {vendor: mistral, model_id: m-free}}\n"
        "    phase_f: {model_a: {vendor: anthropic, model_id: a-unpriced}, "
        "model_b: {vendor: gemini, model_id: g-free}}\n"
        "p1_codegen:\n  budget:\n    prices_usd_per_1m:\n"
        "      g-free: {input: 0.0, output: 0.0}\n      m-free: {input: 0.0, output: 0.0}\n",
        encoding="utf-8")
    with pytest.raises(F.LLMBudgetExceeded, match="a-unpriced"):
        F.prepare_budgeted_clients(["p"], stack_key="phase_f", k=1, cache_root=tmp_path / "c",
                                   client_factory=_no_live_factory, thresholds_path=thresholds)
    # free pair: priced at $0, no Anthropic/Gemini spend -> proceeds (mistral has no cap, $0 is fine)
    calls = []
    clients, rec = F.prepare_budgeted_clients(
        ["p"], stack_key="phase_d", k=1, cache_root=tmp_path / "c",
        client_factory=lambda models: calls.append(list(models)) or [
            types.SimpleNamespace(name=n) for n in models],
        thresholds_path=thresholds)
    assert calls == [["g-free", "m-free"]] and rec["decision"] == "live_calls_for_misses"
    row = rec["preflight_estimate"]["models"]["g-free"]
    assert row["output_tokens_basis"].startswith("scientist.model_stack.max_output_tokens")
    assert row["output_tokens_per_call"] == 100                   # the stack's own ceiling


# ---------------------------------------------------------------------------------------------
# (5) the wall: planned prompt == the source's prompt; zero-miss replay through the real source
# ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def str_case_elig_library():
    if not F._STR_AUDIT.exists():
        pytest.skip("corrected str AuditReport absent (local pipeline output, not shipped with "
                    "the repository)")
    case, _params = F.build_case()
    library = F.load_library()
    return case, _elig(library, _synthetic_available(library)), library


def test_planned_prompt_is_what_the_source_sends_and_replays_without_a_client(
        tmp_path, str_case_elig_library):
    case, elig, library = str_case_elig_library
    pf = _stack("phase_f")
    prompt = F.planned_prompt(case, elig, library, 6)
    ctx = json.loads(prompt.split("CONTEXT:\n", 1)[1])
    assert set(ctx) <= set(ALLOWED_RESEARCHER_FIELDS)              # the wall's allow-list only

    class Recording:
        name = pf["anthropic"]

        def generate(self, p, *, seed):
            self.seen = (p, seed)
            return "[]"
    rec = Recording()
    LLMResearcherSource(rec, cache=None).candidates(case, elig, library, seed=3, m=6,
                                                    model=rec.name, prompt_version="v1",
                                                    generated_at="t")
    assert rec.seen == (prompt, 3)

    cache_root = tmp_path / "cache"
    for model in pf.values():
        _put(cache_root, prompt, model, range(2))
    clients, record = F.prepare_budgeted_clients([prompt], stack_key="phase_f", k=2,
                                                 cache_root=cache_root,
                                                 client_factory=_no_live_factory)
    assert record["decision"] == "replay_only"
    for c in clients:                     # generate() would raise; every seed replays from cache
        res = F.S.run_all_seeds(LLMResearcherSource(c, cache=ResponseCache(cache_root)), case, elig,
                                library, k=2, m=6, model=c.name, prompt_version="v1",
                                generated_at="t")
        assert len(res) == 2 and c.live_calls == 0


def _stub_funnel_stages(monkeypatch, available, seen):
    idx = pd.DatetimeIndex(["2010-01-31"])

    def parent(basis=None):
        seen["basis"] = basis
        return pd.DataFrame({"date": idx}), {"signal_lag": 1}, pd.Series([0.001], index=idx), 1
    monkeypatch.setattr(F, "available_conditioning_variables", lambda: available)
    monkeypatch.setattr(F, "corrected_str_parent", parent)
    monkeypatch.setattr(F, "load_macro_series", lambda lo, hi: {})
    monkeypatch.setattr(F, "bbw4_frame", lambda fd=None: seen.__setitem__("bbw4", fd))
    monkeypatch.setattr(F, "crowding_config_for", lambda fd=None: seen.__setitem__("crowding", fd))
    monkeypatch.setattr(F, "load_crowding_factor_bundle", lambda cfg: None)
    monkeypatch.setattr(F, "load_reporting_delays", lambda: {})
    monkeypatch.setattr(F, "run_experimentalist", lambda *a, **k: types.SimpleNamespace(
        records=[], funnel={}, wrong_signed=[], advanced=()))
    monkeypatch.setattr(F, "economic_funnel", lambda records: {})
    monkeypatch.setattr(F, "agent_quality_funnel", lambda seeds: {})
    monkeypatch.setattr(F.BI, "basis_provenance", lambda b: types.SimpleNamespace(
        to_dict=lambda: {"basis": b, "panels": "stubbed"}))


@pytest.mark.parametrize("flagged", [True, False])
def test_funnel_wiring_zero_miss_replay(monkeypatch, tmp_path, flagged):
    """No client_factory: the DEFAULT live path (build_live_pair, .env) must never be reached."""
    if not F._STR_AUDIT.exists():
        pytest.skip("corrected str AuditReport absent (local pipeline output, not shipped with "
                    "the repository)")
    library = F.load_library()
    available = _synthetic_available(library)
    seen: dict = {}
    _stub_funnel_stages(monkeypatch, available, seen)

    def forbidden(*_a, **_k):
        raise AssertionError("a zero-miss run reached the live-client path (.env / client build)")
    monkeypatch.setattr(F, "build_live_pair", forbidden)
    monkeypatch.setattr(F, "_load_dotenv", forbidden)
    monkeypatch.setattr(F, "_BUDGET_SIDECAR_DIR", tmp_path / "sidecar")
    case, _params = F.build_case()
    prompt = F.planned_prompt(case, _elig(library, available), library, 2)
    models = list(_stack("phase_d").values())
    cache_root = tmp_path / "cache"
    for model in models:
        _put(cache_root, prompt, model, [0])

    kw = (dict(basis="total_return", audit_report=F._STR_AUDIT, factors_dir=tmp_path / "factors")
          if flagged else {})
    out = tmp_path / "out"
    r = F.run_rq4_funnel(phase="dev", k=1, m=2, run_rehearsal=False, out_dir=out,
                         cache_root=cache_root, **kw)
    assert r["entered"] is True and r["generation_errors"] == []
    assert r["sources"] == ["random_eligible", "retrieval_only", *[f"llm_{n}" for n in models]]
    assert sorted(p.name for p in out.iterdir()) == ["rq4_funnel_dev.json"]   # no new file in out
    written = json.loads((out / "rq4_funnel_dev.json").read_text(encoding="utf-8"))
    sidecar = tmp_path / "sidecar" / "rq4_funnel_dev.llm_budget.json"
    if flagged:
        assert seen == {"basis": "total_return", "bbw4": tmp_path / "factors",
                        "crowding": tmp_path / "factors"}
        assert written["inputs"]["basis"] == "total_return"
        assert written["inputs"]["audit_report"] == F.repo_relative(F._STR_AUDIT)
        budget = written["llm_budget"]
        assert not sidecar.exists()
    else:
        assert seen == {"basis": None, "bbw4": None, "crowding": None}
        assert "inputs" not in written and "llm_budget" not in written   # recorded keys only
        budget = json.loads(sidecar.read_text(encoding="utf-8"))
    assert budget["decision"] == "replay_only" and budget["live_clients_constructed"] is False
    assert budget["cache"] == {n: {"hits": 1, "misses": 0} for n in models}
    assert all(u["live_calls"] == 0 for u in budget["metered_usage"].values())


def test_unplanned_call_halts_the_funnel_but_a_vendor_error_stays_a_generation_error(
        monkeypatch, tmp_path):
    if not F._STR_AUDIT.exists():
        pytest.skip("corrected str AuditReport absent (local pipeline output, not shipped with "
                    "the repository)")
    library = F.load_library()
    _stub_funnel_stages(monkeypatch, _synthetic_available(library), {})
    model = _stack("phase_d")["gemini"]
    with pytest.raises(F.UnplannedCallRefused):
        F.run_rq4_funnel(phase="dev", k=1, m=2, run_rehearsal=False, cache_root=tmp_path / "empty",
                         llm_clients=[F.BudgetedClient(model)])     # replay-only, cache empty

    class RateLimited:
        name = model

        def generate(self, prompt, *, seed):
            raise RuntimeError("rate limited")
    r = F.run_rq4_funnel(phase="dev", k=1, m=2, run_rehearsal=False, cache_root=tmp_path / "empty",
                         llm_clients=[RateLimited()])
    assert [e["source"] for e in r["generation_errors"]] == [f"llm_{model}"]


def test_funnel_refuses_before_loading_the_parent(monkeypatch, tmp_path):
    if not F._STR_AUDIT.exists():
        pytest.skip("corrected str AuditReport absent (local pipeline output, not shipped with "
                    "the repository)")
    library = F.load_library()
    seen: dict = {}
    _stub_funnel_stages(monkeypatch, _synthetic_available(library), seen)
    with pytest.raises(F.LLMBudgetExceeded):
        F.run_rq4_funnel(phase="reported", k=5, m=2, run_rehearsal=False,
                         cache_root=tmp_path / "empty", max_usd={"anthropic": 0.0},
                         client_factory=_no_live_factory)
    assert "basis" not in seen                                    # parent never loaded


def test_funnel_main_threads_flags_and_refusal_exit(monkeypatch):
    captured = {}

    def fake_run(**kw):
        captured.update(kw)
        return {"entered": False, "note": "stubbed"}
    monkeypatch.setattr(F, "run_rq4_funnel", fake_run)
    assert F.main(["--basis", "clean", "--no-rehearsal", "--max-usd-gemini", "3"]) == 0
    assert captured["basis"] == "clean" and captured["out_dir"] == BI.basis_dir("clean", "rq4", "funnel")
    assert captured["max_usd"] == {"anthropic": 10.0, "gemini": 3.0}
    captured.clear()
    assert F.main(["--no-rehearsal"]) == 0
    assert (captured["basis"], captured["audit_report"], captured["factors_dir"]) == (None, None, None)
    assert captured["out_dir"] == F._DEFAULT_OUT

    def refuse(**kw):
        raise F.LLMBudgetExceeded("over cap", {"decision": "refused"})
    monkeypatch.setattr(F, "run_rq4_funnel", refuse)
    assert F.main(["--phase", "reported"]) == 2


def test_generative_preflight_spans_every_anchor_and_replays(monkeypatch, tmp_path):
    pf = _stack("phase_f")
    monkeypatch.setattr(G, "planned_prompts",
                        lambda anchors, *, audit_dir=None, m=6: [f"prompt {a}" for a in anchors])
    monkeypatch.setattr(G, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(G, "build_generative_clients", _no_live_factory)
    for model in pf.values():
        for anchor in ("drf", "mom6"):
            _put(tmp_path / "cache", f"prompt {anchor}", model, range(5))
    clients, rec = G.prepare_generative_clients(["drf", "mom6"])
    assert rec["decision"] == "replay_only" and rec["n_prompts"] == 2
    assert all(v == {"hits": 10, "misses": 0} for v in rec["cache"].values())
    # a new anchor's prompt misses -> priced; with a zero cap it is refused before any client
    with pytest.raises(F.LLMBudgetExceeded):
        G.prepare_generative_clients(["drf", "str"], max_usd={"anthropic": 0.0})


def _stub_generative_stages(monkeypatch, tmp_path, available):
    """Every stage of run_capability_generative except case building, eligibility, the wall-bound
    prompt and generation — so the prompt a source sends is the real one."""
    idx = pd.DatetimeIndex(["2010-01-31"])
    monkeypatch.setattr(F, "available_conditioning_variables", lambda: available)
    monkeypatch.setattr(C, "corrected_parent", lambda anchor_id, basis=None: (
        pd.DataFrame({"date": idx}), {"signal_lag": 1}, pd.Series([0.001], index=idx), 1,
        C._ANCHORS[anchor_id]["holding_period"]))
    monkeypatch.setattr(F, "load_macro_series", lambda lo, hi: {})
    monkeypatch.setattr(F, "bbw4_frame", lambda fd=None: None)
    monkeypatch.setattr(F, "crowding_config_for", lambda fd=None: None)
    monkeypatch.setattr(G, "load_crowding_factor_bundle", lambda cfg: None)
    monkeypatch.setattr(G, "load_reporting_delays", lambda: {})
    monkeypatch.setattr(G, "run_experimentalist", lambda *a, **k: types.SimpleNamespace(
        records=[], funnel={}, wrong_signed=[], advanced=()))
    monkeypatch.setattr(G, "economic_funnel", lambda records: {})
    monkeypatch.setattr(G, "agent_quality_funnel", lambda seeds: {})
    monkeypatch.setattr(B, "canonical_proposals", lambda case, elig, library: [])
    monkeypatch.setattr(B, "evaluate_batch", lambda *a, **k: [])
    monkeypatch.setattr(G, "_CACHE_ROOT", tmp_path / "gen_cache")
    monkeypatch.setattr(G, "_BUDGET_SIDECAR_DIR", tmp_path / "gen_sidecar")


@pytest.mark.parametrize("flagged", [False, True])
def test_generative_planned_prompt_is_the_prompt_sent_and_budget_placement(
        monkeypatch, tmp_path, flagged):
    """planned_prompts is NOT stubbed: the pre-flight's planned key must be the key the source
    sends (a mismatch raises UnplannedCallRefused, which halts the run)."""
    if not C._audit_report("drf").exists():
        pytest.skip("corrected drf AuditReport absent (local pipeline output, not shipped with "
                    "the repository)")
    library = F.load_library()
    _stub_generative_stages(monkeypatch, tmp_path, _synthetic_available(library))
    pf = _stack("phase_f")
    requested, built = [], {}

    def factory(models):
        requested.append(list(models))
        built.update({n: _FakeLive(n) for n in models})
        return list(built.values())
    kw = {"audit_dir": C._audit_report("drf").parent} if flagged else {}
    r = G.run_capability_generative("drf", embedder_mode="offline", k=2, m=6,
                                    client_factory=factory, **kw)
    (planned,) = G.planned_prompts(["drf"], audit_dir=C._audit_report("drf").parent, m=6)
    assert requested == [list(pf.values())]
    assert {n: c.calls for n, c in built.items()} == {n: [(planned, 0), (planned, 1)]
                                                      for n in pf.values()}
    assert r["generation_errors"] == [] and r["generative_models"] == list(pf.values())
    sidecar = (tmp_path / "gen_sidecar" /
               "rq4_capability_generative_drf_offline.llm_budget.json")
    if flagged:
        budget = r["llm_budget"]
        assert not sidecar.exists()
    else:
        assert "llm_budget" not in r and "inputs" not in r            # recorded keys only
        budget = json.loads(sidecar.read_text(encoding="utf-8"))
    assert budget["decision"] == "live_calls_for_misses"
    assert all(u["live_calls"] == 2 for u in budget["metered_usage"].values())


def test_unplanned_call_halts_the_generative_sibling(monkeypatch, tmp_path):
    if not C._audit_report("drf").exists():
        pytest.skip("corrected drf AuditReport absent (local pipeline output, not shipped with "
                    "the repository)")
    library = F.load_library()
    _stub_generative_stages(monkeypatch, tmp_path, _synthetic_available(library))
    with pytest.raises(F.UnplannedCallRefused):
        G.run_capability_generative("drf", embedder_mode="offline", k=1,
                                    llm_clients=[F.BudgetedClient(_stack("phase_f")["anthropic"])])


def test_generative_main_threads_flags_and_records_budget(monkeypatch, tmp_path):
    pf = _stack("phase_f")
    monkeypatch.setattr(G, "planned_prompts",
                        lambda anchors, *, audit_dir=None, m=6: [f"prompt {a}" for a in anchors])
    monkeypatch.setattr(G, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(G, "_BUDGET_SIDECAR_DIR", tmp_path / "sidecar")
    monkeypatch.setattr(G, "build_generative_clients", _no_live_factory)
    for model in pf.values():
        _put(tmp_path / "cache", "prompt drf", model, range(5))
    captured = {}

    def fake_run(anchor_id, **kw):
        captured.update(kw, anchor_id=anchor_id)
        return {"entered_would_be": False, "holding_period": 1, "direction": 1,
                "corrected_parent_mean_per_month": 0.0, "n_eligible_mechanisms": 0, "reports": {},
                "generation_errors": [], "canonical_ceiling": None}
    monkeypatch.setattr(G, "run_capability_generative", fake_run)
    sidecar = tmp_path / "sidecar" / "rq4_capability_generative_drf_offline.llm_budget.json"

    out = tmp_path / "out"                                         # flagged: budget embedded
    rc = G.main(["--anchors", "drf", "--embedder", "offline", "--basis", "clean",
                 "--factors-dir", str(tmp_path / "fd"), "--out", str(out)])
    assert rc == 0
    assert (captured["anchor_id"], captured["basis"], captured["factors_dir"]) == (
        "drf", "clean", str(tmp_path / "fd"))
    assert [c.name for c in captured["llm_clients"]] == list(pf.values())
    written = json.loads(G.output_path(out, "drf", "offline").read_text(encoding="utf-8"))
    assert written["llm_budget"]["decision"] == "replay_only"
    assert all(u["live_calls"] == 0 for u in written["llm_budget"]["metered_usage_cumulative"].values())
    assert not sidecar.exists()

    captured.clear()                                               # recorded: budget to the sidecar
    out2 = tmp_path / "out2"
    assert G.main(["--anchors", "drf", "--embedder", "offline", "--out", str(out2)]) == 0
    assert (captured["basis"], captured["audit_dir"], captured["factors_dir"]) == (None, None, None)
    written = json.loads(G.output_path(out2, "drf", "offline").read_text(encoding="utf-8"))
    assert "llm_budget" not in written
    assert sorted(p.name for p in out2.iterdir()) == [G.output_path(out2, "drf", "offline").name]
    budget = json.loads(sidecar.read_text(encoding="utf-8"))
    assert budget["decision"] == "replay_only" and "metered_usage_cumulative" in budget

    def refuse(*a, **k):
        raise F.LLMBudgetExceeded("over cap", {"decision": "refused"})
    monkeypatch.setattr(G, "prepare_generative_clients", refuse)
    captured.clear()
    assert G.main(["--anchors", "drf", "--out", str(out)]) == 2 and captured == {}
