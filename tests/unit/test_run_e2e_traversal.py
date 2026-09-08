"""Unit tests for scripts/run_e2e_traversal.py — the integrated end-to-end
traversal driver. Pure-core tests with injected stubs and tmp_path artefacts;
no real data, no network (conventions of test_run_quant / test_run_reporter_driver)."""

from __future__ import annotations

import dataclasses
import json
import types
from pathlib import Path

import pandas as pd
import pytest

import run_e2e_traversal as drv
from agents.quant.config.runner import StrategyResult


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _write(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def _fake_spec(name="Momentum"):
    return types.SimpleNamespace(
        header=types.SimpleNamespace(strategy_label=types.SimpleNamespace(value=name)))


def _refused_adapt(codes=("REVIEW_REQUIRED",)):
    return types.SimpleNamespace(
        refused=True,
        refusals=[types.SimpleNamespace(code=types.SimpleNamespace(value=c)) for c in codes],
        leg_calls=[],
        to_dict=lambda: {"refused": True})


def _compiled_adapt():
    return types.SimpleNamespace(refused=False, refusals=[], leg_calls=[],
                                 to_dict=lambda: {"refused": False})


def _strategy_result():
    kwargs = {}
    for f in dataclasses.fields(StrategyResult):
        kwargs[f.name] = None
    kwargs.update(
        strategy_label="synthetic",
        monthly_returns=pd.DataFrame({"date": [], "ret": []}),
        summary={"average": 0.001, "t_stat": 2.0, "sharpe": 0.5, "n_months": 12,
                 "annualised_average": 0.012, "bumpiness": 0.1,
                 "first_date": "2002-07", "last_date": "2003-06",
                 "months_per_year": 12, "nw_lags_used": 3,
                 "avg_bonds_per_month": 100.0},
        variant="synthetic", n_legs=1, combiner=None)
    return StrategyResult(**kwargs)


# --------------------------------------------------------------------------
# 1. pinned mapping
# --------------------------------------------------------------------------

def test_pinned_mapping_structure():
    assert len(drv.PINNED_SPECS) == 12
    for key, pin in drv.PINNED_SPECS.items():
        assert not key.startswith("reject")
        assert pin.outcome in {"ok", "review", "zero_specs"}
        assert pin.n_specs >= 0
        if pin.outcome in {"review", "zero_specs"}:
            assert pin.n_specs == 0
    assert sum(p.n_specs for p in drv.PINNED_SPECS.values()) == 32


# --------------------------------------------------------------------------
# 2. extraction record
# --------------------------------------------------------------------------

def test_extraction_record_counts_and_fails_loud(tmp_path):
    run_dir = tmp_path / "paper"
    _write(run_dir / "spec_0.json", {})
    _write(run_dir / "spec_1.json", {})
    _write(run_dir / "run_manifest.json",
           {"operational_profile": {"model_calls": 7, "wall_clock_seconds": 1.5,
                                    "tokens": {"prompt": 10}}})
    pin = drv.PaperPin("X_2020", "paper", "ok", 2)
    row = drv.extraction_record("x", pin, repo_root=tmp_path)
    assert row["outcome"] == "ok" and row["n_specs"] == 2
    assert row["operational_profile"]["model_calls"] == 7

    bad_pin = drv.PaperPin("X_2020", "paper", "ok", 3)
    with pytest.raises(RuntimeError, match="pinned"):
        drv.extraction_record("x", bad_pin, repo_root=tmp_path)


# --------------------------------------------------------------------------
# 3+4. compile stage
# --------------------------------------------------------------------------

def test_compile_paper_refusal_and_executed_rows(tmp_path, monkeypatch):
    run_dir = tmp_path / "paper"
    _write(run_dir / "spec_0.json", {"which": "refuses"})
    _write(run_dir / "spec_1.json", {"which": "compiles"})
    monkeypatch.setattr(drv, "spec_from_dict", lambda d: _fake_spec(d["which"]))

    calls = {"run": 0}

    def fake_adapt(spec, standing_subs=None):
        return _refused_adapt() if spec.header.strategy_label.value == "refuses" \
            else _compiled_adapt()

    def fake_run(result, panel):
        calls["run"] += 1
        return _strategy_result()

    pin = drv.PaperPin("X_2020", "paper", "ok", 2)
    rows, events = drv.compile_paper(
        "x", pin, panel=None, subs=None, artefact_dir=tmp_path / "out",
        repo_root=tmp_path, adapt=fake_adapt, run=fake_run)

    assert [r["outcome"] for r in rows] == ["refused", "executed"]
    assert rows[0]["refusal_codes"] == ["REVIEW_REQUIRED"]
    assert rows[1]["summary"]["t_stat"] == 2.0
    assert calls["run"] == 1  # the refused spec never reaches the runner
    assert events == []
    written = sorted(p.name for p in (tmp_path / "out" / "x").glob("*.json"))
    assert written == ["spec_0_adapt.json", "spec_1_adapt.json"]


def test_compile_paper_events_yield_typed_not_run_rows(tmp_path):
    run_dir = tmp_path / "paper"
    run_dir.mkdir(parents=True)
    _write(run_dir / "events.json", [
        {"kind": "assembly_incomplete", "paper_id": "X_2020",
         "construction_name": "VaR", "detail": "sort_signal unresolved"},
        {"kind": "LibrarianEmissionError", "paper_id": "X_2020",
         "construction_name": "beta"},
    ])
    pin = drv.PaperPin("X_2020", "paper", "ok", 0)
    rows, events = drv.compile_paper("x", pin, panel=None, subs=None,
                                     repo_root=tmp_path)
    assert rows == [{"spec": None, "name": "VaR", "outcome": "not_run",
                     "extraction_event": "assembly_incomplete"}]
    assert len(events) == 1 and events[0]["kind"] == "LibrarianEmissionError"


# --------------------------------------------------------------------------
# 5. subset-equality + run_log allowlist
# --------------------------------------------------------------------------

def test_compare_audit_subset_rule():
    recorded = {"a": 1.5, "nest": {"b": [1, 2]}, "s": "x"}
    fresh_ok = {"a": 1.5, "nest": {"b": [1, 2], "extra": 9}, "s": "x", "new": 0}
    m, additive = drv.subset_mismatches(recorded, fresh_ok)
    assert m == [] and set(additive) == {".nest.extra", ".new"}

    m, _ = drv.subset_mismatches(recorded, {"a": 1.5000001, "nest": {"b": [1, 2]}, "s": "x"})
    assert any(".a" in x for x in m)
    m, _ = drv.subset_mismatches({"l": [1, 2]}, {"l": [1, 2, 3]})
    assert any("length" in x for x in m)

    log = drv._compare_run_log(
        {"git_commit": "new", "window": "dev", "anchors": ["str"]},
        {"git_commit": "old", "window": "dev", "anchors": ["str"]},
        drv.AUDIT_RUN_LOG_ALLOWED)
    assert log["pass"] and "git_commit" in log["allowed_diffs"]

    log = drv._compare_run_log({"window": "dev"}, {"window": "OTHER"},
                               drv.AUDIT_RUN_LOG_ALLOWED)
    assert not log["pass"] and log["mismatches"] == ["run_log.window"]


# --------------------------------------------------------------------------
# 6. quant comparison
# --------------------------------------------------------------------------

def _quant_dir(tmp_path, name, t_stat, git_short):
    d = tmp_path / name
    for a in drv.ANCHOR_STRATEGIES:
        _write(d / f"{a}.json", {"anchor": a, "status": "run",
                                 "strategy_label": a, "variant": "v", "n_legs": 1,
                                 "combiner": None,
                                 "summary": {"average": 0.001, "t_stat": t_stat}})
    _write(d / "coverage.json", {"n_candidates": 3, "run": 3, "refused": 0})
    _write(d / "run_log.json", {"git_short": git_short, "window": "dev",
                                "anchors_requested": list(drv.ANCHOR_STRATEGIES)})
    return d


def test_compare_quant_exact(tmp_path):
    recorded = _quant_dir(tmp_path, "recorded", 2.0, "aaa")
    same = _quant_dir(tmp_path, "same", 2.0, "bbb")   # only an allowed key differs
    assert drv.compare_quant(same, recorded)["pass"]

    drifted = _quant_dir(tmp_path, "drifted", 2.0000001, "aaa")
    cmp = drv.compare_quant(drifted, recorded)
    assert not cmp["pass"] and any("summary" in m for m in cmp["mismatches"])


# --------------------------------------------------------------------------
# 7. funnel byte equality
# --------------------------------------------------------------------------

def test_compare_funnel_byte_equality(tmp_path):
    a = _write(tmp_path / "a.json", {"x": 1})
    b = _write(tmp_path / "b.json", {"x": 1})
    c = _write(tmp_path / "c.json", {"x": 2})
    assert drv.compare_funnel(a, b)["pass"]
    assert not drv.compare_funnel(a, c)["pass"]


# --------------------------------------------------------------------------
# 8. spec field diff
# --------------------------------------------------------------------------

def test_spec_field_diff():
    def inh(tag, value, reason=None):
        return {"tag": tag, "value": value, "evidence": {"unknown_reason": reason}}

    spec_a = {"header": {"run_id": "run-A", "label": inh("STATED", "s")},
              "part2": {"n_groups": inh("STATED", 10),
                        "sort_kind": inh("STATED", "single"),
                        "legs": [{"long_leg": inh("UNKNOWN", None)}],
                        "policy": inh("UNKNOWN", None, "not_stated")}}
    spec_b = {"header": {"run_id": "run-B", "label": inh("STATED", "s")},
              "part2": {"n_groups": inh("UNKNOWN", None),
                        "sort_kind": inh("STATED", "double"),
                        "legs": [{"long_leg": inh("STATED", "highest_signal")}],
                        "policy": inh("UNKNOWN", None, "single_response")}}

    diff = drv.spec_field_diff(spec_a, spec_b)
    assert diff["n_fields"] == 5               # every tagged leaf, incl. inside legs
    assert diff["n_tag_changed"] == 2          # n_groups STATED->UNKNOWN, long_leg UNKNOWN->STATED
    assert diff["n_stated_value_diff"] == 1    # sort_kind 'single' != 'double'
    assert diff["n_evidence_only"] == 1        # policy: same tag+value, evidence churn
    assert {r["path"] for r in diff["rows"]} == {
        ".part2.n_groups", ".part2.sort_kind", ".part2.legs[0].long_leg"}
    # plain header run-identity strings never surface as field diffs
    assert not any("run_id" in r["path"] for r in diff["rows"])


# --------------------------------------------------------------------------
# 9. entry rows with injected case builder
# --------------------------------------------------------------------------

def test_entry_rows_with_injected_build_case(tmp_path):
    def fake_build_case(report, strategy_id, case_id, corrected_quant_config_ref):
        failed = ("lib_gap",) if strategy_id == "str" else ()
        return (types.SimpleNamespace(failed_check_ids=failed),
                types.SimpleNamespace(theta=0.001, q=0.10))

    rows = drv.entry_rows(tmp_path, build_case_fn=fake_build_case)
    assert rows["str"]["entered"] is True
    assert rows["str"]["failed_check_ids"] == ["lib_gap"]
    assert rows["drf"]["entered"] is False and rows["mom6"]["entered"] is False
    assert rows["drf"]["theta"] == 0.001 and rows["drf"]["q"] == 0.10


# --------------------------------------------------------------------------
# 10. matrix + summary rendering
# --------------------------------------------------------------------------

def test_matrix_and_summary_render():
    extractions = {k: {"outcome": p.outcome} for k, p in drv.PINNED_SPECS.items()}
    compile_rows = {k: ([{"outcome": "refused", "refusal_codes": ["REVIEW_REQUIRED"]}]
                        if p.n_specs else [])
                    for k, p in drv.PINNED_SPECS.items()}
    quant_records = {a: {"status": "run"} for a in drv.ANCHOR_STRATEGIES}
    gate12 = {a: {"pass": True} for a in drv.GATE12_ANCHORS}
    audit_ok = {a: True for a in drv.ANCHOR_STRATEGIES}
    entry = {a: {"entered": a == "str"} for a in drv.ANCHOR_STRATEGIES}
    reporter = {a: "verified" for a in drv.ANCHOR_STRATEGIES}

    matrix = drv.build_matrix(extractions, compile_rows, quant_records, gate12,
                              audit_ok, entry, True, reporter)
    assert len(matrix) == 12
    by_key = {r["paper"]: r for r in matrix}
    assert by_key["klz"]["gold_execution"] == "not_applicable"
    assert "entered" in by_key["drr"]["rq4_entry"]
    assert "not entered (typed)" in by_key["jnps"]["rq4_entry"]
    assert by_key["hvz"]["compile"] == "no_specs (terminal at extraction)"
    assert "pass" in by_key["bbw"]["gate12"] and "crf" in by_key["bbw"]["gate12"]

    md = drv.render_summary_md(matrix, {"quant": {"pass": True}},
                               {"n_fields": 60, "n_tag_changed": 2,
                                "n_stated_value_diff": 1},
                               {"exit": 0, "out": "kpp_score.json"})
    assert md.count("|") > 12 * 9
    assert "repair" not in md.lower()


# --------------------------------------------------------------------------
# 11. artefact writes stay under the given dir
# --------------------------------------------------------------------------

def test_artefacts_written_under_artefact_dir_only(tmp_path, monkeypatch):
    run_dir = tmp_path / "paper"
    _write(run_dir / "spec_0.json", {"which": "refuses"})
    monkeypatch.setattr(drv, "spec_from_dict", lambda d: _fake_spec("refuses"))
    before = {p for p in tmp_path.rglob("*")}
    pin = drv.PaperPin("X_2020", "paper", "ok", 1)
    drv.compile_paper("x", pin, panel=None, subs=None,
                      artefact_dir=tmp_path / "artefacts", repo_root=tmp_path,
                      adapt=lambda s, standing_subs=None: _refused_adapt(),
                      run=lambda r, p: (_ for _ in ()).throw(AssertionError))
    new = {p for p in tmp_path.rglob("*")} - before
    assert all(str(p).startswith(str(tmp_path / "artefacts")) for p in new)
