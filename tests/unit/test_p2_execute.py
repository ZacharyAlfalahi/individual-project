"""WS-C (P2) — executed-run assembly: series read-back, the typed empty-series rule, the
Arm-B deterministic-compiler attempt (refused => NO series, never a stand-in), and the
assembled result/report under an authorised below-floor departure. Offline (no LLM, no
engine: the compile chain is injected)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluation.codegen.census import CensusInput, RoutingDecision, run_census  # noqa: E402
from evaluation.codegen.p2_execute import (  # noqa: E402
    CompilerAttempt,
    SeriesReadback,
    arm_b_compiler,
    assemble_executed,
    compiler_attempt,
    empty_series,
    generated_series,
    inter_model_pairs,
    memoised,
    refusal_codes,
)
from evaluation.codegen.p2_selector import select_arms  # noqa: E402

_TH = {
    "arms": {"arm_b_max": 5, "below_floor_min_arm_a": 5},
    "agreement": {"correlation_min": 0.99, "sign_agreement_min": 0.95},
    "divergence_strata": {"high_divergence_corr_lt": 0.90, "medium_divergence_corr_lt": 0.99},
    "taxonomy_sampling": {"strata": ["arm", "divergence_magnitude"]},
    "zoo_list": {"frozen_sha256": "TO_SET"},
    "min_overlap_months": 3,
}
_MODELS = ("model_a", "model_b")


def _series(values, start="2010-01-31"):
    idx = pd.date_range(start, periods=len(values), freq="ME")
    return pd.Series(values, index=idx, name="portfolio_return")


def _census_with(n_refusals, n_compilable):
    inputs = (
        [CensusInput(f"r{i}", {"header": {"paper_id": f"r{i}"}}, True) for i in range(n_refusals)]
        + [CensusInput(f"c{i}", {"header": {"paper_id": f"c{i}"}}, True)
           for i in range(n_compilable)]
    )

    def route(inp):
        return (RoutingDecision(False, True, "refuse_asset_class")
                if inp.paper_id.startswith("r") else RoutingDecision(True, False))

    return run_census(inputs, route)


def _selection(census):
    zoo = [m.paper_id for m in census.members]
    return select_arms(census, zoo, _TH)


# --- series read-back -----------------------------------------------------------

def test_generated_series_reads_ok_runs_and_leaves_failures_none():
    runs = [
        {"paper_id": "r0", "model_id": "model_a", "sandbox_status": "ok",
         "output_path": "/fake/out.csv"},
        {"paper_id": "r0", "model_id": "model_b", "sandbox_status": "wont_run",
         "output_path": None},
    ]
    parsed = _series([0.01, 0.02, 0.03])
    out = generated_series(runs, parse=lambda p: parsed)
    assert out.series["r0"]["model_a"] is parsed
    assert out.series["r0"]["model_b"] is None
    assert out.rejections == ()


def test_generated_series_ignores_output_of_a_non_ok_run():
    """A stale CSV beside a failed run must never be read back as that run's result."""
    runs = [{"paper_id": "r0", "model_id": "model_a", "sandbox_status": "wont_run",
             "output_path": "/fake/stale.csv"}]
    def _boom(_p):
        raise AssertionError("a non-ok run must not be parsed")
    assert generated_series(runs, parse=_boom).series["r0"]["model_a"] is None


def test_duplicate_months_are_rejected_not_raised():
    """Two rows in one calendar month pass the CSV contract but break monthly alignment.
    Rejecting it typed keeps one bad output from voiding an entire paid run."""
    idx = pd.DatetimeIndex(["2010-01-08", "2010-01-15", "2010-02-28"])
    dup = pd.Series([0.01, 0.02, 0.03], index=idx, name="portfolio_return")
    runs = [{"paper_id": "r0", "model_id": "model_a", "sandbox_status": "ok",
             "output_path": "/fake/out.csv"}]

    out = generated_series(runs, parse=lambda p: dup)

    assert out.series["r0"]["model_a"] is None            # not scored
    assert [r["reason"] for r in out.rejections] == ["duplicate_months"]
    assert "2010-01" in out.rejections[0]["detail"]
    # and the pair it feeds is an empty series, i.e. counted, not dropped
    a, _ = inter_model_pairs({"r0": {"model_a": None, "model_b": None}}, ["r0"], _MODELS)["r0"]
    assert len(a) == 0


def test_unreadable_output_is_rejected_not_raised():
    runs = [{"paper_id": "r0", "model_id": "model_a", "sandbox_status": "ok",
             "output_path": "/fake/out.csv"}]

    def _raise(_p):
        raise ValueError("output CSV contains duplicate dates")

    out = generated_series(runs, parse=_raise)
    assert out.series["r0"]["model_a"] is None
    assert out.rejections[0]["reason"] == "unreadable_output"


def test_missing_series_becomes_an_empty_series_not_a_dropped_member():
    pairs = inter_model_pairs({"r0": {"model_a": _series([0.01, 0.02]), "model_b": None}},
                              ["r0"], _MODELS)
    a, b = pairs["r0"]
    assert len(a) == 2 and len(b) == 0          # counted, not dropped


def test_missing_run_record_fails_loud():
    with pytest.raises(KeyError):
        inter_model_pairs({"r0": {"model_a": _series([0.01])}}, ["r0"], _MODELS)


def test_empty_series_is_typed_and_datetime_indexed():
    s = empty_series()
    assert len(s) == 0 and isinstance(s.index, pd.DatetimeIndex)


# --- the Arm-B third implementation ---------------------------------------------

@dataclass
class _Code:
    value: str


@dataclass
class _Refusal:
    code: _Code


@dataclass
class _LegCall:
    refused: bool
    result: object


class _AdaptResult:
    def __init__(self, refused, refusals=(), leg_calls=()):
        self.refused = refused
        self.refusals = refusals
        self.leg_calls = leg_calls


class _RunResult:
    def __init__(self, frame):
        self.monthly_returns = frame


def _monthly(n=4):
    return pd.DataFrame({
        "date": pd.date_range("2010-01-31", periods=n, freq="ME"),
        "strategy_ret": [0.01] * n,
    })


def test_refusal_codes_match_the_rq2_coverage_helper():
    """Parity with scripts/run_t3_coverage.refusal_codes_of, so the P2 record and the RQ2
    coverage record can never disagree about a member's refusal codes."""
    from run_t3_coverage import refusal_codes_of

    result = _AdaptResult(
        True,
        refusals=(_Refusal(_Code("REVIEW_REQUIRED")),),
        leg_calls=(_LegCall(True, _Refusal(_Code("MISSING_BINDING"))),),
    )
    assert list(refusal_codes(result)) == refusal_codes_of(result)


def test_compiler_attempt_refused_carries_codes_and_no_series():
    result = _AdaptResult(True, refusals=(_Refusal(_Code("REVIEW_REQUIRED")),))
    attempt, series = compiler_attempt(
        "c0", {"header": {}}, load_spec=lambda d: d, adapt=lambda s, **k: result,
        run=lambda r, p: pytest.fail("run must not be called on a refusal"),
        subs_provider=lambda: "subs", panel_provider=lambda: "panel")
    assert attempt.outcome == "refused"
    assert attempt.refusal_codes == ("REVIEW_REQUIRED",)
    assert series is None


def test_compiler_attempt_executed_returns_the_engine_series():
    attempt, series = compiler_attempt(
        "c0", {"header": {}}, load_spec=lambda d: d,
        adapt=lambda s, **k: _AdaptResult(False),
        run=lambda r, p: _RunResult(_monthly(5)),
        subs_provider=lambda: "subs", panel_provider=lambda: "panel")
    assert attempt.outcome == "executed" and attempt.n_months == 5
    assert len(series) == 5 and isinstance(series.index, pd.DatetimeIndex)


def test_compiler_attempt_types_an_unloadable_spec():
    def _raise(_d):
        raise ValueError("bad spec")
    attempt, series = compiler_attempt(
        "c0", {"header": {}}, load_spec=_raise, adapt=lambda s, **k: None,
        run=lambda r, p: None, subs_provider=lambda: "subs", panel_provider=lambda: "panel")
    assert attempt.outcome == "spec_unloadable" and series is None


def test_compiler_attempt_types_a_member_with_no_spec():
    attempt, series = compiler_attempt(
        "c0", None, load_spec=lambda d: d, adapt=lambda s, **k: None,
        run=lambda r, p: None, subs_provider=lambda: "subs", panel_provider=lambda: "panel")
    assert attempt.outcome == "spec_unloadable" and series is None


def test_compiler_outcome_vocabulary_is_closed():
    with pytest.raises(ValueError):
        CompilerAttempt("c0", "worked_fine")


def test_panel_provider_is_memoised():
    calls = []

    def provider():
        calls.append(1)
        return ("panel", "subs")

    once = memoised(provider)
    once(), once(), once()
    assert len(calls) == 1


def test_compiler_attempt_types_an_adapter_crash_instead_of_aborting():
    """This runs AFTER the generation spend: an exception in the chain must not void the run."""
    def _raise(*a, **k):
        raise RuntimeError("registry drift")

    attempt, series = compiler_attempt(
        "c0", {"header": {}}, load_spec=lambda d: d, adapt=_raise,
        run=lambda r, p: None, subs_provider=lambda: "subs", panel_provider=lambda: "panel")
    assert attempt.outcome == "compile_error" and series is None
    assert "RuntimeError" in attempt.note


def test_compiler_attempt_types_a_runner_crash():
    def _raise(*a, **k):
        raise ValueError("engine blew up")

    attempt, series = compiler_attempt(
        "c0", {"header": {}}, load_spec=lambda d: d,
        adapt=lambda s, **k: _AdaptResult(False), run=_raise,
        subs_provider=lambda: "subs", panel_provider=lambda: "panel")
    assert attempt.outcome == "compile_error" and series is None


def test_the_dev_panel_is_not_loaded_when_every_spec_refuses():
    """The panel costs seconds and gigabytes; a refusal needs only the standing subs."""
    census = _census_with(2, 3)
    sel = _selection(census)

    def _panel():
        raise AssertionError("the panel must not be materialised when nothing compiles")

    attempts, series = arm_b_compiler(
        census, sel, load_spec=lambda d: d,
        adapt=lambda s, **k: _AdaptResult(True, refusals=(_Refusal(_Code("REVIEW_REQUIRED")),)),
        run=lambda r, p: None, subs_provider=lambda: "subs", panel_provider=_panel)
    assert [a.outcome for a in attempts] == ["refused", "refused"] and series == {}


def test_arm_b_compiler_splits_refused_from_executed():
    census = _census_with(2, 3)
    sel = _selection(census)
    refused_first = {"c0": _AdaptResult(True, refusals=(_Refusal(_Code("REVIEW_REQUIRED")),)),
                     "c1": _AdaptResult(False)}
    attempts, series = arm_b_compiler(
        census, sel, load_spec=lambda d: d,
        adapt=lambda s, **k: refused_first[s["header"]["paper_id"]],
        run=lambda r, p: _RunResult(_monthly(4)),
        subs_provider=lambda: "subs", panel_provider=lambda: "panel")
    assert [a.outcome for a in attempts] == ["refused", "executed"]
    assert set(series) == {"c1"}              # the refused member contributes NO series


# --- assembly -------------------------------------------------------------------

def _driver_result(census, sel, *, statuses=None):
    statuses = statuses or {}
    runs = []
    for pid in sel.members():
        for model_id in _MODELS:
            status = statuses.get((pid, model_id), "ok")
            runs.append({
                "paper_id": pid, "arm": "A" if pid in sel.arm_a else "B",
                "model_id": model_id,
                "code_extracted": True,
                "sandbox_status": status,
                "sandbox_reason": None if status == "ok" else "nonzero_exit",
                "stderr_tail": "" if status == "ok" else "Traceback\nKeyError: 'beta'",
                "output_path": "x",
                "returned_model_version": model_id,
            })
    return {"runs": runs, "generation_errors": [], "models": list(_MODELS),
            "prompt_sha256": {pid: "deadbeef" for pid in sel.members()}}


def test_assemble_below_floor_under_override_emits_numbers_and_labels_them():
    census = _census_with(2, 3)
    sel = _selection(census)
    assert sel.below_floor is True
    base = _series([0.01, -0.02, 0.03, 0.02, 0.015])
    series_by_member = {pid: {m: base.copy() for m in _MODELS} for pid in sel.members()}
    attempts = tuple(CompilerAttempt(pid, "refused", ("REVIEW_REQUIRED",), None, "refused")
                     for pid in sel.arm_b)

    executed = assemble_executed(
        census, sel, _TH, _driver_result(census, sel),
        readback=SeriesReadback(series=series_by_member), compiler_attempts=attempts, compiler_series={},
        model_ids=_MODELS, floor_override="DEP-1", provenance={"r0": {"manifest_phase": "report"}})

    result, report = executed.result, executed.report_md
    assert result["floor_override"] == "DEP-1"
    assert result["below_floor"] is True and result["suppressed"] is False
    assert len(result["metrics"]["agreements"]) == 4        # 2 Arm A + 2 Arm B
    assert result["metrics"]["floor_override"] == "DEP-1"
    assert "descriptive" in result["below_floor_reason"]
    # the report labels the departure, shows the compiler attempts, and never invents a
    # divergence row for a refused member
    assert "BELOW-FLOOR DEPARTURE — DEP-1" in report
    assert "deterministic-compiler attempts" in report
    assert "NOT COMPUTABLE" in report
    assert "sharpe" not in report.lower()
    assert result["compiler_attempts"][0]["outcome"] == "refused"


def test_assemble_counts_a_failed_run_as_worst_divergence_not_a_drop():
    census = _census_with(2, 3)
    sel = _selection(census)
    base = _series([0.01, -0.02, 0.03, 0.02, 0.015])
    series_by_member = {pid: {m: base.copy() for m in _MODELS} for pid in sel.members()}
    broken = sel.arm_a[0]
    series_by_member[broken]["model_b"] = None          # a WONT_RUN cell

    executed = assemble_executed(
        census, sel, _TH, _driver_result(census, sel, statuses={(broken, "model_b"): "wont_run"}),
        readback=SeriesReadback(series=series_by_member), compiler_attempts=(), compiler_series={},
        model_ids=_MODELS, floor_override="DEP-1")

    rows = {a["paper_id"]: a for a in executed.result["metrics"]["agreements"]}
    assert len(rows) == 4                                   # nothing dropped
    assert rows[broken]["n_overlap"] == 0
    assert rows[broken]["agrees"] is False
    assert rows[broken]["stratum"] == "high"
    assert rows[broken]["sufficient_overlap"] is False

    # the surviving run of that pair stays legible on its own terms, and the failed one
    # carries its diagnostic into the report
    runs = {(r["paper_id"], r["model_id"]): r for r in executed.result["runs"]}
    assert runs[(broken, "model_a")]["n_months"] == 5
    assert runs[(broken, "model_b")]["n_months"] is None
    assert "Failure diagnostics" in executed.report_md
    assert "KeyError: 'beta'" in executed.report_md


def test_assemble_above_floor_carries_no_departure_label():
    """The registered path: numbers are emitted on their own authority, unlabelled."""
    census = _census_with(5, 5)
    sel = _selection(census)
    assert sel.below_floor is False
    base = _series([0.01, -0.02, 0.03, 0.02, 0.015])
    series_by_member = {pid: {m: base.copy() for m in _MODELS} for pid in sel.members()}

    executed = assemble_executed(
        census, sel, _TH, _driver_result(census, sel),
        readback=SeriesReadback(series=series_by_member), compiler_attempts=(),
        compiler_series={}, model_ids=_MODELS)

    assert executed.result["floor_override"] is None
    assert executed.result["below_floor"] is False
    assert executed.result["below_floor_reason"] is None
    assert executed.result["metrics"]["distribution"]["agreement_rate"] == 1.0
    assert "BELOW-FLOOR DEPARTURE" not in executed.report_md


def test_a_mismatched_compiler_panel_view_is_stated_in_the_report():
    """An Arm-B divergence is only an implementation difference when both sides saw the
    same panel; when they did not, the report has to say so."""
    census = _census_with(2, 3)
    sel = _selection(census)
    base = _series([0.01, -0.02, 0.03, 0.02, 0.015])
    series_by_member = {pid: {m: base.copy() for m in _MODELS} for pid in sel.members()}
    config = {"compiler_view": {"stale_mask": True, "signal_lag": 1},
              "codegen_panel_view": {"stale_mask": False, "signal_lag": 0},
              "views_match": False}

    executed = assemble_executed(
        census, sel, _TH, _driver_result(census, sel),
        readback=SeriesReadback(series=series_by_member),
        compiler_attempts=(CompilerAttempt(sel.arm_b[0], "executed", (), 5),),
        compiler_series={sel.arm_b[0]: base + 0.001}, model_ids=_MODELS,
        floor_override="DEP-1", compiler_run_config=config)

    assert "DIFFERENT panel views" in executed.report_md
    assert executed.result["compiler_run_config"]["views_match"] is False


def test_a_rejected_series_is_reported_on_its_run_row():
    census = _census_with(2, 3)
    sel = _selection(census)
    base = _series([0.01, -0.02, 0.03, 0.02, 0.015])
    series_by_member = {pid: {m: base.copy() for m in _MODELS} for pid in sel.members()}
    victim = sel.arm_a[0]
    series_by_member[victim]["model_b"] = None
    readback = SeriesReadback(
        series=series_by_member,
        rejections=({"paper_id": victim, "model_id": "model_b",
                     "reason": "duplicate_months", "detail": "more than one row in 2010-01"},))

    executed = assemble_executed(
        census, sel, _TH, _driver_result(census, sel), readback=readback,
        compiler_attempts=(), compiler_series={}, model_ids=_MODELS, floor_override="DEP-1")

    row = next(r for r in executed.result["runs"]
               if r["paper_id"] == victim and r["model_id"] == "model_b")
    assert row["series_rejected"] == "duplicate_months"
    assert executed.result["series_rejections"][0]["paper_id"] == victim


def test_an_empty_reference_arm_is_reported_as_the_result_not_a_gap():
    """When the router compiles nothing, Arm B is empty — the report must say so rather than
    render an empty divergence table."""
    census = _census_with(5, 0)
    sel = _selection(census)
    assert sel.arm_b == () and sel.below_floor is False
    base = _series([0.01, -0.02, 0.03, 0.02, 0.015])
    series_by_member = {pid: {m: base.copy() for m in _MODELS} for pid in sel.members()}

    executed = assemble_executed(
        census, sel, _TH, _driver_result(census, sel),
        readback=SeriesReadback(series=series_by_member), compiler_attempts=(),
        compiler_series={}, model_ids=_MODELS)

    assert "ARM B IS EMPTY" in executed.report_md
    assert executed.result["arm_b"] == []
    assert executed.result["metrics"]["divergences"] == []


def test_assemble_emits_divergence_rows_when_the_compiler_executed():
    census = _census_with(2, 3)
    sel = _selection(census)
    base = _series([0.01, -0.02, 0.03, 0.02, 0.015])
    series_by_member = {pid: {m: base.copy() for m in _MODELS} for pid in sel.members()}
    compiler_series = {pid: base + 0.001 for pid in sel.arm_b}
    attempts = tuple(CompilerAttempt(pid, "executed", (), 5) for pid in sel.arm_b)

    executed = assemble_executed(
        census, sel, _TH, _driver_result(census, sel),
        readback=SeriesReadback(series=series_by_member), compiler_attempts=attempts,
        compiler_series=compiler_series, model_ids=_MODELS, floor_override="DEP-1")

    divergences = executed.result["metrics"]["divergences"]
    assert len(divergences) == len(sel.arm_b) * 2           # each model vs the compiler
    assert all(d["label"] == "neither side is truth" for d in divergences)
    assert "neither side is truth" in executed.report_md
