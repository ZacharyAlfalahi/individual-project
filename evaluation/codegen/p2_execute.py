"""P2 coverage-boundary — executed-run assembly (WS-C).

``p2_driver`` generates and sandboxes; ``p2_metrics`` scores series pairs. This module joins
the two, and only that:

  * **series read-back** — each run's sandboxed output CSV as a return series. A run that
    produced no output (``wont_run`` / malformed / generation error) carries an EMPTY
    series, so the pair is COUNTED as insufficient overlap + worst divergence rather than
    silently dropped (the same discipline as the typed eligibility exclusions).
  * **the Arm-B third implementation** — Arm B's codegen-vs-compiler comparison needs the
    DETERMINISTIC COMPILER's series for the member. This module attempts the real chain
    (``spec_from_dict`` -> ``adapt_spec`` with the verified standing subs -> ``run_strategy``),
    injected by the caller, and records a TYPED refusal when the compiler refuses. A refused
    member yields NO series: the comparison does not exist, and inventing a stand-in
    implementation would fabricate the very thing under test.
  * **assembly** — metrics + taxonomy + eligibility accounting + rendered report.

No model calls, no thresholds of its own, no holdout contact.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from evaluation.codegen.census import CensusResult
from evaluation.codegen.p2_metrics import compute_p2_metrics
from evaluation.codegen.p2_report import render_report
from evaluation.codegen.p2_selector import ArmSelection
from evaluation.codegen.p2_taxonomy import (
    eligibility_exclusion_accounting,
    stratified_failure_sample,
)
from evaluation.codegen.sandbox import parse_output_csv

#: Outcome vocabulary for a compiler attempt on one Arm-B member (closed, like every other
#: disposition in the P2 path — an untyped free string would let a silent drop through).
COMPILER_OUTCOMES = frozenset({"executed", "refused", "spec_unloadable", "compile_error"})


def empty_series() -> pd.Series:
    """The typed stand-in for "this run produced no return series". Scored as n_overlap 0 ->
    non-finite correlation -> worst divergence stratum + insufficient overlap (counted,
    never silently dropped)."""
    return pd.Series([], dtype=float, index=pd.DatetimeIndex([], name="date"),
                     name="portfolio_return")


# ---------------------------------------------------------------------------
# Generated series read-back.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SeriesReadback:
    """What the read-back produced: the per-(member, model) series map, and the TYPED
    rejections of outputs that parsed but cannot be scored. A rejection is a datum about
    the generated code, never a reason to abort a paid run."""

    series: dict[str, dict[str, pd.Series | None]] = field(default_factory=dict)
    rejections: tuple[dict, ...] = ()

    def rejection_for(self, paper_id: str, model_id: str) -> dict | None:
        for r in self.rejections:
            if r["paper_id"] == paper_id and r["model_id"] == model_id:
                return r
        return None


def generated_series(
    runs: Sequence[dict], *, parse: Callable[[Path], pd.Series] = parse_output_csv
) -> SeriesReadback:
    """Read each ``ok`` run's output CSV back as a return series.

    Only a run the sandbox typed ``ok`` has an output to read; every other status keeps
    ``None`` so the caller can distinguish "no series" from "an empty series". Two typed
    rejections keep a bad output from ending the run: an output that fails to parse on
    read-back, and one carrying more than one row in a calendar month — the comparator aligns
    monthly and would raise on the duplicate, voiding an entire paid run over one member's
    CSV."""
    out: dict[str, dict[str, pd.Series | None]] = {}
    rejections: list[dict] = []
    for record in runs:
        pid, model_id = record["paper_id"], record["model_id"]
        series: pd.Series | None = None
        path = record.get("output_path")
        if record.get("sandbox_status") == "ok" and path:
            try:
                candidate = parse(Path(path))
            except Exception as exc:
                rejections.append({
                    "paper_id": pid, "model_id": model_id, "reason": "unreadable_output",
                    "detail": f"{type(exc).__name__}: {exc}"[:200]})
            else:
                months = pd.DatetimeIndex(candidate.index).to_period("M")
                if not months.is_unique:
                    duplicated = sorted({str(m) for m in months[months.duplicated()]})
                    rejections.append({
                        "paper_id": pid, "model_id": model_id, "reason": "duplicate_months",
                        "detail": f"more than one row in {', '.join(duplicated[:5])}"})
                else:
                    series = candidate
        out.setdefault(pid, {})[model_id] = series
    return SeriesReadback(series=out, rejections=tuple(rejections))


def inter_model_pairs(
    series_by_member: dict[str, dict[str, pd.Series | None]],
    member_ids: Sequence[str],
    model_ids: tuple[str, str],
) -> dict[str, tuple[pd.Series, pd.Series]]:
    """The ``compute_p2_metrics`` input: one (model-A, model-B) series pair per member, with
    a missing series replaced by ``empty_series()``. Fail-loud when the driver emitted no run
    at all for a (member, model) — that is a broken run record, not a datum."""
    a_id, b_id = model_ids
    pairs: dict[str, tuple[pd.Series, pd.Series]] = {}
    for pid in member_ids:
        runs = series_by_member.get(pid, {})
        missing = [m for m in (a_id, b_id) if m not in runs]
        if missing:
            raise KeyError(
                f"{pid!r}: no run record for model(s) {missing} — the driver must emit one "
                "run per (member, model); a missing record is a broken run, not a result"
            )
        a, b = runs[a_id], runs[b_id]
        pairs[pid] = (a if a is not None else empty_series(),
                      b if b is not None else empty_series())
    return pairs


def annotated_runs(runs: Sequence[dict], readback: SeriesReadback) -> tuple[dict, ...]:
    """Each run record plus ``n_months`` — the length of the series it produced (``None``
    when it produced none) — and any typed ``series_rejected`` reason. A run that executed is
    then legible on its own terms even when its PAIR is unscorable because the other model
    produced nothing."""
    out: list[dict] = []
    for record in runs:
        series = readback.series.get(record["paper_id"], {}).get(record["model_id"])
        rejection = readback.rejection_for(record["paper_id"], record["model_id"])
        out.append({
            **record,
            "n_months": (int(len(series)) if series is not None else None),
            "series_rejected": (rejection or {}).get("reason"),
            "series_rejected_detail": (rejection or {}).get("detail"),
        })
    return tuple(out)


# ---------------------------------------------------------------------------
# The Arm-B third implementation: the deterministic compiler.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompilerAttempt:
    """One Arm-B member's deterministic-compiler attempt. ``refused`` carries the typed
    refusal codes and NO series — the codegen-vs-compiler comparison simply does not exist
    for that member."""

    paper_id: str
    outcome: str
    refusal_codes: tuple[str, ...] = ()
    n_months: int | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if self.outcome not in COMPILER_OUTCOMES:
            raise ValueError(
                f"{self.paper_id!r}: compiler outcome {self.outcome!r} is not one of "
                f"{sorted(COMPILER_OUTCOMES)}"
            )

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "outcome": self.outcome,
            "refusal_codes": list(self.refusal_codes),
            "n_months": self.n_months,
            "note": self.note,
        }


def refusal_codes(result) -> tuple[str, ...]:
    """Every typed refusal code on an ``AdaptResult`` (spec-level + per-leg). Mirrors
    ``scripts/run_t3_coverage.refusal_codes_of``; parity is asserted in the tests so the P2
    record and the RQ2 coverage record cannot drift apart."""
    codes = [r.code.value for r in result.refusals]
    codes += [lc.result.code.value for lc in result.leg_calls
              if lc.refused and lc.result is not None]
    return tuple(codes)


def compiler_return_series(run_result) -> pd.Series:
    """The compiled strategy's monthly series, rebuilt exactly as the engine's own
    ``_leg_return_series`` does (``.values`` + explicit DatetimeIndex, no dtype drift)."""
    monthly = run_result.monthly_returns
    if len(monthly) == 0:
        return empty_series()
    return pd.Series(
        monthly["strategy_ret"].values,
        index=pd.DatetimeIndex(monthly["date"].values, name="date"),
        name="portfolio_return",
    )


def memoised(provider: Callable[[], object]) -> Callable[[], object]:
    """Call ``provider`` at most once. The dev panel is expensive and is materialised only
    if some Arm-B spec actually compiles; the standing subs are needed for every attempt."""
    box: dict[str, object] = {}

    def once() -> object:
        if "value" not in box:
            box["value"] = provider()
        return box["value"]

    return once


def compiler_attempt(
    paper_id: str,
    spec_dict: object | None,
    *,
    load_spec: Callable[[dict], object],
    adapt: Callable[..., object],
    run: Callable[..., object],
    subs_provider: Callable[[], object],
    panel_provider: Callable[[], object],
) -> tuple[CompilerAttempt, pd.Series | None]:
    """Attempt the deterministic compile+run of one member's EXTRACTED spec.

    Returns ``(attempt, series)``; ``series`` is ``None`` unless the compiler executed. The
    chain is injected (``load_spec`` / ``adapt`` / ``run`` / the two providers) so this
    module stays free of pipeline imports and the tests need no engine. The panel provider
    is consulted only AFTER a spec compiles — a refusal needs the standing subs, never the
    dev panel. A failure anywhere in the chain is a typed outcome, never an abort: this runs
    after the generation spend, so one member must not be able to void the whole run."""
    if spec_dict is None:
        return CompilerAttempt(paper_id, "spec_unloadable",
                               note="member carries no extracted spec"), None
    try:
        spec = load_spec(spec_dict)
    except Exception as exc:      # a malformed emission is a typed datum, not a crash
        return CompilerAttempt(paper_id, "spec_unloadable",
                               note=f"{type(exc).__name__}: {exc}"[:200]), None

    try:
        result = adapt(spec, standing_subs=subs_provider())
    except Exception as exc:
        return CompilerAttempt(paper_id, "compile_error",
                               note=f"adapt raised {type(exc).__name__}: {exc}"[:200]), None
    if result.refused:
        return CompilerAttempt(
            paper_id, "refused", refusal_codes(result), None,
            "the deterministic compiler refused the extracted spec — no compiler "
            "implementation exists for this member, so its codegen-vs-compiler comparison "
            "does not exist (no stand-in series is substituted)",
        ), None

    try:
        run_result = run(result, panel_provider())
    except Exception as exc:
        return CompilerAttempt(paper_id, "compile_error",
                               note=f"run raised {type(exc).__name__}: {exc}"[:200]), None
    if not hasattr(run_result, "monthly_returns"):
        raise RuntimeError(
            f"{paper_id}: run_strategy returned {type(run_result).__name__} for a "
            "non-refused compile — expected a StrategyResult"
        )
    series = compiler_return_series(run_result)
    return CompilerAttempt(paper_id, "executed", (), int(len(series)), None), series


def arm_b_compiler(
    census: CensusResult,
    selection: ArmSelection,
    *,
    load_spec: Callable[[dict], object],
    adapt: Callable[..., object],
    run: Callable[..., object],
    subs_provider: Callable[[], object],
    panel_provider: Callable[[], object],
) -> tuple[tuple[CompilerAttempt, ...], dict[str, pd.Series]]:
    """Every Arm-B member's compiler attempt: ``(attempts, series_by_member)``. Members the
    compiler refused appear in ``attempts`` (typed) and NOT in the series map."""
    subs_once, panel_once = memoised(subs_provider), memoised(panel_provider)
    attempts: list[CompilerAttempt] = []
    series: dict[str, pd.Series] = {}
    for pid in selection.arm_b:
        attempt, compiled = compiler_attempt(
            pid, census.member(pid).extracted_spec,
            load_spec=load_spec, adapt=adapt, run=run,
            subs_provider=subs_once, panel_provider=panel_once)
        attempts.append(attempt)
        if compiled is not None:
            series[pid] = compiled
    return tuple(attempts), series


# ---------------------------------------------------------------------------
# Assembly.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExecutedRun:
    """The assembled executed-run artefacts: the results object and its rendered report."""

    result: dict = field(default_factory=dict)
    report_md: str = ""


def assemble_executed(
    census: CensusResult,
    selection: ArmSelection,
    thresholds: dict,
    driver_result: dict,
    *,
    readback: SeriesReadback,
    compiler_attempts: tuple[CompilerAttempt, ...],
    compiler_series: dict[str, pd.Series],
    model_ids: tuple[str, str],
    floor_override: str | None = None,
    provenance: dict[str, dict] | None = None,
    meta: dict | None = None,
    compiler_run_config: dict | None = None,
    per_stratum: int = 2,
) -> ExecutedRun:
    """Assemble the executed P2 result + report from the driver record, the read-back series
    and the Arm-B compiler attempts. ``floor_override`` is the authorised departure id
    when the selection sits below the registered floor (``None`` on a registered run).
    ``compiler_run_config`` records WHICH panel view the compiler side ran under, so an
    Arm-B divergence can never be read as pure implementation difference when the two sides
    saw different panels."""
    series_by_member = readback.series
    pairs = inter_model_pairs(series_by_member, selection.members(), model_ids)
    metrics = compute_p2_metrics(
        census, selection, pairs, compiler_series or None, thresholds,
        model_ids=model_ids, floor_override=floor_override)
    accounting = eligibility_exclusion_accounting(census)
    taxonomy = stratified_failure_sample(metrics.agreements, per_stratum=per_stratum)
    attempts = tuple(a.to_dict() for a in compiler_attempts)
    runs = annotated_runs(driver_result.get("runs", ()), readback)
    report_md = render_report(
        metrics, selection, census, taxonomy=taxonomy, eligibility=accounting,
        meta=(dict(meta) if meta else None),
        runs=runs, compiler_attempts=attempts, compiler_run_config=compiler_run_config)

    result = {
        "experiment": "p2_coverage_boundary",
        "mode": "executed",
        "suppressed": metrics.suppressed,
        "below_floor": selection.below_floor,
        "below_floor_reason": metrics.below_floor_reason,
        "floor_override": floor_override,
        "arm_a_size": selection.arm_a_size,
        "arm_a": list(selection.arm_a),
        "arm_b": list(selection.arm_b),
        "models": list(driver_result.get("models", [])),
        "observed_dispositions": {
            "compilable": len(census.compilable_set()),
            "refused": len(census.refusal_set()),
            "eligibility_excluded": len(census.eligibility_exclusions()),
        },
        "eligibility_accounting": accounting.to_dict(),
        "census_fate_table": [dict(r) for r in census.fate_table()],
        "metrics": metrics.to_dict(),
        "taxonomy": taxonomy.to_dict(),
        "compiler_attempts": [dict(a) for a in attempts],
        "compiler_run_config": dict(compiler_run_config or {}),
        "series_rejections": [dict(r) for r in readback.rejections],
        "runs": [dict(r) for r in runs],
        "generation_errors": list(driver_result.get("generation_errors", [])),
        "prompt_sha256": dict(driver_result.get("prompt_sha256", {})),
        "spec_provenance": dict(provenance or {}),
    }
    return ExecutedRun(result=result, report_md=report_md)
