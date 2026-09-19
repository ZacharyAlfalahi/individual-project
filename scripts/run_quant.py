"""
scripts/run_quant.py — the RQ2 "B1" end-to-end StrategySpec -> run orchestration driver.

  ./.venv/bin/python scripts/run_quant.py --anchor mom6
  ./.venv/bin/python scripts/run_quant.py --anchor all
  ./.venv/bin/python scripts/run_quant.py --anchor all --basis clean   # consistent-basis run

`--basis {total_return,clean}` swaps the base panel for `scripts/basis_inputs.PANELS[basis][0]` and
(unless `--out` is given) writes under `results/consistent_basis/<basis>/quant/`; the
run_log then also records the basis and the base panel's sha256. No `--basis` => the default
base panel, output dir and run_log below.

This is the deterministic compilation spine RQ2 asks for: a supported anchor strategy
compiles StrategySpec -> adapter -> QuantConfig -> audited
runner and RUNS on the development panel; an unsupported family (or a spec the engine stack
cannot represent) emits a TYPED `ConfigRefusal`, recorded for the RQ2 coverage denominator
rather than crashing. It is "`scripts/run_auditor.py` minus the 2^k bias-toggle lattice":
it runs the *adapter-compiled* rulebook (each anchor's OWN construction), which is distinct
from `scripts/run_anchor_descriptive.py` (that reuses each builder's hand-written run_family).

Development panel ONLY (2002-2021). The frozen holdout is NEVER read. No language model
appears in this path — compilation is deterministic (closed-enum family table + typed
refusal); the LLM explainer sits downstream of the typed outcome, not here.

Design choices (flagged in the run_log, single point of truth):

  * base panel  = `data/development/monthly_panel_total_return.parquet` — the O5 D-Q3
    total-return basis (a legal basis panel: accrued interest + coupon, NOT a forbidden
    endpoint export).
  * run_config  = `run_config.corrected()`, but ONLY its `panel_view` is consumed —
    `views.view()` reads `panel_view` alone (corr family + stale mask on + terminal rows
    kept). The construction block (signal_lag / expost_trim) is NOT applied: B1 runs
    each anchor's OWN compiled QuantConfig (the un-overridden construction), which is the
    key difference from `cell_runner._override_construction` (the lattice path). So a run
    here uses e.g. str's signal_lag=0 and mom6's H=6 exactly as the gold compiles them.
  * safe_rate / benchmark = None for every anchor. The anchors are single-leg long-short
    spreads (P10-P1 deciles / a factor-mimicking portfolio) that net the risk-free rate out
    internally (summarize_returns, characteristic_sort spec section 6: "long-short already
    nets it out"), so no external rf subtraction is owed. It is also uniform: mom6 (H=6) is
    on the overlap path, which `run_from_config` forbids to take either (runner.py:78-87);
    passing None keeps one legal code path across all three anchors.
  * standing substitutions are loaded AND hash-verified (contract section 6). Without them
    str (weighting_base=market_value -> par) and mom6 (expost_trim=truncate -> lab_trim
    delegation) would REFUSE. An absent/edited table is caught fail-loud via verify_hash.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.auditor.ipca_differential.runner import load_dev_signals  # noqa: E402
from shared.licensed_inputs import require_licensed_input  # noqa: E402
from agents.librarian.adapter.adapt import adapt_spec  # noqa: E402
from agents.librarian.adapter.result import AdaptResult  # noqa: E402
from agents.librarian.registries.standing_substitutions import (  # noqa: E402
    STANDING_SUBS_V1_SHA256,
    StandingSubstitutionTable,
    load_standing_substitutions,
)
from agents.quant.config.coverage import layered_coverage  # noqa: E402
from agents.quant.config.ledger_check import (  # noqa: E402
    check_assumptions,
    load_ledger_check_table,
)
from agents.quant.config.runner import StrategyResult, run_strategy  # noqa: E402
from agents.quant.library.run_config import RunConfig, corrected  # noqa: E402
from agents.quant.library.views import view  # noqa: E402
from evaluation.gold_specs.gold_loader import load_gold_spec  # noqa: E402
from scripts import basis_inputs  # noqa: E402
from shared.reporting.run_manifest import (  # noqa: E402
    build_run_manifest,
    write_run_manifest,
)

ANCHORS = ("str", "drf", "mom6")

# The single representative base panel + run_config baseline (documented in the module
# docstring). BASE_PANEL is a dev-window total-return panel; the holdout is never referenced.
BASE_PANEL = REPO_ROOT / "data" / "development" / "monthly_panel_total_return.parquet"
RUN_CONFIG_LABEL = (
    "corrected (panel_view only: corr family + stale_mask on + terminal rows kept); "
    "construction is each anchor's own compiled config, NOT overridden"
)


def resolve_base_panel(basis: str | None) -> Path:
    """The base panel for a run: ``BASE_PANEL`` when no basis is named (the default),
    else the consistent-basis maximal-format panel ``basis_inputs.PANELS[basis][0]``."""
    if basis is None:
        return BASE_PANEL
    return basis_inputs.PANELS[basis_inputs.check_basis(basis)][0]


def default_out_dir(basis: str | None) -> Path:
    """``results/quant/run`` without a basis; with one, the same leaf under
    ``results/consistent_basis/<basis>/quant/`` so a basis run never overwrites a default
    run dir."""
    leaf = "run"
    if basis is None:
        return REPO_ROOT / "results" / "quant" / leaf
    return basis_inputs.basis_dir(basis, "quant", leaf)


def _repo_rel(path: Path) -> str:
    """Repo-relative path string where possible (run_log convention), else the absolute path."""
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------
# Inputs (dev only; holdout untouched) — mirrors run_auditor.py:84-90.
# --------------------------------------------------------------------------

def load_standing_subs_verified() -> StandingSubstitutionTable:
    """The pre-registered standing-substitutions table, hash-checked against the recorded
    constant (contract section 6). Fail-loud if the file is absent/edited — the value-weighted
    (str) and truncation-delegating (mom6) anchors REFUSE without it, so a silent empty table
    would silently change which anchors compile."""
    subs = load_standing_substitutions()
    if not subs.verify_hash(STANDING_SUBS_V1_SHA256):
        raise ValueError(
            "standing-substitutions file hash does not match STANDING_SUBS_V1_SHA256 "
            "(contract section 6): refusing to compile anchors -- the section 6 conventions "
            "must match the recorded pre-registration before any real run."
        )
    return subs


def materialise_panel(
    base_panel: pd.DataFrame,
    signals: pd.DataFrame,
    run_config: RunConfig,
) -> pd.DataFrame:
    """The engine-shape panel every leg runs against, materialised exactly as the auditor
    lattice does: `view(base_panel, run_config, signals=signals)` resolves the family, applies
    the stale mask / terminal-row policy from `run_config.panel_view`, and attaches the
    family-resolved signal columns. Pure — no mutation of the inputs."""
    return view(base_panel, run_config, signals=signals)


def load_inputs(
    run_config: RunConfig, *, base_panel: Path | None = None
) -> tuple[pd.DataFrame, StandingSubstitutionTable]:
    """(view'd engine-shape panel, verified standing-subs). Reads the base dev panel
    (`base_panel`, default `BASE_PANEL`) + the 4 dev signal parquets (`load_dev_signals`), views
    them once, and verifies the standing table. Every path is under data/development/ — the
    holdout is never opened."""
    subs = load_standing_subs_verified()
    base = pd.read_parquet(require_licensed_input(BASE_PANEL if base_panel is None else base_panel, "development panel"))
    signals = load_dev_signals()
    panel = materialise_panel(base, signals, run_config)
    return panel, subs


# --------------------------------------------------------------------------
# Compile + run one anchor — the testable core (branch on AdaptResult.refused).
# --------------------------------------------------------------------------

def adapt_anchor(anchor_id: str, subs: StandingSubstitutionTable) -> AdaptResult:
    """Compile one anchor: gold spec -> `adapt_spec` (with the hash-verified standing subs).
    Same chain the G2 harness uses; returns the `AdaptResult` (which may be `.refused`)."""
    return adapt_spec(load_gold_spec(anchor_id), standing_subs=subs)


def _num(x: object) -> float | None:
    """A JSON-safe finite float, or None for a non-finite value (a raw-family price blow-up
    could yield inf/NaN; None keeps the summary strict-JSON valid). Passes None through."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def summarize_run(result: StrategyResult) -> dict:
    """The curated typed summary recorded per run. `mean_pct_per_month` (= average * 100, the
    headline factor level in the anchor golds' unit) sits beside the raw `average`, the t-stat,
    n_months and the standard engine summary fields."""
    s = result.summary
    return {
        "mean_pct_per_month": _num(s["average"] * 100) if s.get("average") is not None else None,
        "average": _num(s["average"]),
        "t_stat": _num(s["t_stat"]),
        "sharpe": _num(s["sharpe"]),
        "annualised_average": _num(s.get("annualised_average")),
        "bumpiness": _num(s.get("bumpiness")),
        "n_months": int(s["n_months"]),
        "first_date": s.get("first_date"),
        "last_date": s.get("last_date"),
        "months_per_year": s.get("months_per_year"),
        "nw_lags_used": s.get("nw_lags_used"),
        "avg_bonds_per_month": _num(s.get("avg_bonds_per_month")),
    }


def record_anchor(
    anchor_id: str,
    result: AdaptResult,
    panel: pd.DataFrame,
    *,
    safe_rate: pd.DataFrame | None = None,
    benchmark: pd.DataFrame | None = None,
    ledger_refusals: tuple = (),
) -> dict:
    """One anchor's serialisable outcome record. A refused compile is recorded as a TYPED
    refusal (the RQ2 coverage branch); a compiled strategy is run through `run_strategy` and
    recorded with its summary. This is the whole B1 decision, isolated so the unit test can
    exercise BOTH branches without the real loaders."""
    if result.refused:
        return {
            "anchor": anchor_id,
            "status": "refused",
            "variant": result.variant,
            "refusals": [r.to_dict() for r in result.refusals],
        }

    # D28/D29 ledger gate: a strategy that COMPILED but whose STATED Part-2 fields contradict
    # a silently-fixed engine assumption REFUSES (ASSUMPTION_MISMATCH) rather than running a
    # different assumption and calling it a replication. `ledger_refusals` is empty for the
    # strategy whose stated fields match the engine assumptions; it lands in the coverage
    # denominator like any refusal.
    # Threaded here so check_assumptions (ledger_check.py, D28/D29) gates every compiled strategy.
    if ledger_refusals:
        return {
            "anchor": anchor_id,
            "status": "refused",
            "variant": result.variant,
            "refusals": [r.to_dict() for r in ledger_refusals],
        }

    run_result = run_strategy(result, panel, safe_rate=safe_rate, benchmark=benchmark)
    # run_strategy only returns the AdaptResult unrun when result.refused (handled above),
    # so a non-refused compile always yields a StrategyResult here.
    if not isinstance(run_result, StrategyResult):
        raise RuntimeError(
            f"run_strategy returned {type(run_result).__name__} for non-refused anchor "
            f"{anchor_id!r}; a compiled strategy must run to a StrategyResult"
        )
    return {
        "anchor": anchor_id,
        "status": "run",
        "strategy_label": run_result.strategy_label,
        "variant": run_result.variant,
        "n_legs": run_result.n_legs,
        "combiner": run_result.combiner,
        "summary": summarize_run(run_result),
    }


# --------------------------------------------------------------------------
# Coverage aggregation (the RQ2 coverage denominator seed).
# --------------------------------------------------------------------------

def build_coverage(records: list[dict]) -> dict:
    """The RQ2 coverage seed: compiled / run / refused counts over the candidate set, plus
    each anchor's status and its typed refusal codes (empty for a run). `compiled` = strategies
    that compiled StrategySpec -> runnable config (non-refused); `run` = those executed to a
    finite StrategyResult; `refused` = those that emitted a typed refusal at compile time. For
    the supported anchor set compiled == run and refused == 0."""
    per_anchor: dict[str, dict] = {}
    strategies: list[dict] = []
    for r in records:
        codes = [ref["code"] for ref in r.get("refusals", [])]
        per_anchor[r["anchor"]] = {"status": r["status"], "refusal_codes": codes}
        strategies.append({"strategy_id": r["anchor"], "refusal_codes": codes})
    n_run = sum(1 for r in records if r["status"] == "run")
    n_refused = sum(1 for r in records if r["status"] == "refused")
    return {
        "n_candidates": len(records),
        "compiled": n_run,
        "run": n_run,
        "refused": n_refused,
        "per_anchor": per_anchor,
        # §5.2 layered coverage (agents/quant/config/coverage.py): C_semantic / C_binding /
        # C_execution / C_end_to_end + refusals-by-layer. Anchor set = all supported ⇒ 1.0.
        "layered": layered_coverage(strategies),
    }


# --------------------------------------------------------------------------
# Serialisation + run identity (mirrors run_auditor.py write_results / run_log).
# --------------------------------------------------------------------------

def _thresholds_sha256() -> str:
    import hashlib

    return hashlib.sha256((REPO_ROOT / "docs" / "thresholds.yaml").read_bytes()).hexdigest()


def build_run_log(
    subs: StandingSubstitutionTable, run_config: RunConfig, anchors: Iterable[str], records: list[dict],
    *, basis: str | None = None,
) -> dict:
    """The run identity record. Without a basis it is exactly the default run_log (so the
    recorded-run comparators see no additional keys); with one it additionally records the basis
    and the base panel's sha256 beside its path."""
    base_panel = resolve_base_panel(basis)
    log = {
        "standing_subs_version": subs.version,
        "standing_subs_sha256": STANDING_SUBS_V1_SHA256,
        "thresholds_sha256": _thresholds_sha256(),
        "base_panel": _repo_rel(base_panel),
        "run_config_label": RUN_CONFIG_LABEL,
        "run_config_panel_view": {
            "price_family": run_config.panel_view.price_family,
            "stale_mask": run_config.panel_view.stale_mask,
            "include_terminal_rows": run_config.panel_view.include_terminal_rows,
        },
        "run_config_panel_view_hash": run_config.panel_view_hash(),
        "safe_rate": None,
        "benchmark": None,
        "safe_rate_note": (
            "None for every anchor: single-leg long-short spreads net rf internally "
            "(characteristic_sort spec section 6); and the mom6 H=6 overlap path forbids "
            "both safe_rate and benchmark (runner.py:78-87)."
        ),
        "window": "development 2002-2021 (holdout untouched)",
        "anchors_requested": list(anchors),
        "anchors_run": [r["anchor"] for r in records if r["status"] == "run"],
        "anchors_refused": [r["anchor"] for r in records if r["status"] == "refused"],
    }
    if basis is not None:
        log["basis"] = basis
        log["base_panel_sha256"] = basis_inputs.sha256(base_panel)
    return log


def write_results(out_dir: Path, records: list[dict], coverage: dict, run_log: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_log.json").write_text(json.dumps(run_log, indent=2, default=str))
    (out_dir / "coverage.json").write_text(json.dumps(coverage, indent=2, default=str))
    for r in records:
        (out_dir / f"{r['anchor']}.json").write_text(json.dumps(r, indent=2, default=str))


# --------------------------------------------------------------------------
# Orchestration.
# --------------------------------------------------------------------------

def run_all(anchors: Iterable[str], *, out_dir: Path | None = None, basis: str | None = None) -> int:
    """Load inputs once, compile+run every requested anchor, write per-anchor JSON + coverage
    + run_log, print a summary. Exit 0 iff every requested anchor reached a terminal typed
    outcome AND (as all anchors are supported) every one RAN with a finite mean and 0 refusals.
    `basis` (None = the default) selects the base panel and the default output dir."""
    anchors = list(anchors)
    run_config = corrected()
    base_panel = resolve_base_panel(basis)
    panel, subs = load_inputs(run_config, base_panel=base_panel)

    # D28/D29 ledger gate for the corpus set: a compiled anchor whose STATED fields
    # contradict a silently-fixed engine assumption refuses. Loaded once; an anchor whose
    # stated fields match the engine assumptions runs as compiled.
    ledger_table = load_ledger_check_table()
    records = []
    for a in anchors:
        result = adapt_anchor(a, subs)
        ledger_refs = () if result.refused else check_assumptions(load_gold_spec(a), ledger_table)
        records.append(record_anchor(a, result, panel, ledger_refusals=ledger_refs))
    coverage = build_coverage(records)
    run_log = build_run_log(subs, run_config, anchors, records, basis=basis)

    out_dir = out_dir or default_out_dir(basis)
    write_results(out_dir, records, coverage, run_log)
    # WS-8 (O11): a unified per-run execution manifest sidecar — timestamp + code/data/config/
    # output hashes + an operational profile. A SIDECAR: never hashed into a result artefact.
    write_run_manifest(out_dir, build_run_manifest(
        run_id=out_dir.name,
        driver="run_quant",
        timestamp=datetime.now(timezone.utc).isoformat(),
        inputs=[_repo_rel(base_panel)],
        configs=["docs/thresholds.yaml", "agents/quant/config/data/ledger_check_table.yaml"],
        outputs=[str(out_dir / "run_log.json"), str(out_dir / "coverage.json")],
    ))

    run_records = [r for r in records if r["status"] == "run"]
    all_finite = all(r["summary"]["mean_pct_per_month"] is not None for r in run_records)
    all_pass = (
        bool(records)
        and coverage["refused"] == 0
        and len(run_records) == len(records)
        and all_finite
    )
    print(json.dumps(
        {
            "out_dir": str(out_dir),
            "coverage": coverage,
            "levels": {
                r["anchor"]: {
                    "mean_pct_per_month": r["summary"]["mean_pct_per_month"],
                    "t_stat": r["summary"]["t_stat"],
                    "n_months": r["summary"]["n_months"],
                }
                for r in run_records
            },
            "ALL_RUN_FINITE": all_pass,
        },
        indent=2, default=str,
    ))
    return 0 if all_pass else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument(
        "--anchor", choices=(*ANCHORS, "all"), default="all",
        help="which anchor to compile+run (default: all three)",
    )
    ap.add_argument(
        "--out", type=Path, default=None,
        help="output dir (default results/quant/run; with --basis, "
             "results/consistent_basis/<basis>/quant/run)",
    )
    ap.add_argument(
        "--basis", choices=basis_inputs.BASES, default=None,
        help="consistent-basis base panel (basis_inputs.PANELS[basis][0]); "
             "omit for the default monthly_panel_total_return.parquet",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    anchors = ANCHORS if args.anchor == "all" else (args.anchor,)
    return run_all(anchors, out_dir=args.out, basis=args.basis)


if __name__ == "__main__":
    raise SystemExit(main())
