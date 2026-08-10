"""CLI entrypoint for the one-shot holdout builder/evaluator.

    ./.venv/bin/python scripts/run_oneshot_holdout.py --rehearsal      # dev pseudo-window, gate never opens
    ./.venv/bin/python scripts/run_oneshot_holdout.py                  # REAL one-shot — refused here

``--rehearsal`` runs the FULL one-shot holdout pipeline (checklist → seeded stage-1 build → stage-2 evaluation →
manifest) against the development pseudo-window 2018-01..2021-09 using the ``dev_pseudo_builder``
(stored development artefacts only; the gate never opens). It writes the rehearsal-green marker the
real run requires. The real run is REFUSED here: it reads ``data/holdout/`` and runs only on the
explicit written go-ahead with the preconditions met (G6 survivors, P3 moderate-prior artefact, E9).

NOTE (2026-08-10): the machinery is built but has NOT been executed (nothing was run).
This command IS the closeout validation step — run it to validate the wiring end-to-end.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEV_ROOT = REPO_ROOT / "data" / "development"
MARKER_DIR = REPO_ROOT / "agents" / "scientist" / "experimentalist" / "oneshot_holdout" / "markers"
QUARANTINE = REPO_ROOT / "results" / "scientist" / "oneshot_rehearsal_quarantine"
PRIORS = {"wide": 0.02, "moderate": 0.005, "sceptical": 0.0025}   # moderate = fallback σ (rehearsal)


def _rehearsal_survivors_and_benchmarks():
    """Development stand-ins (non-reportable) to exercise the stage-2 evaluation path in rehearsal:
    two factor series as a survivor/parent pair, the BBW factor frame as the benchmark. The REAL run
    derives these from the built holdout inventory instead (see the holdout_builder recipe)."""
    import pandas as pd

    from agents.scientist.experimentalist.oneshot_holdout.run_oneshot_holdout import REHEARSAL_WINDOW
    from agents.scientist.experimentalist.oneshot_holdout.stage2_evaluate import SurvivorInput

    win = REHEARSAL_WINDOW
    bbw = pd.read_parquet(DEV_ROOT / "factors" / "bbw_factors.parquet")
    bbw["date"] = pd.to_datetime(bbw["date"])
    idx = pd.to_datetime(bbw["date"]).dt.to_period("M")
    lo, hi = pd.Period(win.start, "M"), pd.Period(win.end, "M")
    bbw = bbw[(idx >= lo) & (idx <= hi)].reset_index(drop=True)
    # pick two numeric factor columns as survivor / parent stand-ins
    num = [c for c in bbw.columns if c != "date" and pd.api.types.is_numeric_dtype(bbw[c])]
    surv = pd.Series(bbw[num[0]].to_numpy(), index=bbw["date"])
    parent = pd.Series(bbw[num[1] if len(num) > 1 else num[0]].to_numpy(), index=bbw["date"])
    survivors = [SurvivorInput("rehearsal_standin", surv, parent)]
    benchmarks = {"bbw4": bbw[["date"] + num], "dfps4": bbw[["date"] + num]}
    return survivors, benchmarks


def _run_rehearsal() -> int:
    from agents.scientist.experimentalist.oneshot_holdout.frozen import (
        FROZEN_ONESHOT_HOLDOUT_SCRIPT_HASH,
        FROZEN_THRESHOLDS_SHA256,
    )
    from agents.scientist.experimentalist.oneshot_holdout.gate_checklist import ChecklistConfig
    from agents.scientist.experimentalist.oneshot_holdout.panel_builder import dev_pseudo_builder, zero_leakage_check
    from agents.scientist.experimentalist.oneshot_holdout.run_oneshot_holdout import OneshotHoldoutConfig, run_oneshot_holdout

    thresholds = REPO_ROOT / "docs" / "thresholds.yaml"
    checklist = ChecklistConfig(
        release_tag="sc-sci-13-holdout-inference",
        protocol_path=REPO_ROOT / "docs" / "scientist_protocol.yaml",
        thresholds_path=thresholds,
        thresholds_fingerprint=FROZEN_THRESHOLDS_SHA256,      # committed pin — a real check, not self vs self
        frozen_script_hash=FROZEN_ONESHOT_HOLDOUT_SCRIPT_HASH,
        # Rehearsal-mode fallbacks (logged): the real run requires the genuine P3/E9 artefacts.
        p3_artefact_path=None,
        p3_fallback_reason="rehearsal: dev pseudo-window, moderate σ=0.005 fallback (non-reportable)",
        e9_gross_returns_scoping=True,
        manifest_writer_wired=True,
        rehearsal_marker_path=MARKER_DIR / "oneshot_rehearsal_marker.jsonl",
        require_rehearsal=False,           # we are PRODUCING the rehearsal-green here
        repo_root=REPO_ROOT,
    )
    survivors, benchmarks = _rehearsal_survivors_and_benchmarks()
    cfg = OneshotHoldoutConfig(
        rehearsal=True,
        ts=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        checklist=checklist,
        quarantine_dir=QUARANTINE,
        marker_path=MARKER_DIR / "oneshot_marker.jsonl",
        rehearsal_marker_path=MARKER_DIR / "oneshot_rehearsal_marker.jsonl",
        panel_builder=dev_pseudo_builder(DEV_ROOT),
        survivors=survivors,
        benchmarks=benchmarks,
        priors=PRIORS,
        zero_leakage_check=zero_leakage_check,
    )
    report = run_oneshot_holdout(cfg)
    print(f"one-shot holdout rehearsal GREEN={report.green} seed_start={report.seed_start} "
          f"window={report.window.start}..{report.window.end} sub={report.sub_window.start}..{report.sub_window.end}")
    print(f"  seeded artefacts non-NaN at first month: {report.stage1.all_seeded_green()}")
    print(f"  rehearsal-green marker: {MARKER_DIR / 'oneshot_rehearsal_marker.jsonl'}")
    return 0 if report.green else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="one-shot holdout builder/evaluator")
    parser.add_argument("--rehearsal", action="store_true",
                        help="run the full pipeline on the development pseudo-window (gate never opens)")
    args = parser.parse_args(argv)

    if args.rehearsal:
        return _run_rehearsal()

    print(
        "one-shot holdout real run: REFUSED. This is the single code path that reads data/holdout/ and it "
        "executes only on an explicit written go-ahead, with the holdout panel builder wired and "
        "the preconditions met (G6 survivors, P3 moderate-prior artefact, E9). None of that is this "
        "component's job — build + rehearsal-green is where one-shot holdout stops (spec §7).",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
