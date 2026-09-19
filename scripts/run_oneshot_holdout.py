"""CLI entrypoint for the one-shot holdout builder/evaluator.

    ./.venv/bin/python scripts/run_oneshot_holdout.py --rehearsal      # dev pseudo-window, gate never opens
    ./.venv/bin/python scripts/run_oneshot_holdout.py --real           # REAL run — gated (below)

``--rehearsal`` runs the FULL one-shot holdout pipeline (checklist → seeded stage-1 build → stage-2 evaluation →
manifest) against the development pseudo-window 2018-01..2021-09 using the ``dev_pseudo_builder``
(stored development artefacts only; the gate never opens). It writes the rehearsal-green marker the
real run requires. The real run (``--real``, below) reads ``data/holdout/`` and runs only on an
explicit go-ahead with the preconditions met (G6 survivors, P3 moderate-prior artefact, E9).

``--rehearsal`` validates the pipeline end to end.

REAL RUN. ``--real`` refuses unless ALL hold: the unlock env
``SCIENTIST_HOLDOUT_UNLOCK=1``; the holdout-inventory gate (``HOLDOUT_OOS_GATE=APPROVED_HOLDOUT_OPEN`` and
``HOLDOUT_OOS_OPEN=1``); the development preparation artefacts written BEFORE the open by
``scripts/oneshot_holdout_prepare_dev_pins.py`` (the P3 moderate-prior artefact and the development pins); and the §1.3
checklist (tag, SC-SCI-12, thresholds fingerprint, rehearsal-green marker, frozen script hash). The inventory
is built by ``panel_builder.holdout_builder`` (one holdout read), the stage-2 inputs are derived from it by
``scripts/oneshot_holdout_survivor_inputs`` (survivors regenerated from the funnel's cache, executed on the seeded panel;
BBW-4 built on it), and the seeded development portion of every series must reproduce its development pin
before stage 2 runs. Output: <out>/oneshot_report.json; quarantine results/scientist/oneshot_quarantine/.
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
    """Development placeholders (non-reportable) to exercise the stage-2 evaluation path in rehearsal:
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
    # pick two numeric factor columns as survivor / parent placeholders
    num = [c for c in bbw.columns if c != "date" and pd.api.types.is_numeric_dtype(bbw[c])]
    surv = pd.Series(bbw[num[0]].to_numpy(), index=bbw["date"])
    parent = pd.Series(bbw[num[1] if len(num) > 1 else num[0]].to_numpy(), index=bbw["date"])
    survivors = [SurvivorInput("rehearsal_placeholder", surv, parent)]
    benchmarks = {"bbw4": bbw[["date"] + num]}
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
        thresholds_fingerprint=FROZEN_THRESHOLDS_SHA256,      # pinned value — a real check, not self vs self
        frozen_script_hash=FROZEN_ONESHOT_HOLDOUT_SCRIPT_HASH,
        # Rehearsal-mode fallbacks (logged): the real run requires the genuine P3/E9 artefacts.
        p3_artefact_path=None,
        p3_fallback_reason="rehearsal: dev pseudo-window, moderate σ=0.005 fallback (non-reportable)",
        e9_gross_returns_scoping=True,
        manifest_writer_wired=True,
        rehearsal_marker_path=MARKER_DIR / "oneshot_rehearsal_marker.jsonl",
        require_rehearsal=False,           # this run PRODUCES the rehearsal-green marker
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
          f"window={report.window.start}..{report.window.end}")
    print(f"  seeded artefacts non-NaN at first month: {report.stage1.all_seeded_green()}")
    print(f"  rehearsal-green marker: {MARKER_DIR / 'oneshot_rehearsal_marker.jsonl'}")
    return 0 if report.green else 1


ONESHOT_HOLDOUT_RESULTS = REPO_ROOT / "results" / "consistent_basis" / "clean" / "rq4" / "oneshot_holdout"
HOLDOUT_QUARANTINE = REPO_ROOT / "results" / "scientist" / "oneshot_quarantine"


def _run_real(*, out_dir: Path = ONESHOT_HOLDOUT_RESULTS, audit_report: Path | None = None, funnel_artefact: Path | None = None) -> int:
    """The gated real run. Every refusal here happens BEFORE any holdout read."""
    import hashlib
    import json
    import os

    from agents.scientist.experimentalist import holdout
    from agents.scientist.experimentalist.oneshot_holdout.frozen import FROZEN_ONESHOT_HOLDOUT_SCRIPT_HASH, FROZEN_THRESHOLDS_SHA256
    from agents.scientist.experimentalist.oneshot_holdout.gate_checklist import ChecklistConfig
    from agents.scientist.experimentalist.oneshot_holdout.panel_builder import holdout_builder, zero_leakage_check
    from agents.scientist.experimentalist.oneshot_holdout.run_oneshot_holdout import OneshotHoldoutConfig, _dir_code_hash, run_oneshot_holdout
    from scripts import holdout_inventory as HI
    from scripts import oneshot_holdout_survivor_inputs as SI

    root = REPO_ROOT / "results" / "consistent_basis" / "clean"
    audit_report = audit_report or root / "audit" / "full_run" / "str_report.json"
    funnel_artefact = funnel_artefact or (root / "rq4" / "funnel" / "rq4_funnel_reported_minilm.json")
    p3_path, pins_path = out_dir / "p3_moderate_prior.json", out_dir / "oneshot_dev_pins.json"
    for req in (p3_path, pins_path):
        if not req.is_file():
            print(f"REFUSED: development preparation artefact missing: {req} (run scripts/oneshot_holdout_prepare_dev_pins.py)",
                  file=sys.stderr)
            return 2
    # Stage-2 derivation reads these AFTER the single holdout access is spent, so their absence is
    # checked here: a missing input must never cost the one open this run is allowed.
    for req in (audit_report, funnel_artefact):
        if not Path(req).is_file():
            print(f"REFUSED: stage-2 derivation input missing: {req} (the real run reads it after the open, "
                  f"so it is required before the holdout is touched)", file=sys.stderr)
            return 2
    if not holdout.env_unlock_set():
        print("REFUSED: SCIENTIST_HOLDOUT_UNLOCK is not set — the real run executes only on an explicit go-ahead",
              file=sys.stderr)
        return 2
    gate = os.environ.get("HOLDOUT_OOS_GATE")
    try:
        HI._require_gate(gate)
    except HI.HoldoutGateError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    if _dir_code_hash() != FROZEN_ONESHOT_HOLDOUT_SCRIPT_HASH:
        print("REFUSED: the one-shot holdout package hash does not match the pinned value (re-freeze before the real run)",
              file=sys.stderr)
        return 2
    report_out = out_dir / "oneshot_report.json"
    if report_out.exists():
        print(f"REFUSED: {report_out} already exists — the real run has a result; there is no re-run", file=sys.stderr)
        return 2
    pins = json.loads(pins_path.read_text())
    p3 = json.loads(p3_path.read_text())
    priors = {"wide": float(p3["sigma_wide"]), "moderate": float(p3["sigma_moderate"]), "sceptical": float(p3["sigma_sceptical"])}

    checklist = ChecklistConfig(
        release_tag="sc-sci-13-holdout-inference",
        protocol_path=REPO_ROOT / "docs" / "scientist_protocol.yaml",
        thresholds_path=REPO_ROOT / "docs" / "thresholds.yaml",
        thresholds_fingerprint=FROZEN_THRESHOLDS_SHA256,
        frozen_script_hash=FROZEN_ONESHOT_HOLDOUT_SCRIPT_HASH,
        p3_artefact_path=p3_path,
        p3_fallback_reason=None,
        e9_gross_returns_scoping=True,          # E9: the RQ4 evaluation is scoped to GROSS returns (as the funnel)
        manifest_writer_wired=True,
        rehearsal_marker_path=MARKER_DIR / "oneshot_rehearsal_marker.jsonl",
        require_rehearsal=True,
        repo_root=REPO_ROOT,
    )
    holder: dict = {"feed_path": HOLDOUT_QUARANTINE / "_seeded_ipca_feed_corr.parquet"}
    builder = holdout_builder(DEV_ROOT, gate=gate, holdout_daily_dir=REPO_ROOT / "data" / "holdout", holder=holder)

    def derive(stage1):
        seeded = {"maximal": holder["maximal"], "signals": holder["signals"], "macros": holder["macros"],
                  "factors_bbw": holder["full_factors_bbw"], "factors_mktb": holder["full_factors_mktb"]}
        survivors, benchmarks, record = SI.derive_stage2_inputs(
            seeded, audit_report=audit_report, funnel_artefact=funnel_artefact, dev_pins=pins)
        record["macro_seeding"] = holder.get("macro_record")
        return survivors, benchmarks, record

    cfg = OneshotHoldoutConfig(
        rehearsal=False,
        ts=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        checklist=checklist,
        quarantine_dir=HOLDOUT_QUARANTINE,
        marker_path=MARKER_DIR / "oneshot_marker.jsonl",
        rehearsal_marker_path=MARKER_DIR / "oneshot_rehearsal_marker.jsonl",
        panel_builder=builder,
        survivors=[], benchmarks={},
        priors=priors,
        holdout_reader_open=None,                # the inventory's own gate opens the reader inside the builder
        zero_leakage_check=zero_leakage_check,
        derive_inputs=derive,
    )
    report = run_oneshot_holdout(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "purpose": "One-shot holdout: the RQ4 survivors (clean-price basis) evaluated on the protected holdout window; "
                   "descriptive only — sign, level, paired difference, intervals, posteriors; no pass/fail",
        "run_ts": cfg.ts, "window": {"start": report.window.start, "end": report.window.end, "n_months": report.window.n_months},
        "seed_start": report.seed_start, "priors": priors, "p3_artefact": str(p3_path.relative_to(REPO_ROOT)),
        "p3_sha256": hashlib.sha256(p3_path.read_bytes()).hexdigest(),
        "dev_pins": str(pins_path.relative_to(REPO_ROOT)), "dev_pins_sha256": hashlib.sha256(pins_path.read_bytes()).hexdigest(),
        "e9_cost_decision": "gross-returns scoping (the RQ4 evaluation is gross of costs, as the development funnel)",
        "inputs": {"audit_report": str(audit_report.relative_to(REPO_ROOT)), "funnel_artefact": str(funnel_artefact.relative_to(REPO_ROOT)),
                   "basis": "clean", "holdout_daily_dir": "data/holdout"},
        "stage1": {a.name: {"sha256": a.sha256, "n_rows": a.n_rows, "non_nan_at_first_month": a.non_nan_at_first_month}
                   for a in report.stage1.artefacts},
        "survivors": [_survivor_to_dict(r) for r in report.results],
        "manifest": report.manifest,
    }
    report_out.write_text(json.dumps(doc, indent=2, default=str))
    print(f"one-shot holdout real run COMPLETE: {len(report.results)} survivors evaluated on {report.window.start}..{report.window.end}")
    print(f"  report: {report_out}")
    return 0


def _survivor_to_dict(r) -> dict:
    import dataclasses
    d = {"survivor_id": r.survivor_id, "is_extension_1": r.is_extension_1, "benchmarks": {}}
    for name, br in r.benchmarks.items():
        d["benchmarks"][name] = br.to_dict() if hasattr(br, "to_dict") else dataclasses.asdict(br)
    return d


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="one-shot holdout builder/evaluator")
    parser.add_argument("--rehearsal", action="store_true",
                        help="run the full pipeline on the development pseudo-window (gate never opens)")
    parser.add_argument("--real", action="store_true",
                        help="the gated REAL run (unlock env + holdout-inventory gate env + dev preparation)")
    args = parser.parse_args(argv)

    if args.rehearsal:
        return _run_rehearsal()
    if args.real:
        return _run_real()

    print(
        "one-shot holdout: choose --rehearsal (development pseudo-window) or --real (gated; reads "
        "data/holdout/ only with the unlock env, the holdout-inventory gate and the development "
        "preparation artefacts).",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
