"""one-shot holdout orchestrator — one run, two stages (spec §1–§5).

``run_oneshot_holdout`` wires the pre-run checklist (§1.3) → the holdout gate (§1.1, real path only) →
Stage 1 build (§2) → Stage 2 evaluate (§3), governed by the append-only marker state machine
and the pre-registered failure protocol (§4). ``--rehearsal`` runs the FULL pipeline against a
development pseudo-window mirroring the 45-month geometry — the gate never opens, development
data only — and writes the rehearsal-green marker the real run requires (§5).

The real run never executes here without an explicit written go-ahead and the upstream
preconditions (G6 survivors, P3 artefact, E9) — this module encodes the machinery, not the firing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import pandas as pd

from .. import holdout
from .gate_checklist import ChecklistConfig, GateChecklistResult, run_pre_run_checklist
from .manifest_io import write_manifest
from .marker import (
    FRESH_BUILD,
    RESTART_BUILD,
    Marker,
    RehearsalMarker,
)
from .stage1_build import PanelBuilder, Stage1Result, build_holdout_panel
from .stage2_evaluate import SurvivorInput, SurvivorResult, evaluate_survivors
from .windows import Window

# The development pseudo-window mirroring the real 45-month geometry (§5). Fixed, dev-only.
REHEARSAL_WINDOW = Window(start="2018-01", end="2021-09", n_months=45)


class RehearsalNotGreen(RuntimeError):
    """A --rehearsal pass failed a green criterion (§5); the marker is NOT written."""


@dataclass
class OneshotHoldoutConfig:
    rehearsal: bool
    ts: str
    checklist: ChecklistConfig
    quarantine_dir: Path
    marker_path: Path
    rehearsal_marker_path: Path
    panel_builder: PanelBuilder
    survivors: list[SurvivorInput]
    benchmarks: Mapping[str, pd.DataFrame]
    priors: Mapping[str, float]
    holdout_reader_open: Callable[[], None] | None = None      # real path: opens data/holdout access
    zero_leakage_check: Callable[[pd.DataFrame, Window], None] | None = None


@dataclass(frozen=True)
class OneshotHoldoutRunReport:
    rehearsal: bool
    seed_start: str
    window: Window
    stage1: Stage1Result
    results: list[SurvivorResult]
    manifest: dict
    green: bool


def _serialise_results(results: list[SurvivorResult]) -> list[dict]:
    out = []
    for r in results:
        out.append({
            "survivor_id": r.survivor_id,
            "is_extension_1": r.is_extension_1,
            "benchmarks": {
                b: {"full": br.full}
                for b, br in r.benchmarks.items()
            },
        })
    return out


def _manifest(cfg: OneshotHoldoutConfig, check: GateChecklistResult, stage1: Stage1Result,
              results: list[SurvivorResult], *, holdout_processed: bool) -> dict:
    output_blob = json.dumps(_serialise_results(results), sort_keys=True, default=str).encode()
    return {
        "timestamp": cfg.ts,
        "rehearsal": cfg.rehearsal,
        "code_hash": _dir_code_hash(),
        "config_hash": check.thresholds_fingerprint,
        "data_hash": stage1.artefact_hashes(),
        "output_hash": hashlib.sha256(output_blob).hexdigest(),
        "tokens_cost": "n/a",
        "window": vars(check.window),
        "seed_start": stage1.seed_start,
        "holdout_processed": holdout_processed,
    }


def _dir_code_hash() -> str:
    """Hash of the one-shot holdout package source — the frozen-script fingerprint (§1.1). Excludes ``frozen.py``
    (which HOLDS the committed pin) so the pin is over the other modules and cannot depend on itself."""
    here = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for py in sorted(here.glob("*.py")):
        if py.name == "frozen.py":
            continue
        h.update(py.read_bytes())
    return h.hexdigest()


def _open_holdout_gate(cfg: OneshotHoldoutConfig) -> None:
    """Open the artefact-gated single-access holdout gate (never a date). Requires ALL of: the prereg
    tag present, the frozen one-shot holdout-package hash matching the committed pin (**fail-closed** if unpinned —
    an unpinned/tampered script keeps the gate shut), and the env unlock set; then consumes the single
    access and opens the real reader. Called before EVERY real holdout read — including RESUME_EVALUATE,
    which re-reads the holdout in a fresh process and so must re-gate (defence-in-depth)."""
    if not holdout.holdout_gate_open(
        tag_present=holdout.prereg_tag_present("scientist-prereg", repo_root=cfg.checklist.repo_root),
        frozen_script_hash_matches=(
            cfg.checklist.frozen_script_hash is not None
            and _dir_code_hash() == cfg.checklist.frozen_script_hash
        ),
        env_var_set=holdout.env_unlock_set(),
    ):
        raise holdout.HoldoutViolation("holdout gate is shut (artefact conditions unmet)")
    holdout.assert_single_access()
    if cfg.holdout_reader_open is not None:
        cfg.holdout_reader_open()


def _evaluate_and_manifest(cfg: OneshotHoldoutConfig, check, window, stage1, *, holdout_processed: bool):
    # Clip to the registered window (m3) so the "full" statistic is exactly window.n_months — seed
    # months can never leak into the holdout statistic.
    results = evaluate_survivors(cfg.survivors, cfg.benchmarks, window, cfg.priors)
    manifest = _manifest(cfg, check, stage1, results, holdout_processed=holdout_processed)
    return results, manifest


def _run_rehearsal(cfg: OneshotHoldoutConfig, check: GateChecklistResult) -> OneshotHoldoutRunReport:
    window = REHEARSAL_WINDOW.validate()
    stage1 = build_holdout_panel(
        window=window, quarantine_dir=cfg.quarantine_dir,
        panel_builder=cfg.panel_builder, thresholds_path=cfg.checklist.thresholds_path,
        zero_leakage_check=cfg.zero_leakage_check,
    )
    if not stage1.all_seeded_green():
        bad = [a.name for a in stage1.artefacts if not a.non_nan_at_first_month]
        raise RehearsalNotGreen(f"rolling constructions NaN at the first pseudo-window month: {bad}")
    results, manifest = _evaluate_and_manifest(
        cfg, check, window, stage1, holdout_processed=False,
    )
    write_manifest(manifest, cfg.quarantine_dir)              # persist the provenance record (§3)
    RehearsalMarker(cfg.rehearsal_marker_path).write_green(
        ts=cfg.ts, extra={"seed_start": stage1.seed_start, "data_hash": stage1.artefact_hashes(),
                          "pseudo_window": vars(window)},
    )
    return OneshotHoldoutRunReport(True, stage1.seed_start, window, stage1, results, manifest, green=True)


def _run_real(cfg: OneshotHoldoutConfig, check: GateChecklistResult) -> OneshotHoldoutRunReport:
    window = check.window
    marker = Marker(cfg.marker_path)
    action = marker.plan_next()                                 # raises RerunRefused per §4

    if action in (FRESH_BUILD, RESTART_BUILD):
        _open_holdout_gate(cfg)                      # gate + single-access + real reader (never a date)
        marker.append("STAGE1_STARTED", ts=cfg.ts,
                      extra={"restart": action == RESTART_BUILD, "code_hash": _dir_code_hash()})
        stage1 = build_holdout_panel(
            window=window, quarantine_dir=cfg.quarantine_dir,
            panel_builder=cfg.panel_builder, thresholds_path=cfg.checklist.thresholds_path,
            zero_leakage_check=cfg.zero_leakage_check,
        )
        marker.append("STAGE1_COMPLETE", ts=cfg.ts, extra={"data_hash": stage1.artefact_hashes()})
    else:  # RESUME_EVALUATE — the builder re-reads data/holdout, so it MUST re-gate (M2).
        _open_holdout_gate(cfg)
        stage1 = build_holdout_panel(
            window=window, quarantine_dir=cfg.quarantine_dir,
            panel_builder=cfg.panel_builder, thresholds_path=cfg.checklist.thresholds_path,
            zero_leakage_check=cfg.zero_leakage_check,
        )

    marker.append("STAGE2_STARTED", ts=cfg.ts)
    results, manifest = _evaluate_and_manifest(
        cfg, check, window, stage1, holdout_processed=True,
    )
    write_manifest(manifest, cfg.quarantine_dir)             # persist the provenance record before COMPLETE
    marker.append("COMPLETE", ts=cfg.ts, extra={"output_hash": manifest["output_hash"]})
    return OneshotHoldoutRunReport(False, stage1.seed_start, window, stage1, results, manifest, green=False)


def run_oneshot_holdout(cfg: OneshotHoldoutConfig) -> OneshotHoldoutRunReport:
    """Entry point. Runs the §1.3 checklist, then the rehearsal or the real staged run."""
    check = run_pre_run_checklist(cfg.checklist)
    if cfg.rehearsal:
        return _run_rehearsal(cfg, check)
    return _run_real(cfg, check)
