"""Pre-run gate checklist (one-shot holdout §1.3) — every check fail-loud, each individually testable.

The checklist runs BEFORE the holdout gate is ever consulted. It validates the window
(the landed two-source guard), the release provenance (tag reachable, SC-SCI-12 APPROVED
+ discharged), the frozen configuration fingerprint, and the presence of the upstream
artefacts the run depends on (P3 moderate-prior, E9 cost decision, run manifest wiring,
rehearsal-green). It does NOT open the gate — it returns the validated windows and
fingerprint for the orchestrator, or raises on the first failure.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .. import holdout
from .marker import RehearsalMarker
from .windows import Window, derive_sensitivity_subwindow, registered_window

REPO_ROOT = Path(__file__).resolve().parents[4]


class OneshotHoldoutGateError(RuntimeError):
    """A §1.3 pre-run checklist item failed. Fail-loud — nothing missing is ever defaulted
    or waved through; anything absent aborts before the gate opens."""


@dataclass(frozen=True)
class ChecklistConfig:
    """Injectable inputs for the checklist (fixtures in tests; real paths/values at run)."""

    release_tag: str = "sc-sci-13-holdout-inference"
    protocol_path: Path | None = None
    thresholds_path: Path | None = None
    thresholds_fingerprint: str | None = None       # committed frozen sha256 (hex)
    frozen_script_hash: str | None = None            # committed one-shot holdout-package hash; gate stays shut if None
    p3_artefact_path: Path | None = None             # SC-SCI-11 moderate-prior artefact
    p3_fallback_reason: str | None = None            # permits σ=0.005 fallback IF logged
    e9_cost_model_id: str | None = None              # pinned cost model ...
    e9_gross_returns_scoping: bool = False           # ... OR the gross-returns flag
    manifest_writer_wired: bool = False              # Extension-3 run-manifest emitter present
    rehearsal_marker_path: Path | None = None
    require_rehearsal: bool = True                   # False only inside --rehearsal itself
    repo_root: Path = REPO_ROOT


@dataclass(frozen=True)
class GateChecklistResult:
    window: Window
    sub_window: Window
    thresholds_fingerprint: str
    checks_passed: tuple[str, ...] = field(default_factory=tuple)


def _tag_reachable_from_head(tag: str, repo_root: Path) -> bool:
    try:
        subprocess.run(
            ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", tag, "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return True
    except Exception:
        return False


def _find_amendment(protocol: dict, amendment_id: str) -> dict | None:
    """Locate the amendment dict with the given id anywhere in the parsed protocol."""
    def walk(node):
        if isinstance(node, dict):
            if node.get("id") == amendment_id:
                yield node
            for v in node.values():
                yield from walk(v)
        elif isinstance(node, list):
            for v in node:
                yield from walk(v)
    return next(iter(walk(protocol)), None)


# --- individual checks (each raises OneshotHoldoutGateError) ------------------------------------------

def check_window(cfg: ChecklistConfig) -> tuple[Window, Window]:
    """§1.2 — window assertion first (the two-source guard), then DERIVE the sub-window."""
    triple = holdout.assert_evaluation_window(
        protocol_path=cfg.protocol_path, repo_root=cfg.repo_root,
    )
    window = registered_window(triple)
    return window, derive_sensitivity_subwindow(window)


def check_release_tag(cfg: ChecklistConfig) -> None:
    if not _tag_reachable_from_head(cfg.release_tag, cfg.repo_root):
        raise OneshotHoldoutGateError(f"release tag {cfg.release_tag!r} is not reachable from HEAD")


def check_scsci12_approved(cfg: ChecklistConfig) -> None:
    path = Path(cfg.protocol_path) if cfg.protocol_path else cfg.repo_root / "docs" / "scientist_protocol.yaml"
    try:
        protocol = yaml.safe_load(open(path))
    except FileNotFoundError as exc:
        raise OneshotHoldoutGateError(f"scientist_protocol.yaml not found: {path}") from exc
    amendment = _find_amendment(protocol, "SC-SCI-12")
    if amendment is None:
        raise OneshotHoldoutGateError("SC-SCI-12 amendment not found in scientist_protocol.yaml")
    if str(amendment.get("status", "")).strip().upper() != "APPROVED":
        raise OneshotHoldoutGateError(f"SC-SCI-12 status is not APPROVED (got {amendment.get('status')!r})")
    summary = str(amendment.get("summary", ""))
    # Discharge line: the frontier-completeness condition recorded as met/satisfied/verified. Require a
    # standalone positive token (or the specific "PI-verified" phrase) so a negation cannot fail-open
    # the gate — a bare substring would match "UNMET" (contains MET) or "not yet verified".
    upper = summary.upper()
    if not (re.search(r"\b(MET|SATISFIED)\b", upper) or "PI-VERIFIED" in upper):
        raise OneshotHoldoutGateError("SC-SCI-12 discharge line (condition met/satisfied/PI-verified) absent from its summary")


def check_thresholds_fingerprint(cfg: ChecklistConfig) -> str:
    path = Path(cfg.thresholds_path) if cfg.thresholds_path else cfg.repo_root / "docs" / "thresholds.yaml"
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError as exc:
        raise OneshotHoldoutGateError(f"thresholds.yaml not found: {path}") from exc
    if cfg.thresholds_fingerprint is None:
        raise OneshotHoldoutGateError("no frozen thresholds fingerprint supplied to compare against")
    if digest != cfg.thresholds_fingerprint:
        raise OneshotHoldoutGateError(
            f"thresholds.yaml fingerprint mismatch: file={digest[:16]}… "
            f"!= frozen={cfg.thresholds_fingerprint[:16]}…"
        )
    return digest


def check_p3_artefact(cfg: ChecklistConfig) -> None:
    if cfg.p3_artefact_path is not None and Path(cfg.p3_artefact_path).exists():
        if Path(cfg.p3_artefact_path).stat().st_size == 0:
            raise OneshotHoldoutGateError(f"P3 moderate-prior artefact is empty: {cfg.p3_artefact_path}")
        return
    if cfg.p3_fallback_reason:                      # σ=0.005 fallback permitted only if logged
        return
    raise OneshotHoldoutGateError(
        "P3 moderate-prior artefact absent and no fallback reason logged "
        "(σ=0.005 fallback is permitted only with the reason recorded, §1.3)"
    )


def check_e9_cost_decision(cfg: ChecklistConfig) -> None:
    if cfg.e9_cost_model_id or cfg.e9_gross_returns_scoping:
        return
    raise OneshotHoldoutGateError("E9 cost decision not recorded (pin a cost model or set the gross-returns scoping flag)")


def check_manifest_wired(cfg: ChecklistConfig) -> None:
    if not cfg.manifest_writer_wired:
        raise OneshotHoldoutGateError("run-manifest (contract Extension 3) emitter is not wired")


def check_rehearsal_green(cfg: ChecklistConfig) -> None:
    if not cfg.require_rehearsal:
        return
    if cfg.rehearsal_marker_path is None or not RehearsalMarker(cfg.rehearsal_marker_path).is_green():
        raise OneshotHoldoutGateError("rehearsal-green marker absent — the real run refuses without it (§5)")


def run_pre_run_checklist(cfg: ChecklistConfig) -> GateChecklistResult:
    """Run every §1.3 check in order, fail-loud. Returns the validated windows + fingerprint.
    Does NOT open the holdout gate."""
    window, sub_window = check_window(cfg)
    check_release_tag(cfg)
    check_scsci12_approved(cfg)
    fingerprint = check_thresholds_fingerprint(cfg)
    check_p3_artefact(cfg)
    check_e9_cost_decision(cfg)
    check_manifest_wired(cfg)
    check_rehearsal_green(cfg)
    return GateChecklistResult(
        window=window,
        sub_window=sub_window,
        thresholds_fingerprint=fingerprint,
        checks_passed=(
            "window", "release_tag", "scsci12_approved", "thresholds_fingerprint",
            "p3_artefact", "e9_cost_decision", "manifest_wired", "rehearsal_green",
        ),
    )
