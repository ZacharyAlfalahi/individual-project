"""G6 — holdout (spec §9; F8 amendment). Single-access, ARTEFACT-gated — NOT a self-opening date.
The bypass requires ALL of: the `scientist-prereg` tag exists, the frozen holdout-build script's
hash matches its manifest, an explicit env var, and a single-access assertion; every access is
logged. The 2022-01..2025-09 holdout window (45 months per SC-SCI-12 — correcting SC-SCI-10's 48; originally 2022-2024 at the tag) IS
the sanctioned single access (READ NEVER before then). A bare
date never opens the gate — a date opens itself regardless of tag / greenness / real run.
HOLDOUT_VIOLATION on any breach.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml

_ACCESS = {"count": 0}                     # process-lifetime single-access tracker

# SC-SCI-12 (ruled 2026-08-07; frontier-completeness evidence PI-verified 2026-08-08): the holdout
# EVALUATION window. This is the in-code second source for assert_evaluation_window()'s two-source
# agreement check — it MUST equal windows.evaluation_holdout in docs/scientist_protocol.yaml. A
# month-grain window, distinct from the year-grain ingest partition (thresholds.yaml
# holdout_end_year). Data, never a date trigger: nothing in this module opens the gate.
_SC_SCI_12_WINDOW = {"start": "2022-01", "end": "2025-09", "n_months": 45}


class HoldoutViolation(RuntimeError):
    """A holdout access was attempted while the artefact gate was shut, or more than once."""


class WindowAssertionError(RuntimeError):
    """The holdout evaluation window failed its two-source agreement check (SC-SCI-12): the
    governance YAML and the in-code constant disagree, or the YAML block is missing/malformed.
    Raised, NEVER defaulted — a silent default would let a window drift into the one-shot run."""


def prereg_tag_present(tag: str = "scientist-prereg", repo_root=None) -> bool:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    try:
        out = subprocess.run(["git", "-C", str(root), "tag", "-l", tag],
                             capture_output=True, text=True, check=True)
    except Exception:
        return False
    return out.stdout.strip() == tag


def env_unlock_set(var: str = "SCIENTIST_HOLDOUT_UNLOCK") -> bool:
    return bool(os.environ.get(var))


def holdout_gate_open(*, tag_present: bool, frozen_script_hash_matches: bool,
                      env_var_set: bool) -> bool:
    """ALL artefact conditions must hold. There is deliberately no date argument — a date must
    never be the trigger (it would open itself)."""
    return bool(tag_present and frozen_script_hash_matches and env_var_set)


def assert_single_access() -> None:
    """The holdout may be accessed exactly ONCE per process (accesses=1, §4). Every call is logged
    (the caller records it); a second call is a HOLDOUT_VIOLATION."""
    if _ACCESS["count"] >= 1:
        raise HoldoutViolation("holdout already accessed once (accesses=1, spec §4)")
    _ACCESS["count"] += 1


def assert_evaluation_window(protocol_path=None, expected=None, repo_root=None):
    """Fail-loud two-source agreement check on the holdout EVALUATION window (SC-SCI-12).

    Reads ``windows.evaluation_holdout`` {start, end, n_months} from ``scientist_protocol.yaml`` and
    asserts it equals the in-code SC-SCI-12 constant (``expected``, default ``_SC_SCI_12_WINDOW``).
    Two independent sources — the governance YAML and the code constant — must agree, so a drift in
    either is caught BEFORE the one-shot holdout run rather than silently honoured (SC-SCI-12's
    "the frozen holdout script must assert window == 2022-01..2025-09 before its one-shot execution").

    This VALIDATES configuration; it does NOT open the gate and takes NO date argument — a date must
    never be the trigger (see :func:`holdout_gate_open`). Returns the validated
    ``(start, end, n_months)`` on success; raises :class:`WindowAssertionError` on any mismatch,
    missing block, or malformed value. Never defaults.
    """
    exp = dict(_SC_SCI_12_WINDOW if expected is None else expected)
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    path = Path(protocol_path) if protocol_path is not None else root / "docs" / "scientist_protocol.yaml"
    try:
        with open(path) as handle:
            data = yaml.safe_load(handle)
    except FileNotFoundError:
        raise WindowAssertionError(f"scientist_protocol.yaml not found: {path}") from None
    if not isinstance(data, dict):
        raise WindowAssertionError(f"scientist_protocol.yaml empty/malformed: {path}")
    block = (data.get("windows") or {}).get("evaluation_holdout")
    if not isinstance(block, dict):
        raise WindowAssertionError(
            "windows.evaluation_holdout block missing/malformed in scientist_protocol.yaml"
        )
    for key in ("start", "end", "n_months"):
        if key not in block:
            raise WindowAssertionError(
                f"windows.evaluation_holdout.{key} missing in scientist_protocol.yaml"
            )
    # Compare start/end as strings (canonical 'YYYY-MM'); a YAML scalar that parses to any other
    # type or value — e.g. a full ISO date '2022-01-01' → '2022-01-01' — mismatches and fails loud.
    got = {"start": str(block["start"]), "end": str(block["end"]), "n_months": block["n_months"]}
    exp_cmp = {"start": str(exp["start"]), "end": str(exp["end"]), "n_months": exp["n_months"]}
    if got != exp_cmp:
        raise WindowAssertionError(
            "holdout evaluation window disagreement (SC-SCI-12 two-source check): "
            f"scientist_protocol.yaml windows.evaluation_holdout={got} "
            f"!= in-code SC-SCI-12 constant={exp_cmp}. "
            "Refusing to proceed; reconcile before any holdout run."
        )
    return (got["start"], got["end"], got["n_months"])


def _reset_single_access_for_tests() -> None:
    _ACCESS["count"] = 0
