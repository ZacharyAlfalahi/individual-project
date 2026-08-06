"""G6 — holdout (spec §9; F8 amendment). Single-access, ARTEFACT-gated — NOT a self-opening date.
The bypass requires ALL of: the `scientist-prereg` tag exists, the frozen holdout-build script's
hash matches its manifest, an explicit env var, and a single-access assertion; every access is
logged. The 2022-2025 panel build (48 months per SC-SCI-10; originally 2022-2024 at the tag) IS
the sanctioned single access (READ NEVER before then). A bare
date never opens the gate — a date opens itself regardless of tag / greenness / real run.
HOLDOUT_VIOLATION on any breach.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

_ACCESS = {"count": 0}                     # process-lifetime single-access tracker


class HoldoutViolation(RuntimeError):
    """A holdout access was attempted while the artefact gate was shut, or more than once."""


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


def _reset_single_access_for_tests() -> None:
    _ACCESS["count"] = 0
