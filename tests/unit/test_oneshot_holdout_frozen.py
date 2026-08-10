"""one-shot holdout freeze pins (spec §1.3) — tripwires that enforce the re-freeze discipline.

If an oneshot_holdout module changes without re-pinning FROZEN_ONESHOT_HOLDOUT_SCRIPT_HASH, or docs/thresholds.yaml changes
without re-pinning FROZEN_THRESHOLDS_SHA256, these fail — forcing the pin to be updated so the real
one-shot's gate always compares against a *current* committed value, never a self-computed no-op.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from agents.scientist.experimentalist.oneshot_holdout.frozen import (
    FROZEN_ONESHOT_HOLDOUT_SCRIPT_HASH,
    FROZEN_THRESHOLDS_SHA256,
)
from agents.scientist.experimentalist.oneshot_holdout.run_oneshot_holdout import _dir_code_hash

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_frozen_script_hash_matches_current_source():
    # Re-pin FROZEN_ONESHOT_HOLDOUT_SCRIPT_HASH to _dir_code_hash() whenever an oneshot_holdout module (other than frozen.py) changes.
    assert _dir_code_hash() == FROZEN_ONESHOT_HOLDOUT_SCRIPT_HASH, (
        "one-shot holdout source changed without re-pinning FROZEN_ONESHOT_HOLDOUT_SCRIPT_HASH in oneshot_holdout/frozen.py"
    )


def test_frozen_thresholds_sha_matches_file():
    live = hashlib.sha256((REPO_ROOT / "docs" / "thresholds.yaml").read_bytes()).hexdigest()
    assert live == FROZEN_THRESHOLDS_SHA256, (
        "docs/thresholds.yaml changed without re-pinning FROZEN_THRESHOLDS_SHA256 in oneshot_holdout/frozen.py"
    )
