"""Freeze pins for the one-shot holdout real run (§1.3).

These are the *pinned* expected values the real run checks against — not values self-computed at
run time (which would be a no-op that proves nothing). ``FROZEN_THRESHOLDS_SHA256`` pins
``docs/thresholds.yaml``; ``FROZEN_ONESHOT_HOLDOUT_SCRIPT_HASH`` pins the one-shot holdout package source (every ``oneshot_holdout/*.py``
EXCEPT this file — see ``run_oneshot_holdout._dir_code_hash``).

RE-FREEZE DISCIPLINE (load-bearing): any change to ``docs/thresholds.yaml`` requires re-pinning
``FROZEN_THRESHOLDS_SHA256``; any change to an ``oneshot_holdout/*.py`` module (other than this one) requires
re-pinning ``FROZEN_ONESHOT_HOLDOUT_SCRIPT_HASH`` to the new ``_dir_code_hash()``. Until re-pinned, the real
gate fails CLOSED (an empty/mismatched pin keeps the holdout door shut) — the safe default.
"""

from __future__ import annotations

# sha256 of docs/thresholds.yaml. Any thresholds change must re-pin this value; an empty/stale
# value fails the real gate closed.
FROZEN_THRESHOLDS_SHA256 = "d612f6e287dab24bfc4da2a4f95a20a6b0b76172da88317d4f7ae94aec8dfb47"

# _dir_code_hash() over oneshot_holdout/*.py (excluding frozen.py). Any edit to an oneshot_holdout
# module invalidates this and must re-pin it to the new _dir_code_hash(); an empty/stale value
# fails the real gate closed.
FROZEN_ONESHOT_HOLDOUT_SCRIPT_HASH = "8b09b13ee778dec8f51566ef1a3d1ff4713caddd54d37f7343f4b32b1cbb48b2"
