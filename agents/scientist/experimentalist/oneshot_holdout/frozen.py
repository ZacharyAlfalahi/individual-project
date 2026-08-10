"""Committed freeze pins for the one-shot holdout real run (§1.3).

These are the *committed* expected values the real run checks against — not values self-computed at
run time (which would be a no-op that proves nothing). ``FROZEN_THRESHOLDS_SHA256`` pins
``docs/thresholds.yaml``; ``FROZEN_ONESHOT_HOLDOUT_SCRIPT_HASH`` pins the one-shot holdout package source (every ``oneshot_holdout/*.py``
EXCEPT this file — see ``run_oneshot_holdout._dir_code_hash``).

RE-FREEZE DISCIPLINE (load-bearing): any change to ``docs/thresholds.yaml`` requires re-pinning
``FROZEN_THRESHOLDS_SHA256``; any change to an ``oneshot_holdout/*.py`` module (other than this one) requires
re-pinning ``FROZEN_ONESHOT_HOLDOUT_SCRIPT_HASH`` to the new ``_dir_code_hash()``. Until re-pinned, the real
gate fails CLOSED (an empty/mismatched pin keeps the holdout door shut) — the safe default.
"""

from __future__ import annotations

# sha256 of docs/thresholds.yaml at freeze time (2026-08-10).
FROZEN_THRESHOLDS_SHA256 = "f64a82f12804236b34da1b4995ce39c67710a0597638606a720a112cf157a9ad"

# _dir_code_hash() over oneshot_holdout/*.py (excluding frozen.py) at freeze time (2026-08-10), pinned as the last
# build step once every other oneshot_holdout module was final. Any later edit to an oneshot_holdout module invalidates this and
# must re-pin it; an empty/stale value fails the real gate closed.
FROZEN_ONESHOT_HOLDOUT_SCRIPT_HASH = "8632a5eb9cef317d8848b830c018a0896140e2e191c8b90947ef71d50d00069a"
