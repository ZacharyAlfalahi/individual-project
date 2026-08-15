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

# sha256 of docs/thresholds.yaml. Pinned at freeze time (2026-08-10); RE-PINNED 2026-08-14 after the
# RQ2 v1.6 re-scope flipped validation.anchor_criterion "bias_attribution" -> "construction_rulebook"
# (contract §7 gates 1-2; D-Q17). one-shot holdout's real run has not run, so re-pinning is maintenance, not a
# provenance break. Any later thresholds change (e.g. the p2_codegen block in Workstream C) must re-pin.
FROZEN_THRESHOLDS_SHA256 = "d2f2e57a36e11deac26d28adb37f84756fd39c74d421f2358dc29ba7266718a8"

# _dir_code_hash() over oneshot_holdout/*.py (excluding frozen.py). Pinned 2026-08-10; RE-PINNED 2026-08-11 after
# panel_builder.py's imports were repointed from scripts.* to agents/quant/library (the scripts→library
# lift of load_feed + the FISD builders). Any later edit to an oneshot_holdout module invalidates this and must re-pin
# it to the new _dir_code_hash(); an empty/stale value fails the real gate closed.
FROZEN_ONESHOT_HOLDOUT_SCRIPT_HASH = "6ef1fe79c8b42ffbaea128f2bfa182a02f007f99ae94ff233dd93e68a7ee7f01"
