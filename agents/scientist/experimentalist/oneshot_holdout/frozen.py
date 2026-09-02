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
# (contract §7 gates 1-2; D-Q17); RE-PINNED 2026-08-17 after the `corpus:` scale-layer selection-rule
# block was added (T2-SEL-1); RE-PINNED 2026-08-17 after the `p2_codegen:` block was added (Workstream C
# P2 coverage-boundary build); RE-PINNED 2026-08-19 after appending `bkmx_2009` to
# corpus.selection.design_touched_exclusions (T2-SEL-2); RE-PINNED 2026-08-22 after adding
# auditor.practical_significance.vartheta_sensitivity_grid (§8.2.3/§9 materiality sweep); RE-PINNED
# 2026-08-24 after appending `synth_2026` to corpus.selection.design_touched_exclusions (T2-SEL-3,
# the T4(b) synthetic instrument); RE-PINNED 2026-08-24 after adding scientist.equivalence_margin
# (SC-SCI-14 RQ4 Part-B: the development-window TOST margin delta = 1/2*vartheta); RE-PINNED 2026-09-01
# after a comment-only edit to a corpus.selection.design_touched_exclusions entry; RE-PINNED 2026-09-01
# after renaming a cost-scenario key (standing terminology fix); RE-PINNED 2026-09-01 after the T2-SEL-4 freeze of
# corpus.selection (status->frozen, five admission constraints committed, governing_date_rule +
# search_protocol added); RE-PINNED 2026-09-02 after flipping corpus.selection.search_protocol.executed
# to true (T2-SEL-5 — the one-round candidate search ran; log in docs/evaluation/
# t2_candidate_adjudication.md). one-shot holdout's real run
# has not run, so re-pinning is maintenance, not a provenance break. Any later thresholds change must re-pin.
FROZEN_THRESHOLDS_SHA256 = "9aa28ae40c3b58f0423a13c93a464cf7db65d6f6d5b6e204dfa20e90ffb64c58"

# _dir_code_hash() over oneshot_holdout/*.py (excluding frozen.py). Pinned 2026-08-10; RE-PINNED 2026-08-11 after
# panel_builder.py's imports were repointed from scripts.* to agents/quant/library (the scripts→library
# lift of load_feed + the FISD builders); RE-PINNED 2026-08-31 after the 36-month holdout sensitivity
# sub-window was removed from the one-shot holdout modules (windows/gate_checklist/stage1/stage2/run_oneshot_holdout/panel_builder),
# leaving the single registered 45-month window; RE-PINNED 2026-09-01 after a panel_builder.py docstring
# edit. Any later edit to an oneshot_holdout module invalidates this and must
# re-pin it to the new _dir_code_hash(); an empty/stale value fails the real gate closed.
FROZEN_ONESHOT_HOLDOUT_SCRIPT_HASH = "86c360076d08e4bf2e9f419f9fffdb079630894e86eabe2af0b8a546c3b9feed"
