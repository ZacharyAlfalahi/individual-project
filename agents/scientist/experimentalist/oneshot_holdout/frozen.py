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
# t2_candidate_adjudication.md); RE-PINNED 2026-09-02 after raising corpus.selection.t2_target_count
# 3 -> 5 (T2-SEL-6, extension down the frozen selection order); RE-PINNED 2026-09-05 after adding
# the p1_codegen.budget block (WS-C P1 live-run: runtime enforcement of the contract §3/§9 sub-cap);
# RE-PINNED 2026-09-05 after setting p2_codegen.zoo_list.frozen_sha256 to the candidate scale
# zoo-list hash (WS-C P2 census freeze); RE-PINNED 2026-09-05 after adding
# scientist.model_stack.phase_f (WS-D RQ4 funnel: the reported generative pair, build_phase_f_clients);
# RE-PINNED 2026-09-06 after the ratification of p2_codegen.zoo_list.frozen_sha256 (comment
# CANDIDATE -> RATIFIED; the hash value 74e46547… is unchanged — only its status); RE-PINNED
# 2026-09-06 after adding corpus.selection.t2_selected_papers (T2-SEL-7 — the selected T2 set, so
# the consistency guard can distinguish post-selection processing from design contact); RE-PINNED
# 2026-09-06 after merging the two registers (T2-SEL-7 + the P1/P2/phase_f blocks + the
# bbw_2019_masked exclusion entry); RE-PINNED 2026-09-07 after adding
# librarian.relocator_diagnostic (the locator-diagnostic bar registration: qr1 accept_bar null by
# construction, qr2 accept_bar 0.90); RE-PINNED 2026-09-07 after folding
# corpus.selection.t2_selected_papers into design_touched_exclusions and appending the T2-SEL-8
# expanded transfer cohort (10 stems: lee_2022 … li_galvani_2021 — gold-free live-enumeration set).
# one-shot holdout's real run
# has not run, so re-pinning is maintenance, not a provenance break. Any later thresholds change must re-pin.
FROZEN_THRESHOLDS_SHA256 = "8b8d82ea1114a6d9cfb04678775af98d416d83508e18f73d77b81e1c56238402"

# _dir_code_hash() over oneshot_holdout/*.py (excluding frozen.py). Pinned 2026-08-10; RE-PINNED 2026-08-11 after
# panel_builder.py's imports were repointed from scripts.* to agents/quant/library (the scripts→library
# lift of load_feed + the FISD builders); RE-PINNED 2026-08-31 after the 36-month holdout sensitivity
# sub-window was removed from the one-shot holdout modules (windows/gate_checklist/stage1/stage2/run_oneshot_holdout/panel_builder),
# leaving the single registered 45-month window; RE-PINNED 2026-09-01 after a panel_builder.py docstring
# edit. Any later edit to an oneshot_holdout module invalidates this and must
# re-pin it to the new _dir_code_hash(); an empty/stale value fails the real gate closed.
FROZEN_ONESHOT_HOLDOUT_SCRIPT_HASH = "86c360076d08e4bf2e9f419f9fffdb079630894e86eabe2af0b8a546c3b9feed"
