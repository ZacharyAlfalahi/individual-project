"""Relocator calibration tests (scripts/calibrate_relocator.py) — the edit classes,
the bar rule, and the negative-target exclusion; all pure logic, no archives."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from agents.librarian.config.canonical_text import CanonicalText            # noqa: E402
from scripts.calibrate_relocator import (                                   # noqa: E402
    NEGATIVE_TEXTS,
    apply_edit,
    choose_bar,
    negative_targets,
)

_Q = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"


def test_edit_classes_deterministic_and_hash_seeded():
    for cls in ("truncate_head", "truncate_tail", "punctuation_drift",
                "typo_substitution", "ellipsis_elision", "word_substitution"):
        assert apply_edit(_Q, cls, 12345) == apply_edit(_Q, cls, 12345)
    # Different seeds pick different interior words (10 words, all eligible).
    assert apply_edit(_Q, "word_substitution", 0) != apply_edit(_Q, "word_substitution", 1)
    # Edits actually change the quote.
    assert apply_edit(_Q, "truncate_head", 0) != _Q
    assert apply_edit(_Q, "typo_substitution", 0) != _Q


def test_edits_record_drops_noops_and_exact_after_edit():
    # Structurally inapplicable class -> None (caller records skipped_short_quote).
    assert apply_edit("alpha bravo charlie", "ellipsis_elision", 7) is None
    assert apply_edit("...", "typo_substitution", 7) is None
    # No punctuation to drop -> the edit returns the input unchanged (caller
    # records it as a noop and excludes it from recall denominators).
    assert apply_edit(_Q, "punctuation_drift", 7) == _Q


def _roc(rows: dict[float, tuple[int, tuple[int, int], tuple[int, int]]]) -> dict:
    return {bar: {"fa": fa, "r_light": list(rl), "r_mod": list(rm)}
            for bar, (fa, rl, rm) in rows.items()}


def test_bar_rule_picks_expected_bar_on_synthetic_roc():
    # Strictest zero-FA bar meeting both floors wins.
    bar, status = choose_bar(_roc({
        0.80: (2, (100, 100), (100, 100)),
        0.86: (0, (95, 100), (80, 100)),
        0.90: (0, (92, 100), (76, 100)),
        0.94: (0, (80, 100), (50, 100)),
    }), floor_light=0.90, floor_moderate=0.75)
    assert (bar, status) == (0.90, "ok")

    # Zero-FA bars exist but no floor met -> loosest zero-FA bar, labelled.
    bar, status = choose_bar(_roc({
        0.84: (0, (50, 100), (40, 100)),
        0.90: (0, (40, 100), (30, 100)),
    }), floor_light=0.90, floor_moderate=0.75)
    assert (bar, status) == (0.84, "recall_floor_unmet")

    # No zero-FA bar anywhere -> the diagnostic does not run.
    bar, status = choose_bar(_roc({
        0.90: (1, (99, 100), (99, 100)),
        0.98: (1, (99, 100), (99, 100)),
    }), floor_light=0.90, floor_moderate=0.75)
    assert bar is None and status == "no_zero_fa_bar"


def test_negative_targets_exclude_home_paper():
    # drf/crf/lrf share bbw_2019 — home exclusion is by PAPER.
    targets = negative_targets("bbw")
    assert "bbw" not in targets and len(targets) == len(NEGATIVE_TEXTS) - 1
    assert set(targets) <= set(NEGATIVE_TEXTS)
    with pytest.raises(ValueError, match="unknown home paper"):
        negative_targets("nope")


def test_qr2_negative_targets_exclude_same_author_pair():
    # The declared matrix has exactly one same-author pair: {bbw_2019, bbw_2021}.
    assert "bbw2021" in negative_targets("bbw", "qr1")
    assert "bbw2021" not in negative_targets("bbw", "qr2")
    assert "bbw" not in negative_targets("bbw2021", "qr2")
    # Every other home paper keeps its full qr1 target set under qr2.
    for home in ("jnps", "drr", "kpp", "dfps"):
        assert negative_targets(home, "qr2") == negative_targets(home, "qr1")
    with pytest.raises(ValueError, match="unknown protocol"):
        negative_targets("bbw", "qr9")


def test_qr2_truncation_typo_classes_cannot_exact_locate():
    sentence = ("The distress factor earns a significant premium after controlling "
                "for duration and rating in every specification we run here.")
    ct = CanonicalText(
        source_pdf="stub.pdf", source_sha256="deadbeef",
        parser={"name": "stub", "version": "0"},
        normalisation={"ladder_level": "L1", "rules": []},
        pages=("Intro filler. " + sentence + " More filler text.",), status="stub",
    )
    for cls in ("truncate_head_typo", "truncate_tail_typo"):
        edited = apply_edit(sentence, cls, 12345)
        assert edited is not None and edited != sentence
        assert apply_edit(sentence, cls, 12345) == edited          # deterministic
        # A plain truncation still exact-locates; the typo'd
        # variant must not (that is the entire point of the composite class).
        plain = apply_edit(sentence, cls.removesuffix("_typo"), 12345)
        assert ct.locate(plain, level="L1") is not None
        assert ct.locate(edited, level="L1") is None
