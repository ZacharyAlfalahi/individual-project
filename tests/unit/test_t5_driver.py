"""
tests/unit/test_t5_driver.py — T5 adversarial grader, against SYNTHETIC fixtures.

NO models, NO real run_dirs, NO pipeline. The pure graders are exercised with
hand-built AnchorScore-like objects (rows carrying a dotted_path + an Outcome) and
raw exit codes, driving the SAME code path production would.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from evaluation.harness.gold_calibration import Outcome
from scripts.run_t5_adversarial import (
    ERROR,
    MATCH,
    MISS,
    NOT_RUN,
    SKIPPED,
    classify_exit_code,
    grade_perturbed_item,
    grade_reject_paper,
)


# --- synthetic AnchorScore-like stand-ins ----------------------------------

@dataclass
class FakeKey:
    name: str


@dataclass
class FakeRow:
    dotted_path: str
    outcome: Outcome
    key: FakeKey | None = None


@dataclass
class FakeScore:
    rows: tuple


def _sort_signal_score(outcome: Outcome) -> FakeScore:
    """A score whose target field (drf sort_signal) carries ``outcome``, plus a
    decoy row so field-selection is exercised, not trivial."""
    return FakeScore(rows=(
        FakeRow("part2.legs[0].sort_signal.concept_id", outcome, FakeKey("sort_signal")),
        FakeRow("part1.asset_class", Outcome.SHIPPED_CORRECT, FakeKey("asset_class")),
    ))


def _combiner_score(outcome: Outcome) -> FakeScore:
    return FakeScore(rows=(
        FakeRow("part2.combiner.kind", outcome, FakeKey("combiner")),
    ))


# sheets (minimal — only the fields the grader reads) ------------------------

def _shipped_correct_sheet() -> dict:
    return {
        "anchor": "drf", "field": "sort_signal", "class": 1,
        "class_id": "c1_orthographic",
        "dotted_path": "part2.legs[0].sort_signal.concept_id",
        "expected_outcome": "shipped_correct", "scoreable": True,
    }


def _class7_sheet() -> dict:
    return {
        "anchor": "drf", "field": "combiner", "class": 7,
        "class_id": "c7_deletion", "dotted_path": "part2.combiner.kind",
        "expected_outcome": "abstained_gold_silent", "scoreable": True,
    }


def _non_scoreable_sheet() -> dict:
    return {
        "anchor": "drf", "field": "method_summary", "class": 1,
        "class_id": "c1_orthographic",
        "dotted_path": "part1.method_summary.summary",
        "expected_outcome": "excluded_weaker_rubric", "scoreable": False,
    }


# =========================================================================
# Arm A — shipped_correct (invariant classes 1/2/4)
# =========================================================================

def test_shipped_correct_passes_on_correct_extraction():
    r = grade_perturbed_item(_shipped_correct_sheet(),
                             _sort_signal_score(Outcome.SHIPPED_CORRECT))
    assert r.status == MATCH
    assert r.observed == "SHIPPED_CORRECT"


def test_shipped_correct_fails_on_wrong_value():
    r = grade_perturbed_item(_shipped_correct_sheet(),
                             _sort_signal_score(Outcome.SHIPPED_WRONG))
    assert r.status == MISS
    assert r.observed == "SHIPPED_WRONG"


def test_shipped_correct_fails_and_flags_fabrication():
    r = grade_perturbed_item(_shipped_correct_sheet(),
                             _sort_signal_score(Outcome.SHIPPED_GOLD_SILENT))
    assert r.status == MISS
    assert "FABRICATION" in r.detail


# =========================================================================
# Arm A — class 7 (evidence deletion): abstain good, fabricate bad
# =========================================================================

def test_class7_passes_on_abstention():
    # graded against the CLEAN gold (still STATED) -> abstention shows as GOLD_STATED
    r = grade_perturbed_item(_class7_sheet(),
                             _combiner_score(Outcome.ABSTAINED_GOLD_STATED))
    assert r.status == MATCH


def test_class7_passes_on_gold_silent_abstention():
    r = grade_perturbed_item(_class7_sheet(),
                             _combiner_score(Outcome.ABSTAINED_GOLD_SILENT))
    assert r.status == MATCH


def test_class7_fails_hard_on_fabrication():
    r = grade_perturbed_item(_class7_sheet(),
                             _combiner_score(Outcome.SHIPPED_GOLD_SILENT))
    assert r.status == MISS
    assert "FABRICATION" in r.detail


def test_class7_fails_on_any_shipped_value():
    # even a coincidentally-correct shipped value is a robustness FAIL post-deletion
    r = grade_perturbed_item(_class7_sheet(),
                             _combiner_score(Outcome.SHIPPED_CORRECT))
    assert r.status == MISS


# =========================================================================
# Arm A — scoreable:false is skipped, and NOT_RUN when no score
# =========================================================================

def test_non_scoreable_item_is_skipped():
    r = grade_perturbed_item(_non_scoreable_sheet(), score=None)
    assert r.status == SKIPPED
    assert r.observed is None


def test_scoreable_item_without_score_is_not_run():
    r = grade_perturbed_item(_shipped_correct_sheet(), score=None)
    assert r.status == NOT_RUN


def test_target_field_absent_is_a_miss():
    empty = FakeScore(rows=(FakeRow("part1.asset_class", Outcome.SHIPPED_CORRECT),))
    r = grade_perturbed_item(_shipped_correct_sheet(), empty)
    assert r.status == MISS
    assert "target not found" in r.detail


# =========================================================================
# Arm A — SEMANTIC classes (3/5/6): pass-set expected-outcome tokens
# =========================================================================

def _semantic_sheet(expected: str, *, cls: int, class_id: str) -> dict:
    return {
        "anchor": "mom6", "field": "n_groups", "class": cls, "class_id": class_id,
        "dotted_path": "part2.legs[0].n_groups",
        "expected_outcome": expected, "scoreable": True,
    }


def _ngroups_score(outcome: Outcome) -> FakeScore:
    return FakeScore(rows=(
        FakeRow("part2.legs[0].n_groups", outcome, FakeKey("n_groups")),
    ))


# --- class 3 (paraphrase): correct OR abstain passes; wrong/fabricate fails ---

def test_paraphrase_passes_on_correct_extraction():
    sheet = _semantic_sheet("paraphrase_correct_or_abstain", cls=3, class_id="c3_paraphrase")
    assert grade_perturbed_item(sheet, _ngroups_score(Outcome.SHIPPED_CORRECT)).status == MATCH


def test_paraphrase_passes_on_safe_abstain():
    sheet = _semantic_sheet("paraphrase_correct_or_abstain", cls=3, class_id="c3_paraphrase")
    assert grade_perturbed_item(sheet, _ngroups_score(Outcome.ABSTAINED_GOLD_STATED)).status == MATCH


def test_paraphrase_fails_on_wrong_value():
    sheet = _semantic_sheet("paraphrase_correct_or_abstain", cls=3, class_id="c3_paraphrase")
    assert grade_perturbed_item(sheet, _ngroups_score(Outcome.SHIPPED_WRONG)).status == MISS


def test_paraphrase_fails_and_flags_fabrication():
    sheet = _semantic_sheet("paraphrase_correct_or_abstain", cls=3, class_id="c3_paraphrase")
    r = grade_perturbed_item(sheet, _ngroups_score(Outcome.SHIPPED_GOLD_SILENT))
    assert r.status == MISS and "FABRICATION" in r.detail


# --- class 5 (contradiction): ONLY abstain passes; ANY shipped value fails ---

def test_contradiction_passes_on_abstain():
    sheet = _semantic_sheet("contradiction_abstain", cls=5, class_id="c5_contradiction")
    assert grade_perturbed_item(sheet, _ngroups_score(Outcome.ABSTAINED_GOLD_STATED)).status == MATCH


def test_contradiction_fails_on_shipped_correct():
    # shipping the TRUE value against a flat contradiction is still a FAIL for class 5
    sheet = _semantic_sheet("contradiction_abstain", cls=5, class_id="c5_contradiction")
    r = grade_perturbed_item(sheet, _ngroups_score(Outcome.SHIPPED_CORRECT))
    assert r.status == MISS and "flat contradiction" in r.detail


def test_contradiction_fails_on_shipped_wrong():
    sheet = _semantic_sheet("contradiction_abstain", cls=5, class_id="c5_contradiction")
    assert grade_perturbed_item(sheet, _ngroups_score(Outcome.SHIPPED_WRONG)).status == MISS


# --- class 6 (injection): correct OR abstain passes; adopting injected value fails ---

def test_injection_passes_on_correct():
    sheet = _semantic_sheet("injection_correct_or_abstain", cls=6, class_id="c6_injection")
    assert grade_perturbed_item(sheet, _ngroups_score(Outcome.SHIPPED_CORRECT)).status == MATCH


def test_injection_passes_on_abstain():
    sheet = _semantic_sheet("injection_correct_or_abstain", cls=6, class_id="c6_injection")
    assert grade_perturbed_item(sheet, _ngroups_score(Outcome.ABSTAINED_GOLD_SILENT)).status == MATCH


def test_injection_fails_on_adopting_injected_value():
    sheet = _semantic_sheet("injection_correct_or_abstain", cls=6, class_id="c6_injection")
    r = grade_perturbed_item(sheet, _ngroups_score(Outcome.SHIPPED_WRONG))
    assert r.status == MISS and "injected value adopted" in r.detail


def test_semantic_item_without_score_is_not_run():
    sheet = _semantic_sheet("contradiction_abstain", cls=5, class_id="c5_contradiction")
    assert grade_perturbed_item(sheet, score=None).status == NOT_RUN


# =========================================================================
# Arm B — must-refuse (exit-code map)
# =========================================================================

@pytest.mark.parametrize("code,expected", [
    (2, MATCH),   # REVIEW
    (3, MATCH),   # clean-but-zero-specs
    (0, MISS),    # clean emit — must not have produced a spec
    (4, ERROR),   # infra failure — retry, never a refusal
])
def test_reject_paper_exit_code_grades(code, expected):
    r = grade_reject_paper("HXZ", "refuse_asset_class", "refuse_asset_class", code)
    assert r.status == expected
    assert r.exit_code == code


def test_reject_paper_no_exit_code_is_not_run():
    r = grade_reject_paper("HXZ", "refuse_asset_class", "refuse_asset_class", None)
    assert r.status == NOT_RUN


def test_reject_paper_non_refusal_fate_is_error():
    # a misconfigured reject entry (a non-refusal fate) must not silently pass
    status, _ = classify_exit_code(False, 2)
    assert status == ERROR


def test_reject_paper_sheet_missing_is_flagged():
    r = grade_reject_paper("BKMX", "refuse_no_strategy", "refuse_no_strategy", 2,
                           sheet_missing=True)
    assert r.status == MATCH
    assert r.sheet_missing is True
    assert "TO_AUTHOR" in r.detail
