"""The registered development-window equivalence test (TOST): the margin is derived from
thresholds, both one-sided tests must clear, the window minimum is enforced, duplicates are
disclosed, and failing equivalence never becomes evidence of a non-zero alpha."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shared.evaluation.equivalence import (  # noqa: E402
    ASYMMETRY,
    EquivalenceInput,
    EquivalenceInputError,
    duplicate_groups,
    equivalence_family,
    load_margin,
    tost_p_values,
)


def _inp(cid, alpha, se, n_months=231):
    return EquivalenceInput(candidate_id=cid, alpha=alpha, se=se, n_months=n_months)


# --- the margin is derived, never invented --------------------------------------

def test_margin_is_half_the_materiality_floor_from_thresholds():
    margin = load_margin()
    assert margin["delta"] == pytest.approx(margin["factor_of_vartheta"] * margin["vartheta"])
    assert margin["factor_of_vartheta"] == 0.5           # registered
    assert margin["delta"] > 0


def test_a_thresholds_file_without_the_margin_fails_loud(tmp_path):
    bad = tmp_path / "t.yaml"
    bad.write_text("auditor: {}\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_margin(bad)


# --- the test itself -------------------------------------------------------------

def test_an_alpha_at_the_centre_of_a_wide_margin_is_equivalent():
    p_lower, p_upper, p_tost = tost_p_values(alpha=0.0, se=0.0001, df=200, delta=0.0005)
    assert p_tost < 0.01 and p_tost == max(p_lower, p_upper)


def test_an_alpha_far_outside_the_margin_is_not_equivalent():
    _, _, p_tost = tost_p_values(alpha=0.008, se=0.003, df=200, delta=0.0005)
    assert p_tost > 0.5


def test_p_tost_is_the_max_of_the_two_one_sided_tests():
    """Equivalence needs BOTH nulls rejected, so the larger p governs."""
    p_lower, p_upper, p_tost = tost_p_values(alpha=0.0004, se=0.0002, df=100, delta=0.0005)
    assert p_tost == max(p_lower, p_upper)
    assert p_lower != p_upper                       # asymmetric alpha => asymmetric tails


def test_a_wider_standard_error_never_helps_equivalence():
    _, _, tight = tost_p_values(alpha=0.0, se=0.00005, df=200, delta=0.0005)
    _, _, loose = tost_p_values(alpha=0.0, se=0.0005, df=200, delta=0.0005)
    assert loose > tight


def test_a_non_positive_margin_is_refused():
    with pytest.raises(EquivalenceInputError):
        tost_p_values(alpha=0.0, se=0.001, df=100, delta=0.0)


def test_a_non_positive_standard_error_is_refused():
    with pytest.raises(EquivalenceInputError):
        _inp("c", 0.001, 0.0)


def test_too_few_months_leaves_no_degrees_of_freedom():
    with pytest.raises(EquivalenceInputError):
        EquivalenceInput(candidate_id="c", alpha=0.001, se=0.001, n_months=3)


# --- the family ------------------------------------------------------------------

def test_a_short_window_is_typed_and_kept_out_of_the_family():
    family = equivalence_family(
        [_inp("long", 0.0, 0.0001), _inp("short", 0.0, 0.0001, n_months=40)],
        delta=0.0005, q=0.10)
    by_id = {r["candidate_id"]: r for r in family["results"]}
    assert by_id["short"]["status"] == "insufficient_window"
    assert by_id["short"]["p_tost"] is None
    assert family["n_tested"] == 1 and family["n_skipped_insufficient_window"] == 1


def test_bh_runs_over_the_tested_candidates_only():
    family = equivalence_family(
        [_inp("a", 0.0, 0.0001), _inp("b", 0.0, 0.0001), _inp("c", 0.0, 0.0001, n_months=10)],
        delta=0.0005, q=0.10)
    adjusted = [r["adjusted_p"] for r in family["results"] if r["status"] == "tested"]
    assert len(adjusted) == 2 and all(p is not None for p in adjusted)
    assert family["n_equivalent"] == 2


def test_identical_candidates_are_disclosed_and_sensitivity_reported():
    """Two sources proposing the same configuration is one hypothesis, not two."""
    family = equivalence_family(
        [_inp("src_a:prop", 0.00784, 0.0036), _inp("src_b:prop", 0.00784, 0.0036),
         _inp("other:prop", 0.0093, 0.0034)],
        delta=0.0005, q=0.10)
    assert family["duplicate_groups"] == [["src_a:prop", "src_b:prop"]]
    dedup = family["deduplicated_sensitivity"]
    assert dedup["n_distinct"] == 2                      # one representative + the other
    assert "sensitivity" in dedup["note"]


def test_no_duplicates_means_no_sensitivity_block():
    family = equivalence_family([_inp("a", 0.001, 0.0002), _inp("b", 0.002, 0.0002)],
                                delta=0.0005, q=0.10)
    assert family["duplicate_groups"] == []
    assert family["deduplicated_sensitivity"] is None


def test_the_asymmetry_travels_with_the_result():
    family = equivalence_family([_inp("a", 0.008, 0.003)], delta=0.0005, q=0.10)
    assert family["asymmetry"] == ASYMMETRY
    assert "NOT evidence of a non-zero alpha" in family["asymmetry"]
    assert family["n_equivalent"] == 0


def test_duplicate_groups_ignores_merely_similar_candidates():
    assert duplicate_groups([_inp("a", 0.001, 0.0002), _inp("b", 0.0010000001, 0.0002)]) == []
