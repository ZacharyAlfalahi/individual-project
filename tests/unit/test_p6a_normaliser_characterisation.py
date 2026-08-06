"""P6a (WS-B, B3) — characterisation tests pinning the D9 value normaliser on
the known surface-form classes, plus an existence assertion for the D40
dash-class tests.

These tests PIN current behaviour — including the classes the normaliser
deliberately does NOT unify (numeric words, unit synonyms beyond the leading-
int rule). If P6b is ever triggered, its rules must flip a pinned expectation
here explicitly, with before/after diffs — never silently.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.librarian.errors import LibrarianSchemaError
from agents.librarian.pipeline.form_filler import normalise
from agents.librarian.schema import fields as F

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Token folding (enum / signal_ref / default): case, whitespace, hyphens.
# ---------------------------------------------------------------------------

def test_token_folds_case_space_and_hyphen():
    assert normalise("weighting", "Equal Weighted") == "equal_weighted"
    assert normalise("weighting", "value-weighted") == "value_weighted"
    assert normalise("weighting", "  Value   Weighted ") == "value_weighted"


def test_token_is_identity_on_canonical_menu_tokens():
    assert normalise("sort_signal", "var_5pct") == "var_5pct"


# ---------------------------------------------------------------------------
# Int fields: leading-integer surface forms unify; words and non-ints do not.
# ---------------------------------------------------------------------------

def test_int_field_unifies_leading_integer_surface_forms():
    assert normalise(F.N_GROUPS, 6) == 6
    assert normalise(F.N_GROUPS, 6.0) == 6
    assert normalise(F.N_GROUPS, "6") == 6
    assert normalise(F.N_GROUPS, "6 months") == 6
    assert normalise(F.N_GROUPS, "6-month") == 6


def test_int_field_rejects_number_words_and_non_integers():
    # "six months" is NOT unified with 6 — the exact class the P6a review list
    # exists to surface. Pinned as a rejection, not silently coerced.
    with pytest.raises(LibrarianSchemaError):
        normalise(F.N_GROUPS, "six months")
    with pytest.raises(LibrarianSchemaError):
        normalise(F.N_GROUPS, 6.5)


def test_value_kind_override_types_registry_parameters():
    # A registry parameter whose name is not a schema field still normalises as
    # an int when the caller supplies value_kind="int".
    assert normalise("some_registry_param", "36 months", "int") == 36


# ---------------------------------------------------------------------------
# Dates: validated pass-through, never token-folded.
# ---------------------------------------------------------------------------

def test_date_fields_pass_through_unfolded():
    assert normalise(F.SAMPLE_START, "2004-07") == "2004-07"


# ---------------------------------------------------------------------------
# None is silence.
# ---------------------------------------------------------------------------

def test_none_normalises_to_none():
    assert normalise("weighting", None) is None


# ---------------------------------------------------------------------------
# D40 regression anchor (B3): the dash-class tests exist in the suite.
# ---------------------------------------------------------------------------

def test_d40_dash_class_tests_exist():
    src = (_REPO_ROOT / "tests/unit/test_normalise.py").read_text(encoding="utf-8")
    for name in (
        "test_l1_dash_unification_covers_the_whole_class_unconditionally",
        "test_l1_dash_unification_is_length_preserving",
        "test_l1_does_not_fold_content_characters",
    ):
        assert f"def {name}" in src, f"D40 regression test missing: {name}"
