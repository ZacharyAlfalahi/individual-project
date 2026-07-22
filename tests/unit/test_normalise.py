"""
Unit tests for the canonical-text normalisation ladder (parser brief §6, v2).

Covers each level's rules in isolation, the v2 pins that the plan/amendment fixed
(whitespace at L1, de-hyphenation `(\\w)-\\s+(\\w)` at L2), no case-folding,
cumulativeness, idempotence, and the L2 over-join guard on REAL hyphenated
compounds pulled from the three corpus papers.
"""

import pytest

from agents.librarian.config.normalise import LEVELS, RULES, normalise


# --- L0 is the identity ------------------------------------------------------

def test_l0_is_identity():
    s = "Raw ﬁnance text — disas-\nter, with\tweird  spacing."
    assert normalise(s, "L0") == s


def test_bad_level_raises():
    with pytest.raises(ValueError):
        normalise("x", "L3")  # L3 was removed in v2
    with pytest.raises(ValueError):
        normalise("x", "l1")  # case-sensitive level names


# --- L1: NFKC + ligatures + quotes + dashes + whitespace ---------------------

def test_l1_ligature_expansion():
    assert normalise("ﬀ", "L1") == "ff"   # ﬀ
    assert normalise("ﬁ", "L1") == "fi"   # ﬁ
    assert normalise("ﬂ", "L1") == "fl"   # ﬂ
    assert normalise("ﬃ", "L1") == "ffi"  # ﬃ
    assert normalise("ﬄ", "L1") == "ffl"  # ﬄ
    assert normalise("ﬁnance", "L1") == "finance"


def test_l1_nfkc_compatibility():
    # superscript two has an NFKC compatibility decomposition to "2".
    assert normalise("σ²", "L1") == "σ2"  # σ² -> σ2


def test_l1_curly_quote_unification():
    assert normalise("bond’s", "L1") == "bond's"
    assert normalise("“quoted”", "L1") == '"quoted"'
    assert normalise("‘x’", "L1") == "'x'"


def test_l1_dash_unification_covers_the_whole_class_unconditionally():
    """v3 (D40). v2 unified only en/em dash and only when word-flanked, which was
    wrong in two independent ways and cost an anchor: U+2212 MINUS SIGN was never
    covered at all (1,034 occurrences -- the corpus's most common non-ASCII
    character), and the flanking condition meant adding it to the v2 class would
    still not have matched, because papers write " t −6 " with a SPACE before the
    minus."""
    assert normalise("Newey–West", "L1") == "Newey-West"   # en-dash, flanked
    assert normalise("5–10", "L1") == "5-10"               # digits count as \w
    # UNFLANKED now unifies too -- v2 left these, so 286 en-dashes never matched.
    assert normalise("a – b", "L1") == "a - b"
    # the exact shape that broke mom6: minus preceded by a space.
    assert normalise("months t −6 to t −1", "L1") == "months t -6 to t -1"
    # every member of the class folds to the same ASCII hyphen
    for ch in ("‐", "‑", "‒", "–", "—", "―",
               "⁃", "−"):
        assert normalise(f"x{ch}y", "L1") == "x-y", f"U+{ord(ch):04X} not unified"


def test_l1_dash_unification_is_length_preserving():
    """Every member is a 1:1 substitution, which is WHY the v3 ladder change did
    not move a single gold locator offset. A member needing deletion (U+00AD SOFT
    HYPHEN) is deliberately excluded for exactly this reason."""
    for ch in ("‐", "–", "—", "−"):
        src = f"alpha{ch}beta"
        assert len(normalise(src, "L1")) == len(src)


def test_l1_does_not_fold_content_characters():
    """Greek letters and math operators are CONTENT, not typography -- a
    transcriber reproduces them. Folding them to ASCII would destroy meaning
    rather than recover it, so they are deliberately left alone."""
    for ch in ("γ", "β", "σ", "×", "≈", "≥", "∗"):
        assert ch in normalise(f"value {ch} here", "L1")


def test_l1_whitespace_collapse_and_strip():
    assert normalise("a\n  b\tc", "L1") == "a b c"
    assert normalise("  padded  ", "L1") == "padded"
    assert normalise("line one\nline two", "L1") == "line one line two"


def test_l1_strips_space_before_punctuation():
    # PyMuPDF renders "γ ." for "γ."; the space before closing punctuation is removed.
    assert normalise("trend in γ .", "L1") == "trend in γ."
    assert normalise("firm and month ,", "L1") == "firm and month,"
    assert normalise("(losers )", "L1") == "(losers)"
    # opening punctuation and mid-word are untouched.
    assert normalise("a. b", "L1") == "a. b"


def test_l1_does_not_dehyphenate():
    # de-hyphenation is an L2 rule; at L1 the fold artifact survives (whitespace
    # is already collapsed to a single space).
    assert normalise("disas-\nter", "L1") == "disas- ter"


# --- L2: de-hyphenation on top of L1 -----------------------------------------

def test_l2_dehyphenates_fold_artifact():
    # the exact v2 failure case: YAML-folded "disas- ter" and raw "disas-\nter"
    # both join to "disaster" (== PyMuPDF's auto-joined output).
    assert normalise("disas- ter", "L2") == "disaster"
    assert normalise("disas-\nter", "L2") == "disaster"
    assert normalise("disas-  ter", "L2") == "disaster"   # multiple spaces
    assert normalise("kurto-\nsis", "L2") == "kurtosis"


def test_l2_dehyphenation_is_idempotent_with_adjacent_folds():
    # a word broken across TWO lines has two adjacent fold points; the lookahead
    # joins both in one pass, and re-normalising is a no-op.
    assert normalise("extraordi- nar- ily", "L2") == "extraordinarily"
    assert normalise("a- b- c", "L2") == "abc"
    once = normalise("coun- ter- party risk", "L2")
    assert once == "counterparty risk"
    assert normalise(once, "L2") == once


def test_l2_over_join_guard_real_compounds():
    """Real hyphenated compounds pulled from the three corpus papers must be left
    BYTE-IDENTICAL by L2 (hyphen with no following whitespace is never joined). If
    any is altered, the assertion names it -- the finding surfaces here, not later.

    Sources: BBW (short-term, cross-sectional, five-factor, ten-factor, next-month,
    portfolio-level, value-weighted, long-short, highest-VaR, AAA-rated, safety-first);
    BPW (cross-section, bond-level, trade-by-trade, AAA-rated); KPP (risk-return,
    trade-off, out-of-sample, time-series, time-variation, cross-sectional,
    characteristic-managed, bond-month, five-factor).
    """
    compounds = [
        "value-weighted", "cross-sectional", "five-factor", "ten-factor",
        "next-month", "portfolio-level", "long-short", "highest-VaR", "AAA-rated",
        "safety-first", "risk-averse", "trade-by-trade", "bond-level",
        "risk-return", "trade-off", "out-of-sample", "time-series",
        "time-variation", "characteristic-managed", "bond-month",
    ]
    for c in compounds:
        assert normalise(c, "L2") == c, f"L2 over-joined real compound {c!r}"


def test_no_case_folding_at_any_level():
    s = "VaR AAA IPCA BBW"
    for level in LEVELS:
        assert normalise(s, level) == s


# --- structural properties ---------------------------------------------------

def test_idempotence():
    s = "The word ﬁnance and disas-\nter and Newey–West and  spaces."
    for level in LEVELS:
        once = normalise(s, level)
        assert normalise(once, level) == once


def test_rules_manifest_is_cumulative():
    # RULES feeds the freeze config + report; L2 must extend L1.
    assert RULES["L0"] == []
    assert RULES["L1"][:len(RULES["L1"])] == RULES["L2"][:len(RULES["L1"])]
    assert "whitespace_collapse" in RULES["L1"]          # v2: whitespace pinned at L1
    assert "dehyphenation_hyphen_whitespace" in RULES["L2"]
    assert "dehyphenation_hyphen_whitespace" not in RULES["L1"]
