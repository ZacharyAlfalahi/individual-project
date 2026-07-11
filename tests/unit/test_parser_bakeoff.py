"""
Unit tests for the parser bake-off harness: the v3 publisher-watermark strip and
the §2 decision rule. Both are pure (no PDF / no PyMuPDF needed to exercise them).
"""

from scripts.parser_bakeoff.build_canonical_text import strip_publisher_watermark
from scripts.parser_bakeoff.run_bakeoff import (
    PAPERS,
    THIN_CATEGORY_MAX,
    CellStats,
    decide,
)

_WATERMARK = (
    " 15406261, 2011, 3, Downloaded from "
    "https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2011.01655.x by "
    "New York University, Wiley Online Library on [16/06/2026]. See the Terms "
    "and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley "
    "Online Library for rules of use; OA articles are governed by the applicable "
    "Creative Commons License"
)


# --- v3 watermark strip ------------------------------------------------------

def test_strip_removes_watermark_and_reconnects_real_text():
    page = "the average of the overall" + _WATERMARK + " gamma is positive."
    out = strip_publisher_watermark(page)
    assert "Downloaded from" not in out
    assert "onlinelibrary.wiley.com" not in out
    assert "Creative Commons License" not in out
    # the real text on both sides of the injection survives
    assert "the average of the overall" in out
    assert "gamma is positive." in out


def test_strip_is_idempotent():
    page = "text" + _WATERMARK + " more"
    once = strip_publisher_watermark(page)
    assert strip_publisher_watermark(once) == once


def test_strip_leaves_watermark_free_text_untouched():
    clean = "Although corporate bonds and stocks both reflect firm fundamentals."
    assert strip_publisher_watermark(clean) == clean


def test_strip_does_not_eat_ordinary_urls():
    # a non-Wiley-watermark URL in the body must be preserved.
    body = "see https://example.com/paper for details"
    assert strip_publisher_watermark(body) == body


# --- §2 decision rule --------------------------------------------------------

def _cell(level, per_paper, category, corrupted=0):
    pm = sum(m for m, _ in per_paper.values())
    pt = sum(t for _, t in per_paper.values())
    return CellStats(level, per_paper, pm, pt, category, corrupted, 0, [])


def test_decide_picks_lowest_passing_level():
    full = {p: (16, 16) for p in PAPERS}
    l0 = _cell("L0", {p: (0, 16) for p in PAPERS}, {"body": (0, 48)})
    l1 = _cell("L1", full, {"body": (48, 48)})
    l2 = _cell("L2", full, {"body": (48, 48)})
    assert decide({"L0": l0, "L1": l1, "L2": l2}).winner == "L1"


def test_thin_empty_category_flags_not_vetoes():
    full = {p: (16, 16) for p in PAPERS}
    cat = {"body": (48, 48), "page_break": (0, THIN_CATEGORY_MAX)}  # thin, zero matches
    cell = _cell("L1", full, cat)
    fail = _cell("L0", {p: (0, 16) for p in PAPERS}, {"body": (0, 48)})
    v = decide({"L0": fail, "L1": cell, "L2": cell})
    assert v.winner == "L1"
    assert any("page_break" in f for f in v.flags)


def test_nonthin_empty_category_vetoes():
    full = {p: (16, 16) for p in PAPERS}
    cat = {"body": (0, THIN_CATEGORY_MAX + 9)}  # non-thin, zero matches
    cell = _cell("L1", full, cat)
    assert decide({"L0": cell, "L1": cell, "L2": cell}).winner is None


def test_corrupted_match_vetoes():
    full = {p: (16, 16) for p in PAPERS}
    cell = _cell("L1", full, {"body": (48, 48)}, corrupted=1)
    assert decide({"L0": cell, "L1": cell, "L2": cell}).winner is None


def test_subbar_per_paper_vetoes():
    # pooled fine but one paper below the per-paper floor.
    per_paper = {PAPERS[0]: (16, 16), PAPERS[1]: (12, 16), PAPERS[2]: (16, 16)}  # bpw 75%
    cell = _cell("L1", per_paper, {"body": (44, 48)})
    assert decide({"L0": cell, "L1": cell, "L2": cell}).winner is None
