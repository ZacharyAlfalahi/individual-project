"""
Unit tests for the quote matcher (parser brief §5 matching + §7 diagnostics).

The bug-catching tests use the REAL fixture strings from
``evaluation/quote_fixtures/*.yaml`` (not synthetic ``-\\n`` quotes): the fixtures
are YAML-folded to HYPHEN-SPACE ("disas- ter"), which a synthetic ``-\\n`` test
would never reproduce. Each is reconciled against a PyMuPDF-style page (hyphenated
words auto-joined, real line breaks as ``\\n``) to prove the v2 ladder matches at
L2 and correctly fails at L0. The corrupted near-miss set is asserted to fail at
EVERY level, enumerated against the mutations actually present (reword, number_swap).
"""

import re
from pathlib import Path

import pytest
import yaml

from agents.librarian.config.locate import MatchResult, locate_quote, nearest_window

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "evaluation" / "quote_fixtures"
_PAPERS = ("bbw_2019", "bpw_2011", "kpp_2023")

_HYPHEN_FOLD = re.compile(r"(\w)-\s+(\w)")


def _load(paper: str) -> dict:
    with (_FIXTURE_DIR / f"{paper}.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _all_true() -> list[dict]:
    return [q for p in _PAPERS for q in _load(p)["quotes"]]


def _all_corrupted() -> list[tuple[dict, dict]]:
    """Each corrupted fixture paired with the true quote it was derived from."""
    pairs = []
    for p in _PAPERS:
        doc = _load(p)
        by_id = {q["id"]: q for q in doc["quotes"]}
        for c in doc.get("corrupted", []):
            pairs.append((c, by_id[c["derived_from"]]))
    return pairs


def _pymupdf_page(true_text: str) -> str:
    """Mimic how PyMuPDF renders a fixture's source region: auto-join words that
    were hyphenated across a line break, and emit ``\\n`` at each line break."""
    joined = _HYPHEN_FOLD.sub(r"\1\2", true_text)
    return joined.replace(" ", "\n")


# --- reconciliation: the exact v2 bug (real fixtures) ------------------------

@pytest.mark.parametrize("paper", _PAPERS)
def test_true_quotes_reconcile_at_l2_not_l0(paper):
    """Every real true quote matches its PyMuPDF-style page at L2; the folded ones
    do NOT match at L0 (proving the ladder, not luck, is doing the work)."""
    for q in _load(paper)["quotes"]:
        page = _pymupdf_page(q["text"])
        assert locate_quote((page,), q["text"], "L2").matched, (
            f"{q['id']} failed to reconcile at L2"
        )


def test_folded_quotes_miss_at_l0():
    # a quote carrying the hyphen-space fold cannot match the joined page at L0.
    folded = [q for q in _all_true() if _HYPHEN_FOLD.search(q["text"])]
    assert folded, "expected some hyphen-folded fixtures"
    for q in folded:
        page = _pymupdf_page(q["text"])
        assert not locate_quote((page,), q["text"], "L0").matched, (
            f"{q['id']} unexpectedly matched at L0"
        )


# --- ligature / whitespace-only differences ----------------------------------

def test_ligature_quote_matches_only_from_l1():
    pages = ("The word ﬁnance appears on this page.",)  # page carries the glyph
    assert not locate_quote(pages, "finance", "L0").matched
    assert locate_quote(pages, "finance", "L1").matched


def test_whitespace_only_difference_matches_from_l1():
    pages = ("the long short\nfactor return series",)
    assert not locate_quote(pages, "long short factor return series", "L0").matched
    assert locate_quote(pages, "long short factor return series", "L1").matched


# --- cross-page fallback -----------------------------------------------------

def test_cross_page_fallback_records_starting_page():
    pages = ("a sentence that begins on the", "next page and then finishes here")
    quote = "begins on the next page and then finishes"
    res = locate_quote(pages, quote, "L1")
    assert res.matched
    assert res.used_cross_page
    assert res.page == 0  # locator records the STARTING page


def test_within_page_match_is_not_flagged_cross_page():
    pages = ("first page", "the TARGET token lives here")
    res = locate_quote(pages, "TARGET token", "L1")
    assert res.matched and not res.used_cross_page and res.page == 1


# --- corrupted near-miss discipline (zero matches at every level) ------------

def test_corrupted_quotes_never_match_any_level():
    pairs = _all_corrupted()
    assert pairs, "expected corrupted fixtures"
    for corrupted, true in pairs:
        page = _pymupdf_page(true["text"])
        for level in ("L0", "L1", "L2"):
            assert not locate_quote((page,), corrupted["text"], level).matched, (
                f"{corrupted['id']} matched at {level} (corrupted set must never match)"
            )


def test_present_mutation_labels_are_the_known_set():
    # assert against what EXISTS (reword, number_swap) -- no empty-case testing.
    labels = {c["mutation"] for c, _ in _all_corrupted()}
    assert labels, "expected at least one corrupted mutation"
    assert labels <= {"reword", "number_swap"}, f"new mutation label(s): {labels}"


# --- diagnostics + determinism -----------------------------------------------

def test_nearest_window_returns_a_page_slice():
    page = "the average VaR of the highest quintile is 11.88 percent per annum"
    window = nearest_window(page, "average VaR of the lowest quintile")
    assert window and window in page


def test_locate_is_deterministic():
    pages = ("alpha disas-\nter beta", "gamma delta")
    a = locate_quote(pages, "disas- ter beta", "L2")
    b = locate_quote(pages, "disas- ter beta", "L2")
    assert a == b
    assert isinstance(a, MatchResult) and a.matched


def test_miss_returns_structured_result_not_none():
    res = locate_quote(("nothing relevant here",), "absent phrase", "L2")
    assert isinstance(res, MatchResult)
    assert res.matched is False and res.page is None
