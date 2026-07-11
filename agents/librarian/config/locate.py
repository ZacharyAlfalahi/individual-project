"""
Quote matcher / locator (parser brief §5 matching + §7 step-6 diagnostics).

The single source of truth for "does this quote appear in the canonical text, and
where?" -- used both by the runtime locator (``CanonicalText.locate``) and the
offline parser bake-off. Matching is exact-substring after applying the SAME
normalisation ladder (``normalise.normalise``) to both the page text and the quote
at a chosen ``level``; there is no fuzzy / similarity match on the decision path
(D9: the quote gate is binary). ``difflib`` appears only in ``nearest_window``,
which produces a *diagnostic* window for a FAILED quote -- never a match.

Cross-page fallback (§5): a quote is matched per page first; on failure it is tried
against each adjacent page-pair concatenation (joined with a newline, then
normalised at ``level`` so an intra-fold hyphen or a line break at the seam is
handled by the same ladder). The locator records the STARTING page and sets
``used_cross_page`` so the bake-off can report which quotes needed the fallback.

Offset semantics: char offsets are into ``normalise(page, level)`` (store-L0 /
normalise-on-read). At L0 the normalisation is the identity, so offsets are
byte-correct into the raw page -- the frozen behaviour ``CanonicalText.locate``
relied on before the ladder landed.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

from .normalise import normalise


@dataclass(frozen=True)
class MatchResult:
    """Outcome of locating one quote at one ladder level.

      * ``matched``        -- did the quote locate?
      * ``level``          -- the ladder level the search ran at.
      * ``page``           -- starting page of the match (``None`` if no match).
      * ``char_start/end`` -- span into ``normalise(page, level)`` (``None`` if no
                              match). For a cross-page match these index the joined
                              page-pair string, not a single page.
      * ``used_cross_page``-- True iff the match needed the adjacent-pair fallback.
    """

    matched: bool
    level: str
    page: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    used_cross_page: bool = False


def locate_quote(pages: tuple[str, ...], quote: str, level: str) -> MatchResult:
    """Locate ``quote`` in ``pages`` at ``level``.

    Per-page exact substring first (lowest page index wins); on failure, the
    adjacent page-pair fallback. Returns a structured :class:`MatchResult` either
    way (never raises on a miss); an empty / whitespace-only normalised quote is a
    definite non-match (guards against a vacuous ``"".find`` hit)."""
    q = normalise(quote, level)
    if q == "":
        return MatchResult(matched=False, level=level)

    # Per-page.
    for i, page in enumerate(pages):
        idx = normalise(page, level).find(q)
        if idx != -1:
            return MatchResult(
                matched=True,
                level=level,
                page=i,
                char_start=idx,
                char_end=idx + len(q),
                used_cross_page=False,
            )

    # Cross-page fallback: each adjacent pair, joined then normalised so a fold /
    # break at the seam is handled by the active ladder. Records the starting page.
    for i in range(len(pages) - 1):
        joined = normalise(pages[i] + "\n" + pages[i + 1], level)
        idx = joined.find(q)
        if idx != -1:
            return MatchResult(
                matched=True,
                level=level,
                page=i,
                char_start=idx,
                char_end=idx + len(q),
                used_cross_page=True,
            )

    return MatchResult(matched=False, level=level)


def nearest_window(page_text: str, quote: str, width: int | None = None) -> str:
    """Return the slice of ``page_text`` most similar to ``quote`` (diagnostic only).

    For a failed true quote, this makes the REASON visible (parser artifact vs
    quote typo vs missing normalisation rule) per brief §7 step 6. Uses difflib's
    longest common block to anchor the window; falls back to the head of the page
    when there is no shared block at all."""
    if not page_text:
        return ""
    if width is None:
        width = len(quote) + 40
    matcher = difflib.SequenceMatcher(a=page_text, b=quote, autojunk=False)
    block = matcher.find_longest_match(0, len(page_text), 0, len(quote))
    if block.size == 0:
        return page_text[:width]
    start = max(0, block.a - 10)
    end = min(len(page_text), block.a + width)
    return page_text[start:end]
