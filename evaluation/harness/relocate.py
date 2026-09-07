"""
Relocate-then-certify quote-relocator — DIAGNOSTIC INSTRUMENT ONLY.

This module answers one question about an archived model quote that failed the
production quote gate: *does the canonical text contain a span similar enough to
the quote that a relocating locator of pre-registered strength would certify it?*

It is NEVER on the decision path. The production gate
(``agents.librarian.config.locate.locate_quote``) stays exact-substring-only, per
its own "no fuzzy / similarity match on the decision path" guarantee; this module
is imported exclusively by the locator-diagnostic scripts
(``scripts/run_locator_census.py``, ``scripts/calibrate_relocator.py``,
``scripts/run_relocator_rescore.py``) and their tests.

Relocate-then-certify: the model's quote is demoted to a SEARCH KEY. What an
accepted relocation certifies is the paper's own exact span — the L1 ``Locator``
(runtime convention, byte-comparable with STATED evidence) plus the raw-L0 byte
span recovered through ``perturb.map_l1_span_to_l0``. A relocation that cannot be
certified raises ``RelocateError``; an uncertified accept is impossible.

Determinism: ``difflib.SequenceMatcher`` with ``autojunk=False`` everywhere (the
autojunk heuristic silently changes behaviour on pages with high-frequency
characters); ties break on ``(score desc, page asc, offset asc, single-page
first)``; no randomness, no clock.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from difflib import SequenceMatcher

from agents.librarian.config.canonical_text import CanonicalText
from agents.librarian.config.normalise import normalise
from agents.quant.config import Locator

from .perturb import PerturbError, map_l1_span_to_l0


class RelocateError(RuntimeError):
    """Certification or invariant failure inside an accepted relocation.

    Raised instead of ever returning an uncertified accept."""


@dataclass(frozen=True)
class Candidate:
    """The best relocation candidate for one quote in one canonical text.

    ``l1_start``/``l1_end`` index ``normalise(page, "L1")`` for a single-page
    candidate, or the joined pair ``normalise(pages[page] + "\\n" +
    pages[page + 1], "L1")`` when ``cross_page`` — the identical join the
    production matcher searches. ``runner_up`` is the second-best surviving
    candidate's score (0.0 if none); it is recorded descriptively and never
    enters the acceptance predicate."""

    page: int
    l1_start: int
    l1_end: int
    cross_page: bool
    score: float
    runner_up: float


@dataclass(frozen=True)
class RelocateResult:
    """Outcome of ``relocate`` for one quote against one canonical text.

    ``method``: ``"exact"`` | ``"relocated"`` | ``"rejected"``.
    ``reason``: ``"exact_match"`` | ``"accepted_at_bar"`` | ``"below_bar"`` |
    ``"too_short"`` | ``"empty_quote"`` | ``"no_anchor"``.
    ``l0_span``/``l0_text`` are the raw-L0 certification (into ``pages[page]``,
    or the raw joined pair ``pages[page] + "\\n" + pages[page + 1]`` when
    ``cross_page``); populated for relocated accepts only — an exact accept's
    provenance is already the production contract's."""

    accepted: bool
    method: str
    score: float
    locator: Locator | None
    l1_text: str | None
    l0_span: tuple[int, int] | None
    l0_text: str | None
    cross_page: bool
    runner_up: float
    reason: str


# Search-text cache: L1-normalising ~40 pages plus ~39 adjacent-pair joins costs
# more than the matching itself when the same paper is scanned for hundreds of
# quotes (the calibration sweep). Keyed by a digest of the raw pages — never by
# object identity or source_sha256 (two texts could share a source PDF).
_SEARCH_TEXT_CACHE: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}


def _pages_digest(pages: tuple[str, ...]) -> str:
    h = hashlib.sha256()
    for p in pages:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _search_texts(ct: CanonicalText) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(per-page L1 texts, adjacent-pair-join L1 texts) for ``ct``, cached."""
    key = _pages_digest(ct.pages)
    hit = _SEARCH_TEXT_CACHE.get(key)
    if hit is not None:
        return hit
    norm_pages = tuple(normalise(p, "L1") for p in ct.pages)
    joins = tuple(
        normalise(ct.pages[i] + "\n" + ct.pages[i + 1], "L1")
        for i in range(len(ct.pages) - 1)
    )
    _SEARCH_TEXT_CACHE[key] = (norm_pages, joins)
    return norm_pages, joins


def _require_l1(ct: CanonicalText) -> None:
    level = ct.normalisation.get("ladder_level")
    if level != "L1":
        raise RelocateError(
            f"the relocator requires an L1 canonical text (map_l1_span_to_l0 is "
            f"L1-specific); got ladder_level={level!r} for {ct.source_pdf!r}"
        )


def _span_within_one_page(ct: CanonicalText, page: int, s: int, e: int,
                          norm_pages: tuple[str, ...], join: str) -> bool:
    """True iff a pair-join span [s, e) lies wholly inside one of the two pages.

    The join's normalisation glues the seam, so containment is tested from both
    ends rather than assuming ``join == norm_a + " " + norm_b``: the span sits in
    the left page iff it ends before the left page's normalised length, in the
    right page iff it starts after the join's length minus the right page's."""
    if e <= len(norm_pages[page]):
        return True
    if s >= len(join) - len(norm_pages[page + 1]):
        return True
    return False


def score_quote(ct: CanonicalText, quote: str, *, min_anchor_chars: int = 10) -> Candidate | None:
    """Bar-free best relocation candidate for ``quote`` in ``ct``.

    Search: for the L1-normalised quote ``Q`` against every page's L1 text and
    every adjacent-pair join — (1) ``find_longest_match`` anchors the quote in
    the text (texts sharing no block of ``min_anchor_chars`` are skipped);
    (2) a window of ``len(Q) + slack`` (``slack = max(32, len(Q)//4)``) is cut
    around the anchor; (3) the window is trimmed to its outermost matching
    blocks against ``Q``; (4) the trimmed span is scored with
    ``SequenceMatcher.ratio()``. Pair-join candidates lying wholly within one
    page defer to the per-page representation (so the runner-up is a genuinely
    distinct location, not the same span seen through the seam join).

    Returns ``None`` when no text shares an anchor block (or the quote
    normalises to empty). Pure and deterministic."""
    _require_l1(ct)
    q = normalise(quote, "L1")
    if q == "":
        return None

    norm_pages, joins = _search_texts(ct)
    matcher = SequenceMatcher(autojunk=False)
    matcher.set_seq2(q)
    slack = max(32, len(q) // 4)

    candidates: list[tuple[float, int, int, bool, int]] = []  # (score, page, s, cross, e)
    texts: list[tuple[int, bool, str]] = [(i, False, t) for i, t in enumerate(norm_pages)]
    texts += [(i, True, t) for i, t in enumerate(joins)]
    for page, cross, text in texts:
        matcher.set_seq1(text)
        anchor = matcher.find_longest_match(0, len(text), 0, len(q))
        if anchor.size < min_anchor_chars:
            continue
        w0 = max(0, anchor.a - anchor.b - slack)
        w1 = min(len(text), anchor.a + (len(q) - anchor.b) + slack)
        blocks = [
            b for b in SequenceMatcher(a=text[w0:w1], b=q, autojunk=False).get_matching_blocks()
            if b.size > 0
        ]
        if not blocks:
            continue
        s = w0 + blocks[0].a
        e = w0 + blocks[-1].a + blocks[-1].size
        if cross and _span_within_one_page(ct, page, s, e, norm_pages, text):
            continue
        score = SequenceMatcher(a=text[s:e], b=q, autojunk=False).ratio()
        candidates.append((score, page, s, cross, e))

    if not candidates:
        return None
    # score desc, then page asc, offset asc, single-page before cross — total order.
    candidates.sort(key=lambda c: (-c[0], c[1], c[2], c[3]))
    best = candidates[0]
    runner_up = candidates[1][0] if len(candidates) > 1 else 0.0
    return Candidate(
        page=best[1], l1_start=best[2], l1_end=best[4],
        cross_page=best[3], score=best[0], runner_up=runner_up,
    )


def _certify(ct: CanonicalText, cand: Candidate) -> tuple[Locator, str, tuple[int, int], str]:
    """L0-certify an accepted candidate: (locator, l1_text, l0_span, l0_text).

    Raises ``RelocateError`` unless the round-trip holds exactly:
    ``normalise(l0_text, "L1") == l1_text == ct.slice_text(locator)``."""
    if cand.cross_page:
        text_l0 = ct.pages[cand.page] + "\n" + ct.pages[cand.page + 1]
    else:
        text_l0 = ct.pages[cand.page]
    l1_text = normalise(text_l0, "L1")[cand.l1_start:cand.l1_end]
    try:
        l0_start, l0_end = map_l1_span_to_l0(text_l0, cand.l1_start, cand.l1_end, l1_text)
    except PerturbError as exc:
        raise RelocateError(
            f"accepted relocation could not be L0-certified (page {cand.page}, "
            f"span [{cand.l1_start}:{cand.l1_end}], cross_page={cand.cross_page}): {exc}"
        ) from exc
    l0_text = text_l0[l0_start:l0_end]
    if normalise(l0_text, "L1") != l1_text:
        raise RelocateError(
            f"L0 certification round-trip failed on page {cand.page}: the raw span does "
            f"not re-normalise to the relocated L1 window"
        )
    locator = Locator(
        page=cand.page,
        char_start=cand.l1_start,
        char_end=cand.l1_end,
        end_page=(cand.page + 1) if cand.cross_page else None,
    )
    if ct.slice_text(locator) != l1_text:
        raise RelocateError(
            f"certified locator does not slice back to the relocated window on page {cand.page}"
        )
    return locator, l1_text, (l0_start, l0_end), l0_text


def relocate(ct: CanonicalText, quote: str, *, bar: float, min_quote_chars: int,
             min_anchor_chars: int = 10) -> RelocateResult:
    """Exact-or-relocate one quote against one canonical text.

    Order of decision: empty-quote guard (before ``ct.locate``, which raises on
    ``""``) → exact short-circuit (score 1.0) → a-priori length guard on the L1
    quote → best candidate via ``score_quote`` → the single-bar acceptance
    predicate → L0 certification (``RelocateError`` on any failure — never an
    uncertified accept)."""
    if not isinstance(quote, str) or quote.strip() == "":
        return RelocateResult(
            accepted=False, method="rejected", score=0.0, locator=None, l1_text=None,
            l0_span=None, l0_text=None, cross_page=False, runner_up=0.0,
            reason="empty_quote",
        )
    _require_l1(ct)

    exact = ct.locate(quote, level="L1")
    if exact is not None:
        return RelocateResult(
            accepted=True, method="exact", score=1.0, locator=exact,
            l1_text=ct.slice_text(exact, level="L1"), l0_span=None, l0_text=None,
            cross_page=exact.end_page is not None, runner_up=0.0,
            reason="exact_match",
        )

    q = normalise(quote, "L1")
    if len(q) < min_quote_chars:
        return RelocateResult(
            accepted=False, method="rejected", score=0.0, locator=None, l1_text=None,
            l0_span=None, l0_text=None, cross_page=False, runner_up=0.0,
            reason="too_short",
        )

    cand = score_quote(ct, quote, min_anchor_chars=min_anchor_chars)
    if cand is None:
        return RelocateResult(
            accepted=False, method="rejected", score=0.0, locator=None, l1_text=None,
            l0_span=None, l0_text=None, cross_page=False, runner_up=0.0,
            reason="no_anchor",
        )
    if cand.score < bar:
        return RelocateResult(
            accepted=False, method="rejected", score=cand.score, locator=None,
            l1_text=None, l0_span=None, l0_text=None, cross_page=cand.cross_page,
            runner_up=cand.runner_up, reason="below_bar",
        )

    locator, l1_text, l0_span, l0_text = _certify(ct, cand)
    return RelocateResult(
        accepted=True, method="relocated", score=cand.score, locator=locator,
        l1_text=l1_text, l0_span=l0_span, l0_text=l0_text,
        cross_page=cand.cross_page, runner_up=cand.runner_up,
        reason="accepted_at_bar",
    )
