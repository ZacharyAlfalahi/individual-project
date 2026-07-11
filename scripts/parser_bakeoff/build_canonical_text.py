"""
PyMuPDF PDF -> L0 canonical text (parser brief §5), with a determinism gate.

The canonical text stores the parser's RAW per-page text (ladder level L0); the
normalisation ladder is applied later at match time (store-L0 / normalise-on-read),
so a frozen canonical text records ``ladder_level`` and the matcher normalises on
read. Determinism (§7 step 2): the PDF is parsed twice and the page lists must be
byte-identical -- irreducible nondeterminism disqualifies the parser.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

# PyMuPDF (pinned in scripts/requirements_parser.txt) is imported lazily so the
# module and its pure-text helpers load without it; only PDF parsing pulls it in.
def _pymupdf():
    import pymupdf
    return pymupdf

# --- v3 page-canonicalisation (2026-07-10, logged) ----------
# PAGE-ONLY, applied at canonical-text BUILD (NOT part of the symmetric L0-L2 match
# ladder in normalise.py -- quotes never carry a watermark). The Wiley Online Library
# download watermark is injected into the reading-order stream on ~every page of the
# JF PDFs (35/36 BPW, 41/42 KPP; BBW/Elsevier has none), splitting real text and
# breaking page-bottom / footnote / page-break quotes. It is a single contiguous
# block per page with a stable shape:
#   " <ISSN>, <year>, <issue>, Downloaded from https://onlinelibrary.wiley.com/doi/<doi>
#     by <institution>, Wiley Online Library on [<date>]. See the Terms and Conditions
#     (<url>) on Wiley Online Library for rules of use; ... Creative Commons License"
# Removing NON-PAPER text (not loosening matching); anchored tightly so it cannot eat
# real content: the optional numeric prefix, the wiley.com "Downloaded from" URL, and a
# non-greedy run to the sole per-page "Creative Commons License" terminator.
_WILEY_WATERMARK = re.compile(
    r"(?:\s*\d[\d,]*,\s*\d{4},\s*\d+,)?\s*"
    r"Downloaded from https://onlinelibrary\.wiley\.com/\S+ by .*?"
    r"Creative Commons License",
    re.DOTALL,
)


def strip_publisher_watermark(text: str) -> str:
    """Remove the Wiley download-watermark block(s) from one page's text (v3).

    Deterministic and idempotent; leaves watermark-free pages (e.g. BBW) untouched.
    Replaces each block with a single space so the real text on either side of the
    injection reconnects for matching."""
    return _WILEY_WATERMARK.sub(" ", text)


# NOTE — a page-furniture strip (repeated running heads/feet + page-number lines) was
# evaluated and REMOVED (2026-07-11). It recovered ZERO fixture quotes (score 45/48 with
# or without it) while risking silent deletion of real body content (a bare numeric table
# cell like a lone "2008" matches the page-number heuristic). A footnote-block strip was
# also tried and rejected by the gate (it regressed the footnote-category quotes, which
# must locate INTO footnote text). The page-break quotes that furniture stripping was meant
# to help remain unrecovered: their sentence is split at the seam by a page-bottom footnote,
# which needs cross-page-seam handling in the matcher (footnotes kept for footnote quotes,
# skipped only for the seam join) -- outside this builder's scope, not a page-canonicalisation rule.


def parse_pages(pdf_path: str | Path) -> list[str]:
    """Extract per-page plain text with PyMuPDF (reading-order ``"text"`` mode)."""
    doc = _pymupdf().open(pdf_path)
    try:
        return [page.get_text("text") for page in doc]
    finally:
        doc.close()


def build_canonical(pdf_path: str | Path, *, strip_watermark: bool = True) -> dict:
    """Parse ``pdf_path`` into the parser-brief §5 canonical shape.

    Parses TWICE and asserts byte-identical page lists (the §7 determinism gate; the
    watermark is static PDF content, so it is deterministic). Then applies the v3
    page-canonicalisation (``strip_publisher_watermark``) unless disabled. ``status``
    is deliberately omitted here (an L0 build artifact is not yet a frozen canonical
    text -- ``run_bakeoff`` stamps ``status: frozen`` only for the winning cell)."""
    pdf_path = Path(pdf_path)
    raw = pdf_path.read_bytes()
    pages = parse_pages(pdf_path)
    pages_again = parse_pages(pdf_path)
    if pages != pages_again:
        raise RuntimeError(
            f"non-deterministic parse for {pdf_path}: two runs differ (disqualifies "
            "the parser per brief §7 step 2)"
        )
    page_rules: list[str] = []
    if strip_watermark:
        pages = [strip_publisher_watermark(p) for p in pages]
        page_rules.append("strip_publisher_watermark_v3")
    return {
        "source_pdf": str(pdf_path),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "parser": {"name": "pymupdf", "version": _pymupdf().__version__},
        "normalisation": {"ladder_level": "L0", "rules": []},
        "page_canonicalisation": page_rules,
        "pages": pages,
    }
