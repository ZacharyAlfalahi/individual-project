"""
Canonical-text normalisation ladder (parser brief §6) -- **LADDER v2**.

The canonical text is the substrate every evidence check indexes into: model
quotes are verified against it (D9), STATED locators point into it (D6/D7). A
candidate quote matches iff, after applying the *same* normalisation to both the
page text and the quote, the quote is an exact substring of the page. This module
is the single source of truth for that normalisation. It is pure (no I/O, no
state), deterministic, and unit-tested against synthetic and real-fixture cases.

------------------------------------------------------------------------------
Amendment to brief §6 (2026-07-10, PRE-RUN, logged -- NOT a post-hoc loosening)
------------------------------------------------------------------------------
The brief's §6 ladder cannot reconcile the human-authored fixtures against PyMuPDF
output, for a reason that is a *representation mismatch*, not a parser-quality
problem. A word hyphenated across a line break appears as:

  * PAGE  (PyMuPDF ``get_text("text")``): hyphen + NEWLINE -- "disas-\\nter".
  * QUOTE (fixture, typed from the rendered PDF): YAML folds the wrapped source
    line to hyphen + SPACE -- "disas- ter".

The brief's de-hyphenation rule `(\\w)-\\n(\\w)` fires only on the NEWLINE (page)
side, never the SPACE (quote) side, so it leaves the two ASYMMETRIC ("disaster" vs
"disas- ter") at every level -> no match. And the brief put whitespace collapse at
L3 (the top), so nothing bridges PyMuPDF's per-line "\\n" until the most aggressive
level. Two corrections, applied identically to page and quote:

  1. Whitespace normalisation pinned at **L1** (not L3): PyMuPDF emits "\\n" at
     every line break, so every multi-line quote needs newline->space + run-collapse
     to match. This ALONE reconciles the hyphen folds ("disas-\\nter" and
     "disas- ter" both collapse to "disas- ter").
  2. De-hyphenation broadened to `(\\w)-\\s+(\\w)` at **L2** -- a symmetric
     generalisation (joins "disas- ter" -> "disaster" on both sides, and covers a
     parser that DOES auto-join).

Empirical note (bake-off, PyMuPDF 1.28.0): L1 and L2 score IDENTICALLY -- de-hyphenation
recovers zero additional quotes -- which confirms PyMuPDF PRESERVES the line-break
hyphen (it does not auto-join). So whitespace@L1 is the operative fix and L2's
de-hyphenation is defensive (kept for a future auto-joining parser). Rejected
alternative: edit the human-authored fixtures. Rejected -- the ladder fix is general;
the fixture edit is not. This is the brief's own "discovered gap -> logged v2, not
silent edit" path (§6/§2). The freeze config would carry `ladder_version: v2`.

------------------------------------------------------------------------------
The v2 ladder (cumulative; no case-folding at any level)
------------------------------------------------------------------------------
  * L0 -- exact byte substring, no transformation.
  * L1 -- NFKC + explicit ligature expansion + curly-quote unification + en/em-dash
          unification (only where flanked by word chars) + whitespace normalisation
          (every whitespace run -> single space; strip edges) + strip whitespace before
          closing punctuation (PyMuPDF "γ ." -> "γ.").
  * L2 -- L1 + de-hyphenation `(\\w)-\\s+(\\w)` -> join.

L3 is intentionally absent: with whitespace at L1 and de-hyphenation at L2 there is
nothing left for it to add.
"""

from __future__ import annotations

import re
import unicodedata

# The legal ladder levels, in increasing order of aggressiveness. The matcher and
# the bake-off iterate this to pick the LOWEST level that clears the §2 bar.
LEVELS: tuple[str, ...] = ("L0", "L1", "L2")

# Explicit ligature expansion. NFKC already decomposes these, but the explicit map
# is belt-and-braces: it makes the transform independent of NFKC's table version and
# self-documents intent. Exactly the five the brief lists (ﬀ ﬁ ﬂ ﬃ ﬄ).
_LIGATURES: dict[int, str] = {
    0xFB00: "ff",   # ﬀ
    0xFB01: "fi",   # ﬁ
    0xFB02: "fl",   # ﬂ
    0xFB03: "ffi",  # ﬃ
    0xFB04: "ffl",  # ﬄ
}

# Curly / typographic quotes -> straight ASCII. NFKC does NOT do this (they are not
# compatibility-equivalent), so it is a real transform. Case is untouched.
_QUOTES: dict[int, str] = {
    0x2018: "'",   # ‘  left single
    0x2019: "'",   # ’  right single / apostrophe
    0x201C: '"',   # “  left double
    0x201D: '"',   # ”  right double
}

# v3 (2026-07-22, D40): the COMPLETE dash class -> ASCII hyphen, UNCONDITIONALLY.
#
# Supersedes v2's `(?<=\w)[–—](?=\w)`, which was wrong in two independent ways and
# cost an anchor. It covered only en/em dash, so U+2212 MINUS SIGN -- the single
# most common non-ASCII character in this corpus (1,034 occurrences post-L1) --
# was never unified at all; and the word-flanking condition meant that even adding
# U+2212 to the class would NOT have fixed it, because papers write " t −6 ", where
# the minus is preceded by a SPACE and the lookbehind fails.
#
# The defect class is transcription equivalence: a model (or a human) reading a
# rendered PDF types ASCII "-" for every one of these glyphs, so a quote is
# discarded for a character difference no reader can see. Unconditional unification
# is safe for MATCHING because the same transform is applied to both the page and
# the quote -- and it is strictly safer than the flanked rule, which left 286
# unflanked en-dashes unmatched. Every member is a 1:1 substitution, so spans are
# LENGTH-PRESERVING and no locator offset moves (verified: 57/57 gold STATED
# locators slice back byte-exact before and after).
#
# Deliberately NOT included: U+00AD SOFT HYPHEN (invisible -- would need DELETION,
# which is not length-preserving; absent from this corpus, so out of scope until it
# appears). Deliberately NOT unified: Greek letters and math operators (γ, ×, ≈, ∗,
# ′). Those are CONTENT, not typography -- a transcriber reproduces them, and
# folding them to ASCII would destroy meaning rather than recover it.
_DASHES: dict[int, str] = {
    0x2010: "-",  # ‐ hyphen
    0x2011: "-",  # ‑ non-breaking hyphen
    0x2012: "-",  # ‒ figure dash
    0x2013: "-",  # – en dash
    0x2014: "-",  # — em dash
    0x2015: "-",  # ― horizontal bar
    0x2043: "-",  # ⁃ hyphen bullet
    0x2212: "-",  # − MINUS SIGN (Sm, not Pd -- NFKC does not touch it)
}

# Any run of whitespace (incl. newlines/tabs; Unicode \s covers NBSP) -> single space.
_WS_RUN = re.compile(r"\s+")

# Whitespace before closing punctuation -> removed. PyMuPDF inserts a space between
# a special glyph and the following punctuation (e.g. "γ ." for "γ."); this is a
# rendering artifact, so the normalisation is symmetric on page and quote and
# justified independent of matching. (2026-07-10, logged with the v2 amendment.)
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:)])")

# De-hyphenation: a word char + hyphen + ANY whitespace run, with the following word
# char as a LOOKAHEAD (not consumed). Not consuming it means two adjacent fold points
# (a word broken across two lines) both join in one left-to-right pass, so the transform
# is idempotent. v2 broadening of the brief's `(\w)-\n(\w)`; see the module amendment note.
_HYPHEN_BREAK = re.compile(r"(\w)-\s+(?=\w)")

# Ordered, human-readable rule lists per level -- consumed by the freeze config and
# the bake-off report so the recipe is recorded exactly, never re-hardcoded.
_L1_RULES: list[str] = [
    "nfkc",
    "ligature_expansion",
    "curly_quote_unification",
    "dash_unification",          # v3: complete dash class, unconditional
    "whitespace_collapse",
    "strip_space_before_punctuation",
]
_L2_RULES: list[str] = [*_L1_RULES, "dehyphenation_hyphen_whitespace"]
RULES: dict[str, list[str]] = {
    "L0": [],
    "L1": list(_L1_RULES),
    "L2": list(_L2_RULES),
}


def _l1(text: str) -> str:
    """Apply the L1 rules in order.

    Dash unification is now context-free (v3), so it no longer depends on running
    before whitespace collapse; the order is kept for determinism and so the rule
    list reads in application order."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_LIGATURES)
    text = text.translate(_QUOTES)
    text = text.translate(_DASHES)
    text = _WS_RUN.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    return text.strip()


def _l2(text: str) -> str:
    """L2 = L1 output + de-hyphenation. Idempotent: the lookahead (see ``_HYPHEN_BREAK``)
    does not consume the trailing word char, so adjacent fold points both join in one
    pass and re-normalising is a no-op."""
    return _HYPHEN_BREAK.sub(r"\1", _l1(text))


def normalise(text: str, level: str) -> str:
    """Normalise ``text`` at ``level`` (``"L0"``|``"L1"``|``"L2"``).

    The SAME call is applied to a canonical page and to a candidate quote before
    the matcher compares them (``locate.locate_quote``). L0 is the identity, so an
    L0 locator's char offsets are byte-correct into the raw page.
    """
    if level not in LEVELS:
        raise ValueError(f"normalise level must be one of {list(LEVELS)}; got {level!r}")
    if level == "L0":
        return text
    if level == "L1":
        return _l1(text)
    return _l2(text)
