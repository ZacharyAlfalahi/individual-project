"""
Canonical text + status gate (build brief §5.1; parser-brief §5 shape).

The *canonical text* is the substrate every evidence check indexes into: model
quotes are verified against it (D9), STATED locators point into it (D6/D7), and
RQ1's audit trail rests on it. This module carries the loaded object and the
``locate`` entry point; the normalisation ladder (``normalise``) and the matcher
(``locate.locate_quote``) live in sibling modules. The PDF parser that PRODUCES the
pages (and the frozen recipe in ``config/canonical_text.yaml``) is chosen by the
parser bake-off; this module is parser-agnostic -- it consumes ``pages`` however
they were built.

**Status gate (CRITICAL).** A canonical text carries a ``status`` in
``{"stub", "frozen"}``. ``load_canonical_text`` loads *any* status (tests and
scaffolding need to load the stub). But the pipeline entry point must be
*structurally* unable to run a real extraction on a stub: ``require_frozen``
raises ``CanonicalTextNotFrozenError`` unless ``status == "frozen"``. The stub
fixture ships ``status: stub``, so any code path that reaches extraction through
``require_frozen`` refuses -- a stub can never be silently treated as a real
paper. Real quote fixtures are a separate human-only task; this module never
authors them.

**Locator discipline (parser-brief §5).** ``locate(quote, level="L0")`` normalises
both the page text and the quote at ``level`` and returns a ``Locator(page,
char_start, char_end)`` for the first (cross-page-aware) match, or ``None``. It
reuses the frozen ``agents.quant.config.Locator`` (D6/D7): the same locator type
STATED evidence carries, so a located quote and a STATED locator are byte-
comparable. The default ``L0`` is exact substring (identity ladder), preserving the
original byte-into-``pages[page]`` semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from agents.quant.config import Locator

from ..errors import CanonicalTextNotFrozenError, LibrarianSchemaError
from .locate import locate_quote

# The two legal statuses. "stub" = scaffolding text (offline tests); "frozen" =
# a real, bake-off-parsed canonical text cleared to reach extraction.
STATUSES: frozenset[str] = frozenset(("stub", "frozen"))


@dataclass(frozen=True)
class CanonicalText:
    """A parsed, normalised paper as the pipeline sees it (parser-brief §5 shape):

      * ``source_pdf``   -- the source PDF path (audit).
      * ``source_sha256``-- sha256 of the source PDF bytes (freeze identity).
      * ``parser``       -- ``{name, version}`` of the parser that produced it.
      * ``normalisation``-- ``{ladder_level, rules}`` the L0..L3 ladder applied.
      * ``pages``        -- the per-page canonical text; a locator's ``page``
                           indexes into this tuple.
      * ``status``       -- ``"stub"`` | ``"frozen"`` (the status gate, §5.1).

    Frozen + hashable: ``pages`` and the rule list are coerced to tuples so the
    object stays immutable."""

    source_pdf: str
    source_sha256: str
    parser: dict
    normalisation: dict
    pages: tuple[str, ...]
    status: str

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise LibrarianSchemaError(
                f"CanonicalText.status must be one of {sorted(STATUSES)}; got {self.status!r}"
            )
        if not isinstance(self.source_pdf, str) or self.source_pdf.strip() == "":
            raise LibrarianSchemaError("CanonicalText.source_pdf must be a non-empty string")
        if not isinstance(self.source_sha256, str) or self.source_sha256.strip() == "":
            raise LibrarianSchemaError("CanonicalText.source_sha256 must be a non-empty string")
        if not isinstance(self.parser, dict) or "name" not in self.parser or "version" not in self.parser:
            raise LibrarianSchemaError(
                "CanonicalText.parser must be a mapping with 'name' and 'version'"
            )
        if (
            not isinstance(self.normalisation, dict)
            or "ladder_level" not in self.normalisation
            or "rules" not in self.normalisation
        ):
            raise LibrarianSchemaError(
                "CanonicalText.normalisation must be a mapping with 'ladder_level' and 'rules'"
            )
        # Coerce pages to a tuple of strings so the object stays frozen/hashable.
        if isinstance(self.pages, list):
            object.__setattr__(self, "pages", tuple(self.pages))
        if not isinstance(self.pages, tuple) or any(not isinstance(p, str) for p in self.pages):
            raise LibrarianSchemaError("CanonicalText.pages must be a tuple of strings")
        if len(self.pages) == 0:
            raise LibrarianSchemaError("CanonicalText.pages must contain at least one page")

    @property
    def is_frozen(self) -> bool:
        return self.status == "frozen"

    def require_frozen(self) -> "CanonicalText":
        """Return self iff ``status == "frozen"``; otherwise raise
        ``CanonicalTextNotFrozenError``. This is the structural status gate: the
        pipeline entry point calls it before any real extraction, so a stub can
        never reach a real Librarian run."""
        if not self.is_frozen:
            raise CanonicalTextNotFrozenError(
                f"canonical text for {self.source_pdf!r} has status {self.status!r}, not "
                "'frozen'; a real extraction may only run on a frozen canonical text "
                "(build brief §5.1 status gate)"
            )
        return self

    def locate(self, quote: str, level: str = "L0") -> Locator | None:
        """Locate ``quote`` in ``pages`` at ladder ``level``; return a
        ``Locator(page, char_start, char_end)`` for the match, else ``None``.

        Delegates to ``locate.locate_quote`` (the single matching source of truth,
        shared with the parser bake-off): the same normalisation ladder is applied
        to both the page text and the candidate quote before an exact-substring
        compare. ``level`` defaults to ``"L0"`` (exact byte substring, the identity
        ladder), so an L0 locator's offsets are byte-correct into the raw page. At a
        higher level the offsets index ``normalise(page, level)`` -- callers interpret
        them at the ``normalisation.ladder_level`` the canonical text records.

        A quote that only matches via the adjacent page-pair fallback (straddling a
        page boundary) returns ``None`` here: its offsets index the joined page-pair,
        not a single page, so a faithful single-page ``Locator`` cannot be formed yet
        (deferred to the cross-page-seam matcher). The bake-off calls ``locate_quote``
        directly and still scores such matches.
        """
        if not isinstance(quote, str) or quote == "":
            raise LibrarianSchemaError("locate(quote) requires a non-empty string")
        result = locate_quote(self.pages, quote, level)
        if not result.matched:
            return None
        if result.used_cross_page:
            # A cross-page match's offsets index the JOINED page-pair string, not a
            # single page, so they cannot form a faithful single-page Locator
            # (char_end could exceed len(pages[page])). Rather than stamp a malformed
            # locator into the STATED audit trail, decline -- the quote then routes to
            # review (UNKNOWN) instead. Faithful cross-page locators are deferred to the
            # cross-page-seam matcher work (see docs/librarian/validation/parser_bakeoff_report.md). The
            # bake-off uses locate_quote directly, so its cross-page scoring is unaffected.
            return None
        return Locator(
            page=result.page,
            char_start=result.char_start,
            char_end=result.char_end,
        )

    def to_dict(self) -> dict:
        return {
            "source_pdf": self.source_pdf,
            "source_sha256": self.source_sha256,
            "parser": dict(self.parser),
            "normalisation": dict(self.normalisation),
            "pages": list(self.pages),
            "status": self.status,
        }


def load_canonical_text(path: str | Path) -> CanonicalText:
    """Load a canonical-text file (parser-brief §5 shape) into a ``CanonicalText``.

    Loads *any* status (``stub`` or ``frozen``) -- the status gate lives in
    ``require_frozen``, not here, so scaffolding/tests can freely load a stub. A
    missing ``status`` key is a build error (the gate must never be bypassable by
    omission)."""
    p = Path(path)
    if not p.exists():
        raise LibrarianSchemaError(f"canonical text not found at {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise LibrarianSchemaError("canonical text file must be a mapping at top level")

    missing = [
        k
        for k in ("source_pdf", "source_sha256", "parser", "normalisation", "pages", "status")
        if k not in raw
    ]
    if missing:
        raise LibrarianSchemaError(f"canonical text file missing required keys: {missing}")

    return CanonicalText(
        source_pdf=str(raw["source_pdf"]),
        source_sha256=str(raw["source_sha256"]),
        parser=raw["parser"],
        normalisation=raw["normalisation"],
        pages=tuple(str(pg) for pg in raw["pages"]),
        status=str(raw["status"]),
    )
