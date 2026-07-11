"""
Librarian config -- the pipeline's read-side configuration objects.

  * ``canonical_text`` -- the ``CanonicalText`` object (parser-brief §5 shape) +
                          ``load_canonical_text`` (loads any status) + the
                          ``require_frozen`` status gate (build brief §5.1): a
                          stub can never reach a real extraction. ``locate``
                          does cross-page-aware quote location into ``pages``.
  * ``normalise``      -- the L0/L1/L2 canonical-text normalisation ladder (v2).
  * ``locate``         -- the quote matcher (``locate_quote`` + ``MatchResult``)
                          shared by the runtime locator and the parser bake-off.

The PDF parser that produces ``pages`` (and the frozen recipe in
``config/canonical_text.yaml``) is chosen by the parser bake-off; this package is
parser-agnostic.
"""

from __future__ import annotations

from .canonical_text import (
    STATUSES,
    CanonicalText,
    load_canonical_text,
)
from .locate import MatchResult, locate_quote, nearest_window
from .normalise import LEVELS, RULES, normalise

__all__ = [
    "CanonicalText",
    "load_canonical_text",
    "STATUSES",
    "normalise",
    "LEVELS",
    "RULES",
    "locate_quote",
    "MatchResult",
    "nearest_window",
]
