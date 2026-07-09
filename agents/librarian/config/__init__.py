"""
Librarian config -- the pipeline's read-side configuration objects.

  * ``canonical_text`` -- the ``CanonicalText`` object (parser-brief §5 shape) +
                          ``load_canonical_text`` (loads any status) + the
                          ``require_frozen`` status gate (build brief §5.1): a
                          stub can never reach a real extraction. ``locate``
                          does L0 exact-substring quote location into ``pages``.

The real parser + L1-L3 normalisation ladder are a [P2] seam (gated on the
parser bake-off); this package builds the interface against a stub config now.
"""

from __future__ import annotations

from .canonical_text import (
    STATUSES,
    CanonicalText,
    load_canonical_text,
)

__all__ = [
    "CanonicalText",
    "load_canonical_text",
    "STATUSES",
]
