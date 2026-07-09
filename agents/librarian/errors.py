"""
Librarian error taxonomy.

Two branches, deliberately separated (mirrors the Quant side's
``QuantConfigError`` vs ``ConfigRefusal`` split -- see
``agents/quant/config/quant_config.py``):

  * ``LibrarianSchemaError`` -- a *construction-time* / build error: a spec
    object (or one of its parts) is malformed, or a tag was applied outside its
    registered tag-reason row (D24). Raised. Analogous to ``QuantConfigError``:
    it brands off-taxonomy input as a schema bug rather than letting a bare
    ``AttributeError`` / ``TypeError`` escape.

  * ``LibrarianValidationError`` -- a *policy* violation found by
    ``validate_librarian_spec`` on an already-constructed spec (e.g. a DESIGN
    tag survived into Librarian output, D8). These are *collected and returned*
    as a list so a caller can report every problem in one pass -- they are not
    raised by the validator itself.

Both subclass ``ValueError`` so existing ``pytest.raises(ValueError)`` idioms
and broad exception handlers keep working.
"""

from __future__ import annotations


class LibrarianSchemaError(ValueError):
    """Malformed Librarian schema input -- a construction/build error.

    Raised by a dataclass ``__post_init__`` (a fact-bearing field that is not
    provenance-wrapped, a SignalRef with a bad shape) or by
    ``assert_tag_in_registry`` when a (tag, reason) pair is not a registered
    row (D24: a tag applied outside its row is a build error)."""


class LibrarianValidationError(ValueError):
    """A Librarian-output policy violation found by ``validate_librarian_spec``.

    Carries the offending field path + reason. Instances are collected into a
    list and returned (not raised) so all violations in a spec surface at once
    (the four mandatory D8 negatives, etc.)."""

    def __init__(self, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"{field}: {reason}")

    def to_dict(self) -> dict:
        return {"field": self.field, "reason": self.reason}


class CanonicalTextNotFrozenError(LibrarianSchemaError):
    """The pipeline tried to run a real extraction on a canonical text whose
    ``status`` is not ``"frozen"`` (e.g. a ``stub`` fixture).

    This is the structural status gate (§5.1): ``load_canonical_text`` will load
    any status, but ``require_frozen`` refuses to hand a non-frozen text to
    extraction -- a stub can never reach a real Librarian run. Subclasses
    ``LibrarianSchemaError`` (a build/contract error), so ``pytest.raises``
    idioms and broad ``ValueError`` handlers keep working."""
