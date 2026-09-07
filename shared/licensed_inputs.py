"""Fail-loud access to the licensed inputs that are not shipped with the repo.

``data/`` (TRACE/FISD-derived panels and reference tables), ``papers/``, and
``evaluation/canonical_texts/`` hold licensed or copyrighted data that is
intentionally excluded from the repository (see README, "Input data"). The
build/analysis pipeline reads them; the test suite does not. When one is absent
— e.g. on a fresh clone — surface a clear, actionable message at the point of
first use instead of a bare pandas/OS traceback deep in a library.
"""
from __future__ import annotations

from pathlib import Path


class LicensedInputMissing(FileNotFoundError):
    """A licensed, not-shipped input (a dev panel, FISD table, canonical text, …) is absent."""


def require_licensed_input(path: str | Path, what: str = "licensed input") -> Path:
    """Return ``path`` if it exists; else raise :class:`LicensedInputMissing`.

    Call this immediately before a pipeline step first reads a not-shipped input,
    so a fresh clone gets an actionable error rather than a bare ``FileNotFoundError``::

        panel = pd.read_parquet(require_licensed_input(BASE_PANEL, "development panel"))
    """
    p = Path(path)
    if not p.exists():
        raise LicensedInputMissing(
            f"{what} not found at {p}.\n"
            "This is a licensed input that is not shipped with the repository "
            "(see the README section 'Input data (licensed — supply your own)'). "
            "Supply your own copy under data/ to run the pipeline; the test suite "
            "needs none of it and runs on synthetic fixtures."
        )
    return p
