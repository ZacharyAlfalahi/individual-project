"""
Quote-fixture loading seam (the bake-off evaluation seam, [P1]).

RQ1's per-field extraction-fidelity scoring runs against *quote fixtures*: real
paper-gold forms, every field quote-bearing, authored by a human (D34). Authoring
those fixtures is deliberately out of scope for code -- the brief forbids it, and
D34 makes gold a human-only, single-annotator artifact. So the fixtures do not
exist yet: the bake-off has not run.

This module is the **seam** that lets everything downstream of the fixtures be
green *before* the fixtures exist (P1: blank is the safe state -- an absent
fixture directory is abstention, not an error). ``load_quote_fixtures`` returns
an empty list when the directory is absent (or empty), and loads the fixtures
once they are supplied -- no code change required at the seam when
they arrive.

A quote fixture file is a ``.yaml`` under the fixtures directory. This loader is
deliberately thin: it reads each YAML doc into a ``QuoteFixture`` carrying the
raw mapping plus its source path. The scoring harness that *consumes* fixtures is
a separate artifact; this seam only has to make "no fixtures yet" and "some
fixtures now" both work.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .errors import LibrarianSchemaError

#: The default fixtures directory, relative to the repo root. Absent today (the
#: bake-off has not run); a human lands ``.yaml`` fixtures here (D34).
DEFAULT_FIXTURE_DIR = "evaluation/quote_fixtures"


@dataclass(frozen=True)
class QuoteFixture:
    """One loaded quote fixture: the raw mapping the human authored plus the path
    it came from (audit). Intentionally schema-light -- the scoring harness that
    consumes fixtures owns their field schema; this seam only loads them."""

    source_path: str
    content: dict

    def __post_init__(self) -> None:
        if not isinstance(self.content, dict):
            raise LibrarianSchemaError(
                f"quote fixture {self.source_path!r} must be a mapping at top level; "
                f"got {type(self.content).__name__}"
            )


def load_quote_fixtures(dir: str | Path = DEFAULT_FIXTURE_DIR) -> list[QuoteFixture]:
    """Load every ``.yaml`` quote fixture under ``dir``.

    Returns ``[]`` gracefully when ``dir`` does not exist (the bake-off has not
    run -- P1: an absent fixture directory is the safe blank state, never an
    error), and when it exists but holds no ``.yaml`` files. When the directory
    *is* present with fixtures, each is loaded into a ``QuoteFixture`` in sorted
    path order (deterministic).

    Raises ``LibrarianSchemaError`` only for a malformed fixture that *is* present
    (unparseable YAML, or a top-level non-mapping) -- a real fixture that exists
    but is broken is a build error, distinct from the absent-directory blank
    state."""
    d = Path(dir)
    if not d.exists() or not d.is_dir():
        return []  # P1: absent directory -> blank state, not an error.

    fixtures: list[QuoteFixture] = []
    for path in sorted(d.glob("*.yaml")):
        try:
            with path.open("r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise LibrarianSchemaError(
                f"quote fixture {str(path)!r} is not valid YAML: {exc}"
            ) from exc
        if raw is None:
            continue  # an empty .yaml doc is skipped (not a malformed one).
        fixtures.append(QuoteFixture(source_path=str(path), content=raw))
    return fixtures
