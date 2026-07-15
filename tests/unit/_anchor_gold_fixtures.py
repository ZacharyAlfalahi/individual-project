"""
Anchor gold fixtures -- a thin delegation to the real gold loader (G2 Piece B).

The former hand-transcribed 13-field fixture is superseded: ``anchor_gold_spec``
now returns the FULL ``StrategySpec`` parsed from ``evaluation/gold_specs/gold_*.md``
by ``gold_loader.load_gold_spec`` (every schema field, real D7 locators). The
``test_ledger_check`` no-false-positive gate reads the 13 ledger rows, of which the
full spec is a superset -- verified zero mismatches on all three anchors (str, drf,
mom6). A real ``gold_*.md -> StrategySpec`` loader was always G2's job; this is it.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The loader lives under evaluation/ (repo root); ensure it is importable whether
# or not the caller's path config already includes the root (pytest.ini does).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.gold_specs.gold_loader import load_gold_spec  # noqa: E402


def anchor_gold_spec(anchor_id: str):
    """The full gold ``StrategySpec`` for an anchor (``str`` | ``drf`` | ``mom6``)."""
    return load_gold_spec(anchor_id)
