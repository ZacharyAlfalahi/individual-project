"""
Signal resolution (D27): SignalRef -> Binding, each miss type, and the
authorisation override path.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _librarian_fixtures import located_quote, signal_ref, stated  # noqa: E402
from agents.librarian.adapter.signal_resolution import resolve_signal  # noqa: E402
from agents.librarian.schema.signal_ref import DescribedSignal, SignalRef  # noqa: E402
from agents.quant.config.concept_column import (  # noqa: E402
    ConceptColumnRow,
    ConceptColumnTable,
    load_concept_column_table,
)

CT = load_concept_column_table()


def test_grounded_hit_binds_bound_with_column():
    b = resolve_signal(signal_ref("var_5pct"), CT)
    assert b.tag == "BOUND"
    assert b.value == "var_5pct"
    assert b.evidence.column == "var_5pct"


def test_unrecognised_escape_is_missing():
    escape = SignalRef(
        concept_id=signal_ref("unrecognised").concept_id,
        as_described=DescribedSignal(label="odd signal", quotes=(located_quote(),)),
    )
    b = resolve_signal(escape, CT)
    assert b.tag == "MISSING" and b.value is None
    assert "unrecognised" in b.evidence.note


def test_unknown_concept_no_row_is_missing():
    b = resolve_signal(signal_ref("maturity"), CT)
    assert b.tag == "MISSING"
    assert "no concept->column row" in b.evidence.note


def test_param_mismatch_is_missing():
    # A table with a param'd row; a lookup with different params misses.
    table = ConceptColumnTable(
        version="t", registry_version="v1",
        rows=(ConceptColumnRow("windowed", (("months", 6),), "ret6"),),
    )
    ref = SignalRef(
        concept_id=stated("windowed"),
        as_described=signal_ref("windowed").as_described,
        parameters={"months": stated(3)},  # 3 != 6
    )
    assert resolve_signal(ref, table).tag == "MISSING"
    ref_match = SignalRef(
        concept_id=stated("windowed"),
        as_described=signal_ref("windowed").as_described,
        parameters={"months": stated(6)},
    )
    assert resolve_signal(ref_match, table).value == "ret6"


def test_override_column_binds_authorised():
    b = resolve_signal(signal_ref("maturity"), CT, override_column="time_to_maturity")
    assert b.tag == "BOUND" and b.value == "time_to_maturity"
    assert "authorised" in b.evidence.note
