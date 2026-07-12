"""
Concept->column table (D27) loader tests: grounded rows, the registry handshake
attribute, a deterministic content hash, the duplicate-key (mirror-rule) guard, and
the reserved-column guard.
"""

from __future__ import annotations

import pytest

from agents.quant.config.concept_column import (
    ConceptColumnError,
    ConceptColumnRow,
    ConceptColumnTable,
    load_concept_column_table,
)
from agents.quant.library.characteristic_sort import _RESERVED_COLUMNS

# The five v2 grounded rows (from the anchor factor panels). prior_1m_excess_return
# grounded to xret in v2 (resolved 2026-07-12; the engine-contract excess return).
_GROUNDED = {
    "var_5pct": "var_5pct",
    "credit_rating": "rating",
    "past_6m_cumulative_return": "mom6",
    "bpw_gamma": "gamma",
    "prior_1m_excess_return": "xret",
}
# The two concepts deliberately absent from v2 (resolve MISSING).
_DEFERRED = ("maturity", "size")


def test_loads_default_table():
    t = load_concept_column_table()
    assert t.version == "v2"
    assert t.registry_version == "v1"  # the handshake attribute (D27(1))
    assert set(t.concept_ids()) == set(_GROUNDED)


def test_grounded_rows_resolve():
    t = load_concept_column_table()
    for concept, column in _GROUNDED.items():
        assert t.lookup(concept) == column
        assert t.has_row(concept)


def test_deferred_concepts_miss():
    t = load_concept_column_table()
    for concept in _DEFERRED:
        assert t.lookup(concept) is None
        assert not t.has_row(concept)


def test_content_hash_is_deterministic():
    assert load_concept_column_table().content_hash == load_concept_column_table().content_hash


def test_params_are_order_invariant():
    # Two rows differing only in param order key identically (canonical tuple).
    t = ConceptColumnTable(
        version="t", registry_version="v1",
        rows=(ConceptColumnRow("c", (("a", 1), ("b", 2)), "col"),),
    )
    assert t.lookup("c", {"b": 2, "a": 1}) == "col"


def test_duplicate_key_rejected():
    # The mirror rule (D27): one column per (concept, params) -- a duplicate would
    # make a bind AMBIGUOUS, which the adapter forbids.
    with pytest.raises(ConceptColumnError, match="duplicate"):
        ConceptColumnTable(
            version="t", registry_version="v1",
            rows=(
                ConceptColumnRow("c", (), "col_a"),
                ConceptColumnRow("c", (), "col_b"),
            ),
        )


def test_reserved_column_rejected():
    reserved = next(iter(_RESERVED_COLUMNS))
    with pytest.raises(ConceptColumnError, match="reserved"):
        ConceptColumnRow("c", (), reserved)


def test_empty_column_rejected():
    with pytest.raises(ConceptColumnError):
        ConceptColumnRow("c", (), "  ")


def test_missing_registry_version_is_build_error(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("version: v1\nrows:\n  - {concept_id: c, column: col}\n")
    with pytest.raises(ConceptColumnError, match="registry_version"):
        load_concept_column_table(p)
