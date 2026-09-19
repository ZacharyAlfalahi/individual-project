"""
Unit tests for the ledger check table loader: versioned + hash-stamped, and a
malformed row is a build error (never a silent default). Mirrors the
concept_column table's content-hash discipline.
"""

import pytest

from agents.quant.config import (
    LedgerCheckError,
    LedgerCheckRow,
    LedgerCheckTable,
    load_ledger_check_table,
)
from agents.quant.config.ledger_check import _DEFAULT_TABLE_PATH


def _row(**overrides):
    base = dict(
        ledger_item=7,
        block="sort",
        part2_field="bucketing_method",
        engine_fixed="equal_count",
        incompatible_stated=("breakpoint",),
        category="A",
        detail="engine equal-count; paper breakpoint",
    )
    base.update(overrides)
    return LedgerCheckRow(**base)


# --- loads + version + hash -------------------------------------------------

def test_loads_default_table():
    t = load_ledger_check_table()
    assert t.version == "v1"
    assert len(t.rows) == 13  # finalised v1 (2026-07-10): 10-row draft + 5/6/21
    assert {r.block for r in t.rows} == {"common", "sort"}
    assert {r.category for r in t.rows} <= {"A", "C"}
    assert not any(r.borderline for r in t.rows)  # no unresolved rows at freeze


def test_content_hash_deterministic_and_hex():
    a = load_ledger_check_table().content_hash
    b = load_ledger_check_table().content_hash
    assert a == b
    assert len(a) == 64
    int(a, 16)  # parses as hex


def test_content_hash_is_content_not_row_order():
    t = load_ledger_check_table()
    reordered = LedgerCheckTable(version=t.version, rows=tuple(reversed(t.rows)))
    assert reordered.content_hash == t.content_hash


def test_verify_hash():
    t = load_ledger_check_table()
    assert t.verify_hash(t.content_hash)
    assert not t.verify_hash("0" * 64)


# The finalised content_hash of the v1 table. Recorded in the
# YAML header; a silent edit to any row breaks this pin.
_PINNED_SHA256 = "1f0f583db67913115be6f8b1ccb03b7a29af9d2596f88794cd5d421bfb752635"


def test_content_hash_pinned():
    assert load_ledger_check_table().content_hash == _PINNED_SHA256


# --- malformed rows are build errors ----------------------------------------

def test_empty_incompatible_is_build_error():
    # A row with nothing to test is a build error -- structurally bars int-only
    # fields (e.g. realisation_min_survivors) from masquerading as a check.
    with pytest.raises(LedgerCheckError, match="incompatible_stated"):
        _row(incompatible_stated=())


def test_bad_block_is_build_error():
    with pytest.raises(LedgerCheckError, match="block"):
        _row(block="spec")


def test_bad_category_is_build_error():
    with pytest.raises(LedgerCheckError, match="category"):
        _row(category="B")


def test_empty_field_is_build_error():
    with pytest.raises(LedgerCheckError):
        _row(part2_field="")


def test_duplicate_block_field_row_rejected():
    with pytest.raises(LedgerCheckError, match="duplicate"):
        LedgerCheckTable(version="v1", rows=(_row(), _row(ledger_item=99)))


def test_missing_key_in_yaml_is_build_error(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(
        "version: v1\nrows:\n  - {ledger_item: 7, block: sort, part2_field: x}\n"
    )
    with pytest.raises(LedgerCheckError, match="missing required key"):
        load_ledger_check_table(p)


def test_empty_rows_is_build_error(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("version: v1\nrows: []\n")
    with pytest.raises(LedgerCheckError, match="rows"):
        load_ledger_check_table(p)


def test_missing_file_is_build_error(tmp_path):
    with pytest.raises(LedgerCheckError, match="not found"):
        load_ledger_check_table(tmp_path / "nope.yaml")


def test_default_table_path_exists():
    # The committed artifact is where the loader looks by default.
    assert _DEFAULT_TABLE_PATH.exists()
