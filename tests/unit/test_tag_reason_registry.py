"""Unit tests for the D24 tag-reason registry loader + guard."""

import pytest

from agents.librarian.errors import LibrarianSchemaError
from agents.librarian.validators import (
    TagReasonRegistry,
    TagReasonRow,
    assert_tag_in_registry,
    load_tag_reason_registry,
)
from agents.librarian.validators.tag_reason import SEPARATE_NAMESPACES


@pytest.fixture(scope="module")
def registry():
    return load_tag_reason_registry()


# --- loading ---------------------------------------------------------------

def test_registry_loads_with_version_and_rows(registry):
    assert registry.version == "v1"
    assert len(registry.rows) >= 11
    assert all(isinstance(r, TagReasonRow) for r in registry.rows)


def test_stated_quoted_row_present_with_required_evidence(registry):
    row = registry.get("STATED", "quoted")
    assert row is not None
    assert set(row.required_evidence) == {"quote", "locator"}
    assert row.sole_producer == "librarian"


# --- the INFERRED/default/<field> row (D24 batch E2) -----------------------

def test_inferred_default_row_present(registry):
    row = registry.get("INFERRED", "default/<field>")
    assert row is not None
    assert row.namespace == "default"
    assert row.sole_producer == "factory_apply_defaults"


def test_all_four_unknown_reasons_present(registry):
    for reason in ("not_stated", "disagreement", "quote_match_failure", "input_unknown"):
        assert registry.has("UNKNOWN", reason), reason


def test_all_three_design_reasons_present(registry):
    for reason in ("standing_substitution", "override", "derived_from_design"):
        assert registry.has("DESIGN", reason), reason


# --- namespaces reported separately (D24) ----------------------------------

def test_namespaces_reported_separately(registry):
    ns = registry.namespaces()
    # the three separately-reported namespaces each have their own bucket
    for name in SEPARATE_NAMESPACES:
        assert name in ns
    assert len(ns["adapter"]) == 1
    assert len(ns["inference"]) == 1
    assert len(ns["default"]) == 1
    # the adapter INFERRED row lands in the adapter namespace, not ""
    assert ns["adapter"][0].tag == "INFERRED"


# --- the guard: a tag outside its row is a build error ---------------------

def test_valid_tag_reason_passes(registry):
    row = assert_tag_in_registry("STATED", "quoted", registry=registry)
    assert row.tag == "STATED"


def test_tag_outside_its_row_is_build_error(registry):
    # STATED may only be applied for reason 'quoted' -> any other reason raises.
    with pytest.raises(LibrarianSchemaError):
        assert_tag_in_registry("STATED", "input_unknown", registry=registry)
    # an entirely unregistered (tag, reason) pair raises.
    with pytest.raises(LibrarianSchemaError):
        assert_tag_in_registry("BOGUS", "nope", registry=registry)


def test_namespace_mismatch_raises(registry):
    # adapter INFERRED is in the 'adapter' namespace; claiming 'inference' raises.
    # (An empty namespace means "unspecified" and skips the check by design.)
    with pytest.raises(LibrarianSchemaError):
        assert_tag_in_registry("INFERRED", "adapter", namespace="inference", registry=registry)
    # an empty namespace is treated as "unspecified" -> no assertion, passes.
    assert_tag_in_registry("INFERRED", "adapter", namespace="", registry=registry)


# --- structural guards -----------------------------------------------------

def test_duplicate_rows_rejected():
    row = TagReasonRow("STATED", "quoted", "", ("quote",), "librarian", "x")
    with pytest.raises(LibrarianSchemaError):
        TagReasonRegistry(version="v1", rows=(row, row))


def test_missing_registry_file_raises(tmp_path):
    with pytest.raises(LibrarianSchemaError):
        load_tag_reason_registry(tmp_path / "does_not_exist.yaml")
