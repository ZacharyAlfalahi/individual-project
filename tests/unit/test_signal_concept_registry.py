"""
Unit tests for the Signal Concept Registry loader (D22).

Covers: the v1 seed loads; ``content_hash`` is deterministic across two loads;
the ``SignalRegistryLike`` protocol methods behave; and -- the load-bearing
claim -- a ``SignalConceptRegistry`` instance IS a valid ``SignalRegistryLike``
and drives ``validate_librarian_spec`` (a known concept passes; an unknown
concept raises the D22 error; the ``unrecognised`` escape is accepted).
"""

import pytest

from agents.librarian.errors import LibrarianSchemaError
from agents.librarian.registries import (
    SignalConcept,
    SignalConceptRegistry,
    load_signal_concept_registry,
)
from agents.librarian.registries.signal_concept_registry import _DEFAULT_REGISTRY_PATH
from agents.librarian.schema import UNRECOGNISED, DescribedSignal, SignalRef
from agents.librarian.validators import SignalRegistryLike, validate_librarian_spec

from _librarian_fixtures import (
    build_leg,
    build_part2,
    build_spec,
    located_quote,
    signal_ref,
    stated,
)


@pytest.fixture(scope="module")
def registry():
    return load_signal_concept_registry()


# --- v1 seed loads ----------------------------------------------------------

def test_registry_loads_v1_seed(registry):
    assert registry.version == "v1"
    # D22: v1 seeds the 7 audited concepts.
    assert len(registry.concepts) == 7
    assert all(isinstance(c, SignalConcept) for c in registry.concepts)


def test_seed_contains_the_audited_concepts(registry):
    ids = set(registry.ids())
    expected = {
        "prior_1m_excess_return",
        "past_6m_cumulative_return",
        "var_5pct",
        "credit_rating",
        "bpw_gamma",
        "maturity",
        "size",
    }
    assert ids == expected


def test_no_column_names_leak_registry_side(registry):
    # Wall split (D22): a concept carries id/definition/aliases/params only.
    c = registry.get("size")
    assert c is not None
    for banned in ("column", "columns", "column_name", "panel_column"):
        assert not hasattr(c, banned)


# --- content_hash is deterministic -----------------------------------------

def test_content_hash_deterministic_across_loads():
    a = load_signal_concept_registry()
    b = load_signal_concept_registry()
    assert a.content_hash == b.content_hash
    # sha256 hex digest
    assert len(a.content_hash) == 64
    int(a.content_hash, 16)  # parses as hex


def test_content_hash_is_content_not_formatting(registry):
    # The hash is over canonical concept contents, not the id list order.
    reordered = SignalConceptRegistry(
        version=registry.version,
        concepts=tuple(reversed(registry.concepts)),
    )
    assert reordered.content_hash == registry.content_hash


# --- has_concept / parameter_schema behave ---------------------------------

def test_has_concept(registry):
    assert registry.has_concept("size")
    assert registry.has_concept("bpw_gamma")
    assert not registry.has_concept("not_a_concept")
    # exact + binary (D22): an alias is not a key.
    assert not registry.has_concept("momentum")


def test_parameter_schema_returns_mapping(registry):
    # v1 schemas are intentionally empty (registry params are boundary_flags).
    schema = registry.parameter_schema("past_6m_cumulative_return")
    assert dict(schema) == {}


def test_parameter_schema_unknown_concept_raises(registry):
    with pytest.raises(LibrarianSchemaError):
        registry.parameter_schema("not_a_concept")


# --- IS-A SignalRegistryLike + drives validate_librarian_spec (D22) ---------

def test_registry_is_a_signal_registry_like(registry):
    assert isinstance(registry, SignalRegistryLike)


def _leg_with_concept(concept_id):
    return build_leg(sort_signal=signal_ref(concept_id))


def test_known_concept_passes_validation(registry):
    # A leg whose sort_signal names a real registry concept -> clean.
    p2 = build_part2(legs=(_leg_with_concept("past_6m_cumulative_return"),))
    errs = validate_librarian_spec(build_spec(part2=p2), registry=registry)
    assert errs == []


def test_unknown_concept_is_rejected_with_d22_error(registry):
    # A concept id absent from the registry -> the D22 out-of-registry error.
    p2 = build_part2(legs=(_leg_with_concept("mom6"),))  # not seeded in v1
    errs = validate_librarian_spec(build_spec(part2=p2), registry=registry)
    assert len(errs) >= 1
    assert any("not in the Signal Concept Registry" in e.reason for e in errs)
    # the escape value is named in the error (D22).
    assert any(UNRECOGNISED in e.reason for e in errs)


def test_unrecognised_escape_is_accepted(registry):
    # The first-class "not one of ours" escape rides with a non-empty
    # as_described and is deliberately NOT looked up in the registry.
    escape = SignalRef(
        concept_id=stated(UNRECOGNISED),
        as_described=DescribedSignal(label="an out-of-menu signal", quotes=(located_quote(),)),
    )
    p2 = build_part2(legs=(build_leg(sort_signal=escape),))
    errs = validate_librarian_spec(build_spec(part2=p2), registry=registry)
    assert errs == []


# --- structural guards ------------------------------------------------------

def test_duplicate_concept_id_rejected():
    c = SignalConcept(concept_id="size", definition="d", aliases=(), parameter_schema={})
    with pytest.raises(LibrarianSchemaError):
        SignalConceptRegistry(version="v1", concepts=(c, c))


def test_default_registry_path_points_at_seed():
    assert _DEFAULT_REGISTRY_PATH.name == "signal_concept_registry.yaml"
    assert _DEFAULT_REGISTRY_PATH.exists()


def test_missing_registry_file_raises(tmp_path):
    with pytest.raises(LibrarianSchemaError):
        load_signal_concept_registry(tmp_path / "does_not_exist.yaml")


def test_column_key_in_registry_is_build_error(tmp_path):
    bad = tmp_path / "reg.yaml"
    bad.write_text(
        "version: v1\n"
        "concepts:\n"
        "  - id: size\n"
        "    definition: d\n"
        "    column: amt_outstanding\n",  # wall-split violation
        encoding="utf-8",
    )
    with pytest.raises(LibrarianSchemaError):
        load_signal_concept_registry(bad)


def test_unknown_type_token_is_build_error(tmp_path):
    bad = tmp_path / "reg.yaml"
    bad.write_text(
        "version: v1\n"
        "concepts:\n"
        "  - id: size\n"
        "    definition: d\n"
        "    parameter_schema: {window: complex}\n",  # unknown token
        encoding="utf-8",
    )
    with pytest.raises(LibrarianSchemaError):
        load_signal_concept_registry(bad)
