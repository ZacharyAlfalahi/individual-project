"""
Unit tests for the Instrument Concept Registry (schema v1.2) -- the fitted-model
analogue of the sort Signal Concept Registry, loaded by the same generic loader
via a separate file.

Headline properties:
  * the 29 Table A.I instruments load with unique ids + a deterministic hash;
  * the SHARED ids (reused verbatim from the sort registry) have byte-identical
    definitions -- a drift guard so the two files can never diverge on a reused
    identity;
  * ISOLATION -- the sort signal registry's file + content_hash are untouched
    (asserted against a pinned constant), so a defect here cannot perturb the
    hash that flows into every sort spec header;
  * an instance drives ``validate_estimation_block`` (a known instrument passes,
    an unknown one is rejected).
"""

import pytest

from agents.librarian.errors import LibrarianSchemaError
from agents.librarian.registries import (
    SignalConcept,
    load_instrument_concept_registry,
    load_signal_concept_registry,
)
from agents.librarian.validators import validate_estimation_block

from _librarian_fixtures import build_instrument_set, build_kpp_spec, instrument_ref

# The sort signal registry's content_hash at v1 -- pinned so a change to
# signal_concept_registry.yaml (which would perturb every sort spec header) is
# caught here as an isolation breach.
_SIGNAL_REGISTRY_HASH_V1 = (
    "d93fbd5cd1d3e7a384d0bc9cf4a256dbe259f13104fa25c8f802deb7447ad57e"
)

# The two ids reused verbatim from the sort registry.
_SHARED_IDS = ("past_6m_cumulative_return", "credit_rating")


@pytest.fixture(scope="module")
def registry():
    return load_instrument_concept_registry()


# --- the 29 instruments load ------------------------------------------------

def test_registry_loads_29_instruments(registry):
    assert registry.version == "v1"
    assert len(registry.concepts) == 29
    assert all(isinstance(c, SignalConcept) for c in registry.concepts)


def test_ids_are_unique(registry):
    ids = registry.ids()
    assert len(set(ids)) == len(ids)


def test_var_is_a_distinct_new_identity(registry):
    # KPP's Value-at-risk is a new id; the sort registry's generic var_5pct is NOT
    # reused (different definition -- ruling 2026-08-03).
    assert registry.get("bond_var_36m") is not None
    assert registry.get("var_5pct") is None


def test_reused_ids_present(registry):
    for cid in _SHARED_IDS:
        assert registry.get(cid) is not None


# --- content_hash is deterministic -----------------------------------------

def test_content_hash_deterministic_across_loads():
    a = load_instrument_concept_registry()
    b = load_instrument_concept_registry()
    assert a.content_hash == b.content_hash
    assert len(a.content_hash) == 64
    int(a.content_hash, 16)


# --- drift guard: shared ids match the sort registry byte-for-byte ----------

def test_shared_ids_definitions_match_sort_registry(registry):
    sig = load_signal_concept_registry()
    for cid in _SHARED_IDS:
        ins_def = registry.get(cid).definition
        sig_def = sig.get(cid).definition
        assert ins_def == sig_def, f"{cid} definition drifted between the two registries"
        # aliases too -- the reused identity must be identical.
        assert set(registry.get(cid).aliases) == set(sig.get(cid).aliases)


# --- isolation: the sort registry file/hash is untouched --------------------

def test_sort_signal_registry_hash_unchanged():
    # Loading + adding the instrument registry must not perturb the sort registry.
    assert load_signal_concept_registry().content_hash == _SIGNAL_REGISTRY_HASH_V1


# --- wall-split guard still fires on a column key ---------------------------

def test_column_key_is_rejected(tmp_path):
    bad = tmp_path / "bad_registry.yaml"
    bad.write_text(
        "version: v1\n"
        "concepts:\n"
        "  - id: x\n"
        "    definition: a column leaked in\n"
        "    aliases: []\n"
        "    parameter_schema: {}\n"
        "    column: some_panel_col\n",
        encoding="utf-8",
    )
    with pytest.raises(LibrarianSchemaError):
        load_instrument_concept_registry(bad)


# --- drives validate_estimation_block ---------------------------------------

def test_registry_validates_a_known_instrument(registry):
    spec = build_kpp_spec()  # instruments: past_6m_cumulative_return, credit_rating
    assert validate_estimation_block(spec, registry) == []


def test_registry_rejects_an_unknown_instrument(registry):
    spec = build_kpp_spec(
        instruments=build_instrument_set(instruments=(instrument_ref("not_real"),))
    )
    errors = validate_estimation_block(spec, registry)
    assert any("not in the Instrument Concept Registry" in e.reason for e in errors)
