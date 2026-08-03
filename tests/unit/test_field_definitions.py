"""
Unit tests for the frozen per-field definitions loader (RQ1 close-out).

Covers: the v1 file loads; ``verify_hash`` against the recorded canonical sha256
passes (and a tampered file is caught); the byte-``content_hash`` is deterministic;
exactly the ~40 routed fields are present (universe_filter EXCLUDED); and a render
smoke -- for a sample field of each of the 4 ``{definition}``-carrying kinds
(enum / int / date / paper_metric) the rendered prompt contains the frozen
definition, NOT the old ``_humanise`` gloss.
"""

from __future__ import annotations

import pytest

from agents.librarian.errors import LibrarianSchemaError
from agents.librarian.pipeline import real_client as rc
from agents.librarian.pipeline.model_client import FieldQuery
from agents.librarian.registries import (
    FieldDefinitions,
    load_field_definitions,
    load_signal_concept_registry,
)
from agents.librarian.schema import fields as F

# The canonical sha256 of agents/librarian/data/prompts/definitions.yaml (version
# v1), recorded in docs/librarian/specs/part2_schema_and_silence_policy_v1_1.md.
_RECORDED_SHA256 = "c7fd5d5f1d03c5a1d48eeba4dd338981929755f00fe0c0071d16108fee46fa30"

# The 40 routed fields: 9 sort-block enum/int fields (sort_signal + control_axis are
# signal_ref, so excluded) + 28 common + 3 paper_facts. universe_filter EXCLUDED.
_SORT_9 = (
    "sort_kind", "bucketing_method", "n_groups", "stripe_aggregation",
    "control_missing_policy", "long_leg", "signal_transform", "control_n_groups",
    "combiner",
)
_PAPER_FACTS_3 = ("sample_start", "sample_end", "claimed_headline_metric")
_EXPECTED_FIELDS = frozenset(_SORT_9) | frozenset(F.COMMON_FIELDS) | frozenset(_PAPER_FACTS_3)


@pytest.fixture(scope="module")
def defs():
    return load_field_definitions()


@pytest.fixture(scope="module")
def builder():
    return rc.PromptBuilder.load(load_signal_concept_registry())


# --- loads + version --------------------------------------------------------

def test_defs_load_with_version(defs):
    assert defs.version == "v1"
    assert isinstance(defs, FieldDefinitions)


# --- the recorded hash ------------------------------------------------------

def test_verify_hash_matches_recorded(defs):
    assert defs.verify_hash(_RECORDED_SHA256)
    assert defs.content_hash == _RECORDED_SHA256


def test_verify_hash_rejects_wrong_hash(defs):
    assert not defs.verify_hash("0" * 64)


def test_content_hash_deterministic():
    a = load_field_definitions()
    b = load_field_definitions()
    assert a.content_hash == b.content_hash


def test_verify_hash_catches_a_tamper(tmp_path):
    # A silent edit to the frozen file must flip content_hash off the recorded value.
    src = load_field_definitions()  # the real, frozen file
    tampered = tmp_path / "definitions.yaml"
    body = "version: v1\ndefinitions:\n"
    for field, gloss in src.definitions.items():
        body += f"  {field}: \"{gloss} EDITED\"\n"
    tampered.write_text(body, encoding="utf-8")
    reloaded = load_field_definitions(tampered)
    assert not reloaded.verify_hash(_RECORDED_SHA256)
    assert reloaded.content_hash != _RECORDED_SHA256


# --- the exact 40-field set (universe_filter EXCLUDED) ----------------------

def test_all_40_fields_present_and_no_extras(defs):
    assert set(defs.definitions) == set(_EXPECTED_FIELDS)
    assert len(defs) == 40


def test_universe_filter_deliberately_absent(defs):
    # universe_filter is unbound in the manifest (free prose, filled NOT-EXTRACTED),
    # so it is never routed to a {definition} template and must not appear here.
    assert "universe_filter" not in defs
    with pytest.raises(LibrarianSchemaError):
        defs.definition_for("universe_filter")


def test_every_definition_is_a_nonempty_gloss_not_a_value(defs):
    for field, gloss in defs.definitions.items():
        assert isinstance(gloss, str) and gloss.strip()
        # A definition is prose, not a bare menu token: sanity that it is a phrase.
        assert len(gloss.split()) >= 4, field


def test_definition_for_unknown_field_raises(defs):
    with pytest.raises(LibrarianSchemaError):
        defs.definition_for("no_such_field")


def test_missing_file_raises(tmp_path):
    with pytest.raises(LibrarianSchemaError):
        load_field_definitions(tmp_path / "does_not_exist.yaml")


# --- render smoke: the frozen definition is used, NOT _humanise -------------

@pytest.mark.parametrize(
    "field,kind",
    [
        ("weighting_scheme", "enum"),        # a common enum field
        ("n_groups", "int"),                 # a sort-block int field
        ("sample_start", "date"),            # a paper_facts date field
        ("claimed_headline_metric", "paper_metric"),  # the paper_facts metric field
    ],
)
def test_render_uses_the_frozen_definition_not_humanise(builder, defs, field, kind):
    out = builder.render(FieldQuery(field, kind), "Momentum (6m)")
    frozen = defs.definition_for(field)
    # the frozen gloss is rendered verbatim into the DEFINITION slot ...
    assert frozen in out
    # ... and the old light gloss (`field.replace("_", " ")`) is NOT what filled it.
    # (the frozen definition is a full sentence, never the bare humanised field name)
    assert frozen != rc._humanise(field)


def test_builder_definitions_match_the_loaded_file(builder, defs):
    # The builder loads the same frozen resource (no drift between the two paths).
    assert builder._definitions == defs.definitions
