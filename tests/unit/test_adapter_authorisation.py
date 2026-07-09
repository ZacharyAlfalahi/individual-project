"""
Authorisation records (D27) + the registry handshake: the loader, the two entry
types wired through ``adapt_spec``, the bright-line guard, and the version-drift
refusal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _adapter_fixtures import adapter_spec, grounded_leg  # noqa: E402
from _librarian_fixtures import build_header, unknown  # noqa: E402
from agents.librarian.adapter import adapt_spec  # noqa: E402
from agents.librarian.adapter.authorisation import (  # noqa: E402
    AuthorisationRecords,
    BindingSubstitution,
    FieldOverride,
    load_authorisation_records,
)
from agents.librarian.errors import LibrarianSchemaError  # noqa: E402
from agents.quant.config import QuantConfig, RefusalCode  # noqa: E402

_PAPER = "SYNTH-0001"
_LABEL = "Synthetic Momentum"


# --- loader ----------------------------------------------------------------------

def test_absent_file_loads_no_authorisations(tmp_path):
    recs = load_authorisation_records(tmp_path / "missing.yaml")
    assert recs.is_empty


def test_default_template_is_empty():
    assert load_authorisation_records().is_empty


def test_empty_records_list(tmp_path):
    p = tmp_path / "a.yaml"
    p.write_text("version: v1\nrecords: []\n")
    assert load_authorisation_records(p).is_empty


def test_bad_top_level_is_error(tmp_path):
    p = tmp_path / "a.yaml"
    p.write_text("- not a mapping\n")
    with pytest.raises(LibrarianSchemaError):
        load_authorisation_records(p)


# --- FieldOverride structural guard (the bright line, half 1) ---------------------

def test_field_override_on_engine_field_rejected_at_construction():
    with pytest.raises(LibrarianSchemaError, match="Part 2 field"):
        FieldOverride(_PAPER, _LABEL, "groups", 5, "note")  # 'groups' is engine-side


# --- field_override wired through adapt_spec --------------------------------------

def _auth(*, overrides=(), bindings=()):
    return AuthorisationRecords(version="v1", overrides=tuple(overrides), bindings=tuple(bindings))


def test_field_override_sets_design_and_variant():
    auth = _auth(overrides=(FieldOverride(_PAPER, _LABEL, "benchmark_model", "ff5", "authorised"),))
    r = adapt_spec(adapter_spec(), auth=auth)
    assert r.variant is True
    assert not r.refused  # benchmark_model is check-only; the strategy still runs


def test_field_override_bright_line_rejects_unknown_target():
    # Overriding an UNKNOWN field would patch a suspected extraction error (D27).
    auth = _auth(overrides=(FieldOverride(_PAPER, _LABEL, "benchmark_model", "ff5", "authorised"),))
    spec = adapter_spec(benchmark_model=unknown())
    with pytest.raises(LibrarianSchemaError, match="bright line"):
        adapt_spec(spec, auth=auth)


def test_field_override_on_sort_block_field_unsupported_in_v1():
    auth = _auth(overrides=(FieldOverride(_PAPER, _LABEL, "sort_kind", "independent", "note"),))
    with pytest.raises(LibrarianSchemaError, match="per-leg"):
        adapt_spec(adapter_spec(), auth=auth)


# --- binding_substitution --------------------------------------------------------

def test_binding_substitution_binds_authorised_column_and_variants():
    # 'maturity' has no v1 column; a substitution binds it and flags a variant.
    auth = _auth(bindings=(BindingSubstitution(_PAPER, _LABEL, "maturity", "time_to_maturity", "ok"),))
    r = adapt_spec(adapter_spec(legs=(grounded_leg("maturity"),)), auth=auth)
    assert r.variant is True
    assert not r.refused
    assert isinstance(r.leg_calls[0].result, QuantConfig)
    assert r.leg_calls[0].result.score.value == "time_to_maturity"


# --- registry handshake ----------------------------------------------------------

def test_registry_version_drift_refuses_upfront():
    r = adapt_spec(adapter_spec(header=build_header(registry_version="v2")))
    assert r.refused
    assert r.refusals[0].code is RefusalCode.REVIEW_REQUIRED
    assert r.refusals[0].field == "registry_version"
    assert len(r.leg_calls) == 0  # refused BEFORE translating any leg
