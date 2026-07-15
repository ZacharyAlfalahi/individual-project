"""
G2 round-trip gate (evaluation contract §7 gate 2) -- THE decisive G2 test.

For str + drf: the full Librarian->Quant chain (gold_loader -> adapt_spec with the
hash-verified standing register -> to_rulebook) reproduces the independently
hand-authored golden production rulebook BYTE-FOR-BYTE. Plus combiner equality and
the authorised-diff register. mom6's byte-equality is gated on the JNPS freeze.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import pytest  # noqa: E402

from evaluation.gold_specs.gold_loader import is_binding  # noqa: E402
from evaluation.harness.canonical_yaml import assert_rulebook_byte_equal  # noqa: E402
from evaluation.harness.round_trip import (  # noqa: E402
    HarnessError,
    adapt_gold,
    combiner_dict,
    emit_register,
    expected_rulebook,
    load_verified_standing_subs,
    produced_rulebook,
)


@pytest.fixture(scope="module")
def subs():
    return load_verified_standing_subs()


# --- headline: byte-equal rulebooks -----------------------------------------

def test_g2_str_byte_equal(subs):
    r = adapt_gold("str", subs)
    assert not r.refused
    assert_rulebook_byte_equal(produced_rulebook(r), expected_rulebook("str"))


def test_g2_drf_byte_equal(subs):
    r = adapt_gold("drf", subs)
    assert not r.refused
    assert_rulebook_byte_equal(produced_rulebook(r), expected_rulebook("drf"))


def test_g2_combiner_equality(subs):
    for a in ("str", "drf"):
        assert combiner_dict(adapt_gold(a, subs)) == {"kind": "single_leg"}


def test_g2_not_variant(subs):
    for a in ("str", "drf"):
        assert adapt_gold(a, subs).variant is False


# --- the authorised-diff register -------------------------------------------

def test_g2_str_register_one_standing_row(subs):
    r = adapt_gold("str", subs)
    rows = emit_register("str", r, subs, rulebook=produced_rulebook(r))
    assert len(rows) == 1
    row = rows[0]
    assert row.authorisation_class == "standing"
    assert row.authorisation_id == "par_weighting_v1"
    assert row.field == "weighting_base"
    assert (row.paper_value, row.engine_value) == ("market_value", "par")
    assert row.provenance == "DESIGN"
    assert row.variant_effect is False
    assert row.fidelity_aggregate_exclusion is False


def test_g2_drf_register_one_standing_row(subs):
    # drf reproduces byte-equal WITHOUT any adapter intervention (gold states par),
    # but the par-proxy convention divergence is still documented as a standing row.
    r = adapt_gold("drf", subs)
    assert r.standing_subs_applied == ()  # no substitution fired
    rows = emit_register("drf", r, subs, rulebook=produced_rulebook(r))
    assert len(rows) == 1
    row = rows[0]
    assert row.authorisation_class == "standing"
    assert row.authorisation_id == "par_weighting_v1"
    assert row.variant_effect is False
    assert row.fidelity_aggregate_exclusion is False


# --- the harness requires an explicit, hash-verified standing file ----------

def test_harness_requires_verified_standing_file(tmp_path):
    with pytest.raises(HarnessError):
        load_verified_standing_subs(tmp_path / "absent.yaml")


# --- mom6 pending the JNPS canonical-text freeze ----------------------------

@pytest.mark.skipif(
    not is_binding("mom6"),
    reason="mom6 byte-equality gated on the JNPS canonical-text freeze (provisional "
    "locators) + the expost_trim v1-schema seam; activates on freeze with no code change",
)
def test_g2_mom6_byte_equal(subs):  # pragma: no cover -- skipped until JNPS freeze
    r = adapt_gold("mom6", subs)
    assert_rulebook_byte_equal(produced_rulebook(r), expected_rulebook("mom6"))


def test_mom6_loads_and_adapts_without_error(subs):
    # Even while byte-equality is gated, mom6 must load + adapt without raising.
    r = adapt_gold("mom6", subs)
    assert r is not None
    assert not is_binding("mom6")  # documents the pending dependency


def test_g2_mom6_register_lab_trim_delegation(subs):
    # mom6's STATED expost_trim=truncate is delegated to the lab_trim toggle: byte-equal
    # modulo the authorised register, ONE standing row (variant_effect=false). This runs
    # now (it is about the adapter/register, independent of the gated locators). mom6 is
    # equal-weighted, so no par-proxy row.
    r = adapt_gold("mom6", subs)
    assert not r.refused
    rows = emit_register("mom6", r, subs, rulebook=produced_rulebook(r))
    assert len(rows) == 1
    row = rows[0]
    assert row.authorisation_id == "lab_trim_delegation_v1"
    assert row.field == "expost_trim"
    assert (row.paper_value, row.engine_value) == ("truncate", "none")
    assert row.authorisation_class == "standing"
    assert row.variant_effect is False
    assert row.fidelity_aggregate_exclusion is False
