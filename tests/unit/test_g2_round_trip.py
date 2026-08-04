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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from agents.quant.config import run_strategy  # noqa: E402
from evaluation.gold_specs.gold_loader import is_binding  # noqa: E402
from evaluation.harness.canonical_yaml import assert_rulebook_byte_equal  # noqa: E402
from evaluation.harness.round_trip import (  # noqa: E402
    HarnessError,
    adapt_gold,
    assert_composite_rulebook_byte_equal,
    combiner_dict,
    emit_register,
    expected_composite_rulebook,
    expected_rulebook,
    load_verified_standing_subs,
    produced_rulebook,
    produced_rulebook_composite,
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


# --- CRF: the first MULTI-LEG composite G2 (byte-equal per leg AND combiner) --

def test_g2_crf_byte_equal(subs):
    """The headline new G2 capability: the 3-leg CRF composite compiles byte-equal
    to the independently hand-authored golden rulebook -- per leg AND the combiner."""
    r = adapt_gold("crf", subs)
    assert not r.refused          # no silent single-leg fallback (D28)
    assert len(r.leg_calls) == 3
    assert_composite_rulebook_byte_equal(
        produced_rulebook_composite(r), expected_composite_rulebook("crf"))


def test_g2_crf_combiner(subs):
    assert combiner_dict(adapt_gold("crf", subs)) == {
        "kind": "equal_average", "divisor": "available"}


def test_g2_crf_leg_controls(subs):
    """Each leg's distinguishing control column: var_5pct, gamma, xret (the
    prior_1m_excess_return->xret reconciliation). Order-invariant set check."""
    r = adapt_gold("crf", subs)
    controls = {rb["control"] for rb in produced_rulebook_composite(r)["legs"].values()}
    assert controls == {"var_5pct", "gamma", "xret"}


def test_g2_crf_register_one_par_row(subs):
    """CRF is value-weighted (by_size) on every leg, so the par-proxy convention
    yields the single standing row (same shape as drf); no override/variant."""
    r = adapt_gold("crf", subs)
    leg_rb = produced_rulebook_composite(r)["legs"]["var_5pct"]  # any leg: all by_size
    rows = emit_register("crf", r, subs, rulebook=leg_rb)
    assert len(rows) == 1
    row = rows[0]
    assert row.authorisation_id == "par_weighting_v1"
    assert row.variant_effect is False and row.fidelity_aggregate_exclusion is False


def _crf_grid_panel(seed: int = 0) -> pd.DataFrame:
    """5×5×2 grid over 6 months: rating group rg drives the return (worst credit,
    rg=4, earns MORE), the three signal columns (var_5pct/gamma/xret) span quintiles.
    A correct CRF (long worst-credit, short best-credit, averaged over the three
    signal stripes and the three legs) is POSITIVE (~0.02*(4-0) = 0.08)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-31", periods=6, freq="ME")
    rows = []
    for rg in range(5):
        for sg in range(5):
            for k in range(2):
                cusip = f"R{rg}S{sg}K{k}"
                for d in dates:
                    rows.append({"cusip": cusip, "date": d,
                                 "ret": 0.02 * rg + float(rng.normal(0, 0.0005)),
                                 "size": 1.0, "rating": float(rg),
                                 "var_5pct": float(sg), "gamma": float(sg), "xret": float(sg)})
    return pd.DataFrame(rows)


def test_g2_crf_synthetic_sign_through_combiner(subs):
    """Acceptance gate: source-derived fixtures produce the expected ALGEBRAIC sign
    through the adapter->runner->equal_average combiner. Long = worst credit, so
    when worst-credit bonds earn more the CRF composite is positive. (The REALISED
    empirical sign on real data is a diagnostic, never a gate.)"""
    r = adapt_gold("crf", subs)
    res = run_strategy(r, _crf_grid_panel(), safe_rate=None)
    assert res.n_legs == 3 and res.combiner == {"kind": "equal_average", "divisor": "available"}
    mean_crf = res.monthly_returns["strategy_ret"].mean()
    assert mean_crf > 0                       # the algebraic sign gate
    assert mean_crf == pytest.approx(0.08, abs=0.01)   # long worst − short best, averaged


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
    # mom6 is frozen/binding (JNPS freeze landed); a lightweight load+adapt smoke
    # alongside the full test_g2_mom6_byte_equal.
    r = adapt_gold("mom6", subs)
    assert r is not None
    assert is_binding("mom6")  # JNPS canonical text frozen -> binding locators


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
