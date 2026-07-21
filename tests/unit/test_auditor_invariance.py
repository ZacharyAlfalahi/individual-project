"""Stage 6 — the no-op invariance gate (§3.5)."""

from __future__ import annotations

import pytest

from agents.auditor.checks.invariance import invariance_test, run_invariance_tests
from agents.auditor.checks.lattice import run_lattice
from agents.auditor.data.synthetic_panel import SyntheticSpec, make_clean_maximal_panel
from agents.auditor.schemas.toggle import TOGGLE_IDS, ToggleFacts

from _auditor_fixtures import make_score_strategy


def _clean_lattice(seed=5):
    panel, signals = make_clean_maximal_panel(
        SyntheticSpec(n_bonds=40, n_months=48, seed=seed)
    )
    return run_lattice(make_score_strategy(), TOGGLE_IDS, {}, panel, signals=signals)


# --------------------------------------------------------------------------
# On the clean panel the panel toggles and lab_trim are genuine, verified no-ops.
# lib_gap is a special case (below): it shifts the formation window by signal_lag
# months, so it is NOT bit-exact inert — matching §3.5/§10.3 ("near no-op, never
# bit-exact"; its invariance is UNVERIFIED per O-A2).
# --------------------------------------------------------------------------

BIT_EXACT_NO_OP_TOGGLES = ("meas_err", "stale_price", "survivorship", "lab_trim")


@pytest.mark.parametrize("toggle", BIT_EXACT_NO_OP_TOGGLES)
def test_panel_and_trim_toggles_are_no_ops_on_the_clean_panel(toggle):
    lat = _clean_lattice()
    res = invariance_test(lat, toggle)
    assert res.is_no_op
    # both halves: the intervention WAS applied (configs differ) and was inert
    assert res.config_hashes_differ
    assert res.returns_identical and res.n_bonds_identical and res.metrics_identical
    assert "inert across all parallel edges" in res.note


def test_lib_gap_is_a_near_no_op_not_bit_exact_on_the_clean_panel():
    # signal_lag=1 drops the first formation month for every bond, so the NATIVE
    # return series differ even with a time-constant signal. The instrument
    # reports is_no_op=False honestly — it must never be called zero-by-construction
    # until the invariance test passes (§3.5, O-A2). (Its COMMON-support effect is
    # ~0, verified in test_auditor_support.)
    lat = _clean_lattice()
    res = invariance_test(lat, "lib_gap")
    assert not res.is_no_op
    assert res.config_hashes_differ


# --------------------------------------------------------------------------
# The test can FALSIFY: a real effect must fail the no-op assertion
# --------------------------------------------------------------------------

def test_meas_err_is_not_a_no_op_when_families_differ():
    panel, signals = make_clean_maximal_panel(
        SyntheticSpec(n_bonds=40, n_months=48, seed=6)
    )
    panel = panel.copy()
    panel["ret_corr"] = panel["ret_corr"] + 0.03  # corrected family really differs
    lat = run_lattice(make_score_strategy(), TOGGLE_IDS, {}, panel, signals=signals)

    res = invariance_test(lat, "meas_err")
    assert not res.is_no_op
    assert res.config_hashes_differ            # config still differs (correct)
    assert not res.returns_identical           # but behaviour is not inert
    assert "NOT a no-op" in res.note


def test_lib_gap_is_not_a_no_op_with_time_varying_score():
    panel, signals = make_clean_maximal_panel(
        SyntheticSpec(n_bonds=40, n_months=48, seed=7, time_varying_score=True)
    )
    lat = run_lattice(make_score_strategy(), TOGGLE_IDS, {}, panel, signals=signals)
    res = invariance_test(lat, "lib_gap")
    assert not res.is_no_op


# --------------------------------------------------------------------------
# run_invariance_tests only tests declared expect_no_op toggles
# --------------------------------------------------------------------------

def test_run_invariance_tests_covers_only_expect_no_op():
    lat = _clean_lattice()
    facts = {t: ToggleFacts(t, runnable=True) for t in TOGGLE_IDS}
    facts["stale_price"] = ToggleFacts("stale_price", runnable=True, expect_no_op=True)
    facts["meas_err"] = ToggleFacts("meas_err", runnable=True, expect_no_op=True)
    results = run_invariance_tests(lat, facts)
    tested = {r.toggle_id for r in results}
    assert tested == {"stale_price", "meas_err"}
    assert all(r.is_no_op for r in results)  # clean panel => both inert


def test_invariance_on_non_runnable_toggle_raises():
    # A partial lattice without survivorship: it is held, not varied, so its
    # inertness is not testable via the lattice.
    panel, signals = make_clean_maximal_panel(
        SyntheticSpec(n_bonds=30, n_months=36, seed=8)
    )
    runnable = ("meas_err", "stale_price", "lib_gap", "lab_trim")
    lat = run_lattice(
        make_score_strategy(), runnable, {"survivorship": "OFF"}, panel, signals=signals
    )
    with pytest.raises(ValueError, match="not runnable"):
        invariance_test(lat, "survivorship")
