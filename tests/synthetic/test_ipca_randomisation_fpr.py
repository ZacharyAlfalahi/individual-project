"""
The matched-twin randomisation FPR (spec §6.2, D-A51).

Pins: the placebo swap produces genuinely different but exchangeable panels; the permutation
p-value is valid and (aggregated) uniform under exchangeability, so FPR_hat lands inside the binomial
acceptance band; the exact binomial band is correct; and the run is deterministic and supports the
reduced-(Q,R) / rename fallback.
"""

from __future__ import annotations

import warnings

import numpy as np

from agents.auditor.ipca_differential.randomisation_fpr import (
    binomial_acceptance_band,
    make_matched_twin_dataset,
    randomisation_fpr,
    randomisation_pvalue,
    _panels_from_label,
)
from agents.auditor.thresholds import (
    load_ipca_fpr_config,
    load_ipca_lambda,
    load_ipca_projection_gate,
)

LAM = load_ipca_lambda()
GATE = load_ipca_projection_gate()
FCFG = load_ipca_fpr_config()


# ---- the binomial acceptance band (fast, analytic) ------------------------

def test_binomial_band_known_cases():
    assert binomial_acceptance_band(50, 0.05, 0.95) == (0, 6)     # mean 2.5, upper tail cut at 6
    lo, hi = binomial_acceptance_band(20, 0.5, 0.95)
    assert lo <= 10 <= hi and lo >= 5 and hi <= 15                # symmetric around 10


def test_binomial_band_widens_with_coverage():
    lo95, hi95 = binomial_acceptance_band(50, 0.05, 0.95)
    lo99, hi99 = binomial_acceptance_band(50, 0.05, 0.99)
    assert lo99 <= lo95 and hi99 >= hi95                         # higher coverage ⇒ wider/equal band


# ---- the placebo operator (fast) ------------------------------------------

def test_placebo_swap_changes_panels_but_shares_characteristics():
    tw = make_matched_twin_dataset(FCFG.twin_dgp, LAM, seed=0)
    sigma = np.zeros(tw.n_bonds, dtype=int)
    sigma[: tw.n_bonds // 2] = 1                                  # label half the bonds
    f0, f1 = _panels_from_label(tw, sigma)
    # Z (characteristics) is shared by both arms; returns genuinely differ (realised variation ≠ 0).
    assert all(np.array_equal(a, b) for a, b in zip(f0.Z, f1.Z))
    assert any(not np.array_equal(a, b) for a, b in zip(f0.R, f1.R))


# ---- the randomisation test (slower; small Q/R) ---------------------------

def test_single_pvalue_is_valid():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tw = make_matched_twin_dataset(FCFG.twin_dgp, LAM, seed=3)
        p, i_obs, perms = randomisation_pvalue(tw, LAM, GATE, q=19, seed=1)
    assert 1 / 20 <= p <= 1.0
    assert len(perms) == 19 and np.isfinite(i_obs)


def test_fpr_within_acceptance_band_under_true_null():
    """The gate: under exact exchangeability the pipeline preserves p-uniformity, so FPR_hat lands
    inside the binomial acceptance band and the mean p-value is ~uniform."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = randomisation_fpr(FCFG, LAM, GATE, seed=7, r=12, q=15)
    assert res.within_band, f"FPR {res.fpr_hat} (rejects {res.n_reject}) outside band {res.acceptance_band}"
    assert 0.25 <= float(np.mean(res.p_values)) <= 0.75           # p roughly uniform (mean ~0.5)
    assert res.label == "empirical_fpr"


def test_deterministic_under_fixed_seed():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = randomisation_fpr(FCFG, LAM, GATE, seed=5, r=4, q=9)
        b = randomisation_fpr(FCFG, LAM, GATE, seed=5, r=4, q=9)
    assert a.p_values == b.p_values and a.n_reject == b.n_reject


def test_reduced_qr_and_rename_fallback():
    """The (r, q) overrides drive the reduced-compute fallback, and the rename label replaces
    'empirical_fpr' when even reduced compute is infeasible (registered reduced_r/reduced_q live in
    thresholds and are covered by the loader test)."""
    assert (FCFG.reduced_r, FCFG.reduced_q) == (20, 19)            # the registered reduced-(Q,R)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = randomisation_fpr(FCFG, LAM, GATE, seed=1, r=3, q=5, label=FCFG.rename_fallback_label)
    assert res.r_datasets == 3 and res.q_permutations == 5        # overrides respected
    assert res.label == "stochastic null-invariance calibration"   # NEVER called an empirical FPR
    assert res.to_dict()["randomisation_fpr"]["label"] == res.label
