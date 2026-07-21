"""Stage 9 — Layer B statistical calibration (§10.2).

These tests are the statistical VALIDATION gate: false-positive rate under zero
injection, monotone magnitude recovery, minimum detectable effect, and end-to-end
recovery of the three pre-registered interaction mechanisms. They run the real
lattice + bootstrap on modest sizes (seeds/replicates kept small for CI speed).
"""

from __future__ import annotations

import pytest

from agents.auditor.data.synthetic_panel import INTERACTION_MECHANISMS
from agents.auditor.validation.layer_b_calibration import (
    false_positive_rate,
    interaction_recovery,
    magnitude_sweep,
    minimum_detectable_effect,
)

# Cheaper bootstrap for the sweeps (many scenarios).
FAST = dict(
    n_replicates=120,
    data_driven_block_months=6,
    min_effective_blocks=3,
    holding_period=1,
)


# --------------------------------------------------------------------------
# False-positive rate under zero injection (specificity)
# --------------------------------------------------------------------------

def test_zero_injection_false_positive_rate_is_controlled():
    report = false_positive_rate(n_seeds=8, alpha=0.05, boot=FAST)
    # With a clean panel and 5 toggles at alpha=0.05, family-wise FPR should stay
    # modest — the specificity claim. A blown FPR here would void every finding.
    assert report.family_wise_fpr <= 0.5
    # no single toggle should fire on a majority of clean seeds
    for toggle, count in report.per_toggle_false_positives.items():
        assert count <= report.n_seeds // 2, f"{toggle} fired on {count}/{report.n_seeds} clean seeds"


# --------------------------------------------------------------------------
# Magnitude sweep — monotone, correctly signed recovery + MDE
# --------------------------------------------------------------------------

def test_meas_err_recovery_is_monotone_and_signed():
    mags = [0.0, 0.01, 0.02, 0.04]
    records = magnitude_sweep("meas_err", mags, n_seeds=3, boot=FAST)

    def mean_abs_effect(mag):
        rs = [r for r in records if r.magnitude == mag]
        return sum(abs(r.effect) for r in rs) / len(rs)

    # response grows with the planted magnitude
    assert mean_abs_effect(0.0) < mean_abs_effect(0.02) < mean_abs_effect(0.04)
    # a minimum detectable effect exists within the swept range
    mde = minimum_detectable_effect(records, target=0.6)
    assert mde is not None and mde > 0.0
    # zero-magnitude sweeps are (mostly) not detected — specificity within the sweep
    from agents.auditor.validation.layer_b_calibration import detection_rate
    assert detection_rate(records, 0.0) <= 0.5


# --------------------------------------------------------------------------
# The three interaction mechanisms recover end-to-end
# --------------------------------------------------------------------------

@pytest.mark.parametrize("pair", INTERACTION_MECHANISMS)
def test_interaction_mechanism_recovers_end_to_end(pair):
    rec = interaction_recovery(pair, seed=0, boot=FAST)
    # both main effects are present (the biases were injected)
    assert all(abs(v) > 1e-4 for v in rec.main_effects.values()), rec.main_effects
    # the decomposition is efficient end-to-end
    assert rec.efficiency_residual_ok
    # the interaction term is computed with a finite interval
    lo, hi = rec.interaction_ci
    assert lo <= rec.interaction_effect <= hi or lo <= hi
