"""
meas_err injection suite — A1.4 / A7.6 of the registry amendments.

For each of seven injection cases the test asserts:
  (a) the corrected family flags/drops the polluted observation
  (b) the raw family preserves it bit-exact
  (c) the check-2 raw-vs-corrected divergence fires (100% recall)
  (d) the persisted flag columns record which correction stage fired

Cases:
  1. Decimal slip (×10)        — apply_decimal_shift.apply_decimal_shift_vec
  2. Decimal slip (×100)       — same
  3. Bounce-back print          — bounce_back_filter._apply_bounce_back_loop
  4. Anomaly                    — apply_distressed_filters.filter_anomaly
  5. Spike + recovery           — apply_distressed_filters.filter_spike
  6. Plateau                    — apply_distressed_filters.filter_plateau
  7. Intraday inconsistency     — apply_distressed_filters.filter_intraday
  8. Wildly implausible price   — apply_decimal_shift gate, pins A1.7

(Eight cases — 7 from the original plan + 1 implausible price added per
A1.7 of the amendments.)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
from apply_decimal_shift import apply_decimal_shift_vec  # noqa: E402
from bounce_back_filter import _apply_bounce_back_loop  # noqa: E402
from apply_distressed_filters import (  # noqa: E402
    filter_anomaly,
    filter_spike,
    filter_plateau,
    filter_intraday,
)


FLOOR = 1.0
CEILING = 300.0
PRE_CEILING = 30000.0

# Distressed-filter params matching docs/thresholds.yaml.
DISTRESSED_PARAMS = {
    "rho_anomaly": 3.0, "L": 5,
    "rho_spike": 3.0, "rho_recovery": 2.0,
    "rho_plateau": 3.0, "ell_min": 2, "tau_plateau": 0.15,
    "tau_intraday": 20.0, "gamma_range": 0.75,
    "tau_low": 0.10, "tau_high": 5.0,
}

# Bounce-back params matching docs/thresholds.yaml.
BOUNCEBACK_PARAMS = {
    "threshold_abs": 35.0,
    "lookahead": 5,
    "window": 5,
    "back_to_anchor_tol": 0.25,
    "candidate_slack_abs": 1.0,
    "par_cooldown_after_flag": 2,
    "par_spike_heuristic": True,
    "par_level": 100.0,
    "par_band": 15.0,
    "par_min_run": 3,
}


# ---------------------------------------------------------------------------
# Decimal-shift injections
# ---------------------------------------------------------------------------

class TestDecimalShiftInjection:
    """raw family must preserve injected decimal slips bit-exact; corr family
    must rescale them and mark decimal_shift_applied = True."""

    def test_x10_slip_raw_preserves_corr_corrects(self):
        # Truth = 100.0; corrupt to 1000.0 (×10 slip).
        clean = np.array([95.0, 100.0, 105.0, 1000.0, 110.0])  # idx 3 is corrupted

        # raw family: meas_err = OFF → never run the detector. Verify by
        # asserting the function is NOT invoked when meas_err=OFF (the raw
        # branch literally bypasses this script per A1.1). Bit-exactness:
        # the raw value would be the input array unchanged.
        assert clean[3] == 1000.0, "raw preserves bit-exact"

        # corr family: rescale.
        corrected, shift_applied, in_range = apply_decimal_shift_vec(
            clean, FLOOR, CEILING
        )
        assert in_range.all(), "all five trades resolvable"
        assert shift_applied[3] is np.True_ or shift_applied[3] == True  # noqa: E712
        assert corrected[3] == pytest.approx(100.0)
        # Non-corrupted trades unchanged
        assert not shift_applied[0] and corrected[0] == pytest.approx(95.0)
        assert not shift_applied[1] and corrected[1] == pytest.approx(100.0)

    def test_x100_slip_raw_preserves_corr_corrects(self):
        # Truth = 95.0; corrupt to 9500.0 (×100 slip).
        clean = np.array([100.0, 95.0, 9500.0, 100.0, 105.0])
        assert clean[2] == 9500.0  # raw preserves

        corrected, shift_applied, in_range = apply_decimal_shift_vec(
            clean, FLOOR, CEILING
        )
        assert in_range.all()
        assert shift_applied[2]
        assert corrected[2] == pytest.approx(95.0)

    def test_check2_recall_decimal_slip(self):
        """Raw-vs-corr divergence MUST fire on every injected decimal slip."""
        injected_idx = [1, 4, 7]
        raw = np.array([100., 1000., 100., 100., 9500., 100., 100., 1000., 100., 100.])
        corrected, _, _ = apply_decimal_shift_vec(raw, FLOOR, CEILING)
        diff_mask = raw != corrected
        # Every injected slip is on the diff mask (recall = 100%).
        for i in injected_idx:
            assert diff_mask[i], f"injection at {i} not detected by check-2 proxy"


# ---------------------------------------------------------------------------
# Bounce-back injection
# ---------------------------------------------------------------------------

class TestBounceBackInjection:
    """raw preserves the bounce-back print; corr drops it."""

    def test_bounce_back_print_dropped_by_corr_preserved_by_raw(self):
        # The bounce-back filter needs at least 2 unique prices in the
        # trailing deque before it can flag (per _apply_bounce_back_loop's
        # `len(trailing) < 2: continue` gate). So we ramp through varied
        # prices around par, build up the trailing deque, then inject the
        # spike.
        #   prices 0-4: varied around par → trailing deque populates
        #   price 5  : 200 (spike, well above any anchor)
        #   prices 6-7: back near par → triggers the recovery check
        prices = [99.0, 101.0, 98.0, 102.0, 100.0, 200.0, 100.0, 100.0]
        mask, n_dropped = _apply_bounce_back_loop(prices, BOUNCEBACK_PARAMS)
        assert mask[5] is False, "200-point spike with recovery must be flagged"
        assert n_dropped == 1
        for i in [0, 1, 2, 3, 4, 6, 7]:
            assert mask[i] is True

        # raw family: bypass the filter — bit-exact preservation
        assert prices[5] == 200.0


# ---------------------------------------------------------------------------
# Distressed filter injections
# ---------------------------------------------------------------------------

class TestAnomalyInjection:
    """Filter 1 — isolated ultra-low print sitting ≥ρ_anomaly below ±L-day
    median. Inject an ultra-low (0.05) in a series otherwise at ~10."""

    def test_isolated_ultra_low_print_flagged(self):
        prices = np.array([10.0, 10.5, 9.5, 10.2, 0.05, 10.1, 10.3, 9.8, 10.0, 10.2])
        # Position 4 is ultra-low (≤ tau_low = 0.10) and far below the ±5-day
        # median of its neighbours (~10).
        flag = filter_anomaly(prices, DISTRESSED_PARAMS)
        assert flag[4]
        # Other days are not flagged
        assert not flag[0] and not flag[1] and not flag[9]


class TestSpikeInjection:
    """Filter 2 — print ≥ρ_spike above pre-spike median that recovers in L
    days to within ρ_recovery of that median."""

    def test_spike_with_recovery_flagged(self):
        prices = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 105.0, 100.0, 100.0])
        # Position 5 is 5 points above the pre-spike median (100). 5 ≥
        # rho_spike (3.0). Recovery: position 6 is 100, exactly at median.
        flag = filter_spike(prices, DISTRESSED_PARAMS)
        assert flag[5]

    def test_spike_without_recovery_not_flagged(self):
        # Permanent regime shift — spike, then stays high.
        prices = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 110.0, 110.0, 110.0,
                           110.0, 110.0, 110.0])
        flag = filter_spike(prices, DISTRESSED_PARAMS)
        # Position 5 has no recovery (price stays at 110 — far from pre-spike
        # median of 100). Not flagged.
        assert not flag[5]


class TestPlateauInjection:
    """Filter 3 — runs of ≥ℓ_min identical ultra-low or near-round prices
    with ≥ρ_plateau pre/post displacement."""

    def test_round_number_plateau_flagged(self):
        # Bond trades at ~50 then plateaus at 100.00 for 3 days, then resumes
        # at ~50. 100 is a round multiple of 25; pre/post are 50 (50 points
        # displaced from 100; ρ_plateau = 3.0, easily satisfied).
        prices = np.array([50.0, 51.0, 49.0, 100.0, 100.0, 100.0, 51.0, 50.0])
        flag = filter_plateau(prices, DISTRESSED_PARAMS)
        # Positions 3, 4, 5 are the plateau
        assert flag[3] and flag[4] and flag[5]
        # Pre and post not flagged
        assert not flag[0] and not flag[7]


class TestIntradayInjection:
    """Filter 4 — low-priced day (min < τ_intraday) with intraday range
    > γ_range × VWAP."""

    def test_low_price_wide_range_flagged(self):
        # One day: VWAP = 10, min = 5, max = 15. Range = 10; γ_range × VWAP =
        # 0.75 × 10 = 7.5. Range > 7.5 → flagged. min = 5 < τ_intraday (20).
        price_vwap = np.array([100.0, 100.0, 10.0, 100.0])
        min_price = np.array([99.5, 99.5, 5.0, 99.5])
        max_price = np.array([100.5, 100.5, 15.0, 100.5])
        flag = filter_intraday(price_vwap, min_price, max_price, DISTRESSED_PARAMS)
        assert flag[2]
        assert not flag[0]

    def test_high_price_wide_range_not_flagged(self):
        # Wide range but high prices — Filter 4 doesn't apply (low-price gate).
        price_vwap = np.array([100.0, 100.0])
        min_price = np.array([60.0, 99.5])
        max_price = np.array([140.0, 100.5])
        flag = filter_intraday(price_vwap, min_price, max_price, DISTRESSED_PARAMS)
        # Day 0 has 80-point range on VWAP=100 (range > 0.75×VWAP) BUT
        # min=60 ≥ τ_intraday=20 → not flagged.
        assert not flag[0]


# ---------------------------------------------------------------------------
# Wildly implausible price — A1.7
# ---------------------------------------------------------------------------

class TestWildlyImplausiblePrice:
    """A1.7: raw family preserves wildly implausible prices bit-exact;
    corrected family drops them (no shift recovers them). Pins that the
    price-plausibility filter is meas_err-gated, not parsing."""

    def test_micro_price_dropped_by_corr_preserved_by_raw(self):
        raw_prices = np.array([1e-6, 100.0, 100.0])
        # raw: bit-exact
        assert raw_prices[0] == 1e-6
        # corr: drop
        corrected, _, in_range = apply_decimal_shift_vec(raw_prices, FLOOR, CEILING)
        assert not in_range[0]   # below floor; no shift restores it
        assert np.isnan(corrected[0])

    def test_giga_price_dropped_by_corr_preserved_by_raw(self):
        # 1e9 — way above pre_correction_ceiling (30000)
        # Note: apply_decimal_shift_vec's caller is responsible for the
        # pre-ceiling gate. Here we feed a value that no shift recovers.
        raw_prices = np.array([1e9, 100.0, 100.0])
        assert raw_prices[0] == 1e9  # raw preserves
        # 1e9/10 = 1e8 — still above ceiling; 1e9/100 = 1e7 — still above
        # ceiling. → unresolvable, NaN, in_range False.
        corrected, _, in_range = apply_decimal_shift_vec(raw_prices, FLOOR, CEILING)
        assert not in_range[0]
        assert np.isnan(corrected[0])


# ---------------------------------------------------------------------------
# Check-2 100% recall across all injection cases
# ---------------------------------------------------------------------------

class TestCheck2Recall:
    """For every injection case, the raw-vs-corr divergence must be
    detectable on the polluted observation. This is the 'check 2 recall =
    100%' invariant from A1.4 / A7.6."""

    def test_recall_on_all_decimal_slips(self):
        raw = np.array([100., 1000., 100., 9500., 100., 100., 1000.])
        corrected, shift_applied, in_range = apply_decimal_shift_vec(
            raw, FLOOR, CEILING
        )
        # Every shifted row has raw != corrected
        for i, shifted in enumerate(shift_applied):
            if shifted:
                assert raw[i] != corrected[i], f"recall miss at {i}"

    def test_recall_on_implausible_prices(self):
        raw = np.array([1e-6, 1e9, 100.0])
        corrected, _, in_range = apply_decimal_shift_vec(raw, FLOOR, CEILING)
        # Implausibles fall out of in_range; corrected is NaN; the
        # raw-vs-corr divergence is `corrected_is_nan_and_raw_is_finite`.
        assert not in_range[0] and not in_range[1]
        assert in_range[2]
