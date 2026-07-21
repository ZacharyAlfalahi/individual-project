"""Inc2-E — economic significance and deflated Sharpe (§9)."""

from __future__ import annotations

import pytest

from agents.auditor.checks.economic import (
    EconomicBands,
    classify_effect,
    deflated_sharpe_ratio,
    economic_deltas,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    run_economic,
)
from agents.auditor.schemas.lattice_types import MetricSet

BANDS = EconomicBands(small=0.01, moderate=0.05, large=0.10)


def _ms(**over):
    base = dict(
        n_months=120, months_per_year=12, nw_lags_used=4, average=0.01,
        annualised_average=0.12, bumpiness=0.04, sharpe=0.86, t_stat=2.1,
        first_date=None, last_date=None,
    )
    base.update(over)
    return MetricSet.from_summary(base)


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (0.005, "Negligible"), (0.02, "Small"), (0.07, "Moderate"), (0.5, "Large"),
    (-0.07, "Moderate"),  # classification is on |value|
])
def test_classify_effect_bands(value, expected):
    assert classify_effect(value, BANDS) == expected


def test_bands_must_be_ordered():
    with pytest.raises(ValueError):
        EconomicBands(small=0.05, moderate=0.01, large=0.1)


# --------------------------------------------------------------------------
# Deflated Sharpe
# --------------------------------------------------------------------------

def test_psr_increases_with_sharpe():
    lo = probabilistic_sharpe_ratio(0.1, 120)
    hi = probabilistic_sharpe_ratio(0.3, 120)
    assert 0.0 <= lo < hi <= 1.0


def test_expected_max_sharpe_grows_with_trials():
    s1 = expected_max_sharpe(10, sr_std=0.1)
    s2 = expected_max_sharpe(1000, sr_std=0.1)
    assert s2 > s1 > 0


def test_deflation_lowers_the_probability():
    # More trials => higher deflation benchmark => lower DSR for the same Sharpe.
    undeflated = deflated_sharpe_ratio(0.3, 120, 1, sr_std=0.1)
    deflated = deflated_sharpe_ratio(0.3, 120, 500, sr_std=0.1)
    assert deflated < undeflated


def test_expected_max_sharpe_requires_two_trials():
    with pytest.raises(ValueError):
        expected_max_sharpe(1, sr_std=0.1)


# --------------------------------------------------------------------------
# Deltas + assembled result
# --------------------------------------------------------------------------

def test_economic_deltas_signs():
    unc = _ms(average=0.02, annualised_average=0.24, sharpe=1.0, bumpiness=0.04)
    cor = _ms(average=0.01, annualised_average=0.12, sharpe=0.5, bumpiness=0.05)
    d = economic_deltas(unc, cor)
    assert d["delta_annual_return"] == pytest.approx(-0.12)
    assert d["delta_sharpe"] == pytest.approx(-0.5)
    assert d["delta_volatility"] == pytest.approx(0.01)


def test_run_economic_classifies_gap_and_reports_dsr():
    unc = _ms(annualised_average=0.24)
    cor = _ms(annualised_average=0.12, average=0.01, bumpiness=0.04, n_months=120)
    res = run_economic(unc, cor, gap_bands=BANDS, n_trials=50, sr_std=0.1)
    assert res.endpoint_gap == pytest.approx(-0.12)
    assert res.gap_band == "Large"       # |−0.12| >= 0.10
    assert 0.0 <= res.deflated_sharpe_corrected <= 1.0
    assert res.n_trials == 50
    assert "delta_sharpe" in res.to_dict()["deltas"]
