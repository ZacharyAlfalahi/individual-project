"""The deflated-Sharpe deflation benchmark must use the REGISTERED per-period (monthly)
cross-trial Sharpe SD.

An annual-scale dispersion (e.g. 0.5) against a MONTHLY Sharpe inflates the benchmark ~6x and
floors every candidate's deflated Sharpe at 0.0 — a degenerate statistic.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.auditor.checks.economic import deflated_sharpe_ratio, expected_max_sharpe  # noqa: E402
from agents.scientist.experimentalist.robustness import load_registered_sr_std  # noqa: E402


def test_the_registered_value_is_the_monthly_calibration():
    assert load_registered_sr_std("str") == 0.08
    assert load_registered_sr_std() == 0.08          # every registered strategy agrees


def test_an_unknown_strategy_is_refused():
    with pytest.raises(KeyError):
        load_registered_sr_std("not_a_strategy")


def test_a_thresholds_file_without_the_block_is_refused(tmp_path):
    bad = tmp_path / "t.yaml"
    bad.write_text("auditor: {}\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_registered_sr_std("str", bad)


def test_the_registered_benchmark_is_in_monthly_units():
    """An annual-scale dispersion would put the benchmark above any plausible monthly Sharpe."""
    registered = expected_max_sharpe(6, load_registered_sr_std("str"))
    annual_scale = expected_max_sharpe(6, 0.5)
    assert registered < 0.15                      # a monthly Sharpe a real candidate can exceed
    assert annual_scale > 0.6                     # an annual-scale dispersion: ~2.25 annualised
    assert annual_scale > 6 * registered


@pytest.mark.parametrize("t_stat,n_months", [(2.713, 70), (2.165, 231)])
def test_an_annual_scale_dispersion_floors_strong_candidates(t_stat, n_months):
    """Strong candidates (t > 2) scored against an annual-scale dispersion land at or next to
    zero — indistinguishable from worthless."""
    sr_monthly = t_stat / math.sqrt(n_months)
    degenerate = deflated_sharpe_ratio(sr_monthly, n_months, 6, sr_std=0.5)
    registered = deflated_sharpe_ratio(sr_monthly, n_months, 6,
                                       sr_std=load_registered_sr_std("str"))
    assert degenerate < 0.01                      # floored: indistinguishable from worthless
    assert registered > 0.5                       # discriminating
    assert registered > 50 * max(degenerate, 1e-9)
    assert 0.0 <= registered <= 1.0


def test_a_weak_candidate_still_scores_low_under_the_registered_benchmark():
    """The registered benchmark must not simply inflate everything — a near-zero Sharpe stays
    near zero."""
    weak = deflated_sharpe_ratio(0.01, 200, 6, sr_std=load_registered_sr_std("str"))
    assert weak < 0.5
