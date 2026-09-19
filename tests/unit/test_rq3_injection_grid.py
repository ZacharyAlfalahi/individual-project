"""RQ3 per-correction injection grid: verifies the driver's columns against the
reused injection/calibration machinery. Component-level (fast) tests; the full grid's
only expensive column (coverage) is checked on one bias. Deterministic, $0.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_rq3_injection_grid as grid_mod  # noqa: E402
from run_rq3_injection_grid import (  # noqa: E402
    _classify_coverage,
    _recorded_recovered_mde,
    _coverage,
    _injected_bp,
    _signal_background,
)


def test_injected_effect_is_the_noise_free_plant_in_bp():
    # The honest planted truth is the noise-free long-short effect (NOT magnitude x 10000).
    # Survivorship's default plant is ~68 bp/mo (recipe-verified).
    bp = _injected_bp("survivorship", seeds=1)
    assert 55.0 < abs(bp) < 85.0


def test_signal_background_dominates_at_five_times():
    sg = _signal_background("meas_err")
    assert sg["dominates_ge5"] is True                 # engineering-tier: injected >= 5x background
    assert sg["background_machine_zero"] is True        # clean DGP -> background < 0.01 bp/mo


@pytest.mark.skipif(
    not list((REPO_ROOT / "results" / "auditor" / "calibration").glob("run_*/calibration.json")),
    reason="recorded calibration artefact not shipped with the repository",
)
def test_recovered_and_mde_read_from_recorded_calibration():
    m = _recorded_recovered_mde("meas_err")
    assert 280.0 < m["recovered_bp"] < 340.0            # recorded calibration bounds
    assert m["mde_structural"] is False and m["mde_bp"] is not None
    # stale_price's channel is structural (magnitude-independent): mde magnitude is 0 -> flagged.
    assert _recorded_recovered_mde("stale_price")["mde_structural"] is True


def test_degeneracy_guard_predicate():
    # A width-zero (deterministic) CI is degenerate -- never covered/not-covered.
    assert _classify_coverage(0.5, 0.5, 0.5) == "degenerate"
    # A genuine interval passes through with the correct boolean coverage (bounds inclusive).
    assert _classify_coverage(-1.0, 1.0, 0.0) == "covered"
    assert _classify_coverage(-1.0, 1.0, 2.0) == "not_covered"
    assert _classify_coverage(0.0, 1.0, 1.0) == "covered"


def test_degeneracy_floor_constant_is_used(monkeypatch):
    # The documented epsilon floor (1e-12 return units == 1e-8 bp/mo) lives at module top ...
    assert grid_mod._DEGENERATE_CI_HALFWIDTH_FLOOR == 1e-12
    # ... and the predicate reads THAT constant, not an inline magic number: raising it flips a
    # normal-width CI to degenerate; a CI just under it is degenerate at the default too.
    assert _classify_coverage(0.0, 1.9e-12, 1e-12) == "degenerate"     # half-width 0.95e-12 < floor
    monkeypatch.setattr(grid_mod, "_DEGENERATE_CI_HALFWIDTH_FLOOR", 10.0)
    assert grid_mod._classify_coverage(-1.0, 1.0, 0.0) == "degenerate"


def test_interval_coverage_fraction_and_meas_err_degenerate():
    # Per-seed calibration coverage via a REAL block-bootstrap CI (not the degenerate SweepRecord.ci)
    # stays a plain fraction for a genuinely stochastic bias (no degenerate fields attached) ...
    cov = _coverage("survivorship", seeds=2, replicates=30)
    assert cov["interval_coverage"] is not None and 0.0 <= cov["interval_coverage"] <= 1.0
    assert "n_degenerate" not in cov and "exact_recovery" not in cov
    # ... while meas_err's deterministic ON-OFF contrast (per-bond time-constant shift to the raw
    # family only) collapses the CI below the floor -> flagged, with exact recovery at machine
    # precision instead of a coin-flip coverage number.
    dg = _coverage("meas_err", seeds=2, replicates=30)
    assert dg["interval_coverage"] == "degenerate_deterministic"
    assert dg["n_degenerate"] == 2
    assert dg["exact_recovery"] < 1e-12
