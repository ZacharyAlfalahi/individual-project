"""RQ4 funnel power-curve sweep (synthetic positive-control ladder). Plumbing-level tests: the
beta<->bp/mo mapping, sweep aggregation/determinism on a stubbed (fast) funnel, and the CLI-default
regression guard pinning the historical positive-control constants. No real funnel run here (the
positive-control funnel itself is covered by test_scientist_positive_control.py).
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_rq4_positive_control as mod  # noqa: E402


def test_beta_for_bp_mapping_is_linear_on_the_realised_anchor():
    # Anchor: _BETA = 0.00020 <=> 44 bp/mo (the historical run of record's realised mean_return).
    assert mod.beta_for_bp(44.0) == pytest.approx(0.00020, rel=1e-12)
    # Linear: half the bp -> half the beta; and an explicit ladder point.
    assert mod.beta_for_bp(22.0) == pytest.approx(0.00010, rel=1e-12)
    assert mod.beta_for_bp(5.0) == pytest.approx(0.00020 * 5.0 / 44.0, rel=1e-12)
    assert mod.beta_for_bp(0.0) == 0.0


def _stub_funnel(beta=mod._BETA, sigma=mod._SIGMA, seed=mod._SEED):
    """Deterministic fast stand-in for run_positive_control keyed on (beta, seed)."""
    advanced = seed % 2 == 0
    return {"planted_strong": {
        "advanced": advanced, "p_bh": 0.01 if advanced else 0.5,
        "cpcv_median_sharpe": (beta * 1e4 + seed * 1e-9) if advanced else None,
        "mean_return": beta * 24.0}}


def test_sweep_plumbing_aggregation_and_determinism(monkeypatch):
    monkeypatch.setattr(mod, "run_positive_control", _stub_funnel)
    r1 = mod.run_sweep(ladder_bp=(10.0, 44.0), n_seeds=2, base_seed=100)
    r2 = mod.run_sweep(ladder_bp=(10.0, 44.0), n_seeds=2, base_seed=100)
    assert r1 == r2                                     # same seeds -> same results
    m10, m44 = r1["per_magnitude"]

    # Seeds are derived from the base seed and recorded per cell.
    assert [c["seed"] for c in m10["cells"]] == [100, 101]
    assert r1["base_seed"] == 100 and r1["n_seeds"] == 2

    # Betas come from the mapping helper.
    assert m10["beta"] == pytest.approx(mod.beta_for_bp(10.0))
    assert m44["beta"] == pytest.approx(mod.beta_for_bp(44.0))

    # Aggregation: seed 100 advances, 101 does not -> advance rate 0.5 at every magnitude.
    assert m10["advance_rate"] == 0.5 and m10["n_advanced"] == 1
    assert m44["advance_rate"] == 0.5 and m44["n_advanced"] == 1

    # Median CPCV Sharpe is taken over the cells that HAVE a CPCV number (here: seed 100 only).
    assert m10["n_with_cpcv"] == 1
    assert m10["median_cpcv_sharpe"] == pytest.approx(mod.beta_for_bp(10.0) * 1e4 + 100e-9)

    # Empirical MDE = smallest ladder magnitude with advance rate >= 0.5.
    assert r1["empirical_mde_bp"] == 10.0


def test_sweep_mde_is_none_when_no_magnitude_reaches_half(monkeypatch):
    monkeypatch.setattr(mod, "run_positive_control", _stub_funnel)
    # A single odd seed never advances under the stub -> advance rate 0.0, no CPCV, MDE undefined.
    r = mod.run_sweep(ladder_bp=(10.0,), n_seeds=1, base_seed=101)
    assert r["per_magnitude"][0]["advance_rate"] == 0.0
    assert r["per_magnitude"][0]["median_cpcv_sharpe"] is None
    assert r["empirical_mde_bp"] is None


def test_cli_defaults_equal_the_historical_constants():
    # Regression guard: the refactor to CLI parameters must not move the run-of-record defaults.
    args = mod._build_parser().parse_args([])
    assert args.beta == 0.00020
    assert args.sigma == 0.020
    assert args.seed == 20260906
    assert args.sweep is False
    assert args.n_seeds == 20
    assert tuple(args.ladder_bp) == (5.0, 10.0, 15.0, 20.0, 30.0, 44.0)
    # And the module constants themselves are the historical values.
    assert (mod._BETA, mod._SIGMA, mod._SEED) == (0.00020, 0.020, 20260906)
