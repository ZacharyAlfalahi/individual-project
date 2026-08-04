"""
test_shared_eval_spanning.py — the typed spanning contract. Includes the CHARACTERISATION
PIN: spanning_regression reproduces crowding_diagnostic (and the engine) exactly on the
same inputs, so the typed superset never silently diverges from the wired flat-dict
diagnostic. Failure-dominant (D-E18): refusals, rank deficiency, collinearity.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from agents.quant.library.characteristic_sort import regress_on_benchmark
from shared.evaluation.contracts import RefusalCode
from shared.evaluation.crowding import crowding_diagnostic
from shared.evaluation.spanning import spanning_regression

from _shared_eval_fixtures import FACTORS, candidate, crowding_config, factor_frame  # noqa: E402


# ---------------------------------------------------------------------------
# characterisation pin: spanning == crowding == engine
# ---------------------------------------------------------------------------

def test_characterisation_spanning_reproduces_crowding_and_engine() -> None:
    fr = factor_frame(T=200, seed=1)
    y = candidate(fr, a0=0.002, betas={"mktb": 0.5, "drf": 0.3}, noise=0.0005, seed=2)
    cfg = crowding_config()

    sp = spanning_regression(y, config=cfg, factors=fr)
    cw = crowding_diagnostic(y, config=cfg, factors=fr)
    eng = regress_on_benchmark(y, fr, nw_lags=int(math.floor(200 ** 0.25)))

    assert sp.estimable and sp.refusal_code is None
    assert sp.alpha_monthly == pytest.approx(cw["alpha"], rel=1e-12, abs=1e-15)
    assert sp.t_hac == pytest.approx(cw["alpha_t"], rel=1e-12, abs=1e-15)
    assert sp.alpha_monthly == pytest.approx(eng["alpha"], rel=1e-12, abs=1e-15)
    for f in FACTORS:
        assert dict(sp.betas)[f] == pytest.approx(cw[f"beta_{f}"], rel=1e-12, abs=1e-15)
    assert sp.n_obs == 200 and sp.n_controls == 6


def test_provenance_stamps_floor_rule_and_lag() -> None:
    fr = factor_frame(T=209, seed=9)
    y = candidate(fr, betas={f: 0.0 for f in FACTORS}, seed=10)
    sp = spanning_regression(y, config=crowding_config(), factors=fr)
    assert sp.provenance.estimator_id == "newey_west_hac"
    assert sp.provenance.lag_rule == "floor(T**0.25)"
    assert sp.provenance.lag_used == int(math.floor(209 ** 0.25))  # = 3


# ---------------------------------------------------------------------------
# known answers
# ---------------------------------------------------------------------------

def test_planted_alpha_recovered_with_ci() -> None:
    fr = factor_frame(T=200, seed=3)
    y = candidate(fr, a0=0.004, betas={"mktb": 0.6}, noise=0.0003, seed=4)
    sp = spanning_regression(y, config=crowding_config(), factors=fr)
    assert sp.alpha_monthly == pytest.approx(0.004, abs=5e-3)
    assert sp.ci_low is not None and sp.ci_high is not None and sp.ci_low < sp.alpha_monthly < sp.ci_high
    assert sp.r_squared is not None and 0.0 <= sp.r_squared <= 1.0


def test_zero_noise_alpha_exact() -> None:
    fr = factor_frame(T=120, seed=5)
    y = candidate(fr, a0=0.003, betas={"mktb": 0.4, "drf": -0.2}, noise=0.0, seed=6)
    sp = spanning_regression(y, config=crowding_config(min_obs=60), factors=fr)
    assert sp.alpha_monthly == pytest.approx(0.003, abs=1e-9)


# ---------------------------------------------------------------------------
# refusals (failure-dominant)
# ---------------------------------------------------------------------------

def test_insufficient_observations_refuses() -> None:
    fr = factor_frame(T=200, seed=1)
    y = candidate(fr, seed=2)
    sp59 = spanning_regression(y.iloc[:59], config=crowding_config(min_obs=60), factors=fr)
    assert not sp59.estimable and sp59.refusal_code is RefusalCode.INSUFFICIENT_OBSERVATIONS
    assert sp59.alpha_monthly is None and sp59.n_obs == 59  # n_obs populated, estimates None
    sp60 = spanning_regression(y.iloc[:60], config=crowding_config(min_obs=60), factors=fr)
    assert sp60.estimable  # boundary: n = 60 estimates


def test_rank_deficient_duplicated_control_populates_diagnostics() -> None:
    fr = factor_frame(T=120, seed=7)
    fr["mom6"] = fr["str"]  # exact duplicate control -> design rank deficient
    y = candidate(fr, betas={"mktb": 0.4}, seed=8)
    sp = spanning_regression(y, config=crowding_config(min_obs=10), factors=fr)
    assert not sp.estimable and sp.refusal_code is RefusalCode.RANK_DEFICIENT
    assert sp.design_rank == 6 and sp.n_controls == 6  # rank < controls + 1
    # Singular design: condition number is either a huge finite value or nulled-inf; both
    # signal the singularity, and neither is a JSON-unsafe token.
    assert sp.condition_number is None or sp.condition_number > 1e10
    assert sp.alpha_monthly is None  # estimates withheld, conditioning diagnostics kept
    assert len(sp.vif_by_control) == 6
    # perfect collinearity nulls the duplicated pair's VIF (never NaN)
    assert None in dict(sp.vif_by_control).values()


def test_refusal_and_degenerate_results_are_strict_json_safe() -> None:
    # MAJOR-1 regression: no to_dict on a refusal/degenerate branch may emit NaN/inf.
    fr = factor_frame(T=120, seed=17)
    y = candidate(fr, seed=18)
    insufficient = spanning_regression(y.iloc[:40], config=crowding_config(min_obs=60), factors=fr)
    frdup = factor_frame(T=120, seed=19)
    frdup["mom6"] = frdup["str"]
    rank_def = spanning_regression(candidate(frdup, seed=20), config=crowding_config(min_obs=10), factors=frdup)
    for r in (insufficient, rank_def):
        import json
        json.dumps(r.to_dict(), allow_nan=False)  # raises ValueError if any NaN/inf leaks


def test_near_collinear_controls_large_condition_and_loo_swings() -> None:
    fr = factor_frame(T=200, seed=12)
    fr["crf"] = 0.99 * fr["drf"] + 0.01 * fr["crf"]  # crf ~0.99 corr with drf
    y = candidate(fr, a0=0.002, betas={"drf": 0.5}, noise=0.0005, seed=13)
    sp = spanning_regression(y, config=crowding_config(), factors=fr)
    assert sp.estimable  # near-collinear, not exactly singular
    assert sp.condition_number > 50.0
    loo = dict(sp.leave_one_control_out_alpha)
    assert len(loo) == 6  # dropping any one control gives an alpha
    # VIF for the near-collinear pair is elevated.
    vif = dict(sp.vif_by_control)
    assert vif["drf"] > 5.0 or vif["crf"] > 5.0


def test_candidate_identical_to_control_visible_in_max_corr() -> None:
    fr = factor_frame(T=150, seed=14)
    y = pd.Series(fr["mktb"].to_numpy(), index=fr["date"])  # candidate == a control
    sp = spanning_regression(y, config=crowding_config(), factors=fr)
    assert sp.max_abs_pairwise_corr is not None and sp.max_abs_pairwise_corr == pytest.approx(1.0, abs=1e-9)


def test_no_overlap_refuses_as_insufficient() -> None:
    fr = factor_frame(T=60, seed=16)
    y = pd.Series(np.arange(5, dtype=float), index=pd.date_range("2050-01-31", periods=5, freq="ME"))
    sp = spanning_regression(y, config=crowding_config(min_obs=60), factors=fr)
    assert not sp.estimable and sp.refusal_code is RefusalCode.INSUFFICIENT_OBSERVATIONS
    assert sp.n_obs == 0
