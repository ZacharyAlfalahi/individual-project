"""one-shot holdout stage-2 evaluator (spec §3): descriptive-only, the registered window, P3 posterior."""

from __future__ import annotations

import numpy as np
import pandas as pd

from agents.scientist.experimentalist.oneshot_holdout.stage2_evaluate import (
    SurvivorInput,
    evaluate_survivors,
)
from agents.scientist.experimentalist.oneshot_holdout.windows import registered_window

PRIORS = {"wide": 0.02, "moderate": 0.005, "sceptical": 0.0025}


def _synthetic():
    months = pd.date_range("2022-01-31", periods=45, freq="ME")
    rng = np.random.default_rng(3)
    surv = pd.Series(0.004 + rng.normal(0, 0.02, 45), index=months)
    par = pd.Series(0.001 + rng.normal(0, 0.02, 45), index=months)
    factors = pd.DataFrame({
        "date": months,
        "mktb": rng.normal(0, 0.02, 45),
        "drf": rng.normal(0, 0.02, 45),
    })
    return surv, par, factors


def test_evaluator_emits_full_window_with_pinned_lag():
    surv, par, factors = _synthetic()
    reg = registered_window(("2022-01", "2025-09", 45))
    results = evaluate_survivors([SurvivorInput("s1", surv, par)], {"bbw4": factors}, reg, PRIORS)
    rec = results[0].benchmarks["bbw4"]
    assert rec.full["window_label"] == "full_45m" and rec.full["n_obs"] == 45
    assert rec.full["nw_lags_used"] == 2                          # ⌊T^0.25⌋ pinned


def test_evaluator_record_has_no_pass_fail_field():
    surv, par, factors = _synthetic()
    reg = registered_window(("2022-01", "2025-09", 45))
    rec = evaluate_survivors([SurvivorInput("s1", surv, par)], {"bbw4": factors}, reg, PRIORS)[0]
    full = rec.benchmarks["bbw4"].full
    for forbidden in ("pass", "passed", "verdict", "advanced", "survives", "is_significant"):
        assert forbidden not in full


def test_bootstrap_cis_are_labelled_diagnostic_never_confirmatory():
    surv, par, factors = _synthetic()
    reg = registered_window(("2022-01", "2025-09", 45))
    full = evaluate_survivors([SurvivorInput("s1", surv, par)], {"bbw4": factors}, reg, PRIORS)[0] \
        .benchmarks["bbw4"].full
    cis = full["bootstrap_cis"]
    assert len(cis) == 6                                          # 3 statistics x {3, 6}
    assert all(ci["is_confirmatory"] is False for ci in cis)
    labels = {ci["floor_label"] for ci in cis}
    assert labels == {"floor_met", "below_floor"}                # 45/3=15 met, 45/6=7 below


def test_posterior_has_three_priors():
    surv, par, factors = _synthetic()
    reg = registered_window(("2022-01", "2025-09", 45))
    full = evaluate_survivors([SurvivorInput("s1", surv, par)], {"bbw4": factors}, reg, PRIORS)[0] \
        .benchmarks["bbw4"].full
    assert set(full["posterior"]["priors"]) == {"wide", "moderate", "sceptical"}
