"""Stage 7 — synthetic panel injection + Layer B engineering fixtures (§10.2)."""

from __future__ import annotations

import numpy as np
import pytest

from agents.auditor.data.synthetic_panel import (
    SyntheticSpec,
    build_scenario,
    make_clean_maximal_panel,
)
from agents.auditor.schemas.toggle import TOGGLE_IDS
from agents.auditor.validation.layer_b_fixtures import run_scenario, single_bias_fixture


# --------------------------------------------------------------------------
# Each single-bias injection fires and dominates (the engineering gate)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bias", TOGGLE_IDS)
def test_single_bias_injection_fires_and_dominates(bias):
    verdict = single_bias_fixture(bias, seed=0)
    assert verdict.fired, f"{bias}: |E|={verdict.injected_effect:.5f} did not fire"
    assert verdict.dominates, (
        f"{bias}: injected {verdict.injected_effect:.5f} did not dominate "
        f"max other {verdict.max_other_effect:.5f}"
    )


# --------------------------------------------------------------------------
# The clean panel (zero injection) keeps every effect quiet — the FPR baseline
# --------------------------------------------------------------------------

def test_zero_injection_keeps_all_effects_quiet():
    scenario = build_scenario(None, seed=0)
    run = run_scenario(scenario)
    for toggle, effect in run.doe_first_order.items():
        assert abs(effect) < 1e-3, f"{toggle} showed a spurious effect {effect:.6f}"


# --------------------------------------------------------------------------
# Injection specificity — meas_err perturbs only the raw family
# --------------------------------------------------------------------------

def test_meas_err_injection_leaves_corr_family_clean():
    clean, _ = make_clean_maximal_panel(SyntheticSpec(seed=0))
    scenario = build_scenario("meas_err", seed=0)
    # corr family unchanged vs clean; raw family changed
    merged = clean.merge(
        scenario.panel, on=["cusip", "date"], suffixes=("_clean", "_inj")
    )
    assert np.allclose(merged["ret_corr_clean"], merged["ret_corr_inj"])
    assert not np.allclose(merged["ret_raw_clean"], merged["ret_raw_inj"])


def test_survivorship_injection_adds_distress_exits():
    scenario = build_scenario("survivorship", seed=1)
    assert (scenario.panel["exit_reason"] == "defaulted").sum() > 0


def test_lab_trim_scenario_strategy_carries_published_trim():
    scenario = build_scenario("lab_trim", seed=2)
    cfg = scenario.strategy.leg_calls[0].result
    assert cfg.trim_rule.value.method == "truncate"


def test_build_scenario_rejects_unknown_bias():
    with pytest.raises(ValueError, match="unknown bias"):
        build_scenario("look_ahead", seed=0)
