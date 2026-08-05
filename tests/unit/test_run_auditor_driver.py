"""
Validation for `scripts/run_auditor.py` — the core-Auditor real-data driver — WITHOUT
real data. The driver's real path (gold spec -> adapt_spec -> lattice on the dev panel)
cannot run until the pre-run notes are cleared, but its two load-bearing pieces are
exercised here on synthetic panels:

  1. `core_sync_1_verdict` — the pure assertion (pass on exact zeros; fail on a nonzero
     coordinate or a missing one).
  2. `audit_anchor` + the verdict end-to-end through `run_audit`:
       - clean panel  -> stale_price is a genuine no-op -> E_stale == 0 -> PASS
         (this is the CORE-SYNC-1 expectation the dev panel must reproduce);
       - stale injected -> the mask bites -> E_stale != 0 -> FAIL
         (proves the check actually catches a view divergence, i.e. is not vacuous).
"""

from __future__ import annotations

import math
import warnings

import pytest

from agents.auditor.data.synthetic_panel import build_scenario
from agents.auditor.thresholds import SupportGate

import run_auditor
from run_auditor import (
    _MEAS_STALE,
    _STALE,
    audit_anchor,
    core_sync_1_verdict,
    default_anchor_facts,
    load_anchor_expost_trim_off,
    load_anchor_strategy,
    stale_invariance_note,
)

# A permissive gate so the short synthetic panels clear common support (mirrors the
# orchestrator test); the real run uses the fail-loud thresholds gate instead.
_GATE = SupportGate(min_common_months=12, min_common_fraction_of_reference=0.3)


# --------------------------------------------------------------------------
# 1. the pure verdict
# --------------------------------------------------------------------------

def test_verdict_passes_on_exact_zeros():
    v = core_sync_1_verdict({_STALE: 0.0, _MEAS_STALE: 0.0})
    assert v["passed"] is True
    assert v["E_stale"] == 0.0 and v["meas_err×stale_price"] == 0.0
    assert v["failure_reasons"] == []


def test_verdict_fails_on_nonzero_main_effect():
    v = core_sync_1_verdict({_STALE: 0.01, _MEAS_STALE: 0.0})
    assert v["passed"] is False
    assert any("E_stale" in r for r in v["failure_reasons"])


def test_verdict_fails_when_a_coordinate_is_absent():
    # e.g. meas_err or stale_price was not runnable, so the interaction never formed.
    v = core_sync_1_verdict({_STALE: 0.0})
    assert v["passed"] is False
    assert any("interaction absent" in r for r in v["failure_reasons"])


def test_verdict_within_tolerance_passes():
    v = core_sync_1_verdict({_STALE: 1e-12, _MEAS_STALE: -1e-12}, tol=1e-9)
    assert v["passed"] is True


# --------------------------------------------------------------------------
# 2. end-to-end through run_audit on synthetic panels
# --------------------------------------------------------------------------

def _core(scenario):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return audit_anchor(
            scenario.strategy, scenario.panel, default_anchor_facts(), scenario.signals,
            primary_metric="average", support_gate=_GATE,
        )


def test_clean_panel_passes_core_sync_1():
    """The CORE-SYNC-1 expectation: when stale_price masks nothing, both coordinates are
    exact zeros and the invariance check certifies the no-op from the returns side too."""
    core = _core(build_scenario(None))  # clean null: last_trade_date == month-end
    verdict = core_sync_1_verdict(core.saturated.doe)
    assert verdict["passed"] is True
    # bit-identical cells, but the Walsh/DOE transform leaves ~1e-19 float noise, not
    # bit-zero — which is exactly why the check is tolerance-based, not `== 0.0`.
    assert abs(core.saturated.doe[_STALE]) <= run_auditor.DEFAULT_TOL
    assert abs(core.saturated.doe[_MEAS_STALE]) <= run_auditor.DEFAULT_TOL
    # the complementary signal: expect_no_op=True on stale_price -> certified no-op
    inv = stale_invariance_note(core)
    assert inv is not None and inv["is_no_op"] is True


def test_stale_injection_fails_core_sync_1():
    """The check is not vacuous: a panel where the stale mask genuinely bites yields a
    nonzero E_stale, which the driver reports as a FAIL (the wiring-defect signal)."""
    core = _core(build_scenario("stale_price"))
    verdict = core_sync_1_verdict(core.saturated.doe)
    assert verdict["passed"] is False
    assert abs(core.saturated.doe[_STALE]) > run_auditor.DEFAULT_TOL


# --------------------------------------------------------------------------
# 3. the adapt path applies the §6 standing subs (regression: the driver once
#    called adapt_spec WITHOUT the standing-subs table, so `str` and `mom6`
#    refused on the real panel — weighting_base=market_value / expost_trim=truncate).
#    No panel needed: this pins the gold-spec -> adapt_spec half of the real path.
# --------------------------------------------------------------------------

def test_load_anchor_strategy_applies_standing_subs():
    """`str` and `mom6` are ONLY runnable once the pre-registered §6 conventions are
    applied; the driver must pass the standing-subs table (contract §6). `drf` needs no
    sub. A refusal here would mean the driver dropped the standing-subs wiring again."""
    cases = {
        "str": "par_weighting_v1",       # weighting_base market_value -> par
        "mom6": "lab_trim_delegation_v1",  # expost_trim truncate -> none (delegated to lab_trim)
    }
    for anchor, sub_id in cases.items():
        strat = load_anchor_strategy(anchor)
        assert not strat.refused, f"{anchor} refused despite §6 standing subs"
        applied = {a.substitution_id for a in strat.standing_subs_applied}
        assert sub_id in applied, f"{anchor} did not apply {sub_id}; applied={applied}"

    # drf (BBW) is runnable with no standing substitution.
    drf = load_anchor_strategy("drf")
    assert not drf.refused and not drf.standing_subs_applied


# --------------------------------------------------------------------------
# 4. Part E — lab_trim OFF-arm re-injection (spec E). The delegated mom6 base
#    carries NO trim (lab_trim_delegation_v1), so WITHOUT re-injection lab_trim is
#    INERT (a wiring bug, not a real null); expost_trim_off re-injects the paper's
#    published trim on the OFF arm only, making the first-order effect non-zero.
# --------------------------------------------------------------------------

def test_lab_trim_off_reinjection_makes_effect_nonzero():
    import numpy as np

    from agents.auditor.data.synthetic_panel import (
        inject_lab_trim,
        make_clean_maximal_panel,
        score_strategy,
        SyntheticSpec,
    )
    from agents.quant.config import TrimRule

    spec = SyntheticSpec(seed=0)
    panel, signals = make_clean_maximal_panel(spec)
    # Plant extreme long-leg losses (beyond -0.5) that a truncate trim removes.
    panel, signals = inject_lab_trim(panel, signals, 0.8, np.random.default_rng(1000))
    # A strategy with NO base trim -> cfg.trim_rule = none, exactly like the real
    # mom6 anchor after lab_trim_delegation_v1 delegated its trim out of the base.
    strat = score_strategy()
    lab = frozenset({"lab_trim"})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # INERT: no re-injection -> lab_trim OFF = none = ON = none -> ~0 effect
        # (the pre-fix structural zero / wiring bug).
        inert = audit_anchor(
            strat, panel, default_anchor_facts(), signals,
            primary_metric="average", support_gate=_GATE,
        )
        # FIXED: re-inject the published truncate on the OFF arm only.
        fixed = audit_anchor(
            strat, panel, default_anchor_facts(), signals,
            primary_metric="average", support_gate=_GATE,
            expost_trim_off=TrimRule(method="truncate", lo=-0.5, hi=0.5),
        )

    assert abs(inert.saturated.doe[lab]) <= run_auditor.DEFAULT_TOL, (
        "lab_trim must be inert without re-injection (the pre-fix structural zero)"
    )
    assert abs(fixed.saturated.doe[lab]) > run_auditor.DEFAULT_TOL, (
        "lab_trim OFF re-injection must make the first-order effect non-zero (spec E3)"
    )


def test_load_anchor_expost_trim_off_mom6_only():
    # str/drf published no return trim -> None (their lab_trim stays a structural
    # zero, correctly). mom6 -> Jostova's 99.5th right-tail percentile truncate.
    assert load_anchor_expost_trim_off("str") is None
    assert load_anchor_expost_trim_off("drf") is None
    tr = load_anchor_expost_trim_off("mom6")
    assert tr is not None
    assert tr.method == "truncate"
    assert tr.bounds_type == "percentile"
    assert tr.hi == 0.995 and tr.lo is None          # one-sided right (Jostova)
    assert tr.percentile_method == "linear"           # from thresholds, not defaulted


# --------------------------------------------------------------------------
# 5. Part E2b — incomplete-lattice typed verdict. A non-computable cell (e.g.
#    mom6's lab_trim ON cell going unstable on a no-price-range baseline) yields
#    a TYPED verdict, never an imputed / partial-surface attribution.
# --------------------------------------------------------------------------

def test_incomplete_lattice_guard():
    from agents.auditor.checks.orchestrator import (
        IncompleteLatticeError,
        check_lattice_complete,
    )

    # All-finite response surface -> no raise (attribution proceeds).
    Y_ok = {frozenset(): 0.10, frozenset({"lab_trim"}): 0.08}
    check_lattice_complete(Y_ok, "mom6", ("lab_trim",))

    # A non-computable cell (NaN / inf) -> typed verdict, no imputation.
    Y_bad = {
        frozenset(): 0.10,
        frozenset({"lab_trim"}): float("nan"),
        frozenset({"meas_err"}): math.inf,
    }
    with pytest.raises(IncompleteLatticeError) as ei:
        check_lattice_complete(Y_bad, "mom6", ("lab_trim", "meas_err"))
    err = ei.value
    assert err.strategy_label == "mom6"
    assert frozenset({"lab_trim"}) in err.non_computable
    assert frozenset({"meas_err"}) in err.non_computable
    assert frozenset() not in err.non_computable      # the finite cell is not flagged
    assert "no imputation" in str(err).lower()


# --------------------------------------------------------------------------
# 6. Part D — str's meas_err exclusion (not_applicable). str runs a reduced 2^4
#    lattice; the bias_class partition emits ONLY a method component (data-quality
#    + cross-class are None, never a zero — the absence-as-observation guard).
# --------------------------------------------------------------------------

def test_str_meas_err_excluded_reduced_lattice():
    facts = default_anchor_facts("str")
    na = next(f for f in facts if f.toggle_id == "meas_err")
    assert na.not_applicable and not na.runnable
    # None default keeps every toggle runnable (backward-compatible synthetic path).
    assert all(f.runnable for f in default_anchor_facts())

    scenario = build_scenario(None)  # clean single-leg null strategy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        core = audit_anchor(
            scenario.strategy, scenario.panel, facts, scenario.signals,
            primary_metric="average", support_gate=_GATE,
        )
    # Reduced 2^4 lattice: meas_err is EXCLUDED, not measured.
    assert "meas_err" not in core.runnable_toggles
    assert len(core.runnable_toggles) == 4
    assert "meas_err" in core.not_applicable_toggles
    # bias_class partition: ONLY a method component; data-quality + cross-class are
    # None (absence, never a zero observation).
    part = core.to_dict()["bias_class_partition"]
    assert part["data_quality_component"] is None
    assert part["cross_class_modulation"] is None
    assert part["methodological_construction_component"] is not None
