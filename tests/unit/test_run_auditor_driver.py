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

import warnings

from agents.auditor.data.synthetic_panel import build_scenario
from agents.auditor.thresholds import SupportGate

import run_auditor
from run_auditor import (
    _MEAS_STALE,
    _STALE,
    audit_anchor,
    core_sync_1_verdict,
    default_anchor_facts,
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
