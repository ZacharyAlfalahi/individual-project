"""
RQ3 items 5+7 — the full-audit driver (`scripts/run_full_audit.py`).

This is how the full-audit wiring is verified WITHOUT the real confirmatory run (which
requires the registered per-anchor DSR configuration, O-A4). Two independent checks:

  (a) the pure per-anchor path — `audit_anchor_full` on a SYNTHETIC scenario with
      INJECTED `n_trials`/`sr_std` (mirroring `test_auditor_report.py`) -> AuditReport
      -> the deterministic `render_report` emits prose -> `verify_numbers` passes. NO
      LLM, NO real data.

  (b) the DSR gate — `load_anchor_dsr` / `load_dsr_for_anchors` REFUSE fail-loud when
      `auditor.dsr` is absent (a thresholds file without the block). A partial block
      (anchor or field missing) is likewise refused. The committed `docs/thresholds.yaml`
      now carries the RATIFIED block, so the default-path load succeeds — the
      refusals are pinned on synthetic thresholds files.
"""

from __future__ import annotations

import textwrap

import pytest

from agents.auditor import AuditorConfig, render_report, verify_numbers
from agents.auditor.data.synthetic_panel import build_scenario
from agents.auditor.thresholds import (
    AuditorThresholdError,
    BayesParams,
    EconomicGapBands,
    SupportGate,
)

from scripts.run_full_audit import (
    AnchorDsr,
    DsrPreRegistrationAbsent,
    audit_anchor_full,
    load_anchor_dsr,
    load_dsr_for_anchors,
    render_verified_prose,
)

from _auditor_fixtures import all_runnable_facts

# Explicit config (built, not read from thresholds), so the synthetic path never needs
# the not-yet-pre-registered numbers — identical shape to test_auditor_report.CONFIG.
CONFIG = AuditorConfig(
    primary_metric="average",
    percentage_denominator_min=0.0,
    support_gate=SupportGate(min_common_months=12, min_common_fraction_of_reference=0.3),
    n_replicates=200,
    block_length_months=6,
    min_effective_blocks=3,
    vartheta=0.002,
    d_max=0.2,
    fdr_q=0.1,
    bayes=BayesParams(prior_scale=0.1, epsilon=1e-8),
    gap_bands=EconomicGapBands(small=0.005, moderate=0.02, large=0.05),
)


# --------------------------------------------------------------------------
# (a) pure per-anchor path: synthetic scenario + injected DSR -> report -> verified prose
# --------------------------------------------------------------------------

def _report(bias="meas_err", seed=0):
    """Drive the driver's per-anchor entry point exactly as main() would, but on a
    synthetic scenario with INJECTED n_trials/sr_std (mirrors test_auditor_report)."""
    scenario = build_scenario(bias, seed=seed)
    return audit_anchor_full(
        scenario.strategy, scenario.panel, all_runnable_facts(), CONFIG,
        signals=scenario.signals, n_trials=20, sr_std=0.1, seed=1,
    )


def test_audit_anchor_full_builds_a_complete_report():
    d = _report().to_dict()
    for key in ("audit_scope", "bootstrap", "inference", "fdr",
                "bayesian", "compression", "economic"):
        assert key in d, f"missing {key}"
    assert d["economic"]["n_trials"] == 20        # the injected DSR count flows through
    assert d["audit_scope"] == "COMPLETE"


def test_full_audit_path_threads_expost_trim_off_to_the_lattice():
    """LINCHPIN (Part E via the FULL-audit path): expost_trim_off must reach the
    lattice through report.run_full_audit, not just sit on the signature. On a panel
    with planted extreme losses + a no-base-trim strategy (the delegated mom6), the
    lab_trim first-order effect is INERT without re-injection and NON-ZERO with it —
    proving the confirmatory driver exercises Part E, not the raw arm."""
    import numpy as np

    from agents.auditor.data.synthetic_panel import (
        inject_lab_trim,
        make_clean_maximal_panel,
        score_strategy,
        SyntheticSpec,
    )
    from agents.quant.config import TrimRule

    panel, signals = make_clean_maximal_panel(SyntheticSpec(seed=0))
    panel, signals = inject_lab_trim(panel, signals, 0.8, np.random.default_rng(1000))
    strat = score_strategy()  # no base trim = the delegated mom6
    lab = frozenset({"lab_trim"})
    tol = 1e-9

    inert = audit_anchor_full(
        strat, panel, all_runnable_facts(), CONFIG,
        signals=signals, n_trials=20, sr_std=0.1, seed=1,
    )
    fixed = audit_anchor_full(
        strat, panel, all_runnable_facts(), CONFIG,
        signals=signals, n_trials=20, sr_std=0.1, seed=1,
        expost_trim_off=TrimRule(method="truncate", lo=-0.5, hi=0.5),
    )
    assert abs(inert.core.saturated.doe[lab]) <= tol, "lab_trim inert without re-injection"
    assert abs(fixed.core.saturated.doe[lab]) > tol, "expost_trim_off must reach the lattice"


def test_deterministic_render_passes_the_numeric_verifier():
    report = _report("stale_price", seed=1)
    prose = render_report(report)
    result = verify_numbers(prose, report.to_dict())
    assert result.ok, f"untraceable numbers in rendered prose: {result.unverified}"
    assert result.n_checked > 0


def test_render_verified_prose_helper_returns_verified_prose():
    report = _report("meas_err", seed=2)
    prose, result = render_verified_prose(report)
    assert result.ok and result.n_checked > 0
    assert "Audit scope: COMPLETE" in prose
    assert "BH-FDR" in prose
    # the helper's guarantee is the verifier's guarantee
    assert verify_numbers(prose, report.to_dict()).ok


# --------------------------------------------------------------------------
# (b) the DSR pre-registration gate (fail-loud)
# --------------------------------------------------------------------------

def _write_thresholds(tmp_path, body: str):
    p = tmp_path / "thresholds.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def test_dsr_gate_refuses_when_block_absent(tmp_path):
    # An auditor: block with NO dsr sub-block — the absent-configuration case.
    path = _write_thresholds(tmp_path, """
        auditor:
          primary_metric: average
    """)
    with pytest.raises(DsrPreRegistrationAbsent) as exc:
        load_anchor_dsr("str", path)
    assert "auditor.dsr pre-registration absent" in str(exc.value)
    assert "confirmatory_prereg_proposal.md" in exc.value.message


def test_dsr_gate_refuses_via_load_dsr_for_anchors(tmp_path):
    path = _write_thresholds(tmp_path, """
        auditor:
          primary_metric: average
    """)
    with pytest.raises(DsrPreRegistrationAbsent):
        load_dsr_for_anchors(("str", "drf", "mom6"), path)


def test_dsr_gate_loads_the_real_committed_thresholds_now_ratified():
    # The load-bearing proof: `auditor.dsr` was RATIFIED + committed (tag confirmatory-rerun), so against the
    # ACTUAL committed docs/thresholds.yaml (default path) the gate now LOADS rather
    # than refuses.
    # The three synthetic-path refusal tests above still pin the fail-loud behaviour
    # when the block / an anchor / a field is absent.
    dsr = load_dsr_for_anchors(("str", "drf", "mom6"))
    assert set(dsr) == {"str", "drf", "mom6"}
    for anchor in ("str", "drf", "mom6"):
        assert dsr[anchor] == AnchorDsr(n_trials=6, sr_std=0.08)


def test_partial_dsr_block_is_refused_not_defaulted(tmp_path):
    # Block present but the anchor's sr_std is missing -> a partial pre-registration,
    # refused as a generic AuditorThresholdError (NOT the whole-block gate).
    path = _write_thresholds(tmp_path, """
        auditor:
          dsr:
            str:
              n_trials: 6
    """)
    with pytest.raises(AuditorThresholdError, match="sr_std"):
        load_anchor_dsr("str", path)


def test_dsr_block_present_but_missing_anchor_is_refused(tmp_path):
    path = _write_thresholds(tmp_path, """
        auditor:
          dsr:
            str:
              n_trials: 6
              sr_std: 0.08
    """)
    with pytest.raises(AuditorThresholdError, match="mom6"):
        load_anchor_dsr("mom6", path)


def test_ratified_dsr_block_loads(tmp_path):
    # A fully-ratified block loads to typed AnchorDsr inputs — the shape a future
    # approval commit will take (per docs/auditor/confirmatory_prereg_proposal.md §1).
    path = _write_thresholds(tmp_path, """
        auditor:
          dsr:
            str:
              n_trials: 6
              sr_std: 0.08
            drf:
              n_trials: 6
              sr_std: 0.07
            mom6:
              n_trials: 6
              sr_std: 0.09
    """)
    dsr = load_dsr_for_anchors(("str", "drf", "mom6"), path)
    assert dsr["str"] == AnchorDsr(n_trials=6, sr_std=0.08)
    assert dsr["mom6"].sr_std == 0.09
    # and the injected inputs would flow into a real run's economic block unchanged.
    assert all(v.n_trials == 6 for v in dsr.values())
