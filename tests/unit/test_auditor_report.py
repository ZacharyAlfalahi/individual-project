"""Inc2-H — full AuditReport assembly, deterministic renderer, and fail-loud config."""

from __future__ import annotations

import math
import textwrap

import pytest

from agents.auditor import (
    AuditorConfig,
    render_report,
    run_full_audit,
    verify_numbers,
)
from agents.auditor.data.synthetic_panel import build_scenario
from agents.auditor.schemas.toggle import ToggleFacts
from agents.auditor.thresholds import (
    AuditorThresholdError,
    BayesParams,
    EconomicGapBands,
    SupportGate,
)

from _auditor_fixtures import all_runnable_facts

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


def _report(bias="meas_err", seed=0, facts=None):
    scenario = build_scenario(bias, seed=seed)
    return run_full_audit(
        scenario.strategy, scenario.panel, facts or all_runnable_facts(), CONFIG,
        signals=scenario.signals, n_trials=20, sr_std=0.1, seed=1,
    )


# --------------------------------------------------------------------------
# Full report assembly
# --------------------------------------------------------------------------

def test_full_report_has_every_section():
    report = _report()
    d = report.to_dict()
    for key in ("audit_scope", "shapley", "bootstrap", "inference",
                "fdr", "bayesian", "compression", "economic"):
        assert key in d, f"missing {key}"


def test_full_report_confirmatory_family_size():
    d = _report().to_dict()
    # 5 main + 3 validated pairs on a COMPLETE audit
    assert d["fdr"]["n_family"] == 8
    # inference covers the same coordinates
    assert len(d["inference"]) == 8


def test_injected_effect_is_rejected_by_fdr():
    d = _report("meas_err", seed=2).to_dict()
    assert d["fdr"]["decisions"]["meas_err"]["rejected"] is True
    assert d["fdr"]["n_rejected"] >= 1


def test_bootstrap_meta_recorded():
    d = _report().to_dict()
    assert d["bootstrap"]["block_length"] == 6
    assert d["bootstrap"]["n_replicates"] == 200
    assert d["bootstrap"]["t_common"] > 0


def test_stationary_sensitivity_reported_beside_primary():
    # The pre-registered stationary bootstrap (D-A29) is surfaced BESIDE the fixed-block
    # primary — a finite gap CI at the same expected block length, never a verdict input.
    d = _report().to_dict()["bootstrap"]
    assert d["stationary_block_length"] == 6
    lo, hi = d["stationary_gap_ci_low"], d["stationary_gap_ci_high"]
    assert lo <= hi and math.isfinite(lo) and math.isfinite(hi)


def test_materiality_sweep_wired_into_full_audit():
    # The §8.2.3/§9 neighbouring-threshold sweep reaches the Bayesian materiality output.
    import dataclasses

    grid = (0.0005, 0.001, 0.0015, 0.002)
    cfg = dataclasses.replace(CONFIG, vartheta=0.001, vartheta_grid=grid)
    scenario = build_scenario("meas_err", seed=0)
    report = run_full_audit(
        scenario.strategy, scenario.panel, all_runnable_facts(), cfg,
        signals=scenario.signals, n_trials=20, sr_std=0.1, seed=1,
    )
    posteriors = report.to_dict()["bayesian"]["posteriors"]
    assert posteriors
    for post in posteriors.values():
        sweep = dict((v, p) for v, p in post["p_material_sweep"])
        assert set(sweep) == set(grid)


def test_partial_audit_reduces_confirmatory_family():
    facts = [
        ToggleFacts("survivorship", runnable=False,
                    runnable_reason="MISSING_EXIT_DATA", fixed_state="OFF")
        if f.toggle_id == "survivorship" else f
        for f in all_runnable_facts()
    ]
    d = _report("meas_err", seed=3, facts=facts).to_dict()
    assert d["audit_scope"] == "PARTIAL"
    # survivorship main + surv×meas pair drop => 4 main + 2 pairs = 6
    assert d["fdr"]["n_family"] == 6
    assert d["conditioning_statement"]


# --------------------------------------------------------------------------
# Deterministic renderer + numeric verifier round-trip (§11)
# --------------------------------------------------------------------------

def test_rendered_prose_passes_the_numeric_verifier():
    report = _report("stale_price", seed=1)
    prose = render_report(report)
    result = verify_numbers(prose, report.to_dict())
    assert result.ok, f"untraceable numbers in rendered prose: {result.unverified}"
    assert result.n_checked > 0


def test_rendered_prose_mentions_scope_and_fdr():
    report = _report()
    prose = render_report(report)
    assert "Audit scope: COMPLETE" in prose
    assert "BH-FDR" in prose


# --------------------------------------------------------------------------
# Fail-loud config from thresholds
# --------------------------------------------------------------------------

def test_config_from_thresholds_reads_full_block(tmp_path):
    path = tmp_path / "thresholds.yaml"
    path.write_text(textwrap.dedent("""
        auditor:
          primary_metric: average
          support_gate:
            min_common_months: 24
            min_common_fraction_of_reference: 0.5
          bootstrap:
            n_replicates: 1000
            min_effective_blocks: 10
            block_length_months: 6
          shapley:
            percentage_denominator_min: 0.05
          practical_significance:
            vartheta: 0.05
            vartheta_sensitivity_grid: [0.025, 0.05, 0.075, 0.1]
          compression:
            d_max: 0.2
          fdr:
            q: 0.1
          bayes:
            prior_scale: 0.1
            epsilon: 1.0e-8
          economic:
            gap_bands:
              small: 0.01
              moderate: 0.05
              large: 0.10
    """))
    cfg = AuditorConfig.from_thresholds(path)
    assert cfg.vartheta == 0.05
    assert cfg.fdr_q == 0.1
    assert cfg.gap_bands.large == 0.10
    assert cfg.vartheta_grid == (0.025, 0.05, 0.075, 0.1)


def test_config_from_thresholds_fails_loud_on_missing_vartheta(tmp_path):
    path = tmp_path / "thresholds.yaml"
    path.write_text(textwrap.dedent("""
        auditor:
          primary_metric: average
          support_gate:
            min_common_months: 24
            min_common_fraction_of_reference: 0.5
          bootstrap:
            n_replicates: 1000
            min_effective_blocks: 10
            block_length_months: 6
          shapley:
            percentage_denominator_min: 0.05
    """))
    with pytest.raises(AuditorThresholdError, match="practical_significance"):
        AuditorConfig.from_thresholds(path)
