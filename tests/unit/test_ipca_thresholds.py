"""Fail-loud loaders for the Frozen-Loadings IPCA Differential constants (spec §9).

Same integrity requirement as the core auditor thresholds (design §13.2): a missing
pre-registration constant RAISES, never defaults. Additionally pins the status gate:
`load_ipca_execution_config` must refuse any run while the block is a development
contract, so a partial pre-registration can never masquerade as final.
"""

from __future__ import annotations

import textwrap

import pytest

from agents.auditor.thresholds import (
    AuditorThresholdError,
    load_ipca_execution_config,
    load_ipca_fpr_config,
    load_ipca_lambda,
    load_ipca_perturbation_config,
    load_ipca_projection_gate,
    load_ipca_reporting,
)


def _write(tmp_path, body: str):
    p = tmp_path / "thresholds.yaml"
    p.write_text(textwrap.dedent(body))
    return p


# A complete development-contract block (status: development_contract_only).
DEV = """
    auditor:
      ipca_differential:
        status: development_contract_only
        lambda:
          factor_count: 5
          k_sensitivity: [4, 6]
          instrument_count: 8
          characteristic_order: [str_reversal, mom6, var_5pct, gamma_illiq, rating, time_to_maturity, bond_vol]
          als_tolerance: 1.0e-4
          als_max_iter: 1000
          month_weighting: per_month_normalized
          scaling_lane: VOLScaled010
          vol_floor: 0.01
          normalisation_rule: kpp_r2_qr_descending_moment_meanpos
          n_initialisations: 1
          initialisation_seeds: [20260612]
          tie_break: lowest_objective_then_seed
        projection_gate:
          min_cross_section_n: 8
          require_rank: 5
          max_condition_number: 1.0e+10
          pseudoinverse_permitted: false
          pseudoinverse_tolerance: null
          max_failed_month_fraction: 0.10
          failed_month_handling: exclude
        reporting:
          n_pairs: 15
          anchors: [str, mom6, drf]
          focal_pairs: {lib_gap: str, lab_trim: mom6, meas_err: drf}
          no_focal_pair: [survivorship, stale_price]
          benchmark_overlay_exemption: true
    """


def test_committed_thresholds_file_is_valid():
    """The real docs/thresholds.yaml (default path) parses through every loader."""
    lam = load_ipca_lambda()
    assert lam.factor_count == 5
    assert lam.instrument_count == len(lam.characteristic_order) + 1 == 8
    gate = load_ipca_projection_gate()
    assert gate.require_rank == lam.factor_count
    assert gate.pseudoinverse_permitted is False and gate.pseudoinverse_tolerance is None
    rep = load_ipca_reporting()
    assert rep.n_pairs == 15 and set(rep.focal_pairs) == {"lib_gap", "lab_trim", "meas_err"}


def test_committed_fpr_and_perturbation_load():
    """The real docs/thresholds.yaml carries the §6.2/§6.3 constants."""
    fc = load_ipca_fpr_config()
    assert (fc.q_permutations, fc.r_datasets) == (49, 50)
    assert fc.alpha_nominal == 0.05 and fc.acceptance_band_level == 0.95
    assert (fc.twin_dgp.n_bonds, fc.twin_dgp.n_months) == (40, 72)
    assert (fc.reduced_q, fc.reduced_r) == (19, 20)
    pc = load_ipca_perturbation_config()
    assert pc.n_draws == 200 and pc.noise_sd == 0.02


def test_missing_fpr_block_raises(tmp_path):
    path = _write(tmp_path, DEV)                       # DEV registers no randomisation_fpr block
    with pytest.raises(AuditorThresholdError, match="randomisation_fpr"):
        load_ipca_fpr_config(path)


def test_missing_perturbation_block_raises(tmp_path):
    path = _write(tmp_path, DEV)
    with pytest.raises(AuditorThresholdError, match="perturbation_robustness"):
        load_ipca_perturbation_config(path)


def test_happy_path_reads_all(tmp_path):
    path = _write(tmp_path, DEV)
    lam = load_ipca_lambda(path)
    assert lam.k_sensitivity == (4, 6)
    assert lam.initialisation_seeds == (20260612,)
    assert lam.tie_break == "lowest_objective_then_seed"
    gate = load_ipca_projection_gate(path)
    assert gate.max_condition_number == 1.0e10
    assert gate.failed_month_handling == "exclude"
    rep = load_ipca_reporting(path)
    assert rep.anchors == ("str", "mom6", "drf")
    assert rep.no_focal_pair == ("survivorship", "stale_price")


def test_missing_block_raises(tmp_path):
    path = _write(tmp_path, "auditor:\n  primary_metric: average\n")
    with pytest.raises(AuditorThresholdError, match="ipca_differential"):
        load_ipca_lambda(path)


def test_missing_lambda_key_raises(tmp_path):
    body = DEV.replace("          factor_count: 5\n", "")
    path = _write(tmp_path, body)
    with pytest.raises(AuditorThresholdError, match="factor_count"):
        load_ipca_lambda(path)


def test_instrument_count_must_match_characteristic_order(tmp_path):
    body = DEV.replace("instrument_count: 8", "instrument_count: 9")
    path = _write(tmp_path, body)
    with pytest.raises(AuditorThresholdError, match="instrument_count"):
        load_ipca_lambda(path)


def test_invalid_failed_month_handling_raises(tmp_path):
    body = DEV.replace("failed_month_handling: exclude", "failed_month_handling: pretend")
    path = _write(tmp_path, body)
    with pytest.raises(AuditorThresholdError, match="failed_month_handling"):
        load_ipca_projection_gate(path)


def test_pseudoinverse_permitted_requires_tolerance(tmp_path):
    body = DEV.replace("pseudoinverse_permitted: false", "pseudoinverse_permitted: true")
    path = _write(tmp_path, body)
    with pytest.raises(AuditorThresholdError, match="pseudoinverse_tolerance"):
        load_ipca_projection_gate(path)


def test_reporting_n_pairs_must_match_grid(tmp_path):
    body = DEV.replace("n_pairs: 15", "n_pairs: 12")
    path = _write(tmp_path, body)
    with pytest.raises(AuditorThresholdError, match="n_pairs"):
        load_ipca_reporting(path)


def test_execution_config_refuses_development_contract(tmp_path):
    """The status gate: a development contract can never launch an extension run."""
    path = _write(tmp_path, DEV)
    with pytest.raises(AuditorThresholdError, match="not 'preregistered_complete'"):
        load_ipca_execution_config(path)


def test_execution_config_refuses_complete_status_without_deferred_blocks(tmp_path):
    """Flipping status to complete is not enough — the deferred blocks must exist."""
    body = DEV.replace(
        "status: development_contract_only", "status: preregistered_complete"
    )
    path = _write(tmp_path, body)
    with pytest.raises(AuditorThresholdError):
        load_ipca_execution_config(path)


# DEV with status flipped to complete AND the deferred §9 blocks registered (raw 8/10
# space indent to match DEV's siblings-of-status; dedent trims the common 4).
COMPLETE = DEV.replace(
    "status: development_contract_only",
    "status: preregistered_complete\n"
    "        bootstrap:\n"
    "          n_replicates: 500\n"
    "          block_length_months: 6\n"
    "          min_effective_blocks: 10\n"
    "          holding_period_default: 1\n"
    "          alpha: 0.05\n"
    "          conditioning_label: cond\n"
    "        stability_diagnostic:\n"
    "          r_prime: 25\n"
    "          block_length_months: 6\n"
    "          min_effective_blocks: 10\n"
    "          holding_period_default: 1\n"
    "          order_of_magnitude_factor: 10.0\n"
    "          scope: [meas_err, stale_price, survivorship, lib_gap, lab_trim]\n"
    "        randomisation_fpr:\n"
    "          q_permutations: 49\n"
    "          r_datasets: 50\n"
    "          alpha_nominal: 0.05\n"
    "          acceptance_band_level: 0.95\n"
    "          twin_dgp:\n"
    "            n_bonds: 40\n"
    "            n_months: 72\n"
    "            noise_sd: 0.02\n"
    "          reduced_q: 19\n"
    "          reduced_r: 20\n"
    "          rename_fallback_label: rename\n"
    "        perturbation_robustness:\n"
    "          n_draws: 200\n"
    "          noise_sd: 0.02",
)


def test_execution_config_succeeds_when_complete(tmp_path):
    path = _write(tmp_path, COMPLETE)
    cfg = load_ipca_execution_config(path)
    assert cfg.lam.factor_count == 5
    assert cfg.projection_gate.require_rank == 5
    assert cfg.reporting.n_pairs == 15
    # the gate now returns the fully-loaded blocks the run consumes (not just key presence)
    assert cfg.bootstrap.n_replicates == 500
    assert cfg.stability.r_prime == 25
    assert cfg.fpr.q_permutations == 49 and cfg.fpr.twin_dgp.n_bonds == 40
    assert cfg.perturbation.n_draws == 200


def test_execution_config_rejects_malformed_fpr_block(tmp_path):
    """M1 regression: a status flip with a present-but-MALFORMED FPR block (wrong keys) must fail
    the gate here, not silently certify a run that breaks at run time."""
    malformed = COMPLETE.replace(
        "          q_permutations: 49\n", "          q: 49\n"      # wrong key name
    )
    path = _write(tmp_path, malformed)
    with pytest.raises(AuditorThresholdError, match="q_permutations"):
        load_ipca_execution_config(path)
