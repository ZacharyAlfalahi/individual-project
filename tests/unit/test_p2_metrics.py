"""WS-C (P2) — metrics: correlation distribution (primary), agreement rate
(secondary), divergence strata, below-floor suppression, the MDE power guard,
and the fail-loud thresholds loader."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from evaluation.codegen.census import CensusMember, CensusResult
from evaluation.codegen.p2_metrics import (
    MemberAgreement,
    agreement_distribution,
    below_floor,
    codegen_vs_compiler_divergence,
    compute_p2_metrics,
    load_p2_scoring_thresholds,
    mde_by_arm_size,
    member_agreement,
    stratum_of,
)
from evaluation.codegen.p2_selector import select_arms

_TH = {
    "arms": {"arm_b_max": 5, "below_floor_min_arm_a": 5},
    "agreement": {"correlation_min": 0.99, "sign_agreement_min": 0.95},
    "divergence_strata": {"high_divergence_corr_lt": 0.90, "medium_divergence_corr_lt": 0.99},
    "taxonomy_sampling": {"strata": ["arm", "divergence_magnitude"]},
    "zoo_list": {"frozen_sha256": "TO_SET"},
    "min_overlap_months": 3,
}


def _series(values, start="2010-01-31"):
    idx = pd.date_range(start, periods=len(values), freq="ME")
    return pd.Series(values, index=idx, name="portfolio_return")


def _refusal(pid):
    return CensusMember(pid, False, True, "refuse_no_strategy", True,
                        {"header": {"paper_id": pid}})


def _compilable(pid):
    return CensusMember(pid, True, False, None, True, {"header": {"paper_id": pid}})


# --- thresholds loader ----------------------------------------------------------

def test_real_p2_block_loads_and_is_complete():
    block = load_p2_scoring_thresholds()
    assert block["arms"]["arm_b_max"] == 5
    assert block["arms"]["below_floor_min_arm_a"] == 5
    assert block["agreement"]["correlation_min"] == 0.99
    assert block["divergence_strata"]["high_divergence_corr_lt"] == 0.90
    assert block["min_overlap_months"] == 24


def test_thresholds_loader_fails_loud_on_missing_block(tmp_path):
    bad = tmp_path / "thresholds.yaml"
    bad.write_text("other_block: {}\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_p2_scoring_thresholds(bad)


def test_thresholds_loader_fails_loud_on_missing_subkey(tmp_path):
    bad = tmp_path / "thresholds.yaml"
    bad.write_text("p2_codegen:\n  arms: {arm_b_max: 5}\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_p2_scoring_thresholds(bad)


# --- strata + per-member agreement ----------------------------------------------

def test_stratum_bucketing():
    assert stratum_of(0.995, _TH) == "low"       # corr >= 0.99
    assert stratum_of(0.95, _TH) == "medium"     # 0.90 <= corr < 0.99
    assert stratum_of(0.5, _TH) == "high"        # corr < 0.90
    assert stratum_of(float("nan"), _TH) == "high"   # degenerate is the worst


def test_member_agreement_identical_series_agrees_low_divergence():
    s = _series([0.01, -0.02, 0.03, 0.02])
    m = member_agreement("p", "A", s, s.copy(), _TH)
    assert m.correlation == pytest.approx(1.0)
    assert m.sign_agreement == 1.0
    assert m.agrees is True and m.stratum == "low"
    assert m.sufficient_overlap is True


def test_member_agreement_degenerate_candidate_does_not_agree():
    a = _series([0.01, -0.02, 0.03, 0.02])
    b = _series([0.005, 0.005, 0.005, 0.005])    # constant -> corr NaN
    m = member_agreement("p", "A", a, b, _TH)
    assert math.isnan(m.correlation)
    assert m.agrees is False and m.stratum == "high"


def test_member_agreement_insufficient_overlap_not_counted():
    a = _series([0.01, -0.02])                    # only 2 months, min is 3
    m = member_agreement("p", "A", a, a.copy(), _TH)
    assert m.sufficient_overlap is False and m.agrees is False


# --- distribution + agreement rate ----------------------------------------------

def test_agreement_distribution_rate_and_strata():
    members = (
        MemberAgreement("a1", "A", 24, 1.0, 1.0, True, "low", True),
        MemberAgreement("a2", "A", 24, 0.5, 0.8, False, "high", True),
        MemberAgreement("b1", "B", 24, 0.995, 0.96, True, "low", True),
        MemberAgreement("b2", "B", 2, float("nan"), 0.5, False, "high", False),
    )
    dist = agreement_distribution(members)
    assert dist["n_members"] == 4
    assert dist["n_scored"] == 3                  # b2 excluded (insufficient overlap)
    assert dist["n_agree"] == 2
    assert dist["agreement_rate"] == pytest.approx(2 / 3)
    assert dist["n_insufficient_overlap"] == 1
    assert dist["strata_counts"] == {"high": 2, "medium": 0, "low": 2}
    assert dist["n_finite_correlations"] == 3


# --- codegen-vs-compiler divergence (neither side is truth) ---------------------

def test_codegen_vs_compiler_is_labelled_neither_side_is_truth():
    codegen = _series([0.01, -0.02, 0.03, 0.02])
    compiler = _series([0.012, -0.018, 0.031, 0.019])
    d = codegen_vs_compiler_divergence("p", "model_a", codegen, compiler)
    assert d.label == "neither side is truth"
    assert d.metrics["n_overlap"] == 4


# --- below-floor + MDE power guard ----------------------------------------------

def test_below_floor_rule():
    assert below_floor(4, _TH) is True
    assert below_floor(5, _TH) is False


def test_mde_by_arm_size_is_a_power_guard():
    mde = mde_by_arm_size((3, 5, 50, 500))
    # tiny arms are near-useless: n=5 needs ~0.89 difference to detect anything
    assert mde[5]["mde"] == pytest.approx(0.886, abs=0.01)
    assert mde[5]["detectable"] is True          # 0.886 <= 1.0
    assert mde[3]["mde"] > 1.0                    # nothing detectable at n=3
    assert mde[3]["detectable"] is False
    # MDE strictly decreases as n grows (more power)
    assert mde[5]["mde"] > mde[50]["mde"] > mde[500]["mde"]


# --- compute_p2_metrics end-to-end ----------------------------------------------

def test_compute_metrics_below_floor_suppresses_numbers():
    census = CensusResult(tuple(
        [_refusal(f"r{i}") for i in range(3)]
        + [_compilable(f"c{i}") for i in range(6)]
    ))
    zoo = [f"r{i}" for i in range(3)] + [f"c{i}" for i in range(6)]
    sel = select_arms(census, zoo, _TH)
    metrics = compute_p2_metrics(census, sel, {}, None, _TH)
    assert metrics.suppressed is True
    assert metrics.agreements == () and metrics.distribution == {}
    assert metrics.divergences == () and metrics.mde == {}
    assert len(metrics.census_fate_table) == 9   # fate table still emitted


def test_compute_metrics_above_floor_emits_distribution_and_divergence():
    census = CensusResult(tuple(
        [_refusal(f"r{i}") for i in range(5)]
        + [_compilable(f"c{i}") for i in range(5)]
    ))
    zoo = [f"r{i}" for i in range(5)] + [f"c{i}" for i in range(5)]
    sel = select_arms(census, zoo, _TH)
    assert sel.below_floor is False and sel.arm_b == tuple(f"c{i}" for i in range(5))

    base = _series([0.01, -0.02, 0.03, 0.02, 0.015])
    pairs = {pid: (base, base.copy()) for pid in sel.members()}   # identical -> agree
    compiler = {pid: base + 0.001 for pid in sel.arm_b}           # slight divergence

    metrics = compute_p2_metrics(census, sel, pairs, compiler, _TH,
                                 model_ids=("model_a", "model_b"))
    assert metrics.suppressed is False
    assert len(metrics.agreements) == 10                          # 5 A + 5 B
    assert metrics.distribution["agreement_rate"] == pytest.approx(1.0)
    # divergence: 5 Arm-B members × 2 models, all "neither side is truth"
    assert len(metrics.divergences) == 10
    assert all(d.label == "neither side is truth" for d in metrics.divergences)
    assert 5 in metrics.mde


def test_compute_metrics_missing_series_pair_fails_loud():
    census = CensusResult(tuple([_refusal(f"r{i}") for i in range(5)]
                                + [_compilable("c0")]))
    zoo = [f"r{i}" for i in range(5)] + ["c0"]
    sel = select_arms(census, zoo, _TH)
    with pytest.raises(KeyError):
        compute_p2_metrics(census, sel, {}, None, _TH)   # no pairs supplied


# --- the authorised below-floor departure ---------------------------------------

def _below_floor_selection():
    census = CensusResult(tuple(
        [_refusal(f"r{i}") for i in range(2)] + [_compilable(f"c{i}") for i in range(3)]
    ))
    zoo = [f"r{i}" for i in range(2)] + [f"c{i}" for i in range(3)]
    sel = select_arms(census, zoo, _TH)
    assert sel.below_floor is True
    return census, sel


def test_floor_override_computes_below_floor_and_labels_it_descriptive():
    census, sel = _below_floor_selection()
    base = _series([0.01, -0.02, 0.03, 0.02, 0.015])
    pairs = {pid: (base, base.copy()) for pid in sel.members()}

    metrics = compute_p2_metrics(census, sel, pairs, None, _TH, floor_override="DEP-1")

    assert metrics.suppressed is False           # numbers ARE emitted
    assert metrics.floor_override == "DEP-1"
    assert len(metrics.agreements) == 4          # 2 Arm A + 2 Arm B
    # the suppression reason is retained and marked, so a below-floor number can never be
    # read as a registered one
    assert "below_floor_min_arm_a" in metrics.below_floor_reason
    assert "DEP-1" in metrics.below_floor_reason
    assert "descriptive" in metrics.below_floor_reason
    assert metrics.to_dict()["floor_override"] == "DEP-1"


def test_without_the_override_the_same_selection_is_suppressed():
    census, sel = _below_floor_selection()
    base = _series([0.01, -0.02, 0.03, 0.02, 0.015])
    pairs = {pid: (base, base.copy()) for pid in sel.members()}

    metrics = compute_p2_metrics(census, sel, pairs, None, _TH)

    assert metrics.suppressed is True and metrics.agreements == ()
    assert metrics.floor_override is None


def test_override_is_refused_above_the_floor():
    census = CensusResult(tuple([_refusal(f"r{i}") for i in range(5)]
                                + [_compilable(f"c{i}") for i in range(5)]))
    zoo = [f"r{i}" for i in range(5)] + [f"c{i}" for i in range(5)]
    sel = select_arms(census, zoo, _TH)
    base = _series([0.01, -0.02, 0.03, 0.02, 0.015])
    pairs = {pid: (base, base.copy()) for pid in sel.members()}
    with pytest.raises(ValueError, match="ABOVE-floor"):
        compute_p2_metrics(census, sel, pairs, None, _TH, floor_override="DEP-1")


def test_override_must_name_a_departure():
    census, sel = _below_floor_selection()
    with pytest.raises(ValueError, match="non-empty"):
        compute_p2_metrics(census, sel, {}, None, _TH, floor_override="   ")
