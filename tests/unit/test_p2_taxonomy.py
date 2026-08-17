"""WS-C (P2) — taxonomy: pre-registered stratified failure sampler
(arm × divergence-magnitude, worst-first, frozen order) + typed COUNTED
eligibility-exclusion accounting."""

from __future__ import annotations

import math

from evaluation.codegen.census import CensusMember, CensusResult
from evaluation.codegen.p2_metrics import MemberAgreement
from evaluation.codegen.p2_taxonomy import (
    STRATA_AXES,
    eligibility_exclusion_accounting,
    stratified_failure_sample,
)


def _ma(pid, arm, corr, stratum):
    return MemberAgreement(pid, arm, 24, corr, 0.9, False, stratum, True)


def test_strata_axes_are_arm_by_divergence_magnitude():
    assert STRATA_AXES == ("arm", "divergence_magnitude")


def test_stratified_sample_is_worst_first_and_capped():
    members = (
        _ma("a_h1", "A", 0.5, "high"),
        _ma("a_h2", "A", 0.3, "high"),
        _ma("a_h3", "A", float("nan"), "high"),   # non-finite corr = the very worst
        _ma("a_l1", "A", 1.0, "low"),
        _ma("b_m1", "B", 0.95, "medium"),
    )
    sample = stratified_failure_sample(members, per_stratum=2)
    assert sample.strata_axes == ("arm", "divergence_magnitude")
    # cells keyed by (arm, magnitude)
    assert set(sample.cells) == {("A", "high"), ("A", "low"), ("B", "medium")}
    # worst-divergence-first within (A, high): NaN, then 0.3 — capped at 2
    high_a = sample.cells[("A", "high")]
    assert [m.paper_id for m in high_a] == ["a_h3", "a_h2"]
    assert math.isnan(high_a[0].correlation)
    assert [m.paper_id for m in sample.cells[("A", "low")]] == ["a_l1"]
    assert [m.paper_id for m in sample.cells[("B", "medium")]] == ["b_m1"]


def test_stratified_sample_is_deterministic():
    members = (
        _ma("x", "A", 0.4, "high"),
        _ma("y", "A", 0.2, "high"),
    )
    a = stratified_failure_sample(members, per_stratum=5)
    b = stratified_failure_sample(members, per_stratum=5)
    assert a.to_dict() == b.to_dict()


def test_eligibility_accounting_counts_and_types_every_exclusion():
    census = CensusResult((
        CensusMember("ok", True, False, None, True, {"header": {}}),
        CensusMember("x1", False, False, None, False, None,
                     exclusion_reason="text_acquisition_failed"),
        CensusMember("x2", False, False, None, False, None,
                     exclusion_reason="text_acquisition_failed"),
        CensusMember("x3", False, False, None, False, None,
                     exclusion_reason="text_quality_below_bar"),
    ))
    acc = eligibility_exclusion_accounting(census)
    assert acc.total == 3                          # never a silent drop
    assert acc.by_reason == {"text_acquisition_failed": 2, "text_quality_below_bar": 1}
    assert acc.papers_by_reason["text_acquisition_failed"] == ["x1", "x2"]
    assert acc.papers_by_reason["text_quality_below_bar"] == ["x3"]


def test_eligibility_accounting_empty_when_all_eligible():
    census = CensusResult((
        CensusMember("ok1", True, False, None, True, {}),
        CensusMember("ok2", False, True, "refuse_no_strategy", True, {}),
    ))
    acc = eligibility_exclusion_accounting(census)
    assert acc.total == 0 and acc.by_reason == {}
