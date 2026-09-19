"""WS-C (P2) — below-floor coverage-boundary assembly. Offline, no LLM, no disk.

Asserts the zero-spend close-out: a below-floor selection produces suppressed metrics,
a typed eligibility accounting, a report that carries the §6 caveat verbatim and NO
agreement/correlation/Sharpe number, and an explicit 0 model calls / $0.00. Also covers
the reportable derivation and the above-floor guard.
"""

from __future__ import annotations

import pytest

from evaluation.codegen.census import CensusInput, RoutingDecision, run_census
from evaluation.codegen.p2_boundary import assemble_below_floor, derive_reportable
from evaluation.codegen.p2_report import AGREEMENT_CAVEAT, _blockquote
from evaluation.codegen.p2_selector import select_arms

_TH = {
    "arms": {"arm_b_max": 5, "below_floor_min_arm_a": 5},
    "agreement": {"correlation_min": 0.99, "sign_agreement_min": 0.95},
    "divergence_strata": {"high_divergence_corr_lt": 0.90, "medium_divergence_corr_lt": 0.99},
    "taxonomy_sampling": {"strata": ["arm", "divergence_magnitude"]},
    "zoo_list": {"frozen_sha256": "TO_SET"},
    "min_overlap_months": 3,
}


def _below_floor_census(n_refuse=2, n_compilable=3, n_excluded=27):
    """A census with |Arm A| = n_refuse (< floor of 5): refusals, compilables, and typed
    eligibility exclusions (the DFPS-style review-exit shape)."""
    inputs = (
        [CensusInput(f"r{i}", {"header": {}}, True) for i in range(n_refuse)]
        + [CensusInput(f"c{i}", {"header": {}}, True) for i in range(n_compilable)]
        + [CensusInput(f"x{i}", None, False,
                       exclusion_reason="extraction_review_exit_no_spec")
           for i in range(n_excluded)]
    )

    def route(inp):
        return (RoutingDecision(False, True, "refuse_asset_class")
                if inp.paper_id.startswith("r") else RoutingDecision(True, False))

    census = run_census(inputs, route)
    zoo = [m.paper_id for m in census.members]
    return census, select_arms(census, zoo, _TH)


def _prov(n=2):
    return {f"r{i}": {"manifest_phase": "report", "run_dir": "runs/x", "spec_file": "s.json",
                      "spec_sha256": "0" * 64, "code_commit": "fixture-commit"} for i in range(n)}


# --- the zero-spend below-floor close-out -------------------------------------------

def test_below_floor_assembly_is_zero_spend_and_suppressed():
    census, sel = _below_floor_census()
    assert sel.below_floor is True and sel.arm_a_size == 2
    result, md = assemble_below_floor(
        census, sel, _TH, provenance=_prov(2), corpus_status="frozen", zoo_sha_ok=True)
    assert result["suppressed"] is True and result["below_floor"] is True
    assert result["model_calls"] == 0 and result["spend_usd"] == 0.0
    assert result["observed_dispositions"] == {
        "compilable": 3, "refused": 2, "eligibility_excluded": 27}
    # eligibility exclusions are COUNTED + typed, never dropped
    assert result["eligibility_accounting"]["total"] == 27
    assert result["eligibility_accounting"]["by_reason"] == {"extraction_review_exit_no_spec": 27}
    assert len(result["census_fate_table"]) == 32


def test_below_floor_report_carries_caveat_and_no_numbers():
    census, sel = _below_floor_census()
    _, md = assemble_below_floor(
        census, sel, _TH, provenance=_prov(2), corpus_status="frozen", zoo_sha_ok=True)
    # (1) §6 caveat verbatim
    assert _blockquote(AGREEMENT_CAVEAT) in md
    assert "Inter-model agreement is not correctness." in md
    # (2) no agreement/correlation/Sharpe NUMBER-bearing section
    assert "correlation distribution (PRIMARY)" not in md
    assert "Binary agreement rate (SECONDARY)" not in md
    assert "sharpe" not in md.lower()
    # (3) the suppression + the exclusion total ARE present
    assert "SUPPRESSED" in md
    assert "Empty by construction" in md            # §6(a) taxonomy
    assert "**27**" in md                           # eligibility total


# --- reportable derivation ----------------------------------------------------------

def test_reportable_true_only_when_report_frozen_and_sha_ok():
    assert derive_reportable(_prov(2), "frozen", True) is True


@pytest.mark.parametrize("prov,status,sha_ok", [
    ({}, "frozen", True),                                  # no real spec at all
    (_prov(2), "draft_pending_review", True),       # selection not frozen
    (_prov(2), "frozen", False),                           # zoo hash mismatch
])
def test_reportable_false_when_any_precondition_fails(prov, status, sha_ok):
    assert derive_reportable(prov, status, sha_ok) is False


def test_a_non_report_phase_provenance_is_not_reportable():
    bad = {"r0": {"manifest_phase": "dev"}}
    assert derive_reportable(bad, "frozen", True) is False


# --- guard: the above-floor path must not be assembled here -------------------------

def test_assemble_refuses_above_floor():
    census, sel = _below_floor_census(n_refuse=6, n_compilable=6, n_excluded=0)
    assert sel.below_floor is False
    with pytest.raises(ValueError, match="ABOVE-floor"):
        assemble_below_floor(census, sel, _TH, provenance=_prov(6),
                             corpus_status="frozen", zoo_sha_ok=True)
