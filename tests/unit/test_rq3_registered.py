"""The registered RQ3 measurements: prevalence denominators,
completion rates in both readings, and the per-correction OFF-state check. Offline, no engine."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation.rq3_registered.completion import (  # noqa: E402
    ArmCompletion,
    CompletionInputError,
    completion_block,
    ipca_arm,
    lattice_arm,
)
from evaluation.rq3_registered.off_state import (  # noqa: E402
    OffStateInputError,
    off_state_block,
)
from evaluation.rq3_registered.prevalence import (  # noqa: E402
    PrevalenceInputError,
    collect_inputs,
    compute_prevalence,
    corpus_from_inputs,
    variance_from_interval,
)


def _report(runnable, inference, invariance=(), scope="COMPLETE", not_applicable=()):
    return {
        "audit_scope": scope,
        "runnable_toggles": list(runnable),
        "not_applicable_toggles": list(not_applicable),
        "inference": inference,
        "invariance": list(invariance),
    }


def _cell(point, half_width, inert=False):
    return {"point": point, "ci_low": point - half_width, "ci_high": point + half_width,
            "inert": inert}


# --- prevalence denominators (the registered rules) -----------------------------

def test_variance_comes_from_the_reported_interval():
    # a 95% interval of +-1.96*se, against the module's exact z (1.95996...)
    assert variance_from_interval(-0.0196, 0.0196) == pytest.approx(1e-4, rel=1e-4)


def test_a_degenerate_interval_is_zero_variance_not_an_error():
    assert variance_from_interval(0.0, 0.0) == 0.0


def test_an_inverted_interval_fails_loud():
    with pytest.raises(PrevalenceInputError):
        variance_from_interval(0.01, -0.01)


def test_a_non_applicable_coordinate_is_absent_from_its_strategy():
    """D-A31: it is never counted — not in the numerator, not in the denominator."""
    reports = {
        "str": _report(["lib_gap"], {"lib_gap": _cell(0.006, 0.002)},
                       not_applicable=["meas_err"]),
        "drf": _report(["lib_gap", "meas_err"],
                       {"lib_gap": _cell(0.001, 0.002), "meas_err": _cell(0.002, 0.001)}),
    }
    corpus = corpus_from_inputs(collect_inputs(reports))
    assert sorted(corpus) == ["lib_gap", "meas_err"]
    assert [e.strategy_label for e in corpus["meas_err"]] == ["drf"]     # str absent
    assert len(corpus["lib_gap"]) == 2


def test_a_proven_no_op_stays_in_the_denominator():
    """A no-op contributes 0 to the numerator and 1 to the denominator — never an exclusion."""
    reports = {
        "str": _report(["stale_price"], {"stale_price": _cell(0.0, 0.0, inert=True)},
                       invariance=[{"toggle_id": "stale_price", "is_no_op": True,
                                    "returns_identical": True, "config_hashes_differ": True}]),
        "drf": _report(["stale_price"], {"stale_price": _cell(0.0, 0.0, inert=True)},
                       invariance=[{"toggle_id": "stale_price", "is_no_op": True,
                                    "returns_identical": True, "config_hashes_differ": True}]),
    }
    out = compute_prevalence(reports, vartheta=0.001, n_iter=200, burn=50)
    block = out["coordinates"]["stale_price"]
    assert block["n_runnable"] == 2          # both counted
    assert block["n_susceptible"] == 0       # neither susceptible
    assert block["prevalence"] == 0.0        # numerator zero, denominator two


def test_an_incomplete_audit_is_refused_not_folded_in():
    reports = {"str": _report(["lib_gap"], {"lib_gap": _cell(0.006, 0.002)}, scope="PARTIAL")}
    with pytest.raises(PrevalenceInputError, match="COMPLETE"):
        collect_inputs(reports)


def test_a_runnable_toggle_without_an_inference_cell_fails_loud():
    with pytest.raises(PrevalenceInputError, match="no inference cell"):
        collect_inputs({"str": _report(["lib_gap"], {})})


def test_prevalence_rises_with_a_larger_effect():
    def one(point):
        return {"str": _report(["lib_gap"], {"lib_gap": _cell(point, 0.0005)})}
    small = compute_prevalence(one(0.0002), vartheta=0.001, n_iter=400, burn=100)
    large = compute_prevalence(one(0.02), vartheta=0.001, n_iter=400, burn=100)
    assert (large["coordinates"]["lib_gap"]["prevalence"]
            > small["coordinates"]["lib_gap"]["prevalence"])


# --- completion rates -----------------------------------------------------------

def test_the_two_readings_differ_exactly_where_a_refusal_exists():
    arm = ArmCompletion(arm="ipca", unit="cell pairs", completed=9, refused=6)
    assert arm.conservative == pytest.approx(9 / 15)
    assert arm.permissive == 1.0


def test_a_complete_arm_agrees_on_both_readings():
    arm = ArmCompletion(arm="lattice", unit="audits", completed=3, refused=0)
    assert arm.conservative == 1.0 and arm.permissive == 1.0


def test_not_applicable_moves_neither_rate():
    with_na = ArmCompletion(arm="a", unit="audits", completed=3, refused=0, not_applicable=1)
    without = ArmCompletion(arm="a", unit="audits", completed=3, refused=0)
    assert with_na.conservative == without.conservative
    assert with_na.permissive == without.permissive


def test_an_empty_arm_has_no_rate_to_take():
    with pytest.raises(CompletionInputError):
        ArmCompletion(arm="a", unit="audits", completed=0, refused=0)


def test_the_block_refuses_to_pool_the_arms():
    block = completion_block([
        lattice_arm({"str": _report([], {}), "drf": _report([], {})}),
        ipca_arm(9, 6, refusal_code="FEED_OFF_STATE_UNDEFINED"),
    ])
    assert "NOT POOLED" in block["pooling"]
    assert {a["arm"] for a in block["arms"]} == {"correction_lattice", "ipca_differential"}
    assert all("conservative" in a and "permissive" in a for a in block["arms"])


def test_lattice_arm_counts_an_incomplete_audit_as_refused():
    arm = lattice_arm({"str": _report([], {}), "drf": _report([], {}, scope="REFUSED")})
    assert arm.completed == 1 and arm.refused == 1


# --- OFF state ------------------------------------------------------------------

def test_each_correction_reports_its_documented_off_state():
    reports = {"mom6": _report(
        ["meas_err", "lab_trim", "survivorship"],
        {"meas_err": _cell(0.01, 0.002), "lab_trim": _cell(0.01, 0.002),
         "survivorship": _cell(0.01, 0.002)})}
    signatures = {"mom6": {"meas_err_off_family": "jostova_2013",
                           "expost_trim_off": {"method": "truncate"}}}
    block = off_state_block(reports, signatures)

    assert block["n_checked"] == 3 and block["all_documented"] is True
    by_correction = {c["correction"]: c for c in block["checks"]}
    assert by_correction["meas_err"]["off_state"] == "jostova_2013"
    assert by_correction["meas_err"]["source"] == "recorded_signature"
    assert by_correction["survivorship"]["source"] == "as_published_default"
    # the check states what it cannot establish
    assert "configuration-level only" in block["scope_limit"]


def test_a_missing_recorded_signature_fails_loud():
    reports = {"mom6": _report(["meas_err"], {"meas_err": _cell(0.01, 0.002)})}
    with pytest.raises(OffStateInputError, match="recorded signature"):
        off_state_block(reports, {"mom6": {}})


def test_a_strategy_with_no_signature_block_fails_loud():
    reports = {"mom6": _report(["survivorship"], {"survivorship": _cell(0.01, 0.002)})}
    with pytest.raises(OffStateInputError, match="baseline signature"):
        off_state_block(reports, {})


def test_a_proved_no_op_is_listed_with_its_invariance_evidence():
    reports = {"str": _report(
        ["stale_price"], {"stale_price": _cell(0.0, 0.0, inert=True)},
        invariance=[{"toggle_id": "stale_price", "is_no_op": True,
                     "returns_identical": True, "config_hashes_differ": True}])}
    block = off_state_block(reports, {"str": {}})
    assert block["no_ops_proved_inert"] == ["str:stale_price"]
