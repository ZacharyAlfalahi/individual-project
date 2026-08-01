"""
Researcher foundation: the filter-first eligibility filter (§6) and the context-builder wall
(INVARIANT 1). Uses the real library + templates + variable_families (authored backwards from
the census), so eligibility is exercised against genuine entries.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.scientist.researcher import context_builder as cb  # noqa: E402
from agents.scientist.researcher.eligibility import eligible_mechanisms, evaluate  # noqa: E402
from agents.scientist.researcher.library import load_library  # noqa: E402
from agents.scientist.schemas.case import (  # noqa: E402
    DevelopmentWindow,
    HoldoutStatus,
    ScientistCase,
)

LIB = load_library()
# Every conditioning variable the library needs, treated as available (the real data has them;
# this keeps the filter test independent of a data read).
AVAILABLE = {
    "baa_aaa_spread", "vix", "term_spread",
    "rating", "investment_grade", "var_5pct", "gamma_illiq", "bond_vol", "size", "time_to_maturity",
}


def _case() -> ScientistCase:
    return ScientistCase(
        case_id="case_str_lib_gap",
        strategy_id="str",
        corrected_quant_config_ref="qc://corrected/str",
        corrected_run_ref="run://corrected/str",
        audit_report_ref="audit://str",
        failed_check_ids=("lib_gap",),
        failed_check_verdicts={"lib_gap": "FAIL"},
        applicable_toggles=("lib_gap", "lab_trim"),
        development_window=DevelopmentWindow(start="2004-08", end="2021-12"),
        holdout_status=HoldoutStatus(accessible=False),
    )


# ---- version-lock -------------------------------------------------------------------------

def test_library_version_hash_is_stable():
    assert load_library().version_hash == LIB.version_hash        # deterministic
    assert len(LIB.version_hash) == 64                            # sha256 hex


# ---- eligibility (filter-first) -----------------------------------------------------------

def test_characteristic_sort_meets_the_distinct_mechanism_gate():
    elig = eligible_mechanisms(
        LIB.mechanisms, strategy_family="CHARACTERISTIC_SORT", templates=LIB.templates,
        variable_families=LIB.variable_families, available_variables=AVAILABLE,
    )
    # >= 8 conceptually-distinct mechanisms (spec §6 gate) — measured on distinct titles.
    titles = {m["title"] for m in elig}
    assert len(titles) >= 8, f"only {len(titles)} distinct eligible mechanisms"


def test_wrong_family_is_ineligible():
    # mech_007 (nonlinear characteristic interactions) is CHARACTERISTIC_SORT only.
    m = LIB.mechanism("mech_007")
    r = evaluate(m, strategy_family="IPCA", templates=LIB.templates,
                 variable_families=LIB.variable_families, available_variables=AVAILABLE)
    assert not r.eligible
    assert any("not supported" in reason for reason in r.reasons)


def test_missing_variable_makes_ineligible_with_reason():
    # Remove gamma_illiq from availability -> mech_008 (needs liquidity_measure) becomes ineligible.
    avail = AVAILABLE - {"gamma_illiq"}
    r = evaluate(LIB.mechanism("mech_008"), strategy_family="CHARACTERISTIC_SORT",
                 templates=LIB.templates, variable_families=LIB.variable_families,
                 available_variables=avail)
    assert not r.eligible
    assert any("liquidity_measure" in reason for reason in r.reasons)


def test_eligible_mechanism_has_reachable_options():
    r = evaluate(LIB.mechanism("mech_003"), strategy_family="CHARACTERISTIC_SORT",
                 templates=LIB.templates, variable_families=LIB.variable_families,
                 available_variables=AVAILABLE)
    assert r.eligible
    # credit_regime_indicator -> baa_aaa_spread reachable via T1 and/or T3.
    opts = r.reachable["credit_regime_indicator"]
    assert opts and all(v == "baa_aaa_spread" for _, v in opts)


# ---- the wall (INVARIANT 1) ---------------------------------------------------------------

def test_context_contains_only_allow_listed_fields_and_no_magnitudes():
    case = _case()
    results = [
        evaluate(m, strategy_family="CHARACTERISTIC_SORT", templates=LIB.templates,
                 variable_families=LIB.variable_families, available_variables=AVAILABLE)
        for m in LIB.mechanisms
    ]
    ctx = cb.build_context(case, results, LIB)
    assert set(ctx) <= set(cb.ALLOWED_RESEARCHER_FIELDS)
    assert ctx["failed_check_verdicts"] == {"lib_gap": "FAIL"}    # verdicts, not magnitudes
    assert ctx["mechanism_documents"], "eligible mechanisms should populate the context"


def test_wall_rejects_an_injected_magnitude():
    # Durable-red: a magnitude-bearing key anywhere in the context must fail the wall assertion.
    poisoned = {"strategy_spec_economic": {"sharpe": 1.3}}   # 'sharpe' key + float value
    with pytest.raises(AssertionError):
        cb._assert_no_magnitudes(poisoned)


def test_wall_rejects_non_allow_listed_field():
    with pytest.raises(AssertionError):
        cb._assert_no_magnitudes({"corrected_run_ref": "leaked"})   # not on the allow-list


def test_wall_rejects_near_miss_magnitude_key():
    # M1: substring key match — 'net_sharpe' / 'annual_alpha' must be caught, not only 'sharpe'.
    for k in ("net_sharpe", "annual_alpha", "information_ratio", "tstat"):
        with pytest.raises(AssertionError):
            cb._assert_no_magnitudes({"strategy_spec_economic": {k: "x"}})


def test_wall_rejects_numeric_value_under_innocuous_key_in_spec_economic():
    # M1: a magnitude carried as a VALUE under an innocuous key must not slip through.
    with pytest.raises(AssertionError):
        cb._assert_no_magnitudes({"strategy_spec_economic": {"note": 0.083}})


def test_wall_rejects_float_value_anywhere():
    # M1: any float in the (magnitude-free, config-int-only) context is a suspected magnitude.
    with pytest.raises(AssertionError):
        cb._assert_no_magnitudes({"toggle_definitions": {"lib_gap": 1.7}})


def test_wall_rejects_prose_magnitude_but_allows_config_prose():
    # M1: 'sharpe of 1.7' in prose is caught; a config descriptor with numbers is fine.
    with pytest.raises(AssertionError):
        cb._assert_no_magnitudes({"strategy_spec_economic": {"d": "achieved a sharpe of 1.7"}})
    cb._assert_no_magnitudes(
        {"strategy_spec_economic": {"signal": "6-month momentum, 10 deciles, 36-month window"}}
    )  # config numbers, no performance token -> allowed (must not raise)
