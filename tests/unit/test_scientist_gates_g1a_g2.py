"""G1a compilation (panel-transform / native double-sort, F8) and G2 audit-clean (ex-ante
provenance + month-filter lag). Includes the synthetic refusal cases: full-sample threshold ->
G2 fires; contemporaneous regime -> G2 fires; double-sort held>1 -> G1a fails."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.scientist.experimentalist.audit_checks import audit_g2, load_reporting_delays  # noqa: E402
from agents.scientist.experimentalist.compiler import CompiledExtension, PanelTransform, compile_g1a  # noqa: E402
from agents.scientist.researcher.library import load_library  # noqa: E402
from agents.scientist.schemas.outcomes import RefusalCode  # noqa: E402
from agents.scientist.schemas.proposal import decode_proposal  # noqa: E402
from agents.scientist.schemas.case import DevelopmentWindow, HoldoutStatus, ScientistCase  # noqa: E402

LIB = load_library()
AVAILABLE = {"baa_aaa_spread", "vix", "term_spread", "rating", "investment_grade", "var_5pct",
             "gamma_illiq", "bond_vol", "size", "time_to_maturity"}
DELAYS = load_reporting_delays()
CASE = ScientistCase(
    case_id="case_x", strategy_id="str", corrected_quant_config_ref="qc://corrected/str",
    corrected_run_ref="run://c", audit_report_ref="a://c", failed_check_ids=("lib_gap",),
    failed_check_verdicts={"lib_gap": "FAIL"}, applicable_toggles=("lib_gap", "lab_trim"),
    development_window=DevelopmentWindow(start="2004-08", end="2021-12"),
    holdout_status=HoldoutStatus(accessible=False),
)


def _proposal(mechanism, template, variable, lag, form):
    raw = {
        "proposal_id": "p1", "case_id": "case_x", "parent_strategy_id": "str",
        "mechanism_ref": mechanism, "template_ref": template, "rationale": "x", "prediction": "y",
        "config_delta": {"conditioning_variable": variable, "conditioning_lag_months": lag,
                         "interaction_form": form},
        "required_inputs": [variable],
        "generation": {"source": "random_eligible", "seed": 0, "model": "d", "prompt_version": "v0",
                       "library_version": "x", "generated_at": "2026-08-02T00:00:00Z"},
    }
    dec = decode_proposal(raw)
    assert dec.ok, dec.error
    return dec.proposal


# ---- G1a --------------------------------------------------------------------------------

def test_g1a_t4_double_sort_compiles_at_holding1():
    p = _proposal("mech_007", "signal_characteristic_interaction_v1", "var_5pct", 1,
                  "double_sort_independent")
    out, ext = compile_g1a(p, LIB.templates["signal_characteristic_interaction_v1"], CASE,
                           holding_period=1, available_variables=AVAILABLE)
    assert out.passed and out.booleans["compiled"]
    assert ext.template_mode == "double_sort" and ext.control == "var_5pct"


def test_g1a_t4_double_sort_refused_at_holding6():
    p = _proposal("mech_007", "signal_characteristic_interaction_v1", "var_5pct", 1,
                  "double_sort_independent")
    out, ext = compile_g1a(p, LIB.templates["signal_characteristic_interaction_v1"], CASE,
                           holding_period=6, available_variables=AVAILABLE)
    assert not out.passed and out.refusal_code is RefusalCode.COMPILATION_FAILED and ext is None


def test_g1a_t1_compiles_to_month_filter():
    p = _proposal("mech_003", "lagged_binary_regime_interaction_v1", "baa_aaa_spread", 1,
                  "binary_above_historical_median")
    out, ext = compile_g1a(p, LIB.templates["lagged_binary_regime_interaction_v1"], CASE,
                           holding_period=6, available_variables=AVAILABLE)
    assert out.passed and ext.panel_transform.kind == "month_filter"    # runs for any holding


def test_g1a_t3_compiles_to_row_filter():
    p = _proposal("mech_008", "ex_ante_universe_conditioning_v1", "gamma_illiq", 1,
                  "restrict_top_liquidity_tercile")
    out, ext = compile_g1a(p, LIB.templates["ex_ante_universe_conditioning_v1"], CASE,
                           holding_period=1, available_variables=AVAILABLE)
    assert out.passed and ext.panel_transform.kind == "row_filter"


# ---- G2 --------------------------------------------------------------------------------

def _ext(mode, rule, transform=None, control=None):
    return CompiledExtension("p1", "str", mode, rule, "qc://corrected/str",
                             control=control, control_groups=5 if control else None,
                             panel_transform=transform)


def test_g2_valid_month_filter_is_audit_clean():
    ext = _ext("month_filter", "expanding_window_past_only",
               PanelTransform("month_filter", "baa_aaa_spread", 1, "binary_above_historical_median"))
    assert audit_g2(ext, reporting_delays=DELAYS).booleans["audit_clean"]


def test_g2_contemporaneous_regime_fires_timing():
    # lag 0 < reporting_delays.baa_aaa_spread.lag_months (1) -> NEW_TIMING_VIOLATION.
    ext = _ext("month_filter", "expanding_window_past_only",
               PanelTransform("month_filter", "baa_aaa_spread", 0, "binary_above_historical_median"))
    out = audit_g2(ext, reporting_delays=DELAYS)
    assert not out.passed and out.refusal_code is RefusalCode.NEW_TIMING_VIOLATION


def test_g2_full_sample_threshold_fires_provenance():
    ext = _ext("month_filter", "full_sample_median",       # NOT expanding past-only
               PanelTransform("month_filter", "baa_aaa_spread", 1, "binary_above_historical_median"))
    out = audit_g2(ext, reporting_delays=DELAYS)
    assert not out.passed and out.refusal_code is RefusalCode.PARAMETER_PROVENANCE_VIOLATION


def test_g2_row_filter_on_realised_return_field_fires_provenance():
    ext = _ext("row_filter", "ex_ante_committed",
               PanelTransform("row_filter", "xret", 1, "restrict_top_liquidity_tercile"))
    out = audit_g2(ext, reporting_delays=DELAYS)
    assert not out.passed and out.refusal_code is RefusalCode.PARAMETER_PROVENANCE_VIOLATION


def test_g2_valid_row_filter_is_audit_clean():
    ext = _ext("row_filter", "ex_ante_committed",
               PanelTransform("row_filter", "rating", 1, "restrict_investment_grade"))
    assert audit_g2(ext, reporting_delays=DELAYS).booleans["audit_clean"]
