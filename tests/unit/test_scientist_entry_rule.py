"""
Unit tests for the handoff-seam entry rule (R2 / D14 / R3) — the ONE place a magnitude becomes
a verdict.

Covers:
  * `apply_entry_rule` as a pure function of injected (theta, q): the three AND-conjoined
    conditions, the str-lib_gap-style FAIL (strongly significant, performance-reducing), a
    near-null PASS, and the boundary/sign cases;
  * `build_scientist_case` extracting first-order (bh_adjusted_p, signed effect) from a real
    `FdrReport` + `SaturatedBasis` and assembling a magnitude-free `ScientistCase`;
  * `load_entry_rule_params` fail-loud when the `scientist:` block is absent, and its
    reference-resolution happy path (theta/q resolved from the auditor block, never restated).
"""

import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.auditor.checks.fdr import run_fdr  # noqa: E402  (test may touch auditor; the wall
from agents.auditor.schemas.decomposition import SaturatedBasis  # noqa: E402  scans scientist only)
from agents.scientist.schemas.case import DevelopmentWindow, HoldoutStatus  # noqa: E402
from shared.handoff.scientist_case import (  # noqa: E402
    EntryRuleParams,
    ScientistThresholdError,
    ToggleStat,
    apply_entry_rule,
    build_scientist_case,
    load_entry_rule_params,
)

# Injected constants for the tests (NOT the pre-registered values — those live in the
# not-yet-committed scientist: block; here we inject synthetic ones to exercise the rule).
THETA = 0.05
Q = 0.10


def _all_keys(obj) -> set:
    """Every dict key appearing anywhere in a nested dict/list structure."""
    keys: set = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(k)
            keys |= _all_keys(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            keys |= _all_keys(v)
    return keys


# =========================================================================================
# apply_entry_rule — pure function
# =========================================================================================

def test_lib_gap_style_fail():
    # Mirrors str's lib_gap: strongly significant AND performance-reducing AND material -> FAIL.
    stats = {"lib_gap": ToggleStat(bh_adjusted_p=0.001, effect_signed=-0.40)}
    assert apply_entry_rule(stats, theta=THETA, q=Q) == {"lib_gap": "FAIL"}


def test_near_null_pass():
    # Tiny effect, insignificant p -> PASS (fails both the tier and the materiality condition).
    stats = {"meas_err": ToggleStat(bh_adjusted_p=0.80, effect_signed=-0.0001)}
    assert apply_entry_rule(stats, theta=THETA, q=Q) == {"meas_err": "PASS"}


def test_performance_increasing_is_pass():
    # Clears tier AND material, but the correction RAISES the metric (effect_signed > 0) -> PASS.
    stats = {"survivorship": ToggleStat(bh_adjusted_p=0.001, effect_signed=+0.40)}
    assert apply_entry_rule(stats, theta=THETA, q=Q) == {"survivorship": "PASS"}


def test_tier_not_cleared_is_pass():
    # Material AND performance-reducing, but BH-adjusted p above q -> PASS (tier gates entry).
    stats = {"lab_trim": ToggleStat(bh_adjusted_p=0.20, effect_signed=-0.40)}
    assert apply_entry_rule(stats, theta=THETA, q=Q) == {"lab_trim": "PASS"}


def test_magnitude_equal_theta_is_pass_strict_inequality():
    # |effect| == theta is NOT > theta -> PASS (materiality is a strict inequality).
    stats = {"lib_gap": ToggleStat(bh_adjusted_p=0.001, effect_signed=-THETA)}
    assert apply_entry_rule(stats, theta=THETA, q=Q) == {"lib_gap": "PASS"}


def test_p_equal_q_clears_tier():
    # bh_adjusted_p == q DOES clear the tier (<=), so with material+reducing -> FAIL.
    stats = {"lib_gap": ToggleStat(bh_adjusted_p=Q, effect_signed=-0.40)}
    assert apply_entry_rule(stats, theta=THETA, q=Q) == {"lib_gap": "FAIL"}


def test_refused_dominates_even_when_stats_would_fail():
    stats = {"lib_gap": ToggleStat(bh_adjusted_p=0.001, effect_signed=-0.40)}
    out = apply_entry_rule(stats, theta=THETA, q=Q, refused_toggles=frozenset({"lib_gap"}))
    assert out == {"lib_gap": "REFUSED"}


def test_refused_toggle_without_stats_is_refused():
    out = apply_entry_rule({}, theta=THETA, q=Q, refused_toggles=frozenset({"survivorship"}))
    assert out == {"survivorship": "REFUSED"}


def test_mixed_multi_toggle_map():
    stats = {
        "lib_gap": ToggleStat(bh_adjusted_p=0.001, effect_signed=-0.40),   # FAIL
        "meas_err": ToggleStat(bh_adjusted_p=0.90, effect_signed=-0.001),  # PASS
        "stale_price": ToggleStat(bh_adjusted_p=0.001, effect_signed=+0.30),  # PASS (increasing)
    }
    out = apply_entry_rule(stats, theta=THETA, q=Q, refused_toggles=frozenset({"lab_trim"}))
    assert out == {
        "lib_gap": "FAIL",
        "meas_err": "PASS",
        "stale_price": "PASS",
        "lab_trim": "REFUSED",
    }


@pytest.mark.parametrize("bad_q", [0.0, 1.0, -0.1, 1.5])
def test_bad_q_raises(bad_q):
    with pytest.raises(ValueError):
        apply_entry_rule({}, theta=THETA, q=bad_q)


@pytest.mark.parametrize("bad_theta", [-0.01, True])
def test_bad_theta_raises(bad_theta):
    with pytest.raises(ValueError):
        apply_entry_rule({}, theta=bad_theta, q=Q)


def test_togglestat_rejects_non_numbers():
    with pytest.raises(TypeError):
        ToggleStat(bh_adjusted_p="x", effect_signed=-0.4)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ToggleStat(bh_adjusted_p=0.01, effect_signed=True)  # bool is not a real effect


# =========================================================================================
# build_scientist_case — AuditReport extraction (real FdrReport + SaturatedBasis)
# =========================================================================================

def _stub_report(*, doe: dict, pvalues: dict, runnable: tuple, q: float = Q):
    """A duck-typed stand-in exposing exactly the fields build_scientist_case reads:
    core.saturated.doe, core.runnable_toggles, fdr.decisions. Uses the REAL SaturatedBasis and
    the REAL run_fdr so the extraction contract is exercised against genuine auditor types."""
    sat = SaturatedBasis(harsanyi=dict(doe), walsh=dict(doe), doe=dict(doe))
    core = types.SimpleNamespace(saturated=sat, runnable_toggles=runnable)
    fdr = run_fdr(pvalues, q)
    return types.SimpleNamespace(core=core, fdr=fdr)


def test_build_case_str_like_lib_gap():
    doe = {frozenset({"lib_gap"}): -0.40, frozenset({"lab_trim"}): 0.00}
    pvalues = {frozenset({"lib_gap"}): 0.001, frozenset({"lab_trim"}): 0.90}
    report = _stub_report(doe=doe, pvalues=pvalues, runnable=("lib_gap", "lab_trim"))

    case = build_scientist_case(
        report,
        strategy_id="short_term_reversal",
        case_id="sci_case_004",
        corrected_quant_config_ref="qc_str_corrected_v1",
        corrected_run_ref="run_str_corrected",
        audit_report_ref="audit_str_021",
        development_window=DevelopmentWindow(start="2002-07", end="2021-12"),
        holdout_status=HoldoutStatus(accessible=False),
        theta=THETA,
        q=Q,
    )
    assert case.failed_check_ids == ("lib_gap",)
    assert case.failed_check_verdicts == {"lib_gap": "FAIL"}
    assert set(case.applicable_toggles) == {"lib_gap", "lab_trim"}
    # The wall: no magnitude-bearing KEY leaked into the case (exact key match, recursive —
    # substring matching would false-positive on 't_stat' inside 'holdout_status').
    banned_keys = {
        "sharpe", "alpha", "alpha_bbw4", "t_stat", "p_value", "p_raw", "p_bh",
        "effect", "effect_signed", "effect_magnitude", "doe", "bh_adjusted_p", "mean_return",
    }
    assert _all_keys(case.to_dict()).isdisjoint(banned_keys)


def test_build_case_refused_toggle_excluded_from_extraction():
    # lab_trim is REFUSED: it need not (and does not) appear in doe / fdr; it must not raise.
    doe = {frozenset({"lib_gap"}): -0.40}
    pvalues = {frozenset({"lib_gap"}): 0.001}
    report = _stub_report(doe=doe, pvalues=pvalues, runnable=("lib_gap",))

    case = build_scientist_case(
        report,
        strategy_id="short_term_reversal",
        case_id="sci_case_004",
        corrected_quant_config_ref="qc",
        corrected_run_ref="run",
        audit_report_ref="audit",
        development_window=DevelopmentWindow(start="2002-07", end="2021-12"),
        holdout_status=HoldoutStatus(accessible=False),
        theta=THETA,
        q=Q,
        refused_toggles=frozenset({"lab_trim"}),
    )
    assert case.failed_check_ids == ("lib_gap",)
    assert set(case.applicable_toggles) == {"lib_gap", "lab_trim"}  # runnable ∪ refused


def test_build_case_fails_loud_on_missing_coordinate():
    # A runnable toggle with no first-order DOE coordinate -> fail loud, never silent PASS.
    doe = {}  # lib_gap coordinate absent
    pvalues = {frozenset({"lib_gap"}): 0.001}
    report = _stub_report(doe=doe, pvalues=pvalues, runnable=("lib_gap",))
    with pytest.raises(KeyError):
        build_scientist_case(
            report,
            strategy_id="short_term_reversal",
            case_id="c",
            corrected_quant_config_ref="qc",
            corrected_run_ref="run",
            audit_report_ref="audit",
            development_window=DevelopmentWindow(start="2002-07", end="2021-12"),
            holdout_status=HoldoutStatus(accessible=False),
            theta=THETA,
            q=Q,
        )


# =========================================================================================
# load_entry_rule_params — fail-loud + reference resolution
# =========================================================================================

def test_loader_fails_loud_when_scientist_block_absent(tmp_path):
    # A thresholds file with NO `scientist:` block must raise, never default — the fail-loud
    # guarantee. (The real file now HAS the block; see the resolution test below.)
    p = tmp_path / "thresholds.yaml"
    p.write_text("auditor:\n  fdr:\n    q: 0.10\n")
    with pytest.raises(ScientistThresholdError):
        load_entry_rule_params(p)


def test_loader_resolves_references_on_real_file():
    # The committed `scientist:` block references the auditor block (R3); on the REAL
    # docs/thresholds.yaml the loader resolves theta = auditor.practical_significance.vartheta
    # (0.001) and q = auditor.fdr.q (0.10) — the numbers are stated once, never restated here.
    params = load_entry_rule_params()
    assert params.theta == pytest.approx(0.001)
    assert params.q == pytest.approx(0.10)


def test_loader_resolves_references_without_restating_numbers(tmp_path):
    # The scientist block carries REFERENCES into the auditor block (R3); the loader resolves
    # them so the number is stated once and never restated.
    yaml_text = (
        "auditor:\n"
        "  practical_significance:\n"
        "    vartheta: 0.001\n"
        "  fdr:\n"
        "    q: 0.10\n"
        "scientist:\n"
        "  entry_rule:\n"
        "    materiality_threshold_ref: auditor.practical_significance.vartheta\n"
        "    fdr_q_ref: auditor.fdr.q\n"
    )
    p = tmp_path / "thresholds.yaml"
    p.write_text(yaml_text)
    params = load_entry_rule_params(p)
    assert isinstance(params, EntryRuleParams)
    assert params.theta == pytest.approx(0.001)
    assert params.q == pytest.approx(0.10)


def test_loader_accepts_literal_fallback(tmp_path):
    yaml_text = (
        "scientist:\n"
        "  entry_rule:\n"
        "    theta: 0.02\n"
        "    q: 0.10\n"
    )
    p = tmp_path / "thresholds.yaml"
    p.write_text(yaml_text)
    params = load_entry_rule_params(p)
    assert params.theta == pytest.approx(0.02)
    assert params.q == pytest.approx(0.10)


def test_loader_raises_on_dangling_reference(tmp_path):
    yaml_text = (
        "scientist:\n"
        "  entry_rule:\n"
        "    materiality_threshold_ref: auditor.nope.missing\n"
        "    fdr_q_ref: auditor.fdr.q\n"
    )
    p = tmp_path / "thresholds.yaml"
    p.write_text(yaml_text)
    with pytest.raises(ScientistThresholdError):
        load_entry_rule_params(p)
