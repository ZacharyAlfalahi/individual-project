"""
G3 metrics -- contract §3.1 / §3.6, D34 mechanics, D37 conventions.

Two things these tests exist to prevent: a statistic that degenerates silently
(Wald-style zero-width intervals at the ends, NaN in an empty cell), and a
denominator that drifts from the one D37 pre-registered.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402

from evaluation.harness.calibration_report import (  # noqa: E402
    compute_metrics,
    render_headline_table,
)
from evaluation.harness.gold_calibration import score_anchor  # noqa: E402
from evaluation.harness.reportability import (  # noqa: E402
    G3Thresholds,
    ReportabilityError,
    load_g3_thresholds,
)
from evaluation.harness.run_artefacts import load_run  # noqa: E402
from evaluation.harness.stats import Proportion, wilson  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_BBW = _ROOT / "runs" / "g3_v3" / "bbw"
_JNPS = _ROOT / "runs" / "g3_v3" / "jnps"

_Z = 1.959963984540054

_needs_bbw = pytest.mark.skipif(not (_BBW / "trace_0.json").exists(),
                                reason="post-fix BBW dev run absent (runs/ is gitignored)")
_needs_jnps = pytest.mark.skipif(not (_JNPS / "trace_0.json").exists(),
                                 reason="post-fix JNPS dev run absent (runs/ is gitignored)")


# --- Wilson ------------------------------------------------------------------

def test_wilson_matches_hand_computed_references():
    lo, hi = wilson(5, 10, _Z)
    assert lo == pytest.approx(0.23659, abs=1e-4)
    assert hi == pytest.approx(0.76341, abs=1e-4)
    lo, hi = wilson(0, 10, _Z)
    assert lo == pytest.approx(0.0, abs=1e-9)
    assert hi == pytest.approx(0.27753, abs=1e-4)


def test_wilson_is_mirror_symmetric():
    lo_a, hi_a = wilson(3, 10, _Z)
    lo_b, hi_b = wilson(7, 10, _Z)
    assert lo_a == pytest.approx(1 - hi_b, abs=1e-12)
    assert hi_a == pytest.approx(1 - lo_b, abs=1e-12)


def test_wilson_does_not_degenerate_at_the_ends():
    """The reason D34 names Wilson rather than Wald: at k=0 and k=n the Wald
    interval collapses to zero width and reports false certainty."""
    lo, hi = wilson(0, 18, _Z)
    assert lo == 0.0 and hi > 0.0        # not a point mass at zero
    lo, hi = wilson(18, 18, _Z)
    assert hi == 1.0 and lo < 1.0


def test_wilson_returns_none_at_n_zero_never_nan():
    """An empty cell is a real outcome (the §3.2 DISAGREE arm is empty on the
    current artefacts). None forces the renderer to say so; NaN would propagate
    into a table."""
    assert wilson(0, 0, _Z) is None


def test_wilson_rejects_impossible_counts():
    with pytest.raises(ValueError):
        wilson(5, 3, _Z)
    with pytest.raises(ValueError):
        wilson(-1, 3, _Z)


# --- Proportion --------------------------------------------------------------

def _prop(k, n, min_n=20):
    return Proportion(label="x", numerator=k, denominator=n, z=_Z, min_cell_n=min_n)


def test_proportion_renders_empty_uncalibrated_and_calibrated_distinctly():
    assert _prop(0, 0).render() == "x: n=0"
    assert "uncalibrated" in _prop(2, 4).render()
    r = _prop(10, 40).render()
    assert "10/40" in r and "25.0%" in r and "descriptive" in r


def test_raw_counts_are_always_shown():
    """Contract §5.3 house style: never the percentage alone."""
    for k, n in ((0, 0), (2, 4), (10, 40)):
        r = _prop(k, n).render()
        assert (f"{k}/{n}" in r) or (n == 0 and "n=0" in r)


def test_calibration_bar_moves_with_the_threshold_not_a_constant():
    """Pins the no-hard-coded-thresholds rule: change the pre-registered bar and
    the calibrated flag must move."""
    assert _prop(2, 4, min_n=20).calibrated is False
    assert _prop(2, 4, min_n=3).calibrated is True


# --- metrics on the real artefacts -------------------------------------------

@_needs_bbw
def test_drf_metrics_golden():
    art = load_run(_BBW)
    b = compute_metrics(score_anchor("drf", _BBW, artefacts=art), art)
    assert b.universe == 44
    assert (b.coverage.numerator, b.coverage.denominator) == (4, 44)
    assert (b.selective_accuracy.numerator, b.selective_accuracy.denominator) == (3, 4)
    assert (b.over_claim_rate.numerator, b.over_claim_rate.denominator) == (0, 4)
    assert (b.abstention_rate.numerator, b.abstention_rate.denominator) == (40, 44)
    assert (b.missed_evidence_rate.numerator, b.missed_evidence_rate.denominator) == (14, 18)


@_needs_jnps
def test_mom6_metrics_golden():
    art = load_run(_JNPS)
    b = compute_metrics(score_anchor("mom6", _JNPS, artefacts=art), art)
    assert b.universe == 44
    assert (b.coverage.numerator, b.coverage.denominator) == (7, 44)
    assert (b.selective_accuracy.numerator, b.selective_accuracy.denominator) == (6, 7)
    # mom6 shipped nothing WRONG; its only failure was a fabrication.
    assert (b.over_claim_rate.numerator, b.over_claim_rate.denominator) == (1, 7)
    assert (b.missed_evidence_rate.numerator, b.missed_evidence_rate.denominator) == (15, 21)


@_needs_bbw
def test_over_claims_are_in_the_selective_accuracy_denominator():
    """A fabrication is incorrect for P(correct | ships) AND separately counted.
    If it were excluded, a model could raise its selective accuracy by inventing
    values for fields the paper is silent on."""
    art = load_run(_JNPS if (_JNPS / "trace_0.json").exists() else _BBW)
    anchor = "mom6" if (_JNPS / "trace_0.json").exists() else "drf"
    b = compute_metrics(score_anchor(anchor, art.run_dir, artefacts=art), art)
    assert b.selective_accuracy.denominator >= b.over_claim_rate.numerator


@_needs_bbw
def test_the_two_reason_views_genuinely_disagree():
    """The D9 merge checks the quote gate BEFORE the single-response branch, so
    the shipped attribution under-counts one-model-mute fields. If these two ever
    coincide, the metrics layer is reporting one number twice."""
    art = load_run(_BBW)
    b = compute_metrics(score_anchor("drf", _BBW, artefacts=art), art)
    shipped_single = b.shipped_reason_distribution.get("single_response", 0)
    incidence_one_silent = b.condition_incidence.get("one_silent", 0)
    assert incidence_one_silent > shipped_single


@_needs_bbw
def test_shipped_reasons_partition_but_incidence_does_not():
    art = load_run(_BBW)
    b = compute_metrics(score_anchor("drf", _BBW, artefacts=art), art)
    # a partition over the asked fields
    assert sum(b.shipped_reason_distribution.values()) == b.universe
    # NOT a partition, by construction
    assert sum(b.condition_incidence.values()) > b.universe


@_needs_bbw
def test_locator_success_counts_proposed_quotes_not_fields():
    art = load_run(_BBW)
    b = compute_metrics(score_anchor("drf", _BBW, artefacts=art), art)
    pooled = b.locator_success["pooled"]
    assert pooled.denominator == (b.locator_success["model_a"].denominator
                                  + b.locator_success["model_b"].denominator)
    assert pooled.denominator > 0


# --- the D37 denominators ----------------------------------------------------

@_needs_bbw
def test_not_asked_is_in_coverage_but_out_of_missed_evidence():
    """D37 ruling 3, the asymmetry, asserted structurally: the §3.6 denominator
    counts only fields the run was ASKED about, while coverage's denominator is
    the whole gold universe."""
    from evaluation.harness.gold_calibration import Outcome

    art = load_run(_BBW)
    s = score_anchor("drf", _BBW, artefacts=art)
    b = compute_metrics(s, art)
    not_asked = sum(1 for r in s.rows if r.outcome is Outcome.NOT_ASKED)
    headline = [r for r in s.rows if r.outcome not in
                {Outcome.EXCLUDED_WEAKER_RUBRIC, Outcome.EXCLUDED_NOT_FIELD_EXTRACTED}]
    gold_stated_all = sum(1 for r in headline if r.gold_tag == "STATED")
    assert b.coverage.denominator == len(headline)
    assert b.missed_evidence_rate.denominator == gold_stated_all - sum(
        1 for r in headline if r.outcome is Outcome.NOT_ASKED and r.gold_tag == "STATED")
    assert not_asked >= 0


@_needs_bbw
def test_excluded_fields_are_out_of_every_denominator_but_still_counted():
    art = load_run(_BBW)
    s = score_anchor("drf", _BBW, artefacts=art)
    b = compute_metrics(s, art)
    assert b.universe == len(s.rows) - 3          # 2 rubric + 1 strategy_label
    assert b.outcome_distribution["excluded_weaker_rubric"] == 2
    assert b.outcome_distribution["excluded_not_field_extracted"] == 1


# --- the gate ----------------------------------------------------------------

@_needs_bbw
def test_render_refuses_a_non_reportable_bundle_without_explicit_opt_in():
    art = load_run(_BBW)
    b = compute_metrics(score_anchor("drf", _BBW, artefacts=art), art)
    with pytest.raises(ReportabilityError):
        render_headline_table(b)
    out = render_headline_table(b, allow_non_reportable=True)
    assert out.startswith("*** NON-REPORTABLE")
    assert "coverage: 4/44" in out


@_needs_bbw
def test_thresholds_are_injectable_so_the_bar_is_not_a_constant():
    art = load_run(_BBW)
    s = score_anchor("drf", _BBW, artefacts=art)
    loose = G3Thresholds(min_cell_n=2, wilson_z=_Z, interval_label="descriptive")
    b = compute_metrics(s, art, thresholds=loose)
    assert b.selective_accuracy.calibrated is True      # n=4 clears a bar of 2
    strict = compute_metrics(s, art, thresholds=load_g3_thresholds())
    assert strict.selective_accuracy.calibrated is False  # n=4 fails the real bar of 20
