"""
G3 §3.2 agreement calibration (D34, D37 rulings 1/8/9).

These tests exist mostly to stop the table being read as something it is not:
three of its four cells are counterfactual or bounded, and a reader who takes
them for "how accurate is the system?" will draw the wrong conclusion.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402

from evaluation.harness.agreement_calibration import (  # noqa: E402
    ARM_SEMANTICS,
    Arm,
    build_agreement_table,
    classify_arm,
    cohens_kappa,
    render_agreement_table,
)
from evaluation.harness.calibration_report import compute_metrics  # noqa: E402
from evaluation.harness.gold_calibration import score_anchor  # noqa: E402
from evaluation.harness.reportability import ReportabilityError  # noqa: E402
from evaluation.harness.run_artefacts import RunField, load_run  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_BBW = _ROOT / "runs" / "g3_v3" / "bbw"
_JNPS = _ROOT / "runs" / "g3_v3" / "jnps"

_needs_bbw = pytest.mark.skipif(not (_BBW / "trace_0.json").exists(),
                                reason="post-fix BBW dev run absent (runs/ is gitignored)")
_needs_jnps = pytest.mark.skipif(not (_JNPS / "trace_0.json").exists(),
                                 reason="post-fix JNPS dev run absent (runs/ is gitignored)")


def _rf(**over) -> RunField:
    base = dict(
        field="x", a_answered=True, b_answered=True, a_quote="q", b_quote="q",
        a_located=True, b_located=True, a_model_id="a", b_model_id="b",
        normalised_a="v", normalised_b="v", final_tag="STATED",
        shipped_reason="quoted", ship_choice="model_a", not_extracted=False,
    )
    base.update(over)
    return RunField(**base)


# --- arm classification ------------------------------------------------------

def test_classify_arm_covers_all_five_cases():
    assert classify_arm(_rf()) is Arm.AGREE
    assert classify_arm(_rf(b_located=False)) is Arm.AGREE_QUOTE_GATE_FAILED
    assert classify_arm(_rf(normalised_b="other")) is Arm.DISAGREE
    assert classify_arm(_rf(b_answered=False, normalised_b=None)) is Arm.SINGLE_MODEL
    assert classify_arm(_rf(a_answered=False, b_answered=False,
                            normalised_a=None, normalised_b=None)) is Arm.NOT_IN_2X2


def test_arm_classification_ignores_the_trace_agreement_bit():
    """RunField has no agreement attribute, so classify_arm structurally cannot
    read it -- the bit means "shipped STATED", not model concord."""
    import dataclasses

    assert "agreement" not in {f.name for f in dataclasses.fields(RunField)}
    # a field that did NOT ship but whose models agreed is still AGREE-shaped
    lost = _rf(final_tag="UNKNOWN", shipped_reason="quote_match_failure", b_located=False)
    assert classify_arm(lost) is Arm.AGREE_QUOTE_GATE_FAILED


# --- the declared semantics --------------------------------------------------

@_needs_bbw
def test_agree_arm_equals_selective_accuracy_by_construction():
    """D37 ruling 8. If these ever diverge, either the merge changed or one of
    the two is computed wrongly -- both are worth failing on."""
    art = load_run(_BBW)
    s = score_anchor("drf", _BBW, artefacts=art)
    table = build_agreement_table(s, art)
    metrics = compute_metrics(s, art)
    agree = table.pooled[Arm.AGREE].proportion
    sel = metrics.selective_accuracy
    assert (agree.numerator, agree.denominator) == (sel.numerator, sel.denominator)


def test_disagree_cell_is_labelled_an_upper_bound_not_p_correct():
    """D37 ruling 1. The label is load-bearing: there is no shipped candidate to
    score, so calling it P(correct | disagree) would overstate what is known."""
    assert "UPPER BOUND" in ARM_SEMANTICS[Arm.DISAGREE]
    assert "Not P(correct | disagree)" in ARM_SEMANTICS[Arm.DISAGREE]


def test_counterfactual_arms_say_so():
    for arm in (Arm.SINGLE_MODEL, Arm.AGREE_QUOTE_GATE_FAILED):
        assert "COUNTERFACTUAL" in ARM_SEMANTICS[arm]


@_needs_jnps
def test_every_named_arm_gets_a_cell_even_when_empty():
    """An ABSENT row reads as "not measured"; an n=0 row reads as "measured,
    nothing there". mom6 has no disagreements, so this is the live case."""
    art = load_run(_JNPS)
    table = build_agreement_table(score_anchor("mom6", _JNPS, artefacts=art), art)
    assert set(table.pooled) == {Arm.AGREE, Arm.SINGLE_MODEL, Arm.DISAGREE,
                                 Arm.AGREE_QUOTE_GATE_FAILED}
    disagree = table.pooled[Arm.DISAGREE].proportion
    assert disagree.denominator == 0
    assert disagree.value is None and disagree.interval is None
    assert disagree.render().endswith("n=0")


# --- correctness semantics ---------------------------------------------------

def test_a_gold_silent_field_is_incorrect_in_every_arm():
    """Excluding these would make P(correct | agree) optimistic by hiding the
    correlated-fabrication case -- two models agreeing on a value the paper does
    not state. That is the failure the dual-model gate is blind to."""
    from evaluation.harness.agreement_calibration import _matches_gold
    from evaluation.harness.field_pairing import FieldKey

    assert _matches_gold(FieldKey("n_groups"), "UNKNOWN", None, 5) is False
    assert _matches_gold(FieldKey("n_groups"), "STATED", 5, 5) is True
    assert _matches_gold(FieldKey("n_groups"), "STATED", 5, 10) is False
    assert _matches_gold(FieldKey("n_groups"), "STATED", 5, None) is None


@_needs_bbw
def test_quote_gate_losses_were_correct_answers():
    """The substantive finding this cell exists to expose: on BBW both fields
    where the models agreed and the locator lost them were RIGHT. Coverage lost
    to the locator is not evidence-finding failure."""
    art = load_run(_BBW)
    table = build_agreement_table(score_anchor("drf", _BBW, artefacts=art), art)
    cell = table.pooled[Arm.AGREE_QUOTE_GATE_FAILED].proportion
    assert cell.denominator == 2
    assert cell.numerator == 2


# --- kappa -------------------------------------------------------------------

def test_kappa_perfect_and_chance():
    k, _ = cohens_kappa([("a", "a"), ("b", "b"), ("a", "a"), ("b", "b")])
    assert k == pytest.approx(1.0)
    # deliberate disagreement on a balanced 2-category set -> kappa well below 1
    k, _ = cohens_kappa([("a", "b"), ("b", "a"), ("a", "b"), ("b", "a")])
    assert k == pytest.approx(-1.0)


def test_kappa_degenerate_returns_none_with_a_reason_not_nan():
    """The EXPECTED case on this corpus: model_b is silent on most fields, so the
    joint distribution concentrates. None + reason, never NaN, never a raise."""
    k, note = cohens_kappa([("a", "a")] * 10)
    assert k is None and "degenerate" in note
    k, note = cohens_kappa([])
    assert k is None and note == "no observations"


@_needs_bbw
def test_kappa_is_computable_on_the_real_run():
    art = load_run(_BBW)
    table = build_agreement_table(score_anchor("drf", _BBW, artefacts=art), art)
    assert table.kappa is not None
    assert -1.0 <= table.kappa <= 1.0


# --- the gate ----------------------------------------------------------------

@_needs_bbw
def test_render_refuses_a_non_reportable_table_without_opt_in():
    art = load_run(_BBW)
    table = build_agreement_table(score_anchor("drf", _BBW, artefacts=art), art)
    with pytest.raises(ReportabilityError):
        render_agreement_table(table)
    out = render_agreement_table(table, allow_non_reportable=True)
    assert out.startswith("*** NON-REPORTABLE")
    assert "UPPER BOUND" in out and "COUNTERFACTUAL" in out


@_needs_bbw
def test_per_type_cells_exist_and_carry_the_calibration_flag():
    art = load_run(_BBW)
    table = build_agreement_table(score_anchor("drf", _BBW, artefacts=art), art)
    assert table.per_type, "expected at least one (arm, field_type) cell"
    # on a 3-anchor gold set essentially every per-type cell is below D34's bar
    assert any(not c.proportion.calibrated for c in table.per_type.values())
