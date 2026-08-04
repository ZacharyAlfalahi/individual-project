"""
KPP fitted-model sub-metric aggregation (RQ1, contract v1.3 / D42) -- per-field-type
Wilson-interval proportions, reported SEPARATELY from and never pooled into the sort
G3 number.

The headline isolation assertion: aggregating the KPP sub-metric does not touch
``aggregation.ANCHOR_SET`` (the sort pooling set), and the sub-metric declares
``pooled == False``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from evaluation.gold_specs.kpp_gold_loader import load_kpp_gold_spec  # noqa: E402
from evaluation.harness import aggregation  # noqa: E402
from evaluation.harness.estimation_scoring import (  # noqa: E402
    CELL_INSTRUMENT_IDENTITY,
    CELL_INSTRUMENT_SOURCE_CLASS,
    score_kpp,
)
from evaluation.harness.kpp_aggregation import aggregate_kpp  # noqa: E402
from evaluation.harness.reportability import load_g3_thresholds  # noqa: E402

from _librarian_fixtures import (  # noqa: E402
    build_estimation_block,
    build_instrument_set,
    build_kpp_spec,
    instrument_ref,
    stated,
)


def _thresholds():
    return load_g3_thresholds()


# --- non-pooling: ANCHOR_SET is untouched -----------------------------------

def test_anchor_set_is_unchanged_and_submetric_is_not_pooled():
    gold = load_kpp_gold_spec("kpp")
    sub = aggregate_kpp(score_kpp(gold, gold), _thresholds())
    # the sort pooling set is the str/drf/mom6 triple -- KPP is NOT in it.
    assert aggregation.ANCHOR_SET == ("drf", "mom6", "str")
    assert "kpp" not in aggregation.ANCHOR_SET
    assert sub.pooled is False


# --- the 29-instrument identity cell clears the calibration bar -------------

def test_instrument_identity_cell_is_calibrated_on_one_paper():
    gold = load_kpp_gold_spec("kpp")
    sub = aggregate_kpp(score_kpp(gold, gold), _thresholds())
    ident = sub.cells[CELL_INSTRUMENT_IDENTITY]
    assert ident.numerator == 29 and ident.denominator == 29
    assert ident.value == 1.0
    # n=29 >= min_cell_n (20): a single fitted-factor paper yields a CALIBRATED cell.
    assert ident.calibrated is True
    assert ident.interval is not None  # Wilson interval exists
    # source_class likewise reaches n=29 on a perfect self-score.
    assert sub.cells[CELL_INSTRUMENT_SOURCE_CLASS].denominator == 29


def test_small_estimation_cells_report_uncalibrated():
    gold = load_kpp_gold_spec("kpp")
    sub = aggregate_kpp(score_kpp(gold, gold), _thresholds())
    # estimation enum/int cells have small n (<20) -> honestly flagged uncalibrated.
    for name in ("estimation_enum", "estimation_int", "estimation_int_set"):
        assert sub.cells[name].calibrated is False


# --- over-claim + miss flow into the 2x2 ------------------------------------

def test_over_claim_and_recall_reflect_perturbations():
    gold = build_kpp_spec(
        instruments=build_instrument_set(instruments=(
            instrument_ref("past_6m_cumulative_return"),
            instrument_ref("credit_rating"),
        ))
    )
    run = build_kpp_spec(
        instruments=build_instrument_set(instruments=(
            instrument_ref("past_6m_cumulative_return"),
            instrument_ref("duration"),        # false positive
            instrument_ref("bond_var_36m"),    # false positive
        ))
    )
    sub = aggregate_kpp(score_kpp(gold, run), _thresholds())
    ident = sub.cells[CELL_INSTRUMENT_IDENTITY]
    assert ident.numerator == 1 and ident.denominator == 2   # recall 1/2 (credit_rating missed)
    over = sub.instrument_over_claim
    assert over.numerator == 2 and over.denominator == 3     # 2 of 3 run instruments over-claimed


def test_render_is_stable_and_mentions_non_pooling():
    gold = load_kpp_gold_spec("kpp")
    sub = aggregate_kpp(score_kpp(gold, gold), _thresholds())
    text = sub.render()
    assert "NON-POOLED" in text
    assert "instrument_identity" in text
