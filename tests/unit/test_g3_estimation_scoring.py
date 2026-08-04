"""
KPP fitted-model scorer (schema v1.2 / RQ1) -- driven by SYNTHETIC run specs, the
way the sort G3 tests drive synthetic RunArtefacts. Scope A has no live extraction
round-trip; the scorer is validated against constructed gold-vs-run spec pairs.

Covers: a perfect run scores all-correct; a renamed instrument -> one miss + one
false positive; a fabricated instrument on a matched axis; an int mismatch on
n_factors_preferred; a wrong source_class; a run abstention; a prose field is
excluded; and the REAL gold scores 100% against itself.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from evaluation.gold_specs.kpp_gold_loader import load_kpp_gold_spec  # noqa: E402
from evaluation.harness.estimation_scoring import (  # noqa: E402
    CELL_ESTIMATION_ENUM,
    CELL_ESTIMATION_INT,
    CELL_INSTRUMENT_IDENTITY,
    CELL_INSTRUMENT_SOURCE_CLASS,
    Outcome,
    score_kpp,
)

from _librarian_fixtures import (  # noqa: E402
    build_estimation_block,
    build_instrument_set,
    build_kpp_spec,
    instrument_ref,
    stated,
    unknown,
)


def _cells(score, cell):
    return [c for c in score.cells if c.cell == cell]


def _outcomes(score, cell):
    return [c.outcome for c in _cells(score, cell)]


# --- perfect run ------------------------------------------------------------

def test_perfect_run_scores_all_correct():
    gold = build_kpp_spec()
    run = build_kpp_spec()  # identical
    score = score_kpp(gold, run)
    # every scorable estimation enum/int cell is CORRECT; both instruments matched.
    assert all(o == Outcome.CORRECT for o in _outcomes(score, CELL_ESTIMATION_ENUM))
    assert all(o == Outcome.CORRECT for o in _outcomes(score, CELL_ESTIMATION_INT))
    assert all(o == Outcome.CORRECT for o in _outcomes(score, CELL_INSTRUMENT_IDENTITY))
    assert score.n_matched_instruments == 2


def test_real_gold_scores_100_percent_against_itself():
    gold = load_kpp_gold_spec("kpp")
    score = score_kpp(gold, gold)
    ident = _outcomes(score, CELL_INSTRUMENT_IDENTITY)
    assert ident.count(Outcome.CORRECT) == 29
    assert Outcome.INSTRUMENT_MISS not in ident
    assert Outcome.INSTRUMENT_FALSE_POSITIVE not in ident
    sc = _outcomes(score, CELL_INSTRUMENT_SOURCE_CLASS)
    assert sc.count(Outcome.CORRECT) == 29
    assert all(o == Outcome.CORRECT for o in _outcomes(score, CELL_ESTIMATION_ENUM))


# --- instrument miss + false positive ---------------------------------------

def test_renamed_instrument_is_miss_plus_false_positive():
    gold = build_kpp_spec(
        instruments=build_instrument_set(instruments=(
            instrument_ref("past_6m_cumulative_return"),
            instrument_ref("credit_rating"),
        ))
    )
    run = build_kpp_spec(
        instruments=build_instrument_set(instruments=(
            instrument_ref("past_6m_cumulative_return"),
            instrument_ref("duration"),  # renamed: credit_rating -> duration
        ))
    )
    score = score_kpp(gold, run)
    ident = _outcomes(score, CELL_INSTRUMENT_IDENTITY)
    assert ident.count(Outcome.CORRECT) == 1
    assert ident.count(Outcome.INSTRUMENT_MISS) == 1               # credit_rating not emitted
    assert ident.count(Outcome.INSTRUMENT_FALSE_POSITIVE) == 1     # duration over-claimed


# --- estimation value errors ------------------------------------------------

def test_int_mismatch_on_n_factors_preferred():
    gold = build_kpp_spec()
    run = build_kpp_spec(estimation=build_estimation_block(n_factors_preferred=stated(4)))
    score = score_kpp(gold, run)
    int_cells = {c.key: c.outcome for c in _cells(score, CELL_ESTIMATION_INT)}
    assert int_cells["n_factors_preferred"] == Outcome.WRONG
    assert int_cells["oos_split"] == Outcome.CORRECT


def test_wrong_enum_is_scored_wrong():
    gold = build_kpp_spec()
    run = build_kpp_spec(estimation=build_estimation_block(intercept_spec=stated("unrestricted")))
    score = score_kpp(gold, run)
    enum_cells = {c.key: c.outcome for c in _cells(score, CELL_ESTIMATION_ENUM)}
    assert enum_cells["intercept_spec"] == Outcome.WRONG


def test_run_abstention_is_not_wrong():
    gold = build_kpp_spec()
    run = build_kpp_spec(estimation=build_estimation_block(n_factors_preferred=unknown()))
    score = score_kpp(gold, run)
    int_cells = {c.key: c.outcome for c in _cells(score, CELL_ESTIMATION_INT)}
    assert int_cells["n_factors_preferred"] == Outcome.RUN_ABSTAINED  # distinct from WRONG


def test_gold_silent_field_is_excluded():
    # gold does not state oos_split -> the cell is GOLD_SILENT (out of the selective universe).
    gold = build_kpp_spec(estimation=build_estimation_block(oos_split=unknown()))
    run = build_kpp_spec()
    score = score_kpp(gold, run)
    int_cells = {c.key: c.outcome for c in _cells(score, CELL_ESTIMATION_INT)}
    assert int_cells["oos_split"] == Outcome.GOLD_SILENT


def test_prose_fields_excluded():
    gold = build_kpp_spec()
    run = build_kpp_spec()
    score = score_kpp(gold, run)
    prose = [c for c in score.cells if c.outcome == Outcome.EXCLUDED_PROSE]
    # return_variable, characteristic_preprocessing, managed_portfolio_construction
    assert len(prose) == 3


# --- source_class -----------------------------------------------------------

def test_wrong_source_class_scored_wrong():
    gold = build_kpp_spec(
        instruments=build_instrument_set(instruments=(instrument_ref("duration", source_class="bond"),))
    )
    run = build_kpp_spec(
        instruments=build_instrument_set(instruments=(instrument_ref("duration", source_class="equity"),))
    )
    score = score_kpp(gold, run)
    sc = _outcomes(score, CELL_INSTRUMENT_SOURCE_CLASS)
    assert sc == [Outcome.WRONG]
