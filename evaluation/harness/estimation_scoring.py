"""
KPP fitted-model scoring (schema v1.2 / RQ1) -- the parallel analogue of
``gold_calibration.score_anchor``, on the estimation block + instrument set.

``score_kpp(gold_spec, run_spec)`` walks the 11 estimation fields and the
29-instrument set and classifies every cell into a typed ``Outcome`` (mirroring
the sort scorer's selective-accuracy semantics: score only where the gold ships a
value and the run ships a value). It produces a flat list of ``ScoredCell`` rows
that ``kpp_aggregation`` turns into per-field-type Wilson-interval proportions.

Both inputs are ``StrategySpec`` objects carrying ``estimation`` + ``instruments``
(the gold via ``kpp_gold_loader``; a run via the extraction pipeline, or a
synthetic spec in tests). This scorer touches NONE of the sort-path modules
(``field_pairing`` / ``compare_policy`` / ``gold_calibration`` / ``aggregation``).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from agents.librarian.schema.estimation_fields import ESTIMATION_FIELDS, ESTIMATION_FIELD_TYPES

from evaluation.harness.estimation_compare import compare_estimation_value, compare_source_class
from evaluation.harness.instrument_pairing import match_instruments


class Outcome(str, Enum):
    CORRECT = "correct"                      # run shipped a value; equal to gold
    WRONG = "wrong"                          # run shipped a value; not equal
    RUN_ABSTAINED = "run_abstained"          # gold STATED; run UNKNOWN (no value shipped)
    GOLD_SILENT = "gold_silent"              # gold UNKNOWN -- excluded (selective universe)
    EXCLUDED_PROSE = "excluded_prose"        # declared-weaker rubric field
    NOT_COMPARABLE = "not_comparable"        # a value that would not normalise
    INSTRUMENT_MISS = "instrument_miss"      # gold instrument the run did not emit
    INSTRUMENT_FALSE_POSITIVE = "instrument_false_positive"  # run instrument not in gold


# Which field-types feed which reported cell.
CELL_ESTIMATION_ENUM = "estimation_enum"
CELL_ESTIMATION_INT = "estimation_int"
CELL_ESTIMATION_INT_SET = "estimation_int_set"
CELL_ESTIMATION_PROSE = "estimation_prose"
CELL_INSTRUMENT_IDENTITY = "instrument_identity"
CELL_INSTRUMENT_SOURCE_CLASS = "instrument_source_class"

_FTYPE_TO_CELL = {
    "enum": CELL_ESTIMATION_ENUM,
    "int": CELL_ESTIMATION_INT,
    "int_set": CELL_ESTIMATION_INT_SET,
    "prose": CELL_ESTIMATION_PROSE,
}


@dataclass(frozen=True)
class ScoredCell:
    """One scored cell: which reported group it belongs to, an identifier, and its
    typed outcome. ``outcome`` decides num/denom membership in aggregation."""

    cell: str          # one of the CELL_* groups
    key: str           # field name or instrument concept_id
    outcome: Outcome


@dataclass(frozen=True)
class KppScore:
    cells: tuple[ScoredCell, ...]
    n_gold_instruments: int
    n_run_instruments: int
    n_matched_instruments: int


def _is_stated(inh) -> bool:
    return getattr(inh, "tag", None) == "STATED"


def _score_estimation(gold_spec, run_spec,
                      rubric_judgements=None) -> list[ScoredCell]:
    """``rubric_judgements`` (kpp_prose_rubric.md §2.2; None until the rubric is
    IN FORCE): {prose field -> bool rubric adjudication}. A judgement applies ONLY
    to a SHIPPED prose answer; abstained / never-asked prose rows keep their
    coverage channel regardless (D37 conv. 3 -- never read as WRONG)."""
    judgements = rubric_judgements or {}
    cells: list[ScoredCell] = []
    for name in ESTIMATION_FIELDS:
        ftype = ESTIMATION_FIELD_TYPES[name]
        cell = _FTYPE_TO_CELL[ftype]
        g = getattr(gold_spec.estimation, name)
        r = getattr(run_spec.estimation, name) if run_spec.estimation is not None else None

        if ftype == "prose":
            if name in judgements and r is not None and _is_stated(r):
                outcome = compare_estimation_value(
                    name, g.value, r.value, rubric_judgement=judgements[name])
                cells.append(ScoredCell(
                    cell, name,
                    Outcome.CORRECT if outcome.equal else Outcome.WRONG))
            else:
                cells.append(ScoredCell(cell, name, Outcome.EXCLUDED_PROSE))
            continue
        if not _is_stated(g):
            cells.append(ScoredCell(cell, name, Outcome.GOLD_SILENT))
            continue
        if r is None or not _is_stated(r):
            cells.append(ScoredCell(cell, name, Outcome.RUN_ABSTAINED))
            continue
        outcome = compare_estimation_value(name, g.value, r.value)
        if outcome.equal is None:
            cells.append(ScoredCell(cell, name, Outcome.NOT_COMPARABLE))
        else:
            cells.append(ScoredCell(cell, name, Outcome.CORRECT if outcome.equal else Outcome.WRONG))
    return cells


def _score_instruments(gold_spec, run_spec) -> tuple[list[ScoredCell], int, int, int]:
    gold_ins = gold_spec.instruments.instruments if gold_spec.instruments else ()
    run_ins = run_spec.instruments.instruments if run_spec.instruments else ()
    pairing = match_instruments(gold_ins, run_ins)

    cells: list[ScoredCell] = []
    # identity: matched -> CORRECT; missed gold -> MISS; run-only -> FALSE_POSITIVE.
    for m in pairing.matches:
        cells.append(ScoredCell(CELL_INSTRUMENT_IDENTITY, m.concept_id, Outcome.CORRECT))
    for cid in pairing.missed_gold_ids:
        cells.append(ScoredCell(CELL_INSTRUMENT_IDENTITY, cid, Outcome.INSTRUMENT_MISS))
    for cid in pairing.false_positive_run_ids:
        cells.append(ScoredCell(CELL_INSTRUMENT_IDENTITY, cid, Outcome.INSTRUMENT_FALSE_POSITIVE))

    # source_class: scored only on matched pairs where the gold ships a value.
    for m in pairing.matches:
        g_sc = m.gold.source_class
        r_sc = m.run.source_class
        if not _is_stated(g_sc):
            cells.append(ScoredCell(CELL_INSTRUMENT_SOURCE_CLASS, m.concept_id, Outcome.GOLD_SILENT))
            continue
        if not _is_stated(r_sc):
            cells.append(ScoredCell(CELL_INSTRUMENT_SOURCE_CLASS, m.concept_id, Outcome.RUN_ABSTAINED))
            continue
        outcome = compare_source_class(g_sc.value, r_sc.value)
        cells.append(ScoredCell(
            CELL_INSTRUMENT_SOURCE_CLASS, m.concept_id,
            Outcome.CORRECT if outcome.equal else Outcome.WRONG,
        ))
    return cells, len(gold_ins), len(run_ins), len(pairing.matches)


def score_kpp(gold_spec, run_spec, rubric_judgements=None) -> KppScore:
    """Score a KPP run spec against the KPP gold spec. Both must carry an
    ``estimation`` block and an ``instruments`` set. ``rubric_judgements`` is the
    the prose adjudication map (kpp_prose_rubric.md; None until in force)."""
    if gold_spec.estimation is None or gold_spec.instruments is None:
        raise ValueError("score_kpp: gold_spec must carry estimation + instruments (a KPP spec)")
    cells = _score_estimation(gold_spec, run_spec, rubric_judgements=rubric_judgements)
    ins_cells, n_gold, n_run, n_matched = _score_instruments(gold_spec, run_spec)
    cells += ins_cells
    return KppScore(
        cells=tuple(cells),
        n_gold_instruments=n_gold,
        n_run_instruments=n_run,
        n_matched_instruments=n_matched,
    )
