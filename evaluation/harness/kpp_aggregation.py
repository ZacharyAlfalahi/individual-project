"""
KPP fitted-model sub-metric aggregation (schema v1.2 / RQ1, contract v1.3 / D42).

Turns a ``KppScore`` into per-field-type accuracy proportions with Wilson
intervals -- a SEPARATE, explicitly NON-POOLED reported sub-metric. It does NOT
touch ``aggregation.ANCHOR_SET`` and is NOT summed into the sort G3 ``aggregate``:
the fitted-model number is reported beside the sort number, never averaged into
it (the §7 no-pooling mandate). Reuses ``stats.Proportion`` / ``stats.wilson`` and
the pre-registered ``librarian.g3`` thresholds, so the KPP cells use the same
Wilson z and ``min_cell_n`` bar as the sorts.

Cell semantics (selective accuracy = P(correct | the run shipped a value)):
  * estimation_enum / _int / _int_set: numerator = CORRECT, denominator =
    CORRECT + WRONG (gold-silent, run-abstained, prose, not-comparable excluded).
  * instrument_identity: numerator = matched, denominator = matched + missed
    (= the gold instrument count) -- recall of the gold's instrument identities.
    The ~29-instrument denominator is what clears the n>=20 calibration bar (the
    brief's statistical argument): a single fitted-factor paper yields a calibrated
    per-field number on its own.
  * instrument_source_class: numerator = CORRECT, denominator = CORRECT + WRONG on
    matched pairs.
  * instrument_over_claim (reported separately): false positives / run instruments.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from evaluation.harness.estimation_scoring import (
    CELL_INSTRUMENT_IDENTITY,
    KppScore,
    Outcome,
)
from evaluation.harness.reportability import G3Thresholds
from evaluation.harness.stats import Proportion

# Cells that use plain selective accuracy (CORRECT / (CORRECT+WRONG)).
_ACCURACY_CELLS_EXCLUDE = {CELL_INSTRUMENT_IDENTITY}


@dataclass(frozen=True)
class KppSubMetric:
    """The KPP fitted-model RQ1 sub-metric. ``pooled = False`` is a permanent
    property (contract v1.3 / D42): this is reported beside, never inside, the sort
    G3 number."""

    cells: dict[str, Proportion]
    instrument_over_claim: Proportion
    n_gold_instruments: int
    n_run_instruments: int
    n_matched_instruments: int
    excluded_prose: int
    gold_silent: int
    run_abstained: int
    not_comparable: int
    pooled: bool = False

    def render(self) -> str:
        lines = ["KPP fitted-model sub-metric (RQ1, NON-POOLED — reported beside the sort G3):"]
        for name in sorted(self.cells):
            lines.append("  " + self.cells[name].render())
        lines.append("  " + self.instrument_over_claim.render())
        lines.append(
            f"  [instruments: {self.n_matched_instruments}/{self.n_gold_instruments} gold matched; "
            f"{self.n_run_instruments} run-emitted]"
        )
        lines.append(
            f"  [excluded: prose={self.excluded_prose}, gold_silent={self.gold_silent}, "
            f"run_abstained={self.run_abstained}, not_comparable={self.not_comparable}]"
        )
        return "\n".join(lines)


def _prop(label: str, num: int, denom: int, t: G3Thresholds) -> Proportion:
    return Proportion(
        label=label, numerator=num, denominator=denom,
        z=t.wilson_z, min_cell_n=t.min_cell_n, interval_label=t.interval_label,
    )


def aggregate_kpp(score: KppScore, thresholds: G3Thresholds) -> KppSubMetric:
    """Aggregate a ``KppScore`` into the non-pooled fitted-model sub-metric."""
    by_cell: dict[str, list[Outcome]] = defaultdict(list)
    for c in score.cells:
        by_cell[c.cell].append(c.outcome)

    cells: dict[str, Proportion] = {}
    for cell, outcomes in by_cell.items():
        cnt = Counter(outcomes)
        if cell == CELL_INSTRUMENT_IDENTITY:
            correct = cnt[Outcome.CORRECT]
            missed = cnt[Outcome.INSTRUMENT_MISS]
            cells[cell] = _prop(cell, correct, correct + missed, thresholds)
        else:
            correct = cnt[Outcome.CORRECT]
            wrong = cnt[Outcome.WRONG]
            cells[cell] = _prop(cell, correct, correct + wrong, thresholds)

    fp = sum(1 for c in score.cells if c.outcome == Outcome.INSTRUMENT_FALSE_POSITIVE)
    over_claim = _prop("instrument_over_claim", fp, score.n_run_instruments, thresholds)

    tally = Counter(c.outcome for c in score.cells)
    return KppSubMetric(
        cells=cells,
        instrument_over_claim=over_claim,
        n_gold_instruments=score.n_gold_instruments,
        n_run_instruments=score.n_run_instruments,
        n_matched_instruments=score.n_matched_instruments,
        excluded_prose=tally[Outcome.EXCLUDED_PROSE],
        gold_silent=tally[Outcome.GOLD_SILENT],
        run_abstained=tally[Outcome.RUN_ABSTAINED],
        not_comparable=tally[Outcome.NOT_COMPARABLE],
    )
