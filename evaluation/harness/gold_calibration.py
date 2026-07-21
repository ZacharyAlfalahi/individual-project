"""
G3 gold calibration -- the scoring unit (build brief §8 G3, D34, D37).

The G3 gate: *"Extracted specs vs gold specs: field accuracy, the 2x2
P(correct|agree) with CI and n, STATED coverage, per-field-type cells where n
permits. Attribution: adapter validated at G1/G2, so failures here are
extraction."*

This module is the scoring unit only. It turns (gold spec, run
artefacts) into one typed ``ScoredField`` per gold field. The metrics layer, the
phase/reportability gate, and the 2x2 are implemented separately and deliberately live
elsewhere -- so that the outcome taxonomy is fixed and testable before any number
is computed from it.

**The adapter is not involved, by design.** G1/G2 measure translator quality with
known-good forms as input; G3 measures reader quality against gold forms with no
adapter in the path (build brief §1, D30). That separation is what lets an
end-to-end failure be attributed.

**The universe is fixed by the GOLD, never by the run** (contract §8
non-elasticity). A field the run never emitted still produces a row.
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.gold_specs.gold_loader import load_gold_spec  # noqa: E402
from evaluation.harness.compare_policy import (  # noqa: E402
    Comparability,
    CompareOutcome,
    compare_field,
)
from evaluation.harness.field_pairing import FieldKey, pair_fields  # noqa: E402
from evaluation.harness.run_artefacts import RunArtefacts, RunField, load_run  # noqa: E402


class CalibrationError(RuntimeError):
    """The scorer could not run cleanly against these inputs."""


class Outcome(str, Enum):
    """One field's scored outcome.

    ``SHIPPED_GOLD_SILENT`` earns its own member deliberately. Under D34
    exact-match it is simply incorrect, but it is a categorically different
    failure from shipping the wrong value: it is a FABRICATION -- the model
    asserted a fact the paper does not state -- which is the exact thing P1
    ("blank is the safe state") exists to prevent. Typing it separately makes the
    over-claim rate fall out of the taxonomy instead of needing a second pass.
    """

    SHIPPED_CORRECT = "shipped_correct"
    SHIPPED_WRONG = "shipped_wrong"
    SHIPPED_GOLD_SILENT = "shipped_gold_silent"
    SHIPPED_NOT_SCORABLE = "shipped_not_scorable"
    ABSTAINED_GOLD_STATED = "abstained_gold_stated"      # contract §3.6 missed evidence
    ABSTAINED_GOLD_SILENT = "abstained_gold_silent"      # correct abstention
    NOT_ASKED = "not_asked"                              # pipeline scope gap, not a reader failure
    EXCLUDED_WEAKER_RUBRIC = "excluded_weaker_rubric"    # D37 ruling 7
    EXCLUDED_NOT_FIELD_EXTRACTED = "excluded_not_field_extracted"  # D37 ruling 4


SHIPPED_OUTCOMES: frozenset[Outcome] = frozenset({
    Outcome.SHIPPED_CORRECT, Outcome.SHIPPED_WRONG,
    Outcome.SHIPPED_GOLD_SILENT, Outcome.SHIPPED_NOT_SCORABLE,
})
EXCLUDED_OUTCOMES: frozenset[Outcome] = frozenset({
    Outcome.EXCLUDED_WEAKER_RUBRIC, Outcome.EXCLUDED_NOT_FIELD_EXTRACTED,
})


@dataclass(frozen=True)
class ScoredField:
    """One gold field, scored. Carries the dual-model view RECOMPUTED -- never
    read from the trace's ``agreement`` bit, which means "shipped STATED"."""

    anchor_id: str
    paper_id: str
    key: FieldKey
    dotted_path: str
    outcome: Outcome
    gold_tag: str
    gold_value: object
    run_tag: str | None
    run_value: object
    compare: CompareOutcome | None
    models_agree: bool | None
    conditions: frozenset[str]
    shipped_reason: str | None

    @property
    def is_missed_evidence(self) -> bool:
        """Contract §3.6: a gold-STATED field where the model returned not_stated
        or failed the quote gate. NOT_ASKED is excluded (D37 ruling 3) -- a field
        never asked did neither, and attributing a pipeline scope gap to the
        reader would corrupt the architecture decision gate."""
        return self.outcome is Outcome.ABSTAINED_GOLD_STATED


@dataclass(frozen=True)
class AnchorScore:
    anchor_id: str
    paper_id: str
    run_dir: Path
    rows: tuple[ScoredField, ...]

    def counts(self) -> Counter:
        return Counter(r.outcome for r in self.rows)

    def by_outcome(self, outcome: Outcome) -> tuple[ScoredField, ...]:
        return tuple(r for r in self.rows if r.outcome is outcome)


def _classify(paired, run: RunField | None) -> tuple[Outcome, CompareOutcome | None]:
    """Resolution order matters and is deliberate.

    Exclusion is a property of the FIELD (this field is never scored), while
    not-asked is a property of the RUN (this run did not ask). So exclusion
    resolves first: a rubric field that also went unasked is still an exclusion,
    and counting it as a run gap would overstate the pipeline's coverage hole."""
    if paired.excluded_rubric:
        return Outcome.EXCLUDED_WEAKER_RUBRIC, None

    if run is None or run.not_extracted:
        return Outcome.NOT_ASKED, None

    gold_stated = paired.gold_tag == "STATED"

    if run.final_tag != "STATED":
        return (Outcome.ABSTAINED_GOLD_STATED if gold_stated
                else Outcome.ABSTAINED_GOLD_SILENT), None

    # The run shipped a value.
    if not gold_stated:
        return Outcome.SHIPPED_GOLD_SILENT, None

    outcome = compare_field(paired.key, paired.gold_value, run.normalised_a)
    if outcome.comparability is not Comparability.COMPARABLE:
        return Outcome.SHIPPED_NOT_SCORABLE, outcome
    return (Outcome.SHIPPED_CORRECT if outcome.equal else Outcome.SHIPPED_WRONG), outcome


def score_anchor(anchor_id: str, run_dir: str | Path, *,
                 artefacts: RunArtefacts | None = None) -> AnchorScore:
    """Score one anchor's gold against one run.

    ``artefacts`` may be supplied to score a pre-loaded run (the tests use it to
    drive synthetic artefacts through the identical code path as production)."""
    gold = load_gold_spec(anchor_id)
    art = artefacts if artefacts is not None else load_run(run_dir)

    paired, excluded_paths = pair_fields(gold, art)

    rows: list[ScoredField] = []
    for p in paired:
        outcome, cmp_out = _classify(p, p.run)
        rows.append(ScoredField(
            anchor_id=anchor_id,
            paper_id=art.paper_id,
            key=p.key,
            dotted_path=p.dotted_path,
            outcome=outcome,
            gold_tag=p.gold_tag,
            gold_value=p.gold_value,
            run_tag=None if p.run is None else p.run.final_tag,
            run_value=None if p.run is None else p.run.normalised_a,
            compare=cmp_out,
            models_agree=None if p.run is None else p.run.models_agree,
            conditions=frozenset() if p.run is None else p.run.conditions,
            shipped_reason=None if p.run is None else p.run.shipped_reason,
        ))

    # The excluded paths are counted, never dropped -- an exclusion that leaves no
    # trace in the totals is indistinguishable from a field nobody thought about.
    for path in excluded_paths:
        rows.append(ScoredField(
            anchor_id=anchor_id, paper_id=art.paper_id,
            key=FieldKey(path), dotted_path=path,
            outcome=Outcome.EXCLUDED_NOT_FIELD_EXTRACTED,
            gold_tag="", gold_value=None, run_tag=None, run_value=None,
            compare=None, models_agree=None, conditions=frozenset(), shipped_reason=None,
        ))

    return AnchorScore(anchor_id=anchor_id, paper_id=art.paper_id,
                       run_dir=Path(run_dir), rows=tuple(rows))
