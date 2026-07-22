"""
G3 agreement calibration -- the §3.2 2x2 (D34 mechanics, D37 rulings 1/8/9).

Contract §3.2: *"the 2x2 -- P(correct | agree) vs P(correct | single model) vs
P(correct | disagree) -- on normalised values, per field type where n permits."*
D34 adds: *"the 2x2 A/(A+C) with Wilson CI and n; per-field-type cells at n >= 20,
else reported 'uncalibrated'; Cohen's kappa / Krippendorff's alpha over dual-model
output."*

Three things about this table are NOT self-evident and are declared here because
reading it without them would be misleading:

**1. The AGREE arm IS the ship set** (D37 ruling 8). On this pipeline D9 ships a
field exactly when both models answered, both quotes located, and the normalised
values match -- so ``P(correct | agree)`` and selective accuracy are the same
quantity computed two ways. That is a property of the merge, not a coincidence,
and it means the AGREE cell carries no information selective accuracy did not.
Its value is as the BASELINE the other arms are compared against.

**2. The other arms are COUNTERFACTUAL.** The system never ships a single-model
answer or a disagreement, so "correct" in those cells means *"would have been
correct had it shipped"*. They answer "what is the pipeline leaving on the table,
and at what error rate?" -- not "how accurate is the system?". Labelled as such
everywhere they are rendered.

**3. The DISAGREE arm has no single value to score** (D37 ruling 1). When the two
models differ there is no shipped candidate, so this reports
``P(gold in {a, b} | disagree)`` -- whether EITHER model was right -- which is an
UPPER BOUND over any tie-break rule the pipeline might later adopt. It is not
``P(correct | disagree)`` and must never be labelled as such.

**The fourth cell** (D37 ruling 9): ``AGREE_QUOTE_GATE_FAILED`` -- both models
answered and agreed, and neither quote located. Coverage lost to the LOCATOR
rather than to reading, which is precisely the population contract §3.6's
ReAct-vs-k=3 decision turns on. Additive to §3.2's three named cells, never a
substitution.

**Correctness on a gold-SILENT field.** A field where the gold records silence
and the models nonetheless answered is scored INCORRECT in every arm: the right
answer was to abstain. Excluding those fields would make ``P(correct | agree)``
optimistic by hiding exactly the correlated-fabrication case the dual-model gate
is structurally blind to -- which is the single most interesting failure in this
corpus.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.harness.compare_policy import Comparability, compare_field  # noqa: E402
from evaluation.harness.field_pairing import RUBRIC_FIELDS  # noqa: E402
from evaluation.harness.gold_calibration import EXCLUDED_OUTCOMES, AnchorScore  # noqa: E402
from evaluation.harness.reportability import G3Thresholds, load_g3_thresholds  # noqa: E402
from evaluation.harness.run_artefacts import RunField  # noqa: E402
from evaluation.harness.stats import Proportion  # noqa: E402

SILENT = "__SILENT__"


class Arm(str, Enum):
    AGREE = "agree"
    SINGLE_MODEL = "single_model"
    DISAGREE = "disagree"
    AGREE_QUOTE_GATE_FAILED = "agree_quote_gate_failed"
    NOT_IN_2X2 = "not_in_2x2"


# Human-readable statements of what "correct" means in each arm. Rendered with
# the cell, so a reader cannot see the number without the caveat.
ARM_SEMANTICS: dict[Arm, str] = {
    Arm.AGREE: "P(correct | models agree AND both quotes located) -- the SHIP SET; "
               "identical to selective accuracy by construction of the D9 merge",
    Arm.SINGLE_MODEL: "COUNTERFACTUAL: would the lone answer have been correct had it shipped? "
                      "The pipeline never ships these",
    Arm.DISAGREE: "UPPER BOUND: P(gold in {a, b} | disagree) -- whether EITHER model was right. "
                  "Not P(correct | disagree); there is no single value to score",
    Arm.AGREE_QUOTE_GATE_FAILED: "COUNTERFACTUAL: both models agreed and neither quote located. "
                                 "Coverage lost to the LOCATOR, not to reading (§3.6)",
}


@dataclass(frozen=True)
class ArmObservation:
    field: str
    field_type: str
    arm: Arm
    gold_tag: str
    gold_value: object
    candidate_a: object
    candidate_b: object
    correct: bool | None      # None = arm not scoreable for this field
    note: str = ""


def classify_arm(run: RunField) -> Arm:
    """Which §3.2 cell a field falls in, from the RECOMPUTED dual-model view.

    Never from ``trace["agreement"]``, which encodes "shipped STATED"."""
    if not run.a_answered and not run.b_answered:
        return Arm.NOT_IN_2X2
    if run.a_answered != run.b_answered:
        return Arm.SINGLE_MODEL
    # both answered
    if run.models_agree:
        return Arm.AGREE if (run.a_located and run.b_located) else Arm.AGREE_QUOTE_GATE_FAILED
    return Arm.DISAGREE


def _matches_gold(key, gold_tag: str, gold_value, candidate) -> bool | None:
    """Is one model's candidate value the gold answer?

    A gold-SILENT field is incorrect for any candidate: the right answer was to
    abstain, and the model did not."""
    if candidate is None:
        return None
    if gold_tag != "STATED":
        return False
    out = compare_field(key, gold_value, candidate)
    if out.comparability is not Comparability.COMPARABLE:
        return None
    return bool(out.equal)


def build_observations(score: AnchorScore, artefacts) -> list[ArmObservation]:
    """One observation per asked, non-excluded field."""
    from agents.librarian.pipeline.prompts import load_prompt_manifest

    manifest = load_prompt_manifest()
    obs: list[ArmObservation] = []

    for row in score.rows:
        if row.outcome in EXCLUDED_OUTCOMES or row.key.name in RUBRIC_FIELDS:
            continue
        run = artefacts.fields.get(row.key.name)
        if run is None:
            continue
        arm = classify_arm(run)
        if arm is Arm.NOT_IN_2X2:
            continue

        ftype = manifest.field_types.get(row.key.name, "unbound")
        a = run.normalised_a if run.a_answered else None
        b = run.normalised_b if run.b_answered else None
        a_ok = _matches_gold(row.key, row.gold_tag, row.gold_value, a)
        b_ok = _matches_gold(row.key, row.gold_tag, row.gold_value, b)

        if arm is Arm.DISAGREE:
            # D37 ruling 1: an upper bound over any tie-break rule.
            if a_ok is None and b_ok is None:
                correct = None
            else:
                correct = bool(a_ok) or bool(b_ok)
            note = "upper bound: gold in {a, b}"
        elif arm is Arm.SINGLE_MODEL:
            correct = a_ok if a is not None else b_ok
            note = "counterfactual: the pipeline does not ship a lone answer"
        else:
            # AGREE / AGREE_QUOTE_GATE_FAILED: the models concur, so either
            # candidate stands for the pair.
            correct = a_ok if a_ok is not None else b_ok
            note = "" if arm is Arm.AGREE else "counterfactual: lost to the locator, not to reading"

        obs.append(ArmObservation(
            field=row.key.name, field_type=ftype, arm=arm,
            gold_tag=row.gold_tag, gold_value=row.gold_value,
            candidate_a=a, candidate_b=b, correct=correct, note=note,
        ))
    return obs


@dataclass(frozen=True)
class ArmCell:
    arm: Arm
    field_type: str | None       # None = pooled across types
    proportion: Proportion
    semantics: str

    def render(self) -> str:
        scope = self.field_type or "all types"
        return f"{self.arm.value} [{scope}] {self.proportion.render()}"


@dataclass(frozen=True)
class AgreementTable:
    anchor_id: str
    reportability: object
    observations: tuple[ArmObservation, ...]
    pooled: dict[Arm, ArmCell]
    per_type: dict[tuple[Arm, str], ArmCell]
    kappa: float | None
    kappa_note: str
    arm_counts: dict[str, int]


def cohens_kappa(pairs: list[tuple[object, object]]) -> tuple[float | None, str]:
    """Cohen's kappa over the two models' labels.

    Labels are each model's normalised value, or ``SILENT`` -- so the statistic
    measures concordance on BOTH what they said and whether they spoke, which is
    the dual-model behaviour D34 asks about.

    Returns ``(None, reason)`` on a degenerate distribution (``p_e == 1``, i.e.
    every observation in one category) rather than raising or emitting NaN. That
    is the EXPECTED case on this corpus: model_b is silent on most fields, so the
    joint distribution concentrates hard.

    D37 ruling 6: kappa only; the Krippendorff substitution is declared. For
    nominal, pairwise-complete, two-coder data the two nearly coincide, and alpha
    would need a distance metric and a coincidence matrix for no added signal.
    """
    n = len(pairs)
    if n == 0:
        return None, "no observations"
    cats = sorted({str(x) for pair in pairs for x in pair})
    if len(cats) < 2:
        return None, f"degenerate: all observations fall in one category ({cats[0] if cats else '-'})"
    agree = sum(1 for a, b in pairs if a == b)
    p_o = agree / n
    ma = Counter(str(a) for a, _ in pairs)
    mb = Counter(str(b) for _, b in pairs)
    p_e = sum((ma[c] / n) * (mb[c] / n) for c in cats)
    if abs(1.0 - p_e) < 1e-12:
        return None, "degenerate: chance agreement is 1.0, kappa undefined"
    return (p_o - p_e) / (1.0 - p_e), ""


def _kappa_population(score: AnchorScore, artefacts) -> list[tuple[object, object]]:
    """Every asked, non-excluded field as a (model_a label, model_b label) pair,
    with silence as an explicit category. See build_agreement_table."""
    rows: list[tuple[object, object]] = []
    for row in score.rows:
        if row.outcome in EXCLUDED_OUTCOMES or row.key.name in RUBRIC_FIELDS:
            continue
        run = artefacts.fields.get(row.key.name)
        if run is None:
            continue
        rows.append((run.normalised_a if run.a_answered else SILENT,
                     run.normalised_b if run.b_answered else SILENT))
    return rows


def build_agreement_table(score: AnchorScore, artefacts, *,
                          thresholds: G3Thresholds | None = None) -> AgreementTable:
    th = thresholds if thresholds is not None else load_g3_thresholds()
    obs = build_observations(score, artefacts)

    def cell(rows: list[ArmObservation], arm: Arm, ftype: str | None) -> ArmCell:
        scoreable = [o for o in rows if o.correct is not None]
        k = sum(1 for o in scoreable if o.correct)
        label = f"P(correct | {arm.value})" if arm is not Arm.DISAGREE else "P(gold in {a,b} | disagree)"
        return ArmCell(
            arm=arm, field_type=ftype,
            proportion=Proportion(label=label, numerator=k, denominator=len(scoreable),
                                  z=th.wilson_z, min_cell_n=th.min_cell_n,
                                  interval_label=th.interval_label,
                                  note=ARM_SEMANTICS.get(arm, "")),
            semantics=ARM_SEMANTICS.get(arm, ""),
        )

    by_arm: dict[Arm, list[ArmObservation]] = defaultdict(list)
    by_arm_type: dict[tuple[Arm, str], list[ArmObservation]] = defaultdict(list)
    for o in obs:
        by_arm[o.arm].append(o)
        by_arm_type[(o.arm, o.field_type)].append(o)

    # Every named arm gets a cell even when empty -- an absent row reads as
    # "not measured", an n=0 row reads as "measured, nothing there".
    pooled = {arm: cell(by_arm.get(arm, []), arm, None)
              for arm in (Arm.AGREE, Arm.SINGLE_MODEL, Arm.DISAGREE,
                          Arm.AGREE_QUOTE_GATE_FAILED)}
    per_type = {key: cell(rows, key[0], key[1]) for key, rows in by_arm_type.items()}

    # Kappa runs over ALL asked fields, INCLUDING both-silent -- not just the 2x2
    # arms. D34 asks for kappa "over dual-model output", and declining to answer
    # IS output (the model returns answered=false); silence is a category, not a
    # missing value. Restricting to the arms would silently condition the
    # statistic on "at least one model spoke", which is a different question and
    # materially different number (drf 0.214 arms-only vs 0.289 all-asked).
    kappa_rows = _kappa_population(score, artefacts)
    kappa, kappa_note = cohens_kappa(kappa_rows)

    return AgreementTable(
        anchor_id=score.anchor_id, reportability=score.reportability,
        observations=tuple(obs), pooled=pooled, per_type=per_type,
        kappa=kappa, kappa_note=kappa_note,
        arm_counts=dict(Counter(o.arm.value for o in obs)),
    )


def render_agreement_table(table: AgreementTable, *, allow_non_reportable: bool = False) -> str:
    from evaluation.harness.reportability import require_reportable

    require_reportable(table.reportability, allow_non_reportable=allow_non_reportable)
    lines: list[str] = []
    if not table.reportability.reportable:
        lines.append(table.reportability.banner)
    lines.append(f"§3.2 agreement calibration -- {table.anchor_id}")
    for arm in (Arm.AGREE, Arm.SINGLE_MODEL, Arm.DISAGREE, Arm.AGREE_QUOTE_GATE_FAILED):
        c = table.pooled[arm]
        lines.append(f"  {c.proportion.render()}")
        lines.append(f"      {c.semantics}")
    if table.kappa is None:
        lines.append(f"  Cohen's kappa: not computable -- {table.kappa_note}")
    else:
        lines.append(f"  Cohen's kappa (dual-model, incl. silence): {table.kappa:.3f}")
    lines.append(f"  arm counts: {table.arm_counts}")
    return "\n".join(lines)
