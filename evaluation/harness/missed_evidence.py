"""
G3 missed-evidence decomposition -- the §3.6 gate input.

Contract §3.6 defines the missed-evidence rate as *"fraction of gold-STATED
fields (gold carries a locating quote) where the model returned not_stated **or
failed the quote gate**"*, and routes it: coverage < 80% driven by missed
evidence -> build retrieval ReAct; driven by disagreement/decoding noise ->
enable the k=3 ladder.

**Those two clauses bundle mechanisms that need different remedies.** "Returned
not_stated" is a retrieval failure -- the model did not find the evidence, and a
retrieval loop is the matched remedy. "Failed the quote gate" can be something
else entirely: the model found the evidence, read it correctly, and the LOCATOR
refused the span. No number of ReAct iterations fixes a quote that cannot match.

This module does not change the gate. It MEASURES the composition of the
missed-evidence set so the gate is applied to a decomposed quantity rather than a
bundled one. Reporting the decomposition is measurement; changing the routing
rule would be a contract amendment and is not done here.

**Partition, computed from the order-free conditions** -- never from
``final_reason``, which is biased by the D9 merge's fixed branch order (the quote
gate is checked before the single-response branch, so it absorbs fields that were
also one-model-mute):

  NOT_RETRIEVED   neither model answered. A genuine retrieval failure; ReAct is
                  the matched remedy.
  GATE_LOST       at least one model answered with a value that MATCHES GOLD, and
                  no quote located. The evidence was found and correct; only the
                  locator refused it. Neither ReAct nor k=3 addresses this.
  READ_WRONG      models answered but the value is wrong, or they disagreed.
                  Reading or decoding noise; k=3 is the matched remedy.

The three are mutually exclusive and exhaust the missed-evidence set, so the
shares sum to 1 and the "driven by" clause has an unambiguous referent.
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

from evaluation.harness.agreement_calibration import _matches_gold  # noqa: E402
from evaluation.harness.gold_calibration import AnchorScore  # noqa: E402
from evaluation.harness.reportability import G3Thresholds, load_g3_thresholds  # noqa: E402
from evaluation.harness.stats import Proportion  # noqa: E402


class Mechanism(str, Enum):
    NOT_RETRIEVED = "not_retrieved"
    GATE_LOST = "gate_lost"
    MERGE_REFUSED = "merge_refused"
    VALUE_WRONG = "value_wrong"


# The remedy each mechanism routes to under evaluation contract v1.1 §3.6.
MECHANISM_REMEDY: dict[Mechanism, str] = {
    Mechanism.NOT_RETRIEVED: "retrieval ReAct -- the model did not find the evidence",
    Mechanism.GATE_LOST: "LOCATOR remediation (prompt-nudge, then ladder extension) -- the model "
                         "found it and was right; neither ReAct nor k=3 addresses a span that "
                         "cannot match",
    Mechanism.MERGE_REFUSED: "PAIR remediation -- a model had the gold value and the merge refused "
                             "it, typically because the other model was silent. One model's "
                             "systematic abstention against D9's both-must-answer rule; neither "
                             "retrieval nor self-consistency addresses abstention",
    Mechanism.VALUE_WRONG: "k=3 ladder -- no model produced the gold value (reading / decoding noise)",
}


@dataclass(frozen=True)
class MissedField:
    field: str
    mechanism: Mechanism
    gold_value: object
    candidate_a: object
    candidate_b: object
    note: str


@dataclass(frozen=True)
class MissedEvidenceDecomposition:
    anchor_id: str
    total: int
    fields: tuple[MissedField, ...]
    shares: dict[str, Proportion]

    @property
    def counts(self) -> dict[str, int]:
        return dict(Counter(f.mechanism.value for f in self.fields))

    @property
    def dominant(self) -> Mechanism | None:
        """The strict plurality mechanism, or None on a tie.

        None is a real outcome and must not be silently broken: a tie means the
        evidence does not identify one remedy, which is a finding, not a defect."""
        c = Counter(f.mechanism for f in self.fields)
        if not c:
            return None
        ranked = c.most_common()
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            return None
        return ranked[0][0]


def _classify(row, run) -> tuple[Mechanism, str]:
    if run is None or (not run.a_answered and not run.b_answered):
        return Mechanism.NOT_RETRIEVED, "neither model answered"

    a = run.normalised_a if run.a_answered else None
    b = run.normalised_b if run.b_answered else None
    a_ok = _matches_gold(row.key, row.gold_tag, row.gold_value, a)
    b_ok = _matches_gold(row.key, row.gold_tag, row.gold_value, b)

    # Did a model that was RIGHT lose its span? Check per model, so a correct
    # model_a whose quote failed still counts even if model_b was wrong.
    for ok, located, who in ((a_ok, run.a_located, "model_a"), (b_ok, run.b_located, "model_b")):
        if ok and not located:
            return Mechanism.GATE_LOST, f"{who} had the gold value; its quote did not locate"

    if a_ok or b_ok:
        # Right value, and it DID locate -- so the field was lost on AGREEMENT
        # grounds, not evidence grounds: the other model was silent or differed.
        # Neither retrieval nor self-consistency addresses this; it is a property
        # of the PAIR against D9's both-must-answer rule.
        silent = "model_b" if (a_ok and not run.b_answered) else (
            "model_a" if (b_ok and not run.a_answered) else "the other model")
        return Mechanism.MERGE_REFUSED, f"a model had the gold value; {silent} was silent/differed"
    return Mechanism.VALUE_WRONG, "answered, but no model produced the gold value"


def decompose(score: AnchorScore, artefacts, *,
              thresholds: G3Thresholds | None = None) -> MissedEvidenceDecomposition:
    th = thresholds if thresholds is not None else load_g3_thresholds()
    rows = [r for r in score.rows if r.is_missed_evidence]

    out: list[MissedField] = []
    for row in rows:
        run = artefacts.fields.get(row.key.name)
        mech, note = _classify(row, run)
        out.append(MissedField(
            field=row.key.name, mechanism=mech, gold_value=row.gold_value,
            candidate_a=None if run is None or not run.a_answered else run.normalised_a,
            candidate_b=None if run is None or not run.b_answered else run.normalised_b,
            note=note,
        ))

    total = len(out)
    counts = Counter(f.mechanism for f in out)
    shares = {
        m.value: Proportion(label=f"share {m.value}", numerator=counts.get(m, 0),
                            denominator=total, z=th.wilson_z, min_cell_n=th.min_cell_n,
                            interval_label=th.interval_label, note=MECHANISM_REMEDY[m])
        for m in Mechanism
    }
    return MissedEvidenceDecomposition(anchor_id=score.anchor_id, total=total,
                                       fields=tuple(out), shares=shares)


def render_decomposition(d: MissedEvidenceDecomposition) -> str:
    lines = [f"§3.6 missed-evidence decomposition -- {d.anchor_id} (n={d.total})"]
    for m in Mechanism:
        p = d.shares[m.value]
        lines.append(f"  {p.render()}")
        lines.append(f"      remedy: {MECHANISM_REMEDY[m]}")
    dom = d.dominant
    lines.append(f"  dominant mechanism: {dom.value if dom else 'TIE -- no single remedy indicated'}")
    return "\n".join(lines)
