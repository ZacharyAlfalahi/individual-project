"""
G3 metrics -- contract §3.1 and §3.6 (D34 mechanics, D37 conventions).

Turns a scored anchor into the contract's named metrics. This module owns the §3.1
per-anchor metrics and the §3.6 missed-evidence rate; the §3.2 2x2 arms and the
§3.3 aggregation altitudes live in their sibling modules.

**The universe is fixed by the GOLD, never by the run** (contract §8
non-elasticity: "fidelity aggregates cannot be emptied by laundering"). Every
denominator below is derived from gold fields, so a run that emits fewer fields
gets a worse coverage number rather than a smaller denominator.

**Two reason distributions are reported, never one.** The D9 merge attributes
each field to exactly one reason in a FIXED BRANCH ORDER -- the quote gate is
checked before the single-response branch -- so the shipped ``final_reason``
systematically under-counts one-model-mute fields (observed: 19 fields had one
model silent, 8 were labelled ``single_response``). Reporting only the shipped
attribution is misleading about causes; reporting only the order-free incidence
is dishonest about what the system actually emitted. Both, explicitly labelled,
and the incidence is declared NOT a partition -- it sums to more than n by
construction.

**NOT_ASKED is asymmetric, per D37 ruling 3.** It sits IN the coverage
denominator (contract §8 forbids shrinking the universe) and OUT of the §3.6
missed-evidence denominator (§3.6 counts fields where the model "returned
not_stated or failed the quote gate" -- a field never asked did neither).
Attributing a pipeline scope gap to the reader would corrupt the very number the
ReAct-vs-k=3 architecture decision turns on.
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.harness.gold_calibration import (  # noqa: E402
    EXCLUDED_OUTCOMES,
    SHIPPED_OUTCOMES,
    AnchorScore,
    Outcome,
)
from evaluation.harness.reportability import (  # noqa: E402
    G3Thresholds,
    Reportability,
    load_g3_thresholds,
    require_reportable,
)
from evaluation.harness.run_artefacts import RunArtefacts  # noqa: E402
from evaluation.harness.stats import Proportion  # noqa: E402


@dataclass(frozen=True)
class MetricBundle:
    """The §3.1/§3.6 metrics for one anchor.

    ``reportability`` has NO DEFAULT: a bundle cannot be constructed without the
    phase being decided, so no number can reach a renderer unstamped."""

    anchor_id: str
    paper_id: str
    reportability: Reportability
    thresholds: G3Thresholds

    universe: int                       # gold fields in the headline universe
    coverage: Proportion
    selective_accuracy: Proportion
    over_claim_rate: Proportion
    abstention_rate: Proportion
    missed_evidence_rate: Proportion

    # Two views of WHY, see the module docstring. Never merge them.
    shipped_reason_distribution: dict[str, int]      # a partition over asked fields
    condition_incidence: dict[str, int]              # NOT a partition
    outcome_distribution: dict[str, int]
    locator_success: dict[str, Proportion]


def _prop(label, k, n, th: G3Thresholds, note="") -> Proportion:
    return Proportion(label=label, numerator=k, denominator=n, z=th.wilson_z,
                      min_cell_n=th.min_cell_n, interval_label=th.interval_label,
                      note=note)


def compute_metrics(score: AnchorScore, artefacts: RunArtefacts | None = None, *,
                    thresholds: G3Thresholds | None = None) -> MetricBundle:
    """Compute the §3.1 + §3.6 metrics for one scored anchor.

    ``artefacts`` is needed only for the locator-success rate, which is a
    per-(field, model) quantity the scored rows do not carry."""
    th = thresholds if thresholds is not None else load_g3_thresholds()
    rows = score.rows

    # --- the universe (gold-fixed) -----------------------------------------
    # Excluded fields are out of every denominator: strategy_label is not reader
    # output at all, and prose has no scoring policy. They are still COUNTED in
    # the outcome distribution, so the exclusion is visible rather than implied.
    headline = [r for r in rows if r.outcome not in EXCLUDED_OUTCOMES]
    universe = len(headline)

    shipped = [r for r in headline if r.outcome in SHIPPED_OUTCOMES]
    scorable = [r for r in shipped if r.outcome is not Outcome.SHIPPED_NOT_SCORABLE]
    correct = [r for r in shipped if r.outcome is Outcome.SHIPPED_CORRECT]
    over_claims = [r for r in shipped if r.outcome is Outcome.SHIPPED_GOLD_SILENT]
    abstained = [r for r in headline
                 if r.outcome in (Outcome.ABSTAINED_GOLD_STATED, Outcome.ABSTAINED_GOLD_SILENT)]
    not_asked = [r for r in headline if r.outcome is Outcome.NOT_ASKED]

    # --- §3.6: gold-STATED fields the run was ACTUALLY ASKED about ----------
    asked = [r for r in headline if r.outcome is not Outcome.NOT_ASKED]
    gold_stated_asked = [r for r in asked if r.gold_tag == "STATED"]
    missed = [r for r in rows if r.is_missed_evidence]

    coverage = _prop("coverage", len(shipped), universe, th,
                     note=(f"{len(not_asked)} field(s) never asked are IN this denominator "
                           "(contract §8: the universe is gold-fixed)") if not_asked else "")
    selective = _prop("selective accuracy", len(correct), len(scorable), th,
                      note="P(correct | the system ships); over-claims are in the denominator")
    over_claim = _prop("over-claim rate", len(over_claims), len(scorable), th,
                       note="shipped a value where the gold records silence (a fabrication)")
    abstention = _prop("abstention rate", len(abstained), universe, th)
    missed_rate = _prop("missed-evidence rate (§3.6)", len(missed), len(gold_stated_asked), th,
                        note=("gold-STATED fields the run was asked about and did not ship; "
                              "NOT_ASKED excluded per D37 ruling 3"))

    # --- the two WHY views --------------------------------------------------
    shipped_reasons = Counter(r.shipped_reason for r in asked if r.shipped_reason)
    incidence: Counter = Counter()
    for r in asked:
        incidence.update(r.conditions)

    outcomes = Counter(r.outcome.value for r in rows)

    # --- locator success (per model, then pooled) ---------------------------
    locator: dict[str, Proportion] = {}
    if artefacts is not None:
        pooled_k = pooled_n = 0
        for role in ("a", "b"):
            attempted = [f for f in artefacts.fields.values()
                         if getattr(f, f"{role}_answered") and getattr(f, f"{role}_quote")]
            located = [f for f in attempted if getattr(f, f"{role}_located")]
            locator[f"model_{role}"] = _prop(
                f"locator success (model_{role})", len(located), len(attempted), th,
                note="of quotes PROPOSED, the fraction that located in the canonical text")
            pooled_k += len(located)
            pooled_n += len(attempted)
        locator["pooled"] = _prop("locator success (pooled)", pooled_k, pooled_n, th)

    return MetricBundle(
        anchor_id=score.anchor_id, paper_id=score.paper_id,
        reportability=score.reportability, thresholds=th,
        universe=universe,
        coverage=coverage, selective_accuracy=selective, over_claim_rate=over_claim,
        abstention_rate=abstention, missed_evidence_rate=missed_rate,
        shipped_reason_distribution=dict(shipped_reasons),
        condition_incidence=dict(incidence),
        outcome_distribution=dict(outcomes),
        locator_success=locator,
    )


def render_headline_table(bundle: MetricBundle, *, allow_non_reportable: bool = False) -> str:
    """Render the §3.1/§3.6 metrics.

    Refuses a non-reportable bundle unless the caller opts in EXPLICITLY. This is
    the gate: contract §1 forbids a Phase-D number entering the project, and an
    opt-in flag makes every such render greppable rather than accidental."""
    require_reportable(bundle.reportability, allow_non_reportable=allow_non_reportable)

    lines: list[str] = []
    if not bundle.reportability.reportable:
        lines.append(bundle.reportability.banner)
    lines.append(f"G3 gold calibration -- {bundle.anchor_id} ({bundle.paper_id})")
    lines.append(f"  gold universe (headline): {bundle.universe} fields")
    for p in (bundle.coverage, bundle.selective_accuracy, bundle.over_claim_rate,
              bundle.abstention_rate, bundle.missed_evidence_rate):
        lines.append(f"  {p.render()}")
        if p.note:
            lines.append(f"      ({p.note})")
    for name in ("model_a", "model_b", "pooled"):
        p = bundle.locator_success.get(name)
        if p is not None:
            lines.append(f"  {p.render()}")

    lines.append("  shipped reason (a PARTITION over asked fields -- the D9 merge's own "
                 "ordered attribution):")
    for k, v in sorted(bundle.shipped_reason_distribution.items(), key=lambda kv: -kv[1]):
        lines.append(f"      {k:26s} {v}")
    lines.append("  condition incidence (NOT a partition -- sums to more than n by design; "
                 "order-free, so it does not inherit the merge's branch ordering):")
    for k, v in sorted(bundle.condition_incidence.items(), key=lambda kv: -kv[1]):
        lines.append(f"      {k:26s} {v}")
    return "\n".join(lines)
