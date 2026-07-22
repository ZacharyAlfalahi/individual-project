"""
G3 aggregation altitudes -- contract §3.3 (the G = 3 rule).

§3.3: *"report per-paper results, macro-average across papers, micro-average
across fields, and leave-one-paper-out sensitivity. Field-level intervals are
labelled descriptive; a clustered interval may appear as sensitivity analysis
only. No population-level generalisation claim: the unit of generalisation is the
observed gold corpus."*

Four honest limits are built into this module rather than left to the reader.

**1. The anchor set is declared, and shortfall is loud.** The gold corpus is three
anchors; when an anchor has no extraction artefact (e.g. a run that refused at the
crux field), an aggregate over the remainder must never read as if it covered the corpus, so
``anchors_scored`` and ``anchors_expected`` are both carried and every render shows
how many of the corpus were scored.

**2. With G = 2, leave-one-paper-out DEGENERATES to the per-paper table.** Dropping
one of two papers leaves one paper. That is still worth reporting -- it is the
clearest statement of how much a single paper moves a headline number -- but it is
not a sensitivity analysis in any inferential sense, and calling it one would be a
false claim of robustness. The renderer says so.

**3. No dispersion statistic on 2-3 points.** A standard deviation over two or
three papers is not an estimate of anything; the constituents are emitted RAW
instead, per contract §5.3's house style (never the summary alone).

**4. Aggregate reportability is the AND over constituents.** One Phase-D paper
contaminates the aggregate, because §1's rule is about which numbers may be
reported, not about how many. The failing constituent is named in the reason.

Macro vs micro is not a formality here: the anchors differ enormously (locator
success 48% on BBW vs 93% on JNPS), so an unweighted mean of rates and a pooled
count answer different questions and will disagree. Both are reported.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.harness.calibration_report import MetricBundle  # noqa: E402
from evaluation.harness.reportability import (  # noqa: E402
    G3Thresholds,
    Reportability,
    load_g3_thresholds,
    require_reportable,
)
from evaluation.harness.stats import Proportion  # noqa: E402

# The gold corpus. An aggregate that scores fewer must say so.
ANCHOR_SET: tuple[str, ...] = ("drf", "mom6", "str")

# The §3.1/§3.6 metrics carried through every altitude.
_METRICS: tuple[str, ...] = (
    "coverage", "selective_accuracy", "over_claim_rate",
    "abstention_rate", "missed_evidence_rate",
)


@dataclass(frozen=True)
class MacroStat:
    """An unweighted mean across papers, with its constituents.

    ``values`` is emitted alongside the mean deliberately: on G <= 3 the mean is a
    summary of so few points that showing it alone would imply a precision that is
    not there. No standard deviation -- see the module docstring."""

    label: str
    values: tuple[tuple[str, float], ...]      # (anchor_id, rate)
    n_papers: int

    @property
    def mean(self) -> float | None:
        if not self.values:
            return None
        return sum(v for _, v in self.values) / len(self.values)

    def render(self) -> str:
        if self.mean is None:
            return f"{self.label} (macro): n=0 papers"
        parts = ", ".join(f"{a}={v:.1%}" for a, v in self.values)
        return (f"{self.label} (macro, unweighted over {self.n_papers} papers): "
                f"{self.mean:.1%}  [constituents: {parts}]")


@dataclass(frozen=True)
class AggregateBundle:
    anchors_scored: tuple[str, ...]
    anchors_expected: tuple[str, ...]
    reportability: Reportability
    thresholds: G3Thresholds
    per_paper: dict[str, MetricBundle]
    micro: dict[str, Proportion]
    macro: dict[str, MacroStat]
    leave_one_out: dict[str, dict[str, Proportion]]

    @property
    def complete(self) -> bool:
        return set(self.anchors_scored) == set(self.anchors_expected)

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(a for a in self.anchors_expected if a not in self.anchors_scored)


def _aggregate_reportability(bundles: dict[str, MetricBundle]) -> Reportability:
    """AND over constituents. One non-reportable paper makes the aggregate
    non-reportable, and the reason names it -- §1 governs which numbers may be
    reported, not how many are averaged together."""
    if not bundles:
        return Reportability(phase=None, reportable=False,
                             reason="no anchors scored", model_a_id="", model_b_id="")
    bad = [a for a, b in bundles.items() if not b.reportability.reportable]
    first = next(iter(bundles.values())).reportability
    if bad:
        blocker = bundles[bad[0]].reportability
        return Reportability(
            phase=blocker.phase, reportable=False,
            reason=(f"aggregate contains non-reportable anchor(s) {bad}: {blocker.reason}"),
            model_a_id=blocker.model_a_id, model_b_id=blocker.model_b_id,
        )
    phases = {b.reportability.phase for b in bundles.values()}
    if len(phases) > 1:
        return Reportability(
            phase=None, reportable=False,
            reason=f"anchors were produced under different phases {sorted(phases)}; "
                   "calibration is pair-specific and does not pool across phases",
            model_a_id=first.model_a_id, model_b_id=first.model_b_id,
        )
    return Reportability(phase=first.phase, reportable=True,
                         reason="all constituent anchors reportable under one phase",
                         model_a_id=first.model_a_id, model_b_id=first.model_b_id)


def _pool(bundles: list[MetricBundle], metric: str, th: G3Thresholds,
          label_suffix: str = "") -> Proportion:
    """Micro-average: pool the raw counts, then form one rate."""
    k = sum(getattr(b, metric).numerator for b in bundles)
    n = sum(getattr(b, metric).denominator for b in bundles)
    base = getattr(bundles[0], metric).label if bundles else metric
    return Proportion(label=f"{base}{label_suffix}", numerator=k, denominator=n,
                      z=th.wilson_z, min_cell_n=th.min_cell_n,
                      interval_label=th.interval_label)


def aggregate(bundles: dict[str, MetricBundle], *,
              anchors_expected: tuple[str, ...] = ANCHOR_SET,
              thresholds: G3Thresholds | None = None) -> AggregateBundle:
    """Build the §3.3 altitudes from per-anchor metric bundles."""
    th = thresholds if thresholds is not None else load_g3_thresholds()
    ordered = {a: bundles[a] for a in sorted(bundles)}
    as_list = list(ordered.values())

    micro = {m: _pool(as_list, m, th, " (micro)") for m in _METRICS} if as_list else {}

    macro: dict[str, MacroStat] = {}
    for m in _METRICS:
        vals = []
        for anchor, b in ordered.items():
            p = getattr(b, m)
            if p.value is not None:          # an n=0 paper contributes no rate
                vals.append((anchor, p.value))
        macro[m] = MacroStat(label=getattr(as_list[0], m).label if as_list else m,
                             values=tuple(vals), n_papers=len(vals))

    # Leave-one-paper-out: micro over the remainder. With two papers this leaves
    # one, which is the per-paper row -- reported, but not a sensitivity analysis.
    loo: dict[str, dict[str, Proportion]] = {}
    for dropped in ordered:
        rest = [b for a, b in ordered.items() if a != dropped]
        if not rest:
            continue
        loo[dropped] = {m: _pool(rest, m, th, f" (micro, minus {dropped})") for m in _METRICS}

    return AggregateBundle(
        anchors_scored=tuple(ordered),
        anchors_expected=anchors_expected,
        reportability=_aggregate_reportability(ordered),
        thresholds=th,
        per_paper=ordered,
        micro=micro,
        macro=macro,
        leave_one_out=loo,
    )


def render_aggregate(agg: AggregateBundle, *, allow_non_reportable: bool = False) -> str:
    require_reportable(agg.reportability, allow_non_reportable=allow_non_reportable)

    lines: list[str] = []
    if not agg.reportability.reportable:
        lines.append(agg.reportability.banner)
    lines.append(f"§3.3 aggregation -- {len(agg.anchors_scored)} of "
                 f"{len(agg.anchors_expected)} anchors ({', '.join(agg.anchors_scored)})")
    if not agg.complete:
        lines.append(f"  INCOMPLETE: no extraction artefact for {list(agg.missing)}; "
                     "this aggregate does NOT cover the gold corpus")
    lines.append("  Unit of generalisation is the observed gold corpus (§3.3): "
                 "no population-level claim is made.")

    lines.append("  -- micro (pooled across fields) --")
    for m in _METRICS:
        lines.append(f"      {agg.micro[m].render()}")
    lines.append("  -- macro (unweighted across papers) --")
    for m in _METRICS:
        lines.append(f"      {agg.macro[m].render()}")

    lines.append("  -- leave-one-paper-out --")
    if len(agg.anchors_scored) <= 2:
        lines.append(f"      NOTE: with {len(agg.anchors_scored)} papers, LOO leaves "
                     f"{len(agg.anchors_scored) - 1} and degenerates to the per-paper row. "
                     "Reported as the effect of a single paper, NOT as a robustness claim.")
    for dropped, metrics in agg.leave_one_out.items():
        lines.append(f"      drop {dropped}:")
        for m in _METRICS:
            lines.append(f"          {metrics[m].render()}")
    return "\n".join(lines)
