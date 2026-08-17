"""P2 coverage-boundary failure taxonomy (WS-C).

Two deterministic, pre-registered pieces the report consumes:

  * a **stratified failure sampler** — the taxonomy strata are
    ``arm × divergence-magnitude`` (pre-registered in ``docs/thresholds.yaml``
    ``p2_codegen.taxonomy_sampling.strata``). For each (arm, magnitude) cell it
    selects up to ``per_stratum`` members for qualitative inspection, ordered
    worst-divergence-first (lowest correlation) then paper_id — a FROZEN order,
    so the sample is reproducible and precedes the inspection (no cherry-pick).
    "Failure" here means DISAGREEMENT (high/medium divergence); P2 has no oracle,
    so it is never a correctness failure.

  * **typed eligibility-exclusion accounting** — text-acquisition / text-quality
    failures are COUNTED, typed by their ``exclusion_reason``, and reported as an
    explicit line in the taxonomy. They are NEVER silent drops: a paper the
    census could not admit is a datum about coverage, not a gap in the table.

Both are pure functions over the census + the per-member agreements; no model
calls, no randomness.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from evaluation.codegen.census import CensusResult
from evaluation.codegen.p2_metrics import MemberAgreement

#: The pre-registered taxonomy strata axes (mirrors
#: ``p2_codegen.taxonomy_sampling.strata``). Kept here as the code-side default;
#: the thresholds value is the authoritative pre-registration.
STRATA_AXES: tuple[str, ...] = ("arm", "divergence_magnitude")


def _corr_sort_key(m: MemberAgreement) -> tuple:
    """Worst-divergence-first: non-finite correlation is the worst (sorts first),
    then ascending correlation, then paper_id for a total, frozen order."""
    corr = m.correlation
    finite = corr == corr and corr not in (float("inf"), float("-inf"))  # not NaN/inf
    # (0, corr) for finite so lower corr sorts first; (-1, 0) for non-finite so it
    # precedes every finite correlation (the very worst divergence).
    primary = (0, corr) if finite else (-1, 0.0)
    return (primary, m.paper_id)


@dataclass(frozen=True)
class TaxonomySample:
    """The stratified sample: ``cells`` maps ``(arm, magnitude)`` -> the selected
    members (frozen order); ``strata_axes`` records the pre-registered axes."""

    strata_axes: tuple[str, ...]
    per_stratum: int
    cells: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "strata_axes": list(self.strata_axes),
            "per_stratum": self.per_stratum,
            "cells": {
                f"{arm}|{mag}": [m.to_dict() for m in members]
                for (arm, mag), members in self.cells.items()
            },
        }


def stratified_failure_sample(
    agreements: tuple[MemberAgreement, ...],
    *,
    per_stratum: int = 2,
    strata_axes: tuple[str, ...] = STRATA_AXES,
) -> TaxonomySample:
    """Deterministically select up to ``per_stratum`` members per
    ``(arm, divergence_magnitude)`` cell, worst-divergence-first. No RNG — the
    order is frozen by ``_corr_sort_key`` so the sample precedes inspection."""
    if per_stratum < 0:
        raise ValueError("per_stratum must be >= 0")
    buckets: dict[tuple[str, str], list[MemberAgreement]] = {}
    for m in agreements:
        buckets.setdefault((m.arm, m.stratum), []).append(m)
    cells: dict[tuple[str, str], tuple[MemberAgreement, ...]] = {}
    for key, members in buckets.items():
        ordered = sorted(members, key=_corr_sort_key)
        cells[key] = tuple(ordered[:per_stratum])
    return TaxonomySample(
        strata_axes=tuple(strata_axes), per_stratum=per_stratum, cells=cells
    )


@dataclass(frozen=True)
class EligibilityAccounting:
    """Typed COUNTED eligibility exclusions: the total, the per-reason counts, and
    the papers under each reason. Never a silent drop."""

    total: int
    by_reason: dict = field(default_factory=dict)          # reason -> count
    papers_by_reason: dict = field(default_factory=dict)   # reason -> [paper_id]

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "by_reason": dict(self.by_reason),
            "papers_by_reason": {k: list(v) for k, v in self.papers_by_reason.items()},
        }


def eligibility_exclusion_accounting(census: CensusResult) -> EligibilityAccounting:
    """Count and type every eligibility exclusion (text-acquisition / quality
    failure) from the census — COUNTED, typed by ``exclusion_reason``, never
    dropped silently."""
    excluded = census.eligibility_exclusions()
    counts: Counter[str] = Counter()
    papers: dict[str, list[str]] = {}
    for m in excluded:
        reason = m.exclusion_reason or "unspecified"
        counts[reason] += 1
        papers.setdefault(reason, []).append(m.paper_id)
    return EligibilityAccounting(
        total=len(excluded),
        by_reason=dict(counts),
        papers_by_reason=papers,
    )
