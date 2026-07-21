"""
fdr.py — Benjamini-Hochberg over the confirmatory family (§7.2).

The confirmatory family is the five first-order E_i PLUS the three end-to-end-
validated interaction pairs, × the primary metric, × the LOCKED anchors only —
NOT `str` (pilot), and the other seven pairs are exploratory (D-A9, D-A28, D-A33).
Marginals and Shapley allocations enter NO FDR family — they are deterministic
recombinations of the same cell vector, and entering all three would double-count
and brutalise the adjustment on the tests that carry the argument (§7.2).

This module applies BH to whatever family of p-values the caller assembles (a
single strategy's confirmatory coordinates, or the union across locked anchors);
it does not decide the family membership — `confirmatory.confirmatory_coordinates`
and the caller do. `mt_flag` is separate (§7.5) — it concerns multiple testing in
the audited paper's discovery process, not this analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class FdrDecision:
    key: object
    p_value: float
    adjusted_p: float
    rejected: bool
    rank: int


def benjamini_hochberg(
    pvalues: Mapping, q: float
) -> dict:
    """BH step-up at level `q`. Returns a FdrDecision per key with the BH-adjusted
    p-value and the reject flag. Ties break by insertion order; NaN p-values are
    treated as 1.0 (never rejected)."""
    if not (0.0 < q < 1.0):
        raise ValueError(f"q must be in (0,1); got {q}")
    items = [
        (k, (1.0 if p is None or p != p else float(p)))  # p!=p catches NaN
        for k, p in pvalues.items()
    ]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    if m == 0:
        return {}

    # Step-up: adjusted_(i) = min_{j>=i} ( m/j * p_(j) ), enforced monotone from the top.
    adjusted = [0.0] * m
    running_min = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        running_min = min(running_min, items[i][1] * m / rank)
        adjusted[i] = min(1.0, running_min)

    out: dict = {}
    for i, (key, p) in enumerate(items):
        out[key] = FdrDecision(
            key=key, p_value=p, adjusted_p=adjusted[i],
            rejected=adjusted[i] <= q, rank=i + 1,
        )
    return out


@dataclass(frozen=True)
class FdrReport:
    q: float
    decisions: dict            # key -> FdrDecision
    n_family: int
    n_rejected: int
    scope: str = "within_strategy"   # "within_strategy" | "corpus_confirmatory"

    def to_dict(self) -> dict:
        from ..schemas.decomposition import subset_label

        def _label(k):
            if isinstance(k, tuple) and len(k) == 2:
                # (strategy_label, coordinate) key from the corpus assembler.
                strat, coord = k
                coord_label = (
                    subset_label(frozenset(coord))
                    if isinstance(coord, (frozenset, set)) else str(coord)
                )
                return f"{strat}::{coord_label}"
            return subset_label(frozenset(k)) if isinstance(k, (frozenset, set)) else str(k)

        return {
            "q": self.q,
            "scope": self.scope,
            "n_family": self.n_family,
            "n_rejected": self.n_rejected,
            "decisions": {
                _label(k): {
                    "p_value": d.p_value,
                    "adjusted_p": d.adjusted_p,
                    "rejected": d.rejected,
                    "rank": d.rank,
                }
                for k, d in self.decisions.items()
            },
        }


def run_fdr(pvalues: Mapping, q: float, *, scope: str = "within_strategy") -> FdrReport:
    """Apply BH-FDR to a family of p-values. `scope` labels the family: a single
    strategy's coordinates are `within_strategy` (a diagnostic multiplicity control,
    NOT the §7.2 confirmatory verdict); the corpus assembler sets
    `corpus_confirmatory`."""
    decisions = benjamini_hochberg(pvalues, q)
    n_rej = sum(1 for d in decisions.values() if d.rejected)
    return FdrReport(
        q=q, decisions=decisions, n_family=len(decisions), n_rejected=n_rej, scope=scope
    )


def corpus_confirmatory_fdr(
    per_strategy_pvalues: Mapping[str, Mapping], q: float
) -> FdrReport:
    """The pre-registered §7.2 confirmatory FDR: BH over the UNION family across the
    locked anchors (mom6, drf, ...). `per_strategy_pvalues` maps each locked
    strategy label to its {coordinate: p_value}; the family is keyed by
    `(strategy_label, coordinate)` so the adjustment spans anchors, as RQ3's claims
    do (D-A9: per-strategy BH controls nothing at the level the claims are made).

    NOT the within-strategy `run_fdr` — that is a per-strategy diagnostic. `str`
    (pilot) must be excluded by the caller; only locked anchors enter (D-A33)."""
    family: dict = {}
    for strategy_label, coord_pvalues in per_strategy_pvalues.items():
        for coord, p in coord_pvalues.items():
            family[(strategy_label, coord)] = p
    return run_fdr(family, q, scope="corpus_confirmatory")
