"""Development-window equivalence test (TOST) on residual alpha — registered SC-SCI-14.

The protocol registers: ``method: tost``, ``margin_ref: scientist.equivalence_margin``
(δ = 0.5 × ϑ, derived in code, never hardcoded), ``bh_fdr_q: 0.10`` as **its own** family,
``window: development_only`` needing ≥ 60 months, and one asymmetry that governs how every
result here reads:

    **failing to establish equivalence is never evidence of a non-zero alpha.**

Two one-sided tests against the margin:

    H0_lower: alpha ≤ −δ   →  t = (alpha + δ)/se,  p = P(T > t)
    H0_upper: alpha ≥ +δ   →  t = (alpha − δ)/se,  p = P(T < t)
    p_TOST = max(p_lower, p_upper)

Equivalence is established when the BH-adjusted p_TOST clears q — i.e. the residual alpha is
provably inside [−δ, +δ]. This family is never pooled with the proposal families: it asks a
different question of the same candidates, and pooling would change both answers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from scipy import stats

from agents.auditor.checks.fdr import benjamini_hochberg

_REPO_ROOT = Path(__file__).resolve().parents[2]
_THRESHOLDS = _REPO_ROOT / "docs" / "thresholds.yaml"

#: The registered minimum development window for this test.
MIN_MONTHS = 60

ASYMMETRY = ("failing to establish equivalence is NOT evidence of a non-zero alpha — the test "
             "can only ever establish the positive claim that alpha lies inside the margin")


class EquivalenceInputError(ValueError):
    """A candidate cannot be tested as supplied — raised, never silently skipped."""


def load_margin(thresholds_path: Path | None = None) -> dict:
    """δ = ``factor_of_vartheta`` × ϑ, both read from thresholds (fail-loud). The margin is
    DERIVED, never written down separately, so it cannot drift from the materiality floor."""
    doc = yaml.safe_load((thresholds_path or _THRESHOLDS).read_text(encoding="utf-8"))
    try:
        block = doc["scientist"]["equivalence_margin"]
        factor = float(block["factor_of_vartheta"])
        vartheta = float(doc["auditor"]["practical_significance"]["vartheta"])
    except (KeyError, TypeError) as exc:
        raise KeyError(
            "thresholds.yaml is missing scientist.equivalence_margin.factor_of_vartheta or "
            "auditor.practical_significance.vartheta — the TOST margin refuses to be invented"
        ) from exc
    return {"delta": factor * vartheta, "factor_of_vartheta": factor, "vartheta": vartheta,
            "margin_ref": "scientist.equivalence_margin (derived)"}


@dataclass(frozen=True)
class EquivalenceInput:
    """One candidate's development-window residual alpha and its standard error."""

    candidate_id: str
    alpha: float
    se: float
    n_months: int
    n_regressors: int = 4          # BBW-4 benchmark; df = n_months − n_regressors − 1

    @property
    def df(self) -> int:
        return int(self.n_months - self.n_regressors - 1)

    def __post_init__(self) -> None:
        if self.se <= 0:
            raise EquivalenceInputError(
                f"{self.candidate_id}: standard error must be positive, got {self.se}")
        if self.df <= 0:
            raise EquivalenceInputError(
                f"{self.candidate_id}: non-positive degrees of freedom ({self.df})")


@dataclass(frozen=True)
class EquivalenceResult:
    candidate_id: str
    status: str                    # "tested" | "insufficient_window"
    alpha: float
    se: float
    n_months: int
    df: int
    delta: float
    p_lower: float | None = None
    p_upper: float | None = None
    p_tost: float | None = None
    adjusted_p: float | None = None
    equivalent: bool | None = None

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id, "status": self.status, "alpha": self.alpha,
            "se": self.se, "n_months": self.n_months, "df": self.df, "delta": self.delta,
            "p_lower": self.p_lower, "p_upper": self.p_upper, "p_tost": self.p_tost,
            "adjusted_p": self.adjusted_p, "equivalent": self.equivalent,
        }


def tost_p_values(alpha: float, se: float, df: int, delta: float) -> tuple[float, float, float]:
    """``(p_lower, p_upper, p_TOST)`` for one candidate. ``p_TOST`` is the max of the two
    one-sided p-values — equivalence needs BOTH one-sided nulls rejected."""
    if delta <= 0:
        raise EquivalenceInputError(f"the equivalence margin must be positive; got {delta}")
    p_lower = float(stats.t.sf((alpha + delta) / se, df))   # H0: alpha <= -delta
    p_upper = float(stats.t.cdf((alpha - delta) / se, df))  # H0: alpha >= +delta
    return p_lower, p_upper, max(p_lower, p_upper)


def duplicate_groups(inputs: list[EquivalenceInput]) -> list[list[str]]:
    """Candidates that are numerically IDENTICAL (same alpha, se and window) grouped together.

    Two sources can propose the same configuration, and that is
    one hypothesis, not two: left undetected it inflates the BH family and makes the same
    claim twice. Detected here and disclosed; the family stays as registered."""
    seen: dict[tuple, list[str]] = {}
    for item in inputs:
        key = (round(item.alpha, 12), round(item.se, 12), item.n_months)
        seen.setdefault(key, []).append(item.candidate_id)
    return [sorted(ids) for ids in seen.values() if len(ids) > 1]


def equivalence_family(
    inputs: list[EquivalenceInput], *, delta: float, q: float, min_months: int = MIN_MONTHS
) -> dict:
    """The whole family: TOST per candidate, then BH at ``q`` over the tested ones only.

    A candidate whose development window is shorter than the registered minimum is typed
    ``insufficient_window`` and kept OUT of the BH family — testing it would spend a family
    slot on a window the registration says cannot host the test."""
    tested: list[EquivalenceResult] = []
    skipped: list[EquivalenceResult] = []
    for item in inputs:
        if item.n_months < min_months:
            skipped.append(EquivalenceResult(
                candidate_id=item.candidate_id, status="insufficient_window", alpha=item.alpha,
                se=item.se, n_months=item.n_months, df=item.df, delta=delta))
            continue
        p_lower, p_upper, p_tost = tost_p_values(item.alpha, item.se, item.df, delta)
        tested.append(EquivalenceResult(
            candidate_id=item.candidate_id, status="tested", alpha=item.alpha, se=item.se,
            n_months=item.n_months, df=item.df, delta=delta,
            p_lower=p_lower, p_upper=p_upper, p_tost=p_tost))

    decisions = benjamini_hochberg({r.candidate_id: r.p_tost for r in tested}, q) if tested else {}
    results = [
        EquivalenceResult(**{**r.to_dict(),
                             "adjusted_p": decisions[r.candidate_id].adjusted_p,
                             "equivalent": bool(decisions[r.candidate_id].rejected)})
        for r in tested
    ] + skipped

    duplicates = duplicate_groups(inputs)
    dedup_block = None
    if duplicates:
        # Sensitivity: one representative per identical group, so the family counts each
        # distinct hypothesis once. Reported beside the registered family, never instead.
        keep = {group[0] for group in duplicates}
        dropped = {cid for group in duplicates for cid in group[1:]}
        distinct = [r for r in tested if r.candidate_id not in dropped or r.candidate_id in keep]
        dedup_decisions = benjamini_hochberg({r.candidate_id: r.p_tost for r in distinct}, q)
        dedup_block = {
            "note": ("duplicates collapsed to one representative each — the registered family "
                     "above is unchanged; this is a sensitivity, not a re-decision"),
            "n_distinct": len(distinct),
            "n_equivalent": sum(1 for d in dedup_decisions.values() if d.rejected),
            "adjusted_p": {k: d.adjusted_p for k, d in dedup_decisions.items()},
        }

    return {
        "test": "TOST (two one-sided tests) on the development-window residual alpha",
        "margin_delta": delta,
        "bh_q": q,
        "family": "equivalence_test — its OWN family; never pooled with the proposal families",
        "window": f"development only; minimum {min_months} months",
        "asymmetry": ASYMMETRY,
        "n_tested": len(tested),
        "n_skipped_insufficient_window": len(skipped),
        "n_equivalent": sum(1 for r in results if r.equivalent),
        "duplicate_groups": duplicates,
        "deduplicated_sensitivity": dedup_block,
        "results": [r.to_dict() for r in sorted(results, key=lambda r: r.candidate_id)],
    }
