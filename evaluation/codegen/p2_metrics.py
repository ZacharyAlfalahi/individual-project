"""P2 coverage-boundary metrics (WS-C).

The P2 estimand is inter-model AGREEMENT, never Sharpes-as-performance (no
oracle exists to separate "strategy fails to replicate" from "code is wrong" —
the §6 caveat, binding). So this module reads only from
``scoring.inter_model_agreement`` (rung-3 similarity BETWEEN the two models'
generated series) and ``scoring.rung3_series_similarity`` (the Arm-B
codegen-vs-compiler divergence, "neither side is truth"). It never touches
``score_run`` (oracle-requiring) or ``rung4_headline_deltas`` (Sharpe-bearing).

What it produces:

  * the inter-model **correlation DISTRIBUTION** (PRIMARY);
  * the binary **agreement rate** (SECONDARY): a pair "agrees" iff
    ``corr >= correlation_min`` AND ``sign_agreement >= sign_agreement_min``;
  * **divergence strata** — high (corr < 0.90) / medium (0.90 <= corr < 0.99) /
    low (corr >= 0.99), i.e. labelled by DIVERGENCE magnitude (high divergence =
    low correlation);
  * the Arm-B **codegen-vs-compiler divergence** (labelled "neither side is
    truth");
  * an **MDE-by-arm-size** power guard: the smallest agreement-rate difference
    detectable at each arm size.

BELOW-FLOOR rule: if ``|Arm A| < below_floor_min_arm_a`` the selection is
under-powered, so agreement/divergence are SUPPRESSED and only the census fate
table + taxonomy are emitted (``P2Metrics.suppressed``).

Thresholds live in ``docs/thresholds.yaml`` ``p2_codegen`` (fail-loud loader —
a silent default would move the agreement boundary).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import yaml

from evaluation.codegen.census import CensusResult
from evaluation.codegen.p2_selector import ArmSelection
from evaluation.codegen.scoring import inter_model_agreement, rung3_series_similarity

_REPO_ROOT = Path(__file__).resolve().parents[2]
_THRESHOLDS = _REPO_ROOT / "docs" / "thresholds.yaml"

_STRATUM_HIGH = "high"      # high divergence  = low correlation
_STRATUM_MEDIUM = "medium"
_STRATUM_LOW = "low"        # low divergence   = high correlation

_CODEGEN_VS_COMPILER_LABEL = "neither side is truth"


def load_p2_scoring_thresholds(path: Path | None = None) -> dict:
    """The ``p2_codegen`` thresholds block — fail-loud, mirroring
    ``scoring.load_scoring_thresholds``. Raises ``KeyError`` if the block or any
    required sub-key is absent (the metrics refuse invented boundaries)."""
    doc = yaml.safe_load((path or _THRESHOLDS).read_text(encoding="utf-8"))
    try:
        block = doc["p2_codegen"]
    except (KeyError, TypeError) as exc:
        raise KeyError(
            "thresholds.yaml has no p2_codegen block — P2 metrics refuse to run with "
            "invented agreement/divergence boundaries"
        ) from exc
    required = {
        "arms": ("arm_b_max", "below_floor_min_arm_a"),
        "agreement": ("correlation_min", "sign_agreement_min"),
        "divergence_strata": ("high_divergence_corr_lt", "medium_divergence_corr_lt"),
    }
    for sub, keys in required.items():
        if sub not in block or not isinstance(block[sub], dict):
            raise KeyError(f"p2_codegen is missing the {sub!r} sub-block")
        missing = [k for k in keys if k not in block[sub]]
        if missing:
            raise KeyError(f"p2_codegen.{sub} is missing keys: {missing}")
    if "min_overlap_months" not in block:
        raise KeyError("p2_codegen is missing key: min_overlap_months")
    return dict(block)


# ---------------------------------------------------------------------------
# Per-member inter-model agreement.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MemberAgreement:
    """One member's inter-model agreement (model A vs model B generated series)."""

    paper_id: str
    arm: str                    # "A" (refusal set) or "B" (compilable control)
    n_overlap: int
    correlation: float          # may be NaN (degenerate / no overlap)
    sign_agreement: float
    agrees: bool                # corr >= corr_min AND sign_agreement >= sign_min
    stratum: str                # divergence magnitude: high / medium / low
    sufficient_overlap: bool    # n_overlap >= min_overlap_months

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "arm": self.arm,
            "n_overlap": self.n_overlap,
            "correlation": self.correlation,
            "sign_agreement": self.sign_agreement,
            "agrees": self.agrees,
            "stratum": self.stratum,
            "sufficient_overlap": self.sufficient_overlap,
        }


def stratum_of(correlation: float, thresholds: dict) -> str:
    """Divergence stratum by correlation. A non-finite correlation (degenerate
    series / no overlap) is the WORST case -> high divergence."""
    high_lt = float(thresholds["divergence_strata"]["high_divergence_corr_lt"])
    med_lt = float(thresholds["divergence_strata"]["medium_divergence_corr_lt"])
    if not math.isfinite(correlation):
        return _STRATUM_HIGH
    if correlation < high_lt:
        return _STRATUM_HIGH
    if correlation < med_lt:
        return _STRATUM_MEDIUM
    return _STRATUM_LOW


def member_agreement(
    paper_id: str,
    arm: str,
    series_a: pd.Series,
    series_b: pd.Series,
    thresholds: dict,
) -> MemberAgreement:
    """Inter-model agreement for one member from the two models' return series."""
    m = inter_model_agreement(series_a, series_b)
    n = int(m.get("n_overlap", 0))
    correlation = float(m.get("correlation", float("nan")))
    sign_agreement = float(m.get("sign_agreement", float("nan")))
    corr_min = float(thresholds["agreement"]["correlation_min"])
    sign_min = float(thresholds["agreement"]["sign_agreement_min"])
    min_overlap = int(thresholds["min_overlap_months"])
    sufficient = n >= min_overlap
    agrees = bool(
        sufficient
        and math.isfinite(correlation)
        and correlation >= corr_min
        and math.isfinite(sign_agreement)
        and sign_agreement >= sign_min
    )
    return MemberAgreement(
        paper_id=paper_id,
        arm=arm,
        n_overlap=n,
        correlation=correlation,
        sign_agreement=sign_agreement,
        agrees=agrees,
        stratum=stratum_of(correlation, thresholds),
        sufficient_overlap=sufficient,
    )


def agreement_distribution(members: tuple[MemberAgreement, ...]) -> dict:
    """The correlation DISTRIBUTION (primary) + agreement rate (secondary) +
    strata counts. Non-finite correlations and insufficient-overlap members are
    COUNTED separately, never silently dropped."""
    correlations = [m.correlation for m in members]
    finite = [c for c in correlations if math.isfinite(c)]
    insufficient = [m.paper_id for m in members if not m.sufficient_overlap]
    # The agreement rate is computed over members with sufficient overlap (an
    # unreliable near-empty overlap is not evidence of agreement OR disagreement);
    # the count of excluded members is reported alongside so nothing is hidden.
    scored = [m for m in members if m.sufficient_overlap]
    n_agree = sum(1 for m in scored if m.agrees)
    agreement_rate = (n_agree / len(scored)) if scored else float("nan")
    strata_counts = {_STRATUM_HIGH: 0, _STRATUM_MEDIUM: 0, _STRATUM_LOW: 0}
    for m in members:
        strata_counts[m.stratum] += 1
    return {
        "n_members": len(members),
        "correlations": correlations,        # the full distribution (PRIMARY)
        "n_finite_correlations": len(finite),
        "correlation_min": (min(finite) if finite else float("nan")),
        "correlation_median": (float(np.median(finite)) if finite else float("nan")),
        "correlation_max": (max(finite) if finite else float("nan")),
        "agreement_rate": agreement_rate,     # SECONDARY
        "n_agree": n_agree,
        "n_scored": len(scored),
        "n_insufficient_overlap": len(insufficient),
        "insufficient_overlap_papers": insufficient,
        "strata_counts": strata_counts,
    }


# ---------------------------------------------------------------------------
# Arm-B codegen-vs-compiler divergence ("neither side is truth").
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CodegenVsCompiler:
    """One Arm-B (member, model) divergence between the generated series and the
    deterministic compiler's series. NEITHER side is truth (no oracle) — this is
    a divergence magnitude, not a correctness measure."""

    paper_id: str
    model_id: str
    label: str
    metrics: dict = field(default_factory=dict)   # rung3 similarity dict

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "model_id": self.model_id,
            "label": self.label,
            "metrics": self.metrics,
        }


def codegen_vs_compiler_divergence(
    paper_id: str, model_id: str, codegen: pd.Series, compiler: pd.Series
) -> CodegenVsCompiler:
    """rung-3 similarity between one model's generated series and the compiler's
    series for an Arm-B member. Explicitly labelled 'neither side is truth'."""
    return CodegenVsCompiler(
        paper_id=paper_id,
        model_id=model_id,
        label=_CODEGEN_VS_COMPILER_LABEL,
        metrics=rung3_series_similarity(codegen, compiler),
    )


# ---------------------------------------------------------------------------
# Below-floor rule + MDE power guard.
# ---------------------------------------------------------------------------

def below_floor(arm_a_size: int, thresholds: dict) -> bool:
    """True iff Arm A is below the floor -> agreement/divergence are suppressed."""
    return int(arm_a_size) < int(thresholds["arms"]["below_floor_min_arm_a"])


def mde_by_arm_size(
    sizes: tuple[int, ...],
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    worst_case_p: float = 0.5,
) -> dict[int, dict]:
    """Minimum detectable difference in agreement RATE at each arm size — the
    power guard. Two-proportion normal approximation with equal n per group and
    the worst-case variance ``p(1-p)`` (default p=0.5):

        MDE = (z_{1-alpha/2} + z_{power}) * sqrt(2 * p(1-p) / n)

    Reported RAW (an MDE > 1.0 means no proportion difference is detectable at
    all — the honest read for tiny n); ``detectable`` flags MDE <= 1.0. At n=5
    the MDE is ~0.89, i.e. only near-total differences are detectable — the guard
    that keeps P2 descriptive, not inferential."""
    z_alpha = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z_beta = NormalDist().inv_cdf(power)
    factor = z_alpha + z_beta
    var = 2.0 * worst_case_p * (1.0 - worst_case_p)
    out: dict[int, dict] = {}
    for n in sizes:
        n = int(n)
        if n <= 0:
            out[n] = {"mde": float("nan"), "detectable": False}
            continue
        mde = factor * math.sqrt(var / n)
        out[n] = {"mde": mde, "detectable": mde <= 1.0}
    return out


# ---------------------------------------------------------------------------
# The assembled P2 metrics object.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class P2Metrics:
    """Everything the report renders. When ``suppressed`` (below floor), the
    agreement/divergence/MDE fields are empty and only the census fate table is
    carried (the taxonomy is assembled separately in ``p2_taxonomy``)."""

    suppressed: bool
    below_floor_reason: str | None
    census_fate_table: tuple[dict, ...]
    agreements: tuple[MemberAgreement, ...] = ()
    distribution: dict = field(default_factory=dict)
    divergences: tuple[CodegenVsCompiler, ...] = ()
    mde: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "suppressed": self.suppressed,
            "below_floor_reason": self.below_floor_reason,
            "census_fate_table": list(self.census_fate_table),
            "agreements": [a.to_dict() for a in self.agreements],
            "distribution": self.distribution,
            "divergences": [d.to_dict() for d in self.divergences],
            "mde": {str(k): v for k, v in self.mde.items()},
        }


def compute_p2_metrics(
    census: CensusResult,
    selection: ArmSelection,
    inter_model_pairs: dict[str, tuple[pd.Series, pd.Series]],
    arm_b_compiler: dict[str, pd.Series] | None,
    thresholds: dict,
    *,
    model_ids: tuple[str, str] = ("model_a", "model_b"),
) -> P2Metrics:
    """Assemble the P2 metrics from per-member inter-model series pairs and the
    Arm-B compiler series. Honours the below-floor suppression rule.

    ``inter_model_pairs``: member paper_id -> (model-A series, model-B series),
    covering Arm A ∪ Arm B. ``arm_b_compiler``: Arm-B member paper_id ->
    compiler series (the codegen-vs-compiler divergence input); may be ``None``
    or partial when a compiler series is unavailable (those members simply carry
    no divergence row — never a silent Sharpe substitute)."""
    fate_table = census.fate_table()
    if selection.below_floor:
        reason = (
            f"|Arm A|={selection.arm_a_size} < below_floor_min_arm_a="
            f"{int(thresholds['arms']['below_floor_min_arm_a'])} — agreement/divergence "
            "suppressed; only the census fate table + taxonomy are emitted"
        )
        return P2Metrics(
            suppressed=True, below_floor_reason=reason, census_fate_table=fate_table
        )

    agreements: list[MemberAgreement] = []
    for arm, paper_ids in (("A", selection.arm_a), ("B", selection.arm_b)):
        for pid in paper_ids:
            if pid not in inter_model_pairs:
                raise KeyError(
                    f"no inter-model series pair for arm-{arm} member {pid!r} — "
                    "the driver must supply both models' series for every selected member"
                )
            a, b = inter_model_pairs[pid]
            agreements.append(member_agreement(pid, arm, a, b, thresholds))

    distribution = agreement_distribution(tuple(agreements))

    divergences: list[CodegenVsCompiler] = []
    if arm_b_compiler:
        for pid in selection.arm_b:
            compiler = arm_b_compiler.get(pid)
            if compiler is None:
                continue
            a, b = inter_model_pairs[pid]
            divergences.append(
                codegen_vs_compiler_divergence(pid, model_ids[0], a, compiler)
            )
            divergences.append(
                codegen_vs_compiler_divergence(pid, model_ids[1], b, compiler)
            )

    mde = mde_by_arm_size((selection.arm_a_size, len(selection.arm_b)))

    return P2Metrics(
        suppressed=False,
        below_floor_reason=None,
        census_fate_table=fate_table,
        agreements=tuple(agreements),
        distribution=distribution,
        divergences=tuple(divergences),
        mde=mde,
    )
