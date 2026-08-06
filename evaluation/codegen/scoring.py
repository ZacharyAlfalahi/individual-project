"""P1 codegen ablation (WS-C) — the rung-3/4 comparator.

Built from the evaluation contract's §7.3/§7.4 DEFINITIONS (series-level
similarity: correlation, sign agreement, mean difference, tracking error, on
the overlapping sample; headline-statistic deltas with metric-specific
absolute tolerances — the contract deliberately refuses a universal relative
tolerance). No comparator code existed for these rungs before this module;
`summarize_returns` is the reused primitive for the headline statistics.

Verdicts (mini-contract `docs/extensions/contracts/p1_codegen_ablation.md`):

  WONT_RUN    sandbox/output-contract failure, or no extractable code.
  RUNS_WRONG  executes and honours the output contract, but breaches a rung-3
              similarity criterion — the SILENT-ERROR class (FIR analogue).
  RUNS_RIGHT  valid output and every rung-3 criterion passes. Rung-4 deltas
              are always reported and never gate (mirrors §7's
              diagnostics-don't-gate structure).

The RUNS_WRONG rate is a function of the thresholds, so every report publishes
the raw per-run metrics table alongside the verdicts (re-thresholdable by any
reader). Thresholds live in `docs/thresholds.yaml` `p1_codegen.scoring`
(fail-loud loader — a silent default would move the verdict boundary).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from agents.quant.library.characteristic_sort import summarize_returns

_REPO_ROOT = Path(__file__).resolve().parents[2]
_THRESHOLDS = _REPO_ROOT / "docs" / "thresholds.yaml"

_REQUIRED_KEYS = (
    "min_overlap_months",
    "correlation_min",
    "sign_agreement_min",
    "mean_abs_diff_max",
    "tracking_error_max",
    "exact_tier_max_abs_diff",
)


class Verdict(str, Enum):
    WONT_RUN = "WONT_RUN"
    RUNS_WRONG = "RUNS_WRONG"
    RUNS_RIGHT = "RUNS_RIGHT"


@dataclass(frozen=True)
class ScoreResult:
    strategy: str
    verdict: Verdict
    reason: str | None                  # WONT_RUN reason, else None
    exact_tier: bool
    failed_criteria: tuple[str, ...]
    rung3: dict = field(default_factory=dict)
    rung4: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "verdict": self.verdict.value,
            "reason": self.reason,
            "exact_tier": self.exact_tier,
            "failed_criteria": list(self.failed_criteria),
            "rung3": self.rung3,
            "rung4": self.rung4,
        }


def load_scoring_thresholds(path: Path | None = None) -> dict:
    doc = yaml.safe_load((path or _THRESHOLDS).read_text(encoding="utf-8"))
    try:
        block = doc["p1_codegen"]["scoring"]
    except (KeyError, TypeError) as exc:
        raise KeyError(
            "thresholds.yaml has no p1_codegen.scoring block — the comparator refuses to "
            "run with invented tolerances"
        ) from exc
    missing = [k for k in _REQUIRED_KEYS if k not in block]
    if missing:
        raise KeyError(f"p1_codegen.scoring is missing keys: {missing}")
    return dict(block)


def _align_monthly(oracle: pd.Series, cand: pd.Series) -> pd.DataFrame:
    """Inner-join on month period — the overlapping sample of §7.3. Monthly
    alignment tolerates day-of-month conventions (month-end vs month-start)
    without inventing any resampling of the values themselves."""
    o = pd.Series(oracle.values, index=pd.DatetimeIndex(oracle.index).to_period("M"))
    c = pd.Series(cand.values, index=pd.DatetimeIndex(cand.index).to_period("M"))
    if not o.index.is_unique or not c.index.is_unique:
        raise ValueError("duplicate months in a series — cannot align")
    joined = pd.concat({"oracle": o, "cand": c}, axis=1, join="inner").dropna()
    return joined


def rung3_series_similarity(oracle: pd.Series, cand: pd.Series) -> dict:
    """§7.3 series-level similarity on the overlapping sample."""
    joined = _align_monthly(oracle, cand)
    n = int(len(joined))
    if n == 0:
        return {"n_overlap": 0}
    diff = joined["cand"] - joined["oracle"]
    o, c = joined["oracle"].to_numpy(), joined["cand"].to_numpy()
    if n >= 2 and float(np.std(o)) > 0.0 and float(np.std(c)) > 0.0:
        correlation = float(np.corrcoef(o, c)[0, 1])
    else:
        correlation = float("nan")   # degenerate — reported, never silently 1.0
    return {
        "n_overlap": n,
        "correlation": correlation,
        "sign_agreement": float(np.mean(np.sign(o) == np.sign(c))),
        "mean_diff": float(diff.mean()),
        "tracking_error": float(diff.std(ddof=1)) if n >= 2 else float("nan"),
        "max_abs_diff": float(diff.abs().max()),
    }


def rung4_headline_deltas(
    oracle: pd.Series, cand: pd.Series, *, months_per_year: int = 12
) -> dict:
    """§7.4 headline-statistic comparison — reported, never gating."""
    joined = _align_monthly(oracle, cand)
    o = pd.Series(joined["oracle"].values, index=joined.index.to_timestamp("M"))
    c = pd.Series(joined["cand"].values, index=joined.index.to_timestamp("M"))
    so = summarize_returns(o, None, months_per_year)
    sc = summarize_returns(c, None, months_per_year)
    return {
        "delta_average": sc["average"] - so["average"],
        "delta_sharpe": sc["sharpe"] - so["sharpe"],
        "delta_t_stat": sc["t_stat"] - so["t_stat"],
        "oracle_summary": so,
        "candidate_summary": sc,
    }


def score_run(
    strategy: str,
    sandbox_status: str,
    sandbox_reason: str | None,
    oracle: pd.Series | None,
    cand: pd.Series | None,
    thresholds: dict,
) -> ScoreResult:
    """One run's verdict. `sandbox_status`/`sandbox_reason` come from the
    sandbox's `SandboxResult`; `cand` is the parsed output series (None when
    the output contract was breached or no code was extractable)."""
    if sandbox_status != "ok" or cand is None:
        return ScoreResult(
            strategy=strategy, verdict=Verdict.WONT_RUN,
            reason=sandbox_reason or "no_candidate_series",
            exact_tier=False, failed_criteria=(),
        )
    if oracle is None:
        raise ValueError("score_run needs an oracle series for an ok sandbox run")

    rung3 = rung3_series_similarity(oracle, cand)
    if rung3.get("n_overlap", 0) < int(thresholds["min_overlap_months"]):
        return ScoreResult(
            strategy=strategy, verdict=Verdict.RUNS_WRONG, reason=None,
            exact_tier=False, failed_criteria=("insufficient_overlap",),
            rung3=rung3,
        )
    rung4 = rung4_headline_deltas(oracle, cand)

    exact = rung3["max_abs_diff"] <= float(thresholds["exact_tier_max_abs_diff"])
    if exact:
        return ScoreResult(
            strategy=strategy, verdict=Verdict.RUNS_RIGHT, reason=None,
            exact_tier=True, failed_criteria=(), rung3=rung3, rung4=rung4,
        )

    failed: list[str] = []
    corr = rung3["correlation"]
    if not (math.isfinite(corr) and corr >= float(thresholds["correlation_min"])):
        failed.append("correlation")
    if rung3["sign_agreement"] < float(thresholds["sign_agreement_min"]):
        failed.append("sign_agreement")
    if abs(rung3["mean_diff"]) > float(thresholds["mean_abs_diff_max"]):
        failed.append("mean_abs_diff")
    te = rung3["tracking_error"]
    if not (math.isfinite(te) and te <= float(thresholds["tracking_error_max"])):
        failed.append("tracking_error")

    verdict = Verdict.RUNS_RIGHT if not failed else Verdict.RUNS_WRONG
    return ScoreResult(
        strategy=strategy, verdict=verdict, reason=None, exact_tier=False,
        failed_criteria=tuple(failed), rung3=rung3, rung4=rung4,
    )


def inter_model_agreement(series_a: pd.Series, series_b: pd.Series) -> dict:
    """Rung-3 metrics BETWEEN the two models' generated series — the
    agreement-vs-correctness cross-tab input. Agreement is not correctness
    (the WS-E caveat, verbatim in the mini-contract)."""
    return rung3_series_similarity(series_a, series_b)
