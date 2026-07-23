"""
fit_timing.py — the §12.5 fit-timing measurement + the §5.3 multi-start range diagnostic.

Two compute-contingent pieces, decided by MEASUREMENT, never by preference (§5.2 escape hatch):

  * ``time_one_fit`` / ``recommend_initialisations`` (§12.5): time one production fit at a given
    scale and decide whether best-of-M is affordable inside the replicated fits. If
    ``fit_seconds x n_replicate_fits x base_m`` exceeds the budget, production collapses to a single
    fixed initialisation everywhere at once (M -> 1). Either definition; never both.

  * ``multistart_range`` (§5.3, the "ALS local-optimum sensitivity" row): fit both arms from several
    seeded random starts, recompute the interaction bracket I for each, and report the RANGE across
    starts. Deliberately a DIAGNOSTIC, not an interval estimate — distinct local optima can differ in
    span (not merely rotation), and span differences move alpha; the range is that detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import numpy as np
import pandas as pd

from agents.auditor.thresholds import IPCALambda, IPCAProjectionGate
from agents.quant.library.ipca_feed import IPCAFeed

from .differential import differential_from_states
from .production_fit import production_fit, production_fit_seeded


def time_one_fit(feed: IPCAFeed, lam: IPCALambda, *, repeats: int = 3) -> float:
    """Median wall-clock seconds for one ``production_fit`` on ``feed`` (the O-A1 measurement)."""
    times: list[float] = []
    for _ in range(max(1, repeats)):
        t0 = perf_counter()
        production_fit(feed, lam)
        times.append(perf_counter() - t0)
    return float(np.median(times))


@dataclass(frozen=True)
class InitialisationDecision:
    """The §5.2 escape-hatch outcome: best-of-M if affordable, else single-init everywhere."""

    fit_seconds: float
    n_replicate_fits: int
    base_m: int
    budget_seconds: float
    projected_seconds: float
    recommended_n_initialisations: int      # base_m if affordable, else 1
    affordable: bool


def recommend_initialisations(
    fit_seconds: float,
    *,
    n_replicate_fits: int,
    budget_seconds: float,
    base_m: int,
) -> InitialisationDecision:
    """Decide best-of-M vs single-init from the timing measurement (§5.2 / §12.5). best-of-M is
    affordable iff ``fit_seconds x n_replicate_fits x base_m <= budget_seconds``; otherwise collapse
    to a single fixed initialisation (M -> 1) everywhere at once."""
    projected = fit_seconds * n_replicate_fits * base_m
    affordable = base_m <= 1 or projected <= budget_seconds
    return InitialisationDecision(
        fit_seconds=fit_seconds,
        n_replicate_fits=n_replicate_fits,
        base_m=base_m,
        budget_seconds=budget_seconds,
        projected_seconds=projected,
        recommended_n_initialisations=base_m if affordable else 1,
        affordable=affordable,
    )


@dataclass(frozen=True, eq=False)
class MultistartRange:
    """ALS local-optimum sensitivity of I across seeded starts (§5.3). NOT an interval."""

    seeds: tuple[int, ...]
    i_values: tuple[float, ...]
    i_min: float
    i_max: float
    i_range: float
    n_usable: int

    def to_dict(self) -> dict:
        return {
            "multistart_range": {
                "seeds": list(self.seeds),
                "i_values": list(self.i_values),
                "i_min": self.i_min,
                "i_max": self.i_max,
                "i_range": self.i_range,
                "n_usable": self.n_usable,
                "is_interval": False,   # optimizer sensitivity, deliberately not an interval
            }
        }


def multistart_range(
    bias: str,
    anchor_name: str,
    feed_n: IPCAFeed,
    feed_b: IPCAFeed,
    anchor: pd.Series,
    lam: IPCALambda,
    gate: IPCAProjectionGate,
    *,
    seeds: Sequence[int],
) -> MultistartRange:
    """Fit both arms from each seeded random start, recompute I, and report the range across starts.
    Seed 0 in ``seeds`` may be passed as a sentinel for the production (SVD cold) start."""
    i_values: list[float] = []
    for s in seeds:
        theta_n = production_fit_seeded(feed_n, lam, s)
        theta_b = production_fit_seeded(feed_b, lam, s)
        res = differential_from_states(bias, anchor_name, feed_n, feed_b, theta_n, theta_b, anchor, gate)
        i_values.append(res.interaction_bracket_raw.value)
    finite = [v for v in i_values if v == v]
    if finite:
        i_min, i_max = float(min(finite)), float(max(finite))
    else:
        i_min = i_max = float("nan")
    return MultistartRange(
        seeds=tuple(seeds),
        i_values=tuple(i_values),
        i_min=i_min,
        i_max=i_max,
        i_range=(i_max - i_min) if finite else float("nan"),
        n_usable=len(finite),
    )
