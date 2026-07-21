"""
lattice_types.py — the frozen data types for the 2^k lattice (§0.1, §4).

  Combination  -> a choice of OFF/ON per runnable toggle (the input: a RunConfig)
  Cell         -> what you get when you RUN a combination (the output: CellReturns)
  Lattice      -> the set of all combinations plus the adjacency structure
                  (LatticeResult)

The document uses "cell" throughout, since the object we compute with is the
output. The return series is the primary artefact (§4.1) — metrics are functionals
of it, and it is what the bootstrap resamples — so CellReturns stores the monthly
series, not just its summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import pandas as pd

from .toggle import TOGGLE_IDS, ToggleId, ToggleState

# Metric names available for cross-cell attribution (subset of summarize_returns).
# The "primary metric" is selected from these by name (§9.1, pre-registered).
METRIC_NAMES: tuple[str, ...] = (
    "average",
    "annualised_average",
    "bumpiness",
    "sharpe",
    "t_stat",
)


@dataclass(frozen=True)
class MetricSet:
    """A typed view of `characteristic_sort.summarize_returns` output for one
    return series (§9.1). Holds the descriptive dates/counts and the numeric
    metrics an attribution can be computed on."""

    n_months: int
    months_per_year: int
    nw_lags_used: int
    average: float
    annualised_average: float
    bumpiness: float
    sharpe: float
    t_stat: float
    first_date: pd.Timestamp | None = None
    last_date: pd.Timestamp | None = None

    @classmethod
    def from_summary(cls, summary: Mapping[str, object]) -> "MetricSet":
        """Build from the engine's summary dict. Missing numeric keys are an
        engine-contract violation, so we index rather than .get with a default."""
        return cls(
            n_months=int(summary["n_months"]),
            months_per_year=int(summary["months_per_year"]),
            nw_lags_used=int(summary["nw_lags_used"]),
            average=float(summary["average"]),
            annualised_average=float(summary["annualised_average"]),
            bumpiness=float(summary["bumpiness"]),
            sharpe=float(summary["sharpe"]),
            t_stat=float(summary["t_stat"]),
            first_date=summary.get("first_date"),  # type: ignore[arg-type]
            last_date=summary.get("last_date"),  # type: ignore[arg-type]
        )

    def value(self, metric_name: str) -> float:
        """The scalar value of a named metric (for the primary-metric selection)."""
        if metric_name not in METRIC_NAMES:
            raise KeyError(
                f"metric {metric_name!r} is not an attribution metric; "
                f"expected one of {METRIC_NAMES}"
            )
        return float(getattr(self, metric_name))

    def as_metric_dict(self) -> dict:
        """The numeric metrics as a plain dict (for hashing / serialisation)."""
        return {name: float(getattr(self, name)) for name in METRIC_NAMES}


@dataclass(frozen=True, eq=False)
class CellReturns:
    """One executed lattice cell (§0.1). `on_set` is the set of toggles held ON
    (the coalition S in the algebra); `returns` is the monthly long-short series
    on the cell's OWN (native) sample; the hashes are the §3.5 invariance surface.

    eq=False: it holds pandas objects, whose element-wise __eq__ would make a
    dataclass comparison ambiguous."""

    on_set: frozenset[ToggleId]
    run_config: object                 # RunConfig (avoid an import cycle at type level)
    returns: pd.Series                 # date-indexed monthly strategy return
    n_bonds: pd.Series                 # date-indexed bond count (membership proxy)
    metrics_native: MetricSet
    run_config_hash: str
    panel_view_hash: str
    return_hash: str
    n_bonds_hash: str
    metric_hash: str

    def __post_init__(self) -> None:
        bad = self.on_set - set(TOGGLE_IDS)
        if bad:
            raise ValueError(f"on_set contains non-toggle ids: {sorted(bad)}")


@dataclass(frozen=True, eq=False)
class LatticeResult:
    """The full 2^k lattice for one strategy: the runnable toggles (in canonical
    order), the fixed states of any held non-runnable toggles (the conditioning),
    and the executed cells keyed by their ON-set."""

    strategy_label: str
    runnable_toggles: tuple[ToggleId, ...]
    fixed_states: Mapping[ToggleId, ToggleState]
    cells: tuple[CellReturns, ...]
    _by_on_set: dict[frozenset, CellReturns] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        k = len(self.runnable_toggles)
        if len(self.cells) != 2**k:
            raise ValueError(
                f"lattice for {self.strategy_label!r} has {len(self.cells)} cells; "
                f"expected 2^{k} = {2**k} over runnable toggles {self.runnable_toggles}"
            )
        index: dict[frozenset, CellReturns] = {}
        for cell in self.cells:
            if cell.on_set in index:
                raise ValueError(f"duplicate cell for ON-set {sorted(cell.on_set)}")
            index[cell.on_set] = cell
        object.__setattr__(self, "_by_on_set", index)

    def cell_for(self, on_set: frozenset[ToggleId]) -> CellReturns:
        """The cell where exactly `on_set` toggles are ON. Raises if absent."""
        key = frozenset(on_set)
        if key not in self._by_on_set:
            raise KeyError(f"no cell for ON-set {sorted(key)}")
        return self._by_on_set[key]

    @property
    def k(self) -> int:
        return len(self.runnable_toggles)
