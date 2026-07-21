"""
synthetic_panel.py — a parametrised clean maximal-panel DGP + bias injection
(SEAM 2; design item O-A8).

Layer B injects a KNOWN bias of known magnitude into a clean synthetic panel and
runs the full 2^k lattice on it. This module is the generator. It is a real
module (not a test helper) because Layer B fixtures, the calibration sweep and the
recovery sweep all import it.

The clean panel is deliberately built so that on it every toggle recovers ~zero
effect on the COMMON support (the false-positive-rate baseline, §10.2) — the two
price families are identical, every bond trades fresh at month-end, there are no
distress exits, the ranking characteristic is time-constant, and returns are
modest. At the bit-exact NATIVE level, the panel toggles (meas_err, stale_price,
survivorship) and lab_trim are exact no-ops; lib_gap is only a NEAR no-op —
signal_lag shifts the formation window by one month, dropping a boundary month, so
its native return series differs even though its common-support effect is ~0. That
matches the design's "approximately a dummy / near no-op, never bit-exact" (§3.5,
§10.3). A clean panel that showed a spurious COMMON-support effect would corrupt
every specificity claim, so that is the property the FPR baseline rests on.
`inject_bias` perturbs ONLY the as-published (OFF) arm of
one toggle, so the toggle's ON-minus-OFF differential recovers the planted magnitude.

Ground-truth signal: a persistent per-bond characteristic q_i drives next-month
return, ret_{i,t} = alpha * q_i + noise, and the sort ranks on score = q_i. So the
long-short earns approximately alpha * (mean top-group q - mean bottom-group q) —
a known, stable, nonzero base return with no bias in it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agents.librarian.adapter.result import AdaptResult, CombinerInstruction, LegCall
from agents.quant.config import Binding, Evidence, Inherited, build_quant_config

# The published ex-post trim used by the lab_trim scenario: a symmetric absolute
# truncation. lab_trim OFF applies it (the as-published, biased state); lab_trim ON
# removes it. Returns beyond +/-0.5 are dropped by the truncation.
TRUNCATE_TRIM_SPEC: dict = {
    "method": "truncate",
    "target": "return",
    "bounds": {"type": "absolute", "lo": -0.5, "hi": 0.5},
    "sample": "full_sample",
}

# Each toggle's default injection magnitude (return points), chosen large enough
# that the single-bias check fires clearly. The calibration sweep varies these.
DEFAULT_MAGNITUDES: dict[str, float] = {
    "meas_err": 0.02,
    "stale_price": 0.03,
    "survivorship": 0.30,
    "lib_gap": 0.03,
    "lab_trim": 0.8,
}


@dataclass(frozen=True)
class SyntheticSpec:
    """The parameters of a clean synthetic panel (recorded so a scenario is
    reproducible and its planted base alpha is known)."""

    n_bonds: int = 60
    n_months: int = 72
    alpha: float = 0.01
    noise_sd: float = 0.015
    seed: int = 0
    time_varying_score: bool = False   # True => score varies i.i.d. over time
    start: str = "2005-01-31"


def _month_ends(start: str, n: int) -> pd.DatetimeIndex:
    first = pd.Timestamp(start) + pd.offsets.MonthEnd(0)
    return pd.DatetimeIndex(
        [first + pd.offsets.MonthEnd(m) for m in range(n)]
    )


def make_clean_maximal_panel(
    spec: SyntheticSpec | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (maximal_panel, signals) for a clean panel.

    maximal_panel carries the dual `_raw`/`_corr` families (IDENTICAL on the clean
    panel), `exit_reason` (all None), `last_trade_date_{raw,corr}` (= month-end,
    so nothing is stale) and `universe_eligible=True`. signals carries
    `score_raw`/`score_corr` (identical), resolved to `score` by `view()`.
    """
    spec = spec or SyntheticSpec()
    rng = np.random.default_rng(spec.seed)
    dates = _month_ends(spec.start, spec.n_months)
    cusips = [f"BOND{j:04d}" for j in range(spec.n_bonds)]

    q = rng.uniform(-1.0, 1.0, size=spec.n_bonds)  # persistent characteristic

    rows = []
    sig_rows = []
    rf = 0.001
    for i, cusip in enumerate(cusips):
        for t, date in enumerate(dates):
            ret = spec.alpha * q[i] + rng.normal(0.0, spec.noise_sd)
            if spec.time_varying_score:
                score = float(rng.uniform(-1.0, 1.0))
            else:
                score = float(q[i])
            rows.append(
                {
                    "cusip": cusip,
                    "date": date,
                    "size": 1.0,
                    "rf_monthly": rf,
                    "exit_reason": None,
                    "universe_eligible": True,
                    "price_eom_raw": 100.0,
                    "price_eom_corr": 100.0,
                    "ret_raw": ret,
                    "ret_corr": ret,
                    "xret_raw": ret - rf,
                    "xret_corr": ret - rf,
                    "n_trades_raw": 5,
                    "n_trades_corr": 5,
                    "total_vol_raw": 1e6,
                    "total_vol_corr": 1e6,
                    "last_trade_date_raw": date,
                    "last_trade_date_corr": date,
                }
            )
            sig_rows.append(
                {
                    "cusip": cusip,
                    "date": date,
                    "score_raw": score,
                    "score_corr": score,
                }
            )

    panel = pd.DataFrame(rows)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["last_trade_date_raw"] = pd.to_datetime(panel["last_trade_date_raw"])
    panel["last_trade_date_corr"] = pd.to_datetime(panel["last_trade_date_corr"])
    signals = pd.DataFrame(sig_rows)
    signals["date"] = pd.to_datetime(signals["date"])
    return panel, signals


# ---------------------------------------------------------------------------
# The synthetic strategy (sorts on `score`)
# ---------------------------------------------------------------------------

def score_strategy(
    label: str = "synthetic",
    *,
    groups: int = 5,
    weighting: str = "equal",
    signal_lag: int = 0,
    trim: dict | None = None,
) -> AdaptResult:
    """A single-leg strategy ranking on `score` (long top group, short bottom).
    `trim` is a raw engine-style trim spec (e.g. TRUNCATE_TRIM_SPEC) recorded as
    the strategy's as-published trim; the cell runner overrides it per lab_trim
    state. The construction toggles here are the base/as-published values."""
    score = Binding("score", "BOUND", Evidence(column="score"))
    trim_field = (
        None if trim is None
        else Inherited(trim, "DESIGN", Evidence(note="synthetic published trim"))
    )
    cfg = build_quant_config(
        label,
        score,
        groups=Inherited(groups, "DESIGN", Evidence(note="synthetic groups")),
        weighting=Inherited(weighting, "DESIGN", Evidence(note="synthetic weighting")),
        signal_lag=Inherited(signal_lag, "DESIGN", Evidence(note="synthetic base lag")),
        long_group=Inherited(groups - 1, "DESIGN", Evidence(note="top group")),
        short_group=Inherited(0, "DESIGN", Evidence(note="bottom group")),
        holding_period=Inherited(1, "DESIGN", Evidence(note="monthly rebalance")),
        trim=trim_field,
    )
    leg = LegCall(strategy_id=f"{label}::0", kwargs={}, result=cfg)
    return AdaptResult(
        strategy_label=label,
        leg_calls=(leg,),
        combiner=CombinerInstruction(kind="single_leg"),
    )


# ---------------------------------------------------------------------------
# Bias injection — perturb ONLY the as-published (OFF) arm of one toggle
# ---------------------------------------------------------------------------

def _bond_scores(signals: pd.DataFrame) -> pd.Series:
    """Per-bond mean score (= the persistent characteristic q_i on a clean panel)."""
    return signals.groupby("cusip")["score_raw"].mean()


def _leg_bonds(signals: pd.DataFrame, frac: float, *, high: bool) -> set:
    s = _bond_scores(signals).sort_values(ascending=not high)
    n = max(1, int(round(len(s) * frac)))
    return set(s.index[:n])


def inject_meas_err(panel: pd.DataFrame, signals: pd.DataFrame, mag: float):
    """Corrupt the RAW price family only (proportional to the characteristic), so
    meas_err OFF (raw) is biased and ON (corr) is clean."""
    panel = panel.copy()
    q = panel["cusip"].map(_bond_scores(signals))
    panel["ret_raw"] = panel["ret_raw"] + mag * q
    panel["xret_raw"] = panel["xret_raw"] + mag * q
    return panel, signals


def inject_stale_price(panel: pd.DataFrame, signals: pd.DataFrame, mag: float):
    """Make long-leg bonds stale (last trade > theta before month-end) AND inflate
    their return in BOTH families. stale_price OFF includes the inflated stale
    bonds; ON masks them out. (Both families move together, so meas_err stays inert.)"""
    panel = panel.copy()
    high = _leg_bonds(signals, 0.3, high=True)
    mask = panel["cusip"].isin(high)
    old = panel.loc[mask, "date"] - pd.Timedelta(days=45)
    panel.loc[mask, "last_trade_date_raw"] = old
    panel.loc[mask, "last_trade_date_corr"] = old
    for col in ("ret_raw", "ret_corr", "xret_raw", "xret_corr"):
        panel.loc[mask, col] = panel.loc[mask, col] + mag
    return panel, signals


def inject_survivorship(panel: pd.DataFrame, signals: pd.DataFrame, mag: float):
    """Give long-leg bonds a distress exit with a crater return, SPREAD across
    different exit months (real bonds default at different times), and drop each
    bond's rows after its default. survivorship OFF drops the terminal (crater)
    row so the loss vanishes; ON keeps it. Spreading the exits means many
    realisation months carry a crater, so the mean effect is material."""
    panel = panel.copy()
    dates = sorted(panel["date"].unique())
    n = len(dates)
    lo = n // 2                       # defaults occur in the second half
    span = max(1, n - lo - 1)
    high = sorted(_leg_bonds(signals, 0.3, high=True))

    drop_mask = pd.Series(False, index=panel.index)
    for i, bond in enumerate(high):
        exit_t = lo + (i % span)
        exit_date = dates[exit_t]
        rows = panel.index[panel["cusip"] == bond]
        for idx in rows:
            d = panel.loc[idx, "date"]
            if d == exit_date:
                panel.loc[idx, "exit_reason"] = "defaulted"
                panel.loc[idx, ["ret_raw", "ret_corr"]] = -mag
                rf = panel.loc[idx, "rf_monthly"]
                panel.loc[idx, ["xret_raw", "xret_corr"]] = -mag - rf
            elif d > exit_date:
                drop_mask.loc[idx] = True  # bond is dead after default
    panel = panel[~drop_mask].reset_index(drop=True)
    return panel, signals


def inject_lab_trim(panel: pd.DataFrame, signals: pd.DataFrame, mag: float, rng):
    """Plant extreme negative returns (beyond the truncation bound) in long-leg
    bonds in a subset of months, in BOTH families. With the published truncate
    trim, lab_trim OFF drops these big losses (flattering the long leg); ON keeps
    them. The strategy must carry TRUNCATE_TRIM_SPEC (see build_scenario)."""
    panel = panel.copy()
    high = _leg_bonds(signals, 0.3, high=True)
    mask = panel["cusip"].isin(high)
    hit = mask & (rng.random(len(panel)) < 0.25)
    extreme = -(0.5 + mag)   # beyond the truncate bound of -0.5
    panel.loc[hit, "ret_raw"] = extreme
    panel.loc[hit, "ret_corr"] = extreme
    # Keep the panel's own xret == ret - rf invariant on the injected rows.
    rf = panel.loc[hit, "rf_monthly"]
    panel.loc[hit, "xret_raw"] = extreme - rf
    panel.loc[hit, "xret_corr"] = extreme - rf
    return panel, signals


def _make_lib_gap_panel(spec: SyntheticSpec, mag: float):
    """A panel where next-month return is predicted by the CONTEMPORANEOUS
    formation-month score: ret_{t} = mag * score_{t-1} + noise, with i.i.d.
    time-varying scores. So lag=0 (rank on score_t) captures the signal and lag=1
    (rank on score_{t-1}) misses it — lib_gap OFF outperforms ON."""
    rng = np.random.default_rng(spec.seed)
    dates = _month_ends(spec.start, spec.n_months)
    cusips = [f"BOND{j:04d}" for j in range(spec.n_bonds)]
    rf = 0.001
    rows, sig_rows = [], []
    for cusip in cusips:
        scores = rng.uniform(-1.0, 1.0, size=spec.n_months)
        prev = 0.0
        for t, date in enumerate(dates):
            ret = mag * prev + rng.normal(0.0, spec.noise_sd)
            prev = scores[t]
            rows.append({
                "cusip": cusip, "date": date, "size": 1.0, "rf_monthly": rf,
                "exit_reason": None, "universe_eligible": True,
                "price_eom_raw": 100.0, "price_eom_corr": 100.0,
                "ret_raw": ret, "ret_corr": ret,
                "xret_raw": ret - rf, "xret_corr": ret - rf,
                "n_trades_raw": 5, "n_trades_corr": 5,
                "total_vol_raw": 1e6, "total_vol_corr": 1e6,
                "last_trade_date_raw": date, "last_trade_date_corr": date,
            })
            sig_rows.append({
                "cusip": cusip, "date": date,
                "score_raw": float(scores[t]), "score_corr": float(scores[t]),
            })
    panel = pd.DataFrame(rows)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["last_trade_date_raw"] = pd.to_datetime(panel["last_trade_date_raw"])
    panel["last_trade_date_corr"] = pd.to_datetime(panel["last_trade_date_corr"])
    signals = pd.DataFrame(sig_rows)
    signals["date"] = pd.to_datetime(signals["date"])
    return panel, signals


# ---------------------------------------------------------------------------
# Scenario builder
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Scenario:
    """A complete Layer B scenario: the (possibly perturbed) maximal panel, the
    signals, the strategy, and which biases were injected (empty for the
    zero-injection null)."""

    panel: pd.DataFrame
    signals: pd.DataFrame
    strategy: AdaptResult
    injected: tuple[str, ...]
    magnitude: float


def build_scenario(
    bias: str | None,
    magnitude: float | None = None,
    *,
    seed: int = 0,
    spec: SyntheticSpec | None = None,
) -> Scenario:
    """Build a single-bias (or zero-injection) scenario. `bias=None` is the clean
    null; otherwise one of the five toggle ids. The magnitude defaults to
    DEFAULT_MAGNITUDES[bias]."""
    base = spec or SyntheticSpec()
    spec = SyntheticSpec(**{**base.__dict__, "seed": seed})
    rng = np.random.default_rng(seed + 1000)

    if bias is None:
        panel, signals = make_clean_maximal_panel(spec)
        return Scenario(panel, signals, score_strategy(), (), 0.0)

    if bias not in DEFAULT_MAGNITUDES:
        raise ValueError(f"unknown bias {bias!r}; expected one of {sorted(DEFAULT_MAGNITUDES)}")
    mag = DEFAULT_MAGNITUDES[bias] if magnitude is None else magnitude

    if bias == "lib_gap":
        panel, signals = _make_lib_gap_panel(spec, mag)
        return Scenario(panel, signals, score_strategy(), ("lib_gap",), mag)

    panel, signals = make_clean_maximal_panel(spec)
    strat = score_strategy()
    if bias == "meas_err":
        panel, signals = inject_meas_err(panel, signals, mag)
    elif bias == "stale_price":
        panel, signals = inject_stale_price(panel, signals, mag)
    elif bias == "survivorship":
        panel, signals = inject_survivorship(panel, signals, mag)
    elif bias == "lab_trim":
        panel, signals = inject_lab_trim(panel, signals, mag, rng)
        strat = score_strategy(trim=TRUNCATE_TRIM_SPEC)
    return Scenario(panel, signals, strat, (bias,), mag)


# The three pre-registered end-to-end interaction mechanisms (§10.2), chosen to
# span architectural layers: two panel-layer, two construction-layer, one cross-layer.
INTERACTION_MECHANISMS: tuple[tuple[str, str], ...] = (
    ("meas_err", "stale_price"),    # two panel-layer toggles
    ("lib_gap", "lab_trim"),        # two construction-layer toggles
    ("survivorship", "meas_err"),   # across layers
)


def build_interaction_scenario(
    pair: tuple[str, str],
    *,
    seed: int = 0,
    spec: SyntheticSpec | None = None,
) -> Scenario:
    """Compose the two injections of an interaction mechanism onto ONE panel, so
    the full lattice can recover the interaction end-to-end. The injections are
    applied to overlapping long-leg bonds, so their joint effect is non-additive."""
    key = frozenset(pair)
    valid = {frozenset(p) for p in INTERACTION_MECHANISMS}
    if key not in valid:
        raise ValueError(
            f"{pair} is not a registered interaction mechanism {INTERACTION_MECHANISMS}"
        )
    base = spec or SyntheticSpec()
    spec = SyntheticSpec(**{**base.__dict__, "seed": seed})
    rng = np.random.default_rng(seed + 2000)

    if key == frozenset(("lib_gap", "lab_trim")):
        panel, signals = _make_lib_gap_panel(spec, DEFAULT_MAGNITUDES["lib_gap"])
        panel, signals = inject_lab_trim(
            panel, signals, DEFAULT_MAGNITUDES["lab_trim"], rng
        )
        strat = score_strategy(trim=TRUNCATE_TRIM_SPEC)
        return Scenario(panel, signals, strat, ("lib_gap", "lab_trim"), 0.0)

    panel, signals = make_clean_maximal_panel(spec)
    strat = score_strategy()
    injected = []
    for bias in pair:  # deterministic order as given
        mag = DEFAULT_MAGNITUDES[bias]
        if bias == "meas_err":
            panel, signals = inject_meas_err(panel, signals, mag)
        elif bias == "stale_price":
            panel, signals = inject_stale_price(panel, signals, mag)
        elif bias == "survivorship":
            panel, signals = inject_survivorship(panel, signals, mag)
        injected.append(bias)
    return Scenario(panel, signals, strat, tuple(injected), 0.0)
