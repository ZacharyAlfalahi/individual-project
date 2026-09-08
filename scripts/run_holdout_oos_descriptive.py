"""Descriptive out-of-sample runner for the corrected anchors (see the internal holdout
pre-registration).

Two modes. ``--rehearsal`` validates the descriptive OOS *code* on the DEVELOPMENT
pseudo-window only (non-reportable stand-in numbers; the holdout is never read). ``--real``
is the gated single-access holdout open (step 3): CLOSED BY DEFAULT, it refuses unless the
two-part operator confirmation is set (see run_real), then builds the seeded dev+holdout
inventory via ``holdout_inventory.load_holdout_inputs``.

What it computes, per anchor {str, drf, mom6}, DESCRIPTIVELY (no pass/fail, no selection):
  * Item 1 — corrected (all-ON lattice cell): OOS mean/month, Newey-West t, NW interval, n,
    and persistence Δ = mean_OOS − mean_in-sample (in-sample = recorded artefact baseline).
  * Item 2 — as-published (all-OFF lattice cell) beside it, plus the paired differential
    (corrected − as-published) — the RQ3 bias-persistence read. Reported as a PAIR, not a pick.
  * Item 3 — the traded-liquidity negative control (LRF corrected long-short) as an OOS level.

Method is faithful REUSE, never a re-implemented sort: the same audited lattice chain the
in-sample RQ3/RQ4 machinery uses (``corrected_parent`` generalised — build_lattice_configs →
_override_construction → run_characteristic_sort / run_with_holding_period), each cell
self-verified against ``run_cell`` at zero tolerance. NW statistics come from the audited
``_nw_auto_lags``/``_nw_hac_variance`` primitives, so the reported t matches the sanctioned
``summarize_returns`` t exactly.

DEV / HOLDOUT DISCIPLINE: every read in ``--rehearsal`` is under ``data/development/`` via
``load_dev_inputs``; every constructed series is asserted to end before 2022-01. The real mode
refuses without reading anything.

Usage:
  ./.venv/bin/python scripts/run_holdout_oos_descriptive.py --rehearsal   # dev pseudo-window
  ./.venv/bin/python scripts/run_holdout_oos_descriptive.py --real        # gated holdout open (closed by default)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Light, auditor-free import: the audited NW primitives + the sanctioned summary. The heavy
# lattice / dev-input imports are LAZY (inside the construction functions) so importing this
# module for unit tests of the pure stats/assembly layer does not pull agents.auditor.
from agents.quant.library.characteristic_sort import (  # noqa: E402
    _nw_auto_lags,
    _nw_hac_variance,
)

THRESHOLDS_FILE = _REPO_ROOT / "docs" / "thresholds.yaml"
_DEV_DATA_ROOT = _REPO_ROOT / "data" / "development"   # the ONLY partition this runner may read
_HOLDOUT_FLOOR = pd.Timestamp("2022-01-01")            # any month >= this is holdout territory

# Fixed anchor order — reported in full, always, in this order (no selection surface).
ANCHOR_ORDER = ("str", "drf", "mom6")
NEG_CONTROL_KEY = "_negative_control"

# The dev pseudo-window the rehearsal evaluates on (exactly as run_oneshot_holdout rehearses):
# development months only, the gate never opens. 2018-01..2021-09 = 45 months (matches the
# holdout window length; the numbers are non-reportable stand-ins that exercise the arithmetic).
PSEUDO_WINDOW = ("2018-01", "2021-09")
# The REAL holdout window (SC-SCI-12) — recorded for provenance only; never opened here.
HOLDOUT_WINDOW = ("2022-01", "2025-09")

# Recorded in-sample corrected (all-ON) baselines — the single source of truth for the
# persistence Δ and the zero-tolerance self-verify. Values are the corrected_parent_mean_per_month
# produced by the SAME corrected_parent chain this runner reuses, so a correct rehearsal
# reproduces them to floating-point precision.
#   str  — results/scientist/rq4_funnel/rq4_funnel_reported.json
#   drf  — results/scientist/rq4_capability/rq4_capability_drf_minilm.json
#   mom6 — results/scientist/rq4_capability/rq4_capability_mom6_minilm.json
PINNED_CORRECTED_IN_SAMPLE = {
    "str": 0.0019679237643625653,
    "drf": 0.0033477066753361266,
    "mom6": -0.0028481829436300056,
}
# As-published (all-OFF) + negative-control in-sample baselines, pinned in
# the internal holdout pre-registration (computed from dev, self-verified). Used as the
# persistence-Δ reference in the gated real run.
PINNED_AS_PUBLISHED_IN_SAMPLE = {
    "str": 0.007949651424605457,
    "drf": 0.0034386749221172516,
    "mom6": -0.030217602497509634,
}
PINNED_NEG_CONTROL_IN_SAMPLE = 7.76033181823128e-05
_BASELINE_ATOL = 1e-9   # recomputed via the identical chain => machine-precision agreement


# --------------------------------------------------------------------------------------------
# Pure statistics core (unit-tested with synthetic known answers; no dev data required)
# --------------------------------------------------------------------------------------------

def nw_level(series, *, nw_lags: int | None = None, alpha: float = 0.05,
             months_per_year: int = 12) -> dict:
    """Newey-West level of a monthly long-short return series: mean, NW-HAC se, NW t, and the
    analytic (1-alpha) NW confidence interval. Uses the audited ``_nw_auto_lags`` /
    ``_nw_hac_variance`` primitives, so ``nw_t`` equals ``summarize_returns``'s ``t_stat`` for the
    same series and lag choice. Degenerate cases (T<2, sd=0, non-finite/≤0 variance) return None
    for the se/t/CI rather than a laundered zero — strict-JSON safe."""
    s = pd.Series(series).dropna()
    if isinstance(s.index, pd.DatetimeIndex):
        s = s.sort_index()
    T = int(len(s))
    out: dict = {
        "n_months": T,
        "mean_per_month": None,
        "mean_pct_per_month": None,
        "nw_se": None,
        "nw_t": None,
        "nw_lags_used": 0,
        "ci_alpha": float(alpha),
        "ci_low": None,
        "ci_high": None,
    }
    if T == 0:
        return out
    mean = float(s.mean())
    out["mean_per_month"] = mean
    out["mean_pct_per_month"] = mean * 100.0
    if T < 2:
        return out
    L = _nw_auto_lags(T) if nw_lags is None else max(0, min(int(nw_lags), T - 1))
    out["nw_lags_used"] = int(L)
    y = s.to_numpy(dtype=float)
    X = np.ones((T, 1))
    var = _nw_hac_variance(y - mean, X, L)[0, 0]
    if not math.isfinite(var) or var <= 0.0:
        return out
    se = math.sqrt(var)
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    out["nw_se"] = se
    out["nw_t"] = mean / se
    out["ci_low"] = mean - z * se
    out["ci_high"] = mean + z * se
    return out


def descriptive_record(series, in_sample_mean, *, label: str, nw_lags: int | None = None,
                       alpha: float = 0.05) -> dict:
    """A full descriptive row: the NW level of ``series`` plus the persistence Δ against the
    in-sample baseline (None when either the OOS mean or the baseline is absent)."""
    lvl = nw_level(series, nw_lags=nw_lags, alpha=alpha)
    ism = None if in_sample_mean is None else float(in_sample_mean)
    delta = (None if (lvl["mean_per_month"] is None or ism is None)
             else lvl["mean_per_month"] - ism)
    return {"label": label, "in_sample_mean_per_month": ism,
            "persistence_delta_per_month": delta, **lvl}


def _paired_diff_record(a: pd.Series, b: pd.Series, *, label: str,
                        in_sample_mean=None, nw_lags: int | None = None,
                        alpha: float = 0.05) -> dict:
    """Descriptive row for the paired differential (a − b) on the two series' COMMON months.
    Reported as a pair, never a selected winner — this is the snoop-safe RQ3 bias-persistence read."""
    ai, bi = pd.Series(a), pd.Series(b)
    common = ai.index.intersection(bi.index)
    diff = ai.reindex(common) - bi.reindex(common)
    return descriptive_record(diff, in_sample_mean, label=label, nw_lags=nw_lags, alpha=alpha)


def _slice_to_window(s: pd.Series, window: tuple[str, str]) -> pd.Series:
    """Slice a month-indexed series to the inclusive [start, end] month window."""
    if len(s) == 0:
        return s
    lo, hi = pd.Period(window[0], "M"), pd.Period(window[1], "M")
    idx = pd.PeriodIndex(pd.to_datetime(pd.Series(s.index)), freq="M")
    mask = np.asarray((idx >= lo) & (idx <= hi))
    return s[mask]


def build_report(series_by_key: dict, in_sample_means: dict, *, window: tuple[str, str],
                 alpha: float = 0.05, nw_lags: int | None = None) -> dict:
    """Assemble the full descriptive report from full-span series: slice each to ``window`` and
    emit every row. EVERY anchor gets corrected + as-published + their paired differential; the
    negative control (if present) gets its level. No row is ever dropped or ranked — the
    all-reported / no-selection rule is structural here."""
    anchors: dict = {}
    for a in ANCHOR_ORDER:
        cells = series_by_key[a]
        ism = in_sample_means.get(a, {})
        corr_w = _slice_to_window(cells["corrected"], window)
        pub_w = _slice_to_window(cells["as_published"], window)
        diff_ism = (None if (ism.get("corrected") is None or ism.get("as_published") is None)
                    else float(ism["corrected"]) - float(ism["as_published"]))
        anchors[a] = {
            "corrected": descriptive_record(
                corr_w, ism.get("corrected"), label=f"{a}:corrected", nw_lags=nw_lags, alpha=alpha),
            "as_published": descriptive_record(
                pub_w, ism.get("as_published"), label=f"{a}:as_published", nw_lags=nw_lags, alpha=alpha),
            "corrected_minus_as_published": _paired_diff_record(
                corr_w, pub_w, label=f"{a}:corrected_minus_as_published",
                in_sample_mean=diff_ism, nw_lags=nw_lags, alpha=alpha),
        }
    negative_control = None
    if NEG_CONTROL_KEY in series_by_key:
        nc = _slice_to_window(series_by_key[NEG_CONTROL_KEY]["corrected"], window)
        negative_control = descriptive_record(
            nc, in_sample_means.get(NEG_CONTROL_KEY, {}).get("corrected"),
            label="negative_control:traded_liquidity_corrected", nw_lags=nw_lags, alpha=alpha)
    return {
        "window": {"start": window[0], "end": window[1]},
        "ci_alpha": float(alpha),
        "nw_lags": ("auto" if nw_lags is None else int(nw_lags)),
        "anchors": anchors,
        "negative_control": negative_control,
        "reporting_rule": ("all rows reported; descriptive (a sign, a level, an interval); "
                           "no pass/fail; no selection among candidates"),
    }


# --------------------------------------------------------------------------------------------
# Construction layer (dev-only; heavy imports are lazy; exercised by the rehearsal run)
# --------------------------------------------------------------------------------------------

def _guard_no_holdout(paths) -> None:
    """Fail loud if any configured input path touches the holdout partition."""
    for p in paths:
        if "holdout" in Path(p).parts:
            raise RuntimeError(f"REFUSED: path touches holdout: {p}")


def _assert_dev_only(dates, what: str) -> None:
    """Fail loud if any loaded/constructed month reaches the holdout floor (2022-01). This is the
    real data-boundary guard: it fires on the ACTUAL data, not a static path, so repointing the dev
    loader at the holdout (or a seeded panel leaking holdout months) is caught before any statistic."""
    s = pd.to_datetime(pd.Series(dates)).dropna()
    if s.empty:
        raise RuntimeError(f"{what}: no dated rows — cannot certify dev-only")
    if pd.Timestamp(s.max()) >= _HOLDOUT_FLOOR:
        raise RuntimeError(f"{what}: reads into the holdout (max {s.max()} >= {_HOLDOUT_FLOOR.date()})")


def load_dev_panels():
    """The development maximal panel + signals via the auditor's dev loader (dev-only; never
    reads the holdout). Returned once and shared by every construction below. Both the declared
    source path and the ACTUAL loaded dates are guarded against the holdout."""
    from agents.auditor.ipca_differential.runner import load_dev_inputs
    _guard_no_holdout([_DEV_DATA_ROOT])                       # declared source is the dev partition
    maximal, signals, _registry = load_dev_inputs()
    _assert_dev_only(maximal["date"], "dev maximal panel")   # actual data ends before the holdout
    if "date" in getattr(signals, "columns", []):
        _assert_dev_only(signals["date"], "dev signals")
    return maximal, signals


def _cell_series(anchor_id: str, strategy, expost_trim_off, maximal, signals, configs,
                 on_set: frozenset, *, allow_holdout: bool = False) -> pd.Series:
    """The monthly long-short return series for one lattice cell of an anchor, built via the same
    audited chain as ``corrected_parent`` and self-verified against ``run_cell`` at zero tolerance.
    Fails loud if the anchor is not a single-leg sort, if the cell is missing, or if the series
    reads into the holdout (a dev-only invariant)."""
    from agents.auditor.checks.cell_runner import _override_construction, run_cell
    from agents.quant.config.quant_config import to_rulebook
    from agents.quant.library.characteristic_sort import run_characteristic_sort
    from agents.quant.library.overlap import run_with_holding_period
    from agents.quant.library.views import view

    try:
        rc = next(rc for oset, rc in configs if oset == on_set)
    except StopIteration as exc:
        raise RuntimeError(
            f"{anchor_id}: lattice cell {sorted(on_set)} absent from build_lattice_configs") from exc

    panel = view(maximal, rc, signals=signals)
    overridden = _override_construction(strategy, rc, expost_trim_off)
    if len(overridden.leg_calls) != 1:
        raise RuntimeError(f"{anchor_id}: expected a single-leg anchor, "
                           f"got {len(overridden.leg_calls)} legs — seam reduction invalid")
    qc = overridden.leg_calls[0].result
    holding_period = int(qc.holding_period.value)
    base_rulebook = to_rulebook(qc)

    if holding_period == 1:
        mr = run_characteristic_sort(panel, base_rulebook)["monthly_returns"]
    else:
        mr = run_with_holding_period(panel, base_rulebook, holding_period)
    series = pd.Series(mr["strategy_ret"].to_numpy(),
                       index=pd.DatetimeIndex(mr["date"].to_numpy())).sort_index()

    cell = run_cell(strategy, rc, on_set, panel, expost_trim_off=expost_trim_off)
    cell_ret = cell.returns.sort_index()
    if not (series.index.equals(cell_ret.index)
            and np.allclose(series.to_numpy(), cell_ret.to_numpy(), rtol=0, atol=0)):
        raise RuntimeError(f"{anchor_id} {sorted(on_set)}: cell series diverges from run_cell")
    if len(series) == 0:
        raise RuntimeError(f"{anchor_id} {sorted(on_set)}: empty cell series")
    if not allow_holdout:
        _assert_dev_only(series.index, f"{anchor_id} {sorted(on_set)} cell series")
    return series


def anchor_cells(anchor_id: str, maximal, signals, *, allow_holdout: bool = False) -> dict:
    """Both endpoint cells for an anchor over the (dev) panel: corrected (all-ON over the runnable
    toggles) and as-published (all-OFF). Mirrors ``corrected_parent``'s scope derivation."""
    from agents.auditor.checks.lattice import build_lattice_configs
    from agents.auditor.checks.preflight import derive_scope
    from scripts.run_auditor import (
        default_anchor_facts,
        load_anchor_expost_trim_off,
        load_anchor_meas_err_off_family,
        load_anchor_strategy,
    )

    strategy = load_anchor_strategy(anchor_id)
    facts = default_anchor_facts(anchor_id)
    meas_err_off_family = load_anchor_meas_err_off_family(anchor_id)
    expost_trim_off = load_anchor_expost_trim_off(anchor_id, None)

    pf = derive_scope(strategy.strategy_label, facts)
    configs = build_lattice_configs(
        pf.runnable_toggles, pf.fixed_states, lib_gap_lags=(0, 1),
        not_applicable_toggles=pf.not_applicable_toggles, meas_err_off_family=meas_err_off_family)
    all_on = frozenset(pf.runnable_toggles)
    all_off = frozenset()
    return {
        "corrected": _cell_series(anchor_id, strategy, expost_trim_off, maximal, signals, configs,
                                  all_on, allow_holdout=allow_holdout),
        "as_published": _cell_series(anchor_id, strategy, expost_trim_off, maximal, signals, configs,
                                     all_off, allow_holdout=allow_holdout),
    }


def negative_control_corrected(maximal, signals, *, allow_holdout: bool = False) -> pd.Series:
    """The traded-liquidity negative control as a corrected long-short LEVEL series: the LRF
    corrected-price-family premium, reused from the registered negative-control run path (no
    re-implemented sort). Expected near-null out-of-sample."""
    from agents.auditor.validation.negative_control_run import _series_of, lrf_family_returns

    lrf_signals = signals[["cusip", "date", "gamma_illiq_raw", "gamma_illiq_corr"]].rename(
        columns={"gamma_illiq_raw": "gamma_raw", "gamma_illiq_corr": "gamma_corr"})
    corr_mr = lrf_family_returns(maximal, lrf_signals, "corr")
    series = _series_of(corr_mr, "strategy_ret")
    if len(series) == 0:
        raise RuntimeError("negative control: empty series")
    if not allow_holdout:
        _assert_dev_only(series.index, "negative control series")
    return series


def _verify_pinned_baselines(full_dev_corrected: dict) -> None:
    """Zero-tolerance self-verify: the recomputed all-ON full-dev mean must reproduce the recorded
    corrected_parent baseline for every anchor. A mismatch means the construction chain drifted —
    fail loud, never grade against a wrong parent."""
    for a in ANCHOR_ORDER:
        got = float(full_dev_corrected[a])
        want = PINNED_CORRECTED_IN_SAMPLE[a]
        if abs(got - want) > _BASELINE_ATOL:
            raise RuntimeError(
                f"{a}: recomputed corrected full-dev mean {got!r} != recorded baseline "
                f"{want!r} (|Δ|={abs(got - want):.3e} > {_BASELINE_ATOL:.0e}) — chain drifted")


# --------------------------------------------------------------------------------------------
# Rehearsal orchestration + provenance
# --------------------------------------------------------------------------------------------

def _git(cmd: list[str]) -> str:
    try:
        return subprocess.run(["git", *cmd], cwd=_REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _thresholds_sha256() -> str:
    return hashlib.sha256(THRESHOLDS_FILE.read_bytes()).hexdigest()


def run_rehearsal(*, window: tuple[str, str] = PSEUDO_WINDOW, out_dir: Path | None = None,
                  write: bool = True) -> dict:
    """Full dev-only rehearsal: build every anchor's endpoint cells + the negative control over
    the development panel, self-verify the corrected baselines at zero tolerance, then produce the
    descriptive report on the pseudo-window. NO holdout read; the numbers are non-reportable
    stand-ins that prove the wiring end-to-end before any real open."""
    maximal, signals = load_dev_panels()   # guards the dev source + asserts loaded data is pre-holdout

    series_by_key: dict = {}
    full_dev_corrected: dict = {}
    computed_in_sample: dict = {}
    for a in ANCHOR_ORDER:
        cells = anchor_cells(a, maximal, signals)
        series_by_key[a] = cells
        full_dev_corrected[a] = float(cells["corrected"].mean())
        computed_in_sample[a] = {
            "corrected": PINNED_CORRECTED_IN_SAMPLE[a],           # recorded single source of truth
            "as_published": float(cells["as_published"].mean()),  # computed -> candidate [to pin]
        }
    nc = negative_control_corrected(maximal, signals)
    series_by_key[NEG_CONTROL_KEY] = {"corrected": nc}
    computed_in_sample[NEG_CONTROL_KEY] = {"corrected": float(nc.mean())}  # computed -> candidate [to pin]

    _verify_pinned_baselines(full_dev_corrected)

    report = build_report(series_by_key, computed_in_sample, window=window)

    artifact = {
        "purpose": ("DESCRIPTIVE corrected-anchor OOS runner — REHEARSAL on the development "
                    "pseudo-window (build-before-open). Non-reportable stand-in numbers proving "
                    "the wiring; NO holdout was read. See the internal holdout pre-registration."),
        "mode": "rehearsal",
        "reads_holdout": False,
        "development_only": True,
        "pseudo_window": {"start": window[0], "end": window[1]},
        "real_holdout_window": {"start": HOLDOUT_WINDOW[0], "end": HOLDOUT_WINDOW[1]},
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "thresholds_sha256": _thresholds_sha256(),
        "baseline_self_verify": {
            "committed_corrected_baselines": PINNED_CORRECTED_IN_SAMPLE,
            "recomputed_corrected_full_dev_means": full_dev_corrected,
            "atol": _BASELINE_ATOL,
            "passed": True,   # a failure raises before reaching here
        },
        "in_sample_means_used": computed_in_sample,
        "report": report,
    }

    if write:
        out = out_dir or (_REPO_ROOT / "results" / "scientist" / "holdout_oos_rehearsal")
        out.mkdir(parents=True, exist_ok=True)
        dest = out / "oos_descriptive_rehearsal_dev_pseudo.json"
        tmp = dest.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(artifact, f, indent=2, allow_nan=False)
        tmp.replace(dest)
        artifact["_written_to"] = str(dest.relative_to(_REPO_ROOT))
    return artifact


def _print_rehearsal_summary(artifact: dict) -> None:
    rep = artifact["report"]
    print(f"OOS descriptive REHEARSAL (dev pseudo-window "
          f"{rep['window']['start']}..{rep['window']['end']}) — baseline self-verify PASSED")
    for a in ANCHOR_ORDER:
        row = rep["anchors"][a]["corrected"]
        m, t, n = row["mean_pct_per_month"], row["nw_t"], row["n_months"]
        ms = f"{m:+.4f}" if m is not None else "  none"
        ts = f"{t:+.3f}" if t is not None else " none"
        print(f"  {a:5s} corrected: mean {ms}%/mo  NW-t {ts}  n {n}  "
              f"Δ {row['persistence_delta_per_month']}")
    if artifact.get("_written_to"):
        print(f"  written: {artifact['_written_to']}")


def _dev_portion_mean(series: pd.Series) -> float:
    """Mean of the series over its DEVELOPMENT months only (< 2022-01) — the seeded panel's dev
    half, which must reproduce the recorded dev baseline."""
    idx = pd.PeriodIndex(pd.to_datetime(pd.Series(series.index)), freq="M")
    dev = series[np.asarray(idx < pd.Period(HOLDOUT_WINDOW[0], "M"))]
    return float(dev.mean()) if len(dev) else float("nan")


def run_real(*, gate: str | None = None, window: tuple[str, str] = HOLDOUT_WINDOW,
             out_dir: Path | None = None, write: bool = True) -> dict:
    """GATED real holdout run (step 3). Builds the seeded (dev+holdout) inventory via
    ``holdout_inventory.load_holdout_inputs`` — which REFUSES without the explicit gate
    (token arg + HOLDOUT_OOS_OPEN=1 env) — then runs the identical construction and produces the
    descriptive report over the holdout window. Unreachable without the gated open. Belt-and-braces
    self-verify: the seeded panel's DEV half must reproduce each recorded corrected baseline, so a
    broken seeded build fails loud rather than emitting confidently-wrong holdout numbers."""
    import holdout_inventory

    # Cheap early gate check: a sealed --real refuses IMMEDIATELY, before the heavy dev self-check
    # precondition runs. load_holdout_inputs re-checks the gate (defense in depth).
    holdout_inventory._require_gate(gate)

    # PRE-OPEN capability guard: refuse BEFORE reading the holdout if a required construction family
    # cannot be built, so the single irreversible open is never spent on a run that will crash. The
    # drf/mom6 as-published (all-OFF) cells need the per-paper PROFILE family (bbw_2019/jostova_2013);
    # the guard fires only if holdout_inventory.PROFILE_FAMILIES_SUPPORTED is ever set False.
    _PROFILE_REQUIRING_ANCHORS = ("drf", "mom6")
    if not holdout_inventory.PROFILE_FAMILIES_SUPPORTED:
        blocked = [a for a in ANCHOR_ORDER if a in _PROFILE_REQUIRING_ANCHORS]
        if blocked:
            raise holdout_inventory.HoldoutGateError(
                f"pre-open guard: the as-published cells for {blocked} require the per-paper profile "
                "families (bbw_2019/jostova_2013), which the seeded holdout inventory does not yet "
                "build. Refusing BEFORE opening the holdout — build the profile families first "
                "(holdout_inventory.PROFILE_FAMILIES_SUPPORTED). The holdout is NOT read.")
    if not holdout_inventory.FISD_HOLDOUT_LAYER_SUPPORTED:
        raise holdout_inventory.HoldoutGateError(
            "pre-open guard: the seeded inventory attaches FISD (size, rating, investment_grade, "
            "maturity, exit_reason) from the DEV-only files (merge_fisd), which end 2021-12. On the "
            "holdout that yields NaN rating (drf/lrf use rating as their bivariate CONTROL axis -> "
            "every bond dropped at eligibility -> empty cells) and NaN size for new-issue bonds (the "
            "VW weight for ALL anchors -> dropped). Refusing BEFORE opening the holdout — build the "
            "holdout-covering FISD layer first (holdout_inventory.FISD_HOLDOUT_LAYER_SUPPORTED). "
            "The holdout is NOT read.")

    # Fail-closed precondition (self-defending open): the machine-precision dev reproduction must
    # pass before we ever open the holdout. Dev-only, no holdout read; raises on any divergence.
    holdout_inventory.selfcheck_on_dev()

    maximal, signals, _reg = holdout_inventory.load_holdout_inputs(
        holdout_daily_dir=_REPO_ROOT / "data" / "holdout", gate=gate)   # closed by default

    series_by_key: dict = {}
    in_sample_means: dict = {}
    dev_self_verify: dict = {}
    for a in ANCHOR_ORDER:
        cells = anchor_cells(a, maximal, signals, allow_holdout=True)
        series_by_key[a] = cells
        # Verify BOTH cells' dev halves reproduce the recorded baselines — corrected (corr family)
        # AND as-published (raw for str, the per-paper profile family for drf/mom6). This is the one
        # place the profile-family construction is numerically checked on data we can trust.
        for cell_kind, want in (("corrected", PINNED_CORRECTED_IN_SAMPLE[a]),
                                ("as_published", PINNED_AS_PUBLISHED_IN_SAMPLE[a])):
            got = _dev_portion_mean(cells[cell_kind])
            if not math.isfinite(got) or abs(got - want) > 1e-6:
                raise RuntimeError(f"{a}.{cell_kind}: seeded dev-portion mean {got!r} != recorded "
                                   f"baseline {want!r} — seeded build drifted; refusing to report")
            dev_self_verify.setdefault(a, {})[cell_kind] = {
                "recomputed_dev_mean": got, "committed_baseline": want}
        in_sample_means[a] = {"corrected": PINNED_CORRECTED_IN_SAMPLE[a],
                              "as_published": PINNED_AS_PUBLISHED_IN_SAMPLE[a]}
    nc = negative_control_corrected(maximal, signals, allow_holdout=True)
    series_by_key[NEG_CONTROL_KEY] = {"corrected": nc}
    in_sample_means[NEG_CONTROL_KEY] = {"corrected": PINNED_NEG_CONTROL_IN_SAMPLE}

    report = build_report(series_by_key, in_sample_means, window=window)
    artifact = {
        "purpose": ("DESCRIPTIVE corrected-anchor OOS check on the PROTECTED HOLDOUT (single "
                    "confirmed open). Per the internal holdout pre-registration."),
        "mode": "real_holdout",
        "reads_holdout": True,
        "window": {"start": window[0], "end": window[1]},
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "thresholds_sha256": _thresholds_sha256(),
        "dev_portion_self_verify": {"atol": 1e-6, "passed": True, "anchors": dev_self_verify},
        "in_sample_means_used": in_sample_means,
        "report": report,
    }
    if write:
        out = out_dir or (_REPO_ROOT / "results" / "scientist" / "holdout_oos_real")
        out.mkdir(parents=True, exist_ok=True)
        dest = out / "oos_descriptive_holdout.json"
        tmp = dest.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(artifact, f, indent=2, allow_nan=False)
        tmp.replace(dest)
        artifact["_written_to"] = str(dest.relative_to(_REPO_ROOT))
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Descriptive corrected-anchor OOS runner (build-before-open)")
    parser.add_argument("--rehearsal", action="store_true",
                        help="run the descriptive check on the development pseudo-window "
                             "(dev-only; the holdout is never read)")
    parser.add_argument("--real", action="store_true",
                        help="attempt the GATED real holdout run (reads /data/holdout/ ONLY if the "
                             "gate env is set: HOLDOUT_OOS_GATE + HOLDOUT_OOS_OPEN=1; refuses "
                             "otherwise). This is the single irreversible open — step 3.")
    args = parser.parse_args(argv)

    if args.rehearsal:
        artifact = run_rehearsal()
        _print_rehearsal_summary(artifact)
        return 0

    if args.real:
        import holdout_inventory
        try:
            artifact = run_real(gate=os.environ.get("HOLDOUT_OOS_GATE"))
        except holdout_inventory.HoldoutGateError as exc:
            # Sealed: gate closed OR the pre-open guard fired — the holdout was NOT read.
            print(f"Holdout OOS real run REFUSED (holdout sealed, NOT read): {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            # A non-gate failure. Pre-open steps (dev self-check, capability guards) can raise
            # here too, so the read may NOT have happened; treat it as indeterminate.
            print(f"Holdout OOS real run FAILED during execution: {exc!r}\n"
                  "The partition MAY have been read (if the failure was at/after the open) -- "
                  "do NOT re-open without review.",
                  file=sys.stderr)
            return 3
        print(f"Holdout OOS REAL run complete -> {artifact.get('_written_to')}")
        return 0

    print(
        "Holdout OOS descriptive: no mode selected. Use --rehearsal for the dev-only validation. "
        "The real run (--real) opens /data/holdout/ and is REFUSED unless the gate env is "
        "set (HOLDOUT_OOS_GATE + HOLDOUT_OOS_OPEN=1) — the single irreversible open, step 3 "
        "(see the internal holdout pre-registration).",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
