"""Descriptive out-of-sample runner for the corrected anchors.

Two modes. ``--rehearsal`` validates the descriptive OOS *code* on the DEVELOPMENT
pseudo-window only (non-reportable stand-in numbers; the holdout is never read). ``--real``
is the gated single-access holdout open: CLOSED BY DEFAULT, it refuses unless the
two-part operator confirmation is set (see run_real), then builds the seeded dev+holdout
inventory via ``holdout_inventory.load_holdout_inputs``. Every pre-open guard fires before
the partition is touched.

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

Percentile-trimmed as-published cells and the total-return companion:
  * mom6's published trim (lab_trim OFF, H=6) resolves its 99.5th-percentile cutoff over the
    cell's own panel and applies it to every held-month return (``agents/quant/library/overlap.py``).
    On the seeded (2002–2025) panel the cutoff therefore includes holdout returns (design decision D4),
    so that one cell's dev-half self-verify compares against the dev-only cell rebuilt at the SEEDED
    cutoff; every other cell keeps its strict pin check. The persistence-Δ reference stays the dev pin.
  * The total-return substrate is the default-flat substrate (a defaulted bond trades flat from its
    default month).
  * A real-run artefact is never overwritten (the run refuses before opening the holdout if the
    destination exists).

DEV / HOLDOUT DISCIPLINE: every read in ``--rehearsal`` is under ``data/development/`` (via
``load_dev_inputs`` and ``load_dev_total_return_panel``); every constructed series is asserted to
end before 2022-01.

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
import sys
import tempfile
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
from shared.licensed_inputs import require_licensed_input  # noqa: E402

THRESHOLDS_FILE = _REPO_ROOT / "docs" / "thresholds.yaml"
_DEV_DATA_ROOT = _REPO_ROOT / "data" / "development"   # the ONLY partition this runner may read
_HOLDOUT_FLOOR = pd.Timestamp("2022-01-01")            # any month >= this is holdout territory
# The recorded default-flat development total-return panels. Must equal
# holdout_inventory.DEV_TR_PANEL / DEV_TR_PROFILES (asserted in tests).
_DEV_TR_PANEL = _DEV_DATA_ROOT / "monthly_panel_total_return_default_flat.parquet"
_DEV_TR_PROFILES = _DEV_DATA_ROOT / "monthly_panel_profiles_total_return_default_flat.parquet"

# Fixed anchor order — reported in full, always, in this order (no selection surface).
ANCHOR_ORDER = ("str", "drf", "mom6")
NEG_CONTROL_KEY = "_negative_control"

# The dev pseudo-window the rehearsal evaluates on (exactly as run_oneshot_holdout rehearses):
# development months only, the gate never opens. 2018-01..2021-09 = 45 months (matches the
# holdout window length; the numbers are non-reportable stand-ins that exercise the arithmetic).
PSEUDO_WINDOW = ("2018-01", "2021-09")
# The REAL holdout window (SC-SCI-12).
HOLDOUT_WINDOW = ("2022-01", "2025-09")

# Recorded in-sample corrected (all-ON) baselines — the single source of truth for the
# persistence Δ and the zero-tolerance self-verify. Values are the corrected_parent_mean_per_month
# produced by the SAME corrected_parent chain this runner reuses, so a correct rehearsal
# reproduces them to floating-point precision. The corrected cells carry no trim, so the
# overlap-path trim resolution does not affect them.
#   str  — results/scientist/rq4_funnel/rq4_funnel_reported.json
#   drf  — results/scientist/rq4_capability/rq4_capability_drf_minilm.json
#   mom6 — results/scientist/rq4_capability/rq4_capability_mom6_minilm.json
PINNED_CORRECTED_IN_SAMPLE = {
    "str": 0.0019679237643625653,
    "drf": 0.0033477066753361266,
    "mom6": -0.0028481829436300056,
}
# As-published (all-OFF) + negative-control in-sample baselines (computed from dev with the percentile
# trim resolved on the overlap path; self-verified by --rehearsal). Used as the
# persistence-Δ reference in the gated real run.
PINNED_AS_PUBLISHED_IN_SAMPLE = {
    "str": 0.007949651424605457,
    "drf": 0.0034386749221172516,
    "mom6": 0.006478265466729551,
}
PINNED_NEG_CONTROL_IN_SAMPLE = 7.76033181823128e-05
_BASELINE_ATOL = 1e-9   # recomputed via the identical chain => machine-precision agreement

# --- TOTAL-RETURN companion — default-flat substrate ---------------------------------------------
# The holdout report emits BOTH substrates side by side: clean (frozen primary, above) and
# total-return (companion). The total-return substrate runs the IDENTICAL lattice chain, differing
# ONLY in the base panel — the default-flat total-return return leg, clean-price signals unchanged.
# All pins are computed from the recorded default-flat dev panels (same provenance as above).
PINNED_CORRECTED_IN_SAMPLE_TOTAL_RETURN: dict = {
    "str": 0.002403036582046719,
    "drf": 0.004128692478377258,
    "mom6": -0.0033834994394345854,
}
PINNED_AS_PUBLISHED_IN_SAMPLE_TOTAL_RETURN: dict = {
    "str": 0.008498566766636377,
    "drf": 0.004562406883788531,
    "mom6": 0.004314723032145246,
}
PINNED_NEG_CONTROL_IN_SAMPLE_TOTAL_RETURN = 0.00048241375857034435

# The realised as-published trim cutoff of each percentile-trimmed anchor on the DEVELOPMENT panel
# (the cutoff behind the as-published dev pin), per substrate. Only mom6 carries a percentile trim.
PINNED_TRIM_CUTOFF_DEV: dict = {
    "clean": {"mom6": 0.12982514054489847},
    "total_return": {"mom6": 0.13623757934357994},
}


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


def load_dev_total_return_panel() -> pd.DataFrame:
    """The development TOTAL-RETURN maximal panel: the recorded default-flat
    ``monthly_panel_total_return_default_flat`` (raw/corr total-return legs) LEFT-merged with the
    recorded ``monthly_panel_profiles_total_return_default_flat`` (drf/mom6 as-published profile
    families) — EXACTLY the two base panels the default-flat total-return RQ3 audit consumes
    (``scripts/run_full_audit_total_return.py``). Dev-only; the same clean signals are reused for the
    sort variable. Both the source path and the loaded dates are guarded against the holdout."""
    _guard_no_holdout([_DEV_DATA_ROOT])
    maximal_tr = pd.read_parquet(require_licensed_input(_DEV_TR_PANEL, "total-return development panel"))
    maximal_tr = maximal_tr.merge(
        pd.read_parquet(require_licensed_input(_DEV_TR_PROFILES, "total-return profile panel")),
        on=["cusip", "date"], how="left")
    _assert_dev_only(maximal_tr["date"], "dev total-return maximal panel")
    return maximal_tr


def _assemble_substrate(maximal, signals, *, allow_holdout: bool) -> tuple[dict, dict]:
    """Build every anchor's endpoint cells (corrected all-ON, as-published all-OFF) + the negative
    control for ONE substrate panel, via the identical audited lattice chain. Returns
    ``(series_by_key, corrected_dev_means)`` where ``corrected_dev_means[a]`` is the corrected
    cell's DEVELOPMENT-portion mean (full span in the dev/rehearsal case; the dev half of the
    seeded series in the real case) — the quantity the pinned-baseline self-verify checks. The two
    substrates (clean, total-return) are assembled by two calls to this one function; only the
    ``maximal`` panel differs (signals — the sort variable — are the clean signals for both)."""
    series_by_key: dict = {}
    corrected_dev_means: dict = {}
    for a in ANCHOR_ORDER:
        cells = anchor_cells(a, maximal, signals, allow_holdout=allow_holdout)
        series_by_key[a] = cells
        corrected_dev_means[a] = (_dev_portion_mean(cells["corrected"]) if allow_holdout
                                  else float(cells["corrected"].mean()))
    nc = negative_control_corrected(maximal, signals, allow_holdout=allow_holdout)
    series_by_key[NEG_CONTROL_KEY] = {"corrected": nc}
    return series_by_key, corrected_dev_means


def _substrate_in_sample_means(series_by_key: dict, *, allow_holdout: bool,
                               pinned_corrected: dict, pinned_as_published: dict,
                               pinned_neg_control) -> dict:
    """The in-sample baseline map for one substrate: the pinned value where present (the
    single source of truth for the persistence Δ), else the computed dev-portion mean as a
    candidate-to-pin (a not-yet-pinned value on its first run)."""
    def dev_mean(s):
        return _dev_portion_mean(s) if allow_holdout else float(s.mean())
    out: dict = {}
    for a in ANCHOR_ORDER:
        cells = series_by_key[a]
        pc, pp = pinned_corrected.get(a), pinned_as_published.get(a)
        out[a] = {
            "corrected": float(pc) if pc is not None else dev_mean(cells["corrected"]),
            "as_published": float(pp) if pp is not None else dev_mean(cells["as_published"]),
        }
    nc = series_by_key[NEG_CONTROL_KEY]["corrected"]
    out[NEG_CONTROL_KEY] = {"corrected": (float(pinned_neg_control) if pinned_neg_control is not None
                                          else dev_mean(nc))}
    return out


def _real_substrate(maximal, signals, *, pinned_corrected: dict, pinned_as_published: dict,
                    pinned_neg_control, pinned_trim_cutoff: dict, dev_loader,
                    atol: float = 1e-6) -> tuple[dict, dict, dict]:
    """One substrate of the GATED real run: build every seeded (dev+holdout) endpoint cell + the
    negative control, self-verify each cell's DEV-portion mean (fail loud on any drift, refusing to
    report confidently-wrong holdout numbers), and return ``(series_by_key, in_sample_means,
    dev_self_verify)``.

    Self-verify per cell:
      * every corrected cell, and every as-published cell WITHOUT a percentile trim: the seeded dev
        half must reproduce the recorded dev pin (atol).
      * an as-published cell WITH a percentile trim (mom6): its cutoff is resolved on the seeded
        panel, so it includes holdout returns (D4) and the dev pin (dev-only cutoff) is not the
        right target. The seeded dev half must instead reproduce the dev-only cell rebuilt at the
        SEEDED cutoff (atol) — an exact equivalent check on the same construction. The persistence-Δ
        reference stays the dev pin; the dev-only mean at the seeded cutoff and both cutoffs are
        recorded beside it.
    ``dev_loader()`` returns the substrate's development ``(dev_maximal, dev_signals)``; it is called
    only if a percentile-trimmed anchor is present."""
    series_by_key: dict = {}
    in_sample_means: dict = {}
    dev_self_verify: dict = {}
    dev_panels = None
    for a in ANCHOR_ORDER:
        cells = anchor_cells(a, maximal, signals, allow_holdout=True)
        series_by_key[a] = cells

        got = _dev_portion_mean(cells["corrected"])
        _check_close(got, pinned_corrected[a], atol, f"{a}.corrected: seeded dev-portion mean")
        dev_self_verify.setdefault(a, {})["corrected"] = {
            "recomputed_dev_mean": got, "committed_baseline": float(pinned_corrected[a])}

        got = _dev_portion_mean(cells["as_published"])
        cutoff = as_published_trim_cutoff(a, maximal, signals)
        if cutoff is None:
            _check_close(got, pinned_as_published[a], atol, f"{a}.as_published: seeded dev-portion mean")
            dev_self_verify[a]["as_published"] = {
                "recomputed_dev_mean": got, "committed_baseline": float(pinned_as_published[a])}
            in_sample_means[a] = {"corrected": float(pinned_corrected[a]),
                                  "as_published": float(pinned_as_published[a])}
            continue
        if dev_panels is None:
            dev_panels = dev_loader()
        dev_at_seeded = as_published_series_at_cutoff(a, dev_panels[0], dev_panels[1],
                                                      cutoff["trim_rule_absolute"])
        want = float(dev_at_seeded.mean())
        _check_close(got, want, atol,
                     f"{a}.as_published: seeded dev-portion mean vs dev-only cell at the seeded cutoff")
        # Series, not just means: identical months and every month within atol (a missing/extra month or
        # offsetting differences cannot pass on the mean alone).
        seeded = cells["as_published"]
        seeded_months = pd.PeriodIndex(pd.to_datetime(pd.Series(seeded.index)), freq="M")
        seeded_dev = seeded[np.asarray(seeded_months < pd.Period(HOLDOUT_WINDOW[0], "M"))]
        dev_sorted = dev_at_seeded.sort_index()
        if not seeded_dev.index.equals(dev_sorted.index):
            raise RuntimeError(f"{a}.as_published: seeded dev-half months differ from the dev-only cell at the "
                               "seeded cutoff — seeded build drifted; refusing to report")
        max_abs_month = float(np.max(np.abs(seeded_dev.to_numpy() - dev_sorted.to_numpy())))
        if not (math.isfinite(max_abs_month) and max_abs_month <= atol):
            raise RuntimeError(f"{a}.as_published: seeded dev-half series vs dev-only cell at the seeded cutoff "
                               f"max |Δ| {max_abs_month!r} > atol {atol:.0e} — refusing to report")
        dev_self_verify[a]["as_published"] = {
            "check": "seeded dev half == dev-only cell rebuilt at the SEEDED trim cutoff (D4)",
            "recomputed_dev_mean": got,
            "dev_only_cell_mean_at_seeded_cutoff": want,
            "committed_baseline_dev_cutoff": float(pinned_as_published[a]),
            "seeded_trim_cutoff": cutoff["realised"],
            "dev_trim_cutoff_pinned": float(pinned_trim_cutoff[a]),
        }
        in_sample_means[a] = {"corrected": float(pinned_corrected[a]),
                              "as_published": float(pinned_as_published[a]),
                              "as_published_dev_mean_at_seeded_cutoff": want}
    nc = negative_control_corrected(maximal, signals, allow_holdout=True)
    series_by_key[NEG_CONTROL_KEY] = {"corrected": nc}
    in_sample_means[NEG_CONTROL_KEY] = {"corrected": float(pinned_neg_control)}
    return series_by_key, in_sample_means, dev_self_verify


def _check_close(got: float, want, atol: float, what: str) -> None:
    """Fail loud unless ``got`` and a non-None ``want`` are both finite and within ``atol`` (a NaN on
    either side would otherwise pass, since every comparison with NaN is False)."""
    if (want is None or not math.isfinite(got) or not math.isfinite(float(want))
            or abs(got - float(want)) > atol):
        raise RuntimeError(f"{what} {got!r} != {want!r} (atol {atol:.0e}) — seeded build drifted; "
                           "refusing to report")


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


def _anchor_setup(anchor_id: str):
    """(strategy, expost_trim_off, lattice configs, all-ON set) — the per-anchor wiring shared by every
    cell construction here, mirroring ``corrected_parent`` / ``run_full_audit.run_anchor_full``."""
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
    return strategy, expost_trim_off, configs, frozenset(pf.runnable_toggles)


def anchor_cells(anchor_id: str, maximal, signals, *, allow_holdout: bool = False) -> dict:
    """Both endpoint cells for an anchor over the (dev) panel: corrected (all-ON over the runnable
    toggles) and as-published (all-OFF). Mirrors ``corrected_parent``'s scope derivation."""
    strategy, expost_trim_off, configs, all_on = _anchor_setup(anchor_id)
    return {
        "corrected": _cell_series(anchor_id, strategy, expost_trim_off, maximal, signals, configs,
                                  all_on, allow_holdout=allow_holdout),
        "as_published": _cell_series(anchor_id, strategy, expost_trim_off, maximal, signals, configs,
                                     frozenset(), allow_holdout=allow_holdout),
    }


def as_published_trim_cutoff(anchor_id: str, maximal, signals) -> dict | None:
    """The realised as-published (all-OFF) trim cutoff on THIS panel, resolved by the engine's own
    ``resolve_trim_rule`` over the exact panel + rulebook the cell runs — or None when the anchor's
    as-published cell carries no percentile trim (str, drf). Returns
    ``{"trim_rule_absolute": <engine trim dict>, "realised": <float threshold>}``."""
    from agents.auditor.checks.cell_runner import _override_construction
    from agents.quant.config.quant_config import to_rulebook
    from agents.quant.library.characteristic_sort import resolve_trim_rule
    from agents.quant.library.views import view

    strategy, expost_trim_off, configs, _all_on = _anchor_setup(anchor_id)
    if expost_trim_off is None or expost_trim_off.bounds_type != "percentile":
        return None
    rc = next(rc for oset, rc in configs if oset == frozenset())
    panel = view(maximal, rc, signals=signals)
    qc = _override_construction(strategy, rc, expost_trim_off).leg_calls[0].result
    trim_abs, realised = resolve_trim_rule(panel, to_rulebook(qc))
    if realised is None or "hi" not in realised or "lo" in realised:
        raise RuntimeError(f"{anchor_id}: expected a one-sided right percentile trim; got {realised!r}")
    return {"trim_rule_absolute": trim_abs, "realised": float(realised["hi"]["threshold"])}


def as_published_series_at_cutoff(anchor_id: str, maximal, signals, trim_rule_absolute: dict,
                                  *, allow_holdout: bool = False) -> pd.Series:
    """The as-published (all-OFF) cell series with the published trim FIXED at an absolute cutoff
    (``trim_rule_absolute``, as returned by ``as_published_trim_cutoff``) instead of re-resolved on
    this panel — the construction the D4 self-verify compares against."""
    from agents.quant.config import TrimRule

    strategy, _expost_trim_off, configs, _all_on = _anchor_setup(anchor_id)
    bounds = trim_rule_absolute["bounds"]
    if bounds.get("type") != "absolute":
        raise ValueError(f"{anchor_id}: trim_rule_absolute must carry absolute bounds; got {bounds!r}")
    fixed = TrimRule(method=trim_rule_absolute["method"], lo=bounds.get("lo"), hi=bounds.get("hi"))
    return _cell_series(anchor_id, strategy, fixed, maximal, signals, configs, frozenset(),
                        allow_holdout=allow_holdout)


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


def _verify_pinned_baselines(full_dev_corrected: dict,
                             pinned: dict = PINNED_CORRECTED_IN_SAMPLE, *,
                             label: str = "clean", what: str = "corrected") -> None:
    """Zero-tolerance self-verify: the recomputed full-dev mean must reproduce the recorded
    baseline for every anchor. A mismatch means the construction chain drifted — fail loud, never
    grade against a wrong parent. An anchor whose pinned baseline is None (a not-yet-pinned value)
    is SKIPPED — the computed value is reported as a candidate; it is never silently graded."""
    for a in ANCHOR_ORDER:
        want = pinned.get(a)
        if want is None:
            continue
        got = float(full_dev_corrected[a])
        if not (math.isfinite(got) and abs(got - float(want)) <= _BASELINE_ATOL):   # NaN never passes
            raise RuntimeError(
                f"[{label}] {a}: recomputed {what} full-dev mean {got!r} != recorded baseline "
                f"{want!r} (|Δ|={abs(got - float(want)):.3e} > {_BASELINE_ATOL:.0e}) — chain drifted")


def _missing_pins() -> list[str]:
    """Every pin the gated real run needs, listed if unset. The real run refuses BEFORE opening the
    holdout unless this is empty, so an incomplete re-pin can never spend an open."""
    missing: list[str] = []
    for name, pins in (("PINNED_CORRECTED_IN_SAMPLE", PINNED_CORRECTED_IN_SAMPLE),
                       ("PINNED_AS_PUBLISHED_IN_SAMPLE", PINNED_AS_PUBLISHED_IN_SAMPLE),
                       ("PINNED_CORRECTED_IN_SAMPLE_TOTAL_RETURN", PINNED_CORRECTED_IN_SAMPLE_TOTAL_RETURN),
                       ("PINNED_AS_PUBLISHED_IN_SAMPLE_TOTAL_RETURN",
                        PINNED_AS_PUBLISHED_IN_SAMPLE_TOTAL_RETURN)):
        missing += [f"{name}[{a}]" for a in ANCHOR_ORDER if pins.get(a) is None]
    if PINNED_NEG_CONTROL_IN_SAMPLE is None:
        missing.append("PINNED_NEG_CONTROL_IN_SAMPLE")
    if PINNED_NEG_CONTROL_IN_SAMPLE_TOTAL_RETURN is None:
        missing.append("PINNED_NEG_CONTROL_IN_SAMPLE_TOTAL_RETURN")
    for substrate, cut in PINNED_TRIM_CUTOFF_DEV.items():
        missing += [f"PINNED_TRIM_CUTOFF_DEV[{substrate}][{a}]" for a, v in cut.items() if v is None]
    return missing


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _display_path(path: Path) -> str:
    """Repo-relative path when ``path`` lies inside the repo, else the absolute path — so a message or
    provenance field never raises for an out-of-repo destination (e.g. a test's tmp directory)."""
    try:
        return str(Path(path).resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------------------------
# Rehearsal orchestration + provenance
# --------------------------------------------------------------------------------------------

def _thresholds_sha256() -> str:
    return hashlib.sha256(THRESHOLDS_FILE.read_bytes()).hexdigest()


def _rehearse_substrate(maximal, signals, *, label: str, pinned_corrected: dict,
                        pinned_as_published: dict, pinned_neg_control, pinned_trim_cutoff: dict,
                        window: tuple[str, str]) -> dict:
    """One substrate of the rehearsal: build every cell on the dev panel, hard-verify EVERY pinned
    value (corrected, as-published, negative control, dev trim cutoff) at machine precision, and
    report the computed values beside the pins (a None pin reports its computed candidate)."""
    series_by_key, full_dev_corrected = _assemble_substrate(maximal, signals, allow_holdout=False)
    _verify_pinned_baselines(full_dev_corrected, pinned_corrected, label=label, what="corrected")
    full_dev_as_published = {a: float(series_by_key[a]["as_published"].mean()) for a in ANCHOR_ORDER}
    _verify_pinned_baselines(full_dev_as_published, pinned_as_published, label=label,
                             what="as_published")
    nc_mean = float(series_by_key[NEG_CONTROL_KEY]["corrected"].mean())
    if pinned_neg_control is not None and not (
            math.isfinite(nc_mean) and abs(nc_mean - float(pinned_neg_control)) <= _BASELINE_ATOL):
        raise RuntimeError(f"[{label}] negative control: recomputed {nc_mean!r} != pin "
                           f"{pinned_neg_control!r} — chain drifted")
    cutoffs: dict = {}
    for a in ANCHOR_ORDER:
        cut = as_published_trim_cutoff(a, maximal, signals)
        if cut is None:
            continue
        cutoffs[a] = cut["realised"]
        pin = pinned_trim_cutoff.get(a)
        if pin is not None and cut["realised"] != float(pin):
            raise RuntimeError(f"[{label}] {a}: dev trim cutoff {cut['realised']!r} != pin {pin!r}")
    in_sample = _substrate_in_sample_means(
        series_by_key, allow_holdout=False, pinned_corrected={}, pinned_as_published={},
        pinned_neg_control=None)   # the COMPUTED dev means (every pinned one was just verified)
    return {
        "report": build_report(series_by_key, in_sample, window=window),
        "in_sample_means_computed": in_sample,
        "dev_trim_cutoffs_computed": cutoffs,
        "pins_verified": {
            "corrected": {a: pinned_corrected.get(a) for a in ANCHOR_ORDER},
            "as_published": {a: pinned_as_published.get(a) for a in ANCHOR_ORDER},
            "negative_control": pinned_neg_control,
            "dev_trim_cutoff": dict(pinned_trim_cutoff),
            "atol": _BASELINE_ATOL,
            "note": "every non-None pin was reproduced (a failure raises before this record exists)",
        },
    }


def run_rehearsal(*, window: tuple[str, str] = PSEUDO_WINDOW, out_dir: Path | None = None,
                  write: bool = True) -> dict:
    """Full dev-only rehearsal: build every anchor's endpoint cells + the negative control over
    BOTH substrates — clean (frozen primary) and total-return (default-flat companion) — hard-verify
    every pinned value, then produce a descriptive report per substrate on the pseudo-window. NO
    holdout read; the numbers are non-reportable stand-ins that exercise the full construction."""
    maximal, signals = load_dev_panels()   # guards the dev source + asserts loaded data is pre-holdout
    clean = _rehearse_substrate(
        maximal, signals, label="clean", pinned_corrected=PINNED_CORRECTED_IN_SAMPLE,
        pinned_as_published=PINNED_AS_PUBLISHED_IN_SAMPLE,
        pinned_neg_control=PINNED_NEG_CONTROL_IN_SAMPLE,
        pinned_trim_cutoff=PINNED_TRIM_CUTOFF_DEV["clean"], window=window)
    del maximal
    maximal_tr = load_dev_total_return_panel()
    total_return = _rehearse_substrate(
        maximal_tr, signals, label="total_return",
        pinned_corrected=PINNED_CORRECTED_IN_SAMPLE_TOTAL_RETURN,
        pinned_as_published=PINNED_AS_PUBLISHED_IN_SAMPLE_TOTAL_RETURN,
        pinned_neg_control=PINNED_NEG_CONTROL_IN_SAMPLE_TOTAL_RETURN,
        pinned_trim_cutoff=PINNED_TRIM_CUTOFF_DEV["total_return"], window=window)

    artifact = {
        "purpose": ("DESCRIPTIVE corrected-anchor OOS runner — REHEARSAL on the development "
                    "pseudo-window (build-before-open). BOTH substrates (clean primary + default-flat "
                    "total-return companion), percentile trim resolved on the overlap path. "
                    "Non-reportable stand-in numbers proving the wiring; NO holdout was read."),
        "mode": "rehearsal",
        "reads_holdout": False,
        "development_only": True,
        "substrates": ["clean", "total_return"],
        "pseudo_window": {"start": window[0], "end": window[1]},
        "real_holdout_window": {"start": HOLDOUT_WINDOW[0], "end": HOLDOUT_WINDOW[1]},
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "thresholds_sha256": _thresholds_sha256(),
        "dev_total_return_panels": {
            str(_DEV_TR_PANEL.relative_to(_REPO_ROOT)): _sha256(_DEV_TR_PANEL),
            str(_DEV_TR_PROFILES.relative_to(_REPO_ROOT)): _sha256(_DEV_TR_PROFILES),
        },
        "clean": clean,
        "total_return": total_return,
    }

    if write:
        out = out_dir or (_REPO_ROOT / "results" / "scientist" / "holdout_oos_rehearsal")
        out.mkdir(parents=True, exist_ok=True)
        dest = out / "oos_descriptive_rehearsal_dev_pseudo.json"
        tmp_fd, tmp_name = tempfile.mkstemp(dir=out, prefix=f"{dest.stem}.", suffix=".json.tmp")
        tmp = Path(tmp_name)
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(artifact, f, indent=2, allow_nan=False)
        tmp.replace(dest)
        artifact["_written_to"] = str(_display_path(dest))
    return artifact


def _print_rehearsal_summary(artifact: dict) -> None:
    win = artifact["pseudo_window"]
    print(f"OOS descriptive REHEARSAL (dev pseudo-window {win['start']}..{win['end']}) — "
          f"both substrates; every pinned value reproduced")
    for title in ("clean", "total_return"):
        sub = artifact[title]
        print(f"  [{title}]  dev trim cutoffs {sub['dev_trim_cutoffs_computed']}")
        for a in ANCHOR_ORDER:
            ism = sub["in_sample_means_computed"][a]
            print(f"    {a:5s} in-sample corrected {ism['corrected']!r}  as_published {ism['as_published']!r}")
        print(f"    negative control in-sample {sub['in_sample_means_computed'][NEG_CONTROL_KEY]['corrected']!r}")
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
    """GATED real holdout run. Builds the seeded (dev+holdout) inventory via
    ``holdout_inventory.load_holdout_inputs`` — which REFUSES without the explicit gate
    (token arg + HOLDOUT_OOS_OPEN=1 env) — then runs the identical construction and produces the
    descriptive report over the holdout window. Every cell's seeded dev half is self-verified
    (``_real_substrate``), so a broken seeded build fails loud rather than emitting
    confidently-wrong holdout numbers."""
    import holdout_inventory

    # Cheap early gate check: a sealed --real refuses IMMEDIATELY, before the heavy dev self-check
    # precondition runs. load_holdout_inputs re-checks the gate (defense in depth).
    holdout_inventory._require_gate(gate)

    # PRE-OPEN: every pin must be set — including a dev trim-cutoff pin, on both substrates, for EVERY
    # anchor whose as-published cell carries a percentile trim (else _real_substrate would raise a
    # KeyError after the holdout is open) — the run must record its artefact, and the destination must
    # not already hold one.
    missing = _missing_pins()
    for a in ANCHOR_ORDER:
        trim = _anchor_setup(a)[1]
        if trim is not None and trim.bounds_type == "percentile":
            for substrate in ("clean", "total_return"):
                if PINNED_TRIM_CUTOFF_DEV.get(substrate, {}).get(a) is None:
                    missing.append(f"PINNED_TRIM_CUTOFF_DEV[{substrate}][{a}] (trim cutoff pin)")
    if missing:
        raise holdout_inventory.HoldoutGateError(
            f"pre-open guard: unset pins {missing} — run --rehearsal and pin the verified values first. "
            "The holdout is NOT read.")
    if not write:
        raise holdout_inventory.HoldoutGateError(
            "pre-open guard: write=False — a real run reads the holdout and must record its artefact. "
            "The holdout is NOT read.")
    out = out_dir or (_REPO_ROOT / "results" / "scientist" / "holdout_oos_real")
    dest = out / "oos_descriptive_holdout.json"
    if write and dest.exists():
        raise holdout_inventory.HoldoutGateError(
            f"pre-open guard: {_display_path(dest)} already exists — a real-run artefact is "
            "never overwritten. The holdout is NOT read.")

    # PRE-OPEN capability guard: refuse BEFORE reading the holdout if a required construction family
    # cannot be built, so the irreversible open is never spent on a run that will crash. The
    # drf/mom6 as-published (all-OFF) cells run at a per-paper PROFILE family (bbw_2019/jostova_2013).
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
    if not holdout_inventory.TOTAL_RETURN_LAYER_SUPPORTED:
        raise holdout_inventory.HoldoutGateError(
            "pre-open guard: the total-return companion substrate requires the seeded total-return "
            "base panel (assemble_total_return_maximal), which is not validated "
            "(holdout_inventory.TOTAL_RETURN_LAYER_SUPPORTED). Refusing BEFORE opening the holdout. "
            "The holdout is NOT read.")

    # Fail-closed precondition (self-defending open): the machine-precision dev reproduction must
    # pass before the holdout is ever opened — INCLUDING the seeded default-flat total-return base panel
    # (selfcheck_on_dev reproduces the recorded default-flat dev total-return artefacts). Dev-only,
    # no holdout read; raises on any divergence.
    holdout_inventory.selfcheck_on_dev()

    # ONE holdout read serves BOTH substrates: `with_total_return` derives the seeded (dev+holdout)
    # TOTAL-RETURN base panel from the SAME seeded maximal via the audited accrual transform.
    maximal, maximal_tr, signals, _reg = holdout_inventory.load_holdout_inputs(
        holdout_daily_dir=_REPO_ROOT / "data" / "holdout", gate=gate,   # closed by default
        with_total_return=True)

    # clean substrate (frozen primary)
    series_by_key, in_sample_means, dev_self_verify = _real_substrate(
        maximal, signals, pinned_corrected=PINNED_CORRECTED_IN_SAMPLE,
        pinned_as_published=PINNED_AS_PUBLISHED_IN_SAMPLE,
        pinned_neg_control=PINNED_NEG_CONTROL_IN_SAMPLE,
        pinned_trim_cutoff=PINNED_TRIM_CUTOFF_DEV["clean"], dev_loader=load_dev_panels)
    report = build_report(series_by_key, in_sample_means, window=window)

    # total-return substrate (companion) — identical chain, seeded default-flat total-return panel
    def _dev_tr_loader():
        # The dev total-return panel + the dev CLEAN signals (the sort variable for both substrates),
        # loaded without re-reading the dev clean maximal (RAM: the seeded panels are held too).
        from agents.auditor.ipca_differential.runner import load_dev_signals
        dev_signals = load_dev_signals()
        if "date" in getattr(dev_signals, "columns", []):
            _assert_dev_only(dev_signals["date"], "dev signals")
        return load_dev_total_return_panel(), dev_signals
    series_by_key_tr, in_sample_means_tr, dev_self_verify_tr = _real_substrate(
        maximal_tr, signals, pinned_corrected=PINNED_CORRECTED_IN_SAMPLE_TOTAL_RETURN,
        pinned_as_published=PINNED_AS_PUBLISHED_IN_SAMPLE_TOTAL_RETURN,
        pinned_neg_control=PINNED_NEG_CONTROL_IN_SAMPLE_TOTAL_RETURN,
        pinned_trim_cutoff=PINNED_TRIM_CUTOFF_DEV["total_return"], dev_loader=_dev_tr_loader)
    report_tr = build_report(series_by_key_tr, in_sample_means_tr, window=window)

    artifact = {
        "purpose": ("DESCRIPTIVE corrected-anchor OOS check on the PROTECTED HOLDOUT (single "
                    "confirmed open), with the percentile trim resolved on the overlap path and the "
                    "default-flat total-return substrate. BOTH substrates: clean (frozen primary) + "
                    "total-return (companion)."),
        "mode": "real_holdout",
        "reads_holdout": True,
        "substrates": ["clean", "total_return"],
        "window": {"start": window[0], "end": window[1]},
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "thresholds_sha256": _thresholds_sha256(),
        "dev_total_return_panels": {
            str(_DEV_TR_PANEL.relative_to(_REPO_ROOT)): _sha256(_DEV_TR_PANEL),
            str(_DEV_TR_PROFILES.relative_to(_REPO_ROOT)): _sha256(_DEV_TR_PROFILES),
        },
        "dev_portion_self_verify": {"atol": 1e-6, "passed": True, "anchors": dev_self_verify},
        "dev_portion_self_verify_total_return": {"atol": 1e-6, "passed": True,
                                                 "anchors": dev_self_verify_tr},
        "in_sample_means_used": in_sample_means,
        "in_sample_means_used_total_return": in_sample_means_tr,
        "report": report,
        "report_total_return": report_tr,
    }
    if write:
        out.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_name = tempfile.mkstemp(dir=out, prefix=f"{dest.stem}.", suffix=".json.tmp")
        tmp = Path(tmp_name)
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(artifact, f, indent=2, allow_nan=False)
        # No-clobber publish: os.link refuses if dest appeared after the pre-open check (e.g. a concurrent
        # run). This run read the holdout, so its artefact is never discarded — kept under a conflict name.
        try:
            os.link(tmp, dest)
        except FileExistsError:
            # mkstemp allocates a name nothing else holds (dateless, no PID reuse), so this
            # run's holdout artefact can never overwrite an earlier conflict record.
            keep_fd, keep_name = tempfile.mkstemp(dir=out, prefix=f"{dest.stem}.conflict_", suffix=".json")
            os.close(keep_fd)
            keep = Path(keep_name)
            tmp.replace(keep)
            raise RuntimeError(f"{_display_path(dest)} appeared during the run; this run's artefact is kept "
                               f"at {_display_path(keep)} — review before anything is reported")
        tmp.unlink()
        artifact["_written_to"] = _display_path(dest)
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
                             "otherwise, and refuses before any read if a pin is unset or the "
                             "destination artefact exists).")
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
            # Sealed: gate closed OR a pre-open guard fired — the holdout was NOT read.
            print(f"Holdout OOS real run REFUSED (holdout NOT read): {exc}", file=sys.stderr)
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
        "set (HOLDOUT_OOS_GATE + HOLDOUT_OOS_OPEN=1).",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
