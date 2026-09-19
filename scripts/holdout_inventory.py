"""Seeded holdout inventory builder for the descriptive corrected-anchor OOS check.

BUILD-BEFORE-OPEN. This module builds the `(maximal, signals)` inventory the descriptive runner
(`run_holdout_oos_descriptive.py`) consumes — the same format `load_dev_inputs()` returns — but
seeded from the FULL development history concatenated with the holdout daily layer, so every
rolling construction (var_5pct 36-month, mom6, gamma, bond_vol) is warm at 2022-01.

It reads `data/holdout/` ONLY through `load_holdout_inputs`, behind a two-part operator
confirmation (a token arg AND an env var) that is CLOSED by default. `main()`
default refuses; the runner's real mode opens it only when that confirmation is explicitly
set. The assembly + signal layer is
faithful and dev-validated: `selfcheck_on_dev()` rebuilds the recorded dev `monthly_panel_maximal`
and the four signal parquets from the dev daily layer and asserts reproduction within the
self-check tolerance (max abs diff <= 1e-9).

Faithful reuse — no re-implemented aggregation/sort. Reuses `build_monthly_panel.aggregate_family`
/`merge_fisd`, and the signal builders' `compute_dual_family`/`compute_gamma`. The ONLY logic
reproduced here is `build_panel`'s ~20-line ret/xret adjacency glue (it has no path-parameterised
entry point), pinned against the recorded panel by the self-check (max abs diff <= 1e-9).

SEAM SAFETY: the dev→holdout return continuity is handled by concatenating the DAILY layers and
computing `ret` once over the continuous per-cusip price series (exactly as `build_panel` does),
never by concatenating two independently-return-computed monthly panels.

WHAT IS NOT DEV-VALIDATABLE (flagged): the raw→cleaned-daily chain (preprocess_trace →
apply_decimal_shift → bounce_back_filter → apply_distressed_filters) runs only on holdout raw and
has no dev raw to reproduce against; it is a documented prerequisite that must produce the holdout
`trace_daily_raw` + `trace_daily_corr_filtered` this module consumes. Confirmed only at the open.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_bond_vol as _BV  # noqa: E402
import build_gamma_illiq as _G  # noqa: E402
import build_mom6_signal as _M  # noqa: E402
import build_monthly_panel as _B  # noqa: E402
import build_profile_monthly_panel as _PM  # noqa: E402
import build_var_5pct as _V  # noqa: E402
from shared.licensed_inputs import require_licensed_input  # noqa: E402

DEV_ROOT = _REPO_ROOT / "data" / "development"
HOLDOUT_FLOOR = pd.Timestamp("2022-01-01")
RAW_DAILY_NAME = "trace_daily_raw.parquet"
CORR_DAILY_NAME = "trace_daily_corr_filtered.parquet"   # corr = post-distressed daily (build_panel CORR_DAILY)
# Per-paper baseline profile families (drf as-published -> bbw_2019, mom6 as-published -> jostova_2013).
# Each is that profile's dedup-ON cleaned daily layer (build_profile_monthly_panel.PROFILE_FAMILIES).
PROFILE_IDS = ("bbw_2019", "jostova_2013")
_PROFILE_DAILY_NAMES = {pid: f"trace_daily_{pid}__dedup_on.parquet" for pid in PROFILE_IDS}

# Two-part operator-confirmation guard. CLOSED by default: both the token arg AND the env var
# must be set to open the holdout read. A deliberate confirmation latch, not
# cryptographic authorization -- both values are in-source; nothing in this repo sets them.
_GATE_TOKEN = "APPROVED_HOLDOUT_OPEN"
_GATE_ENV = "HOLDOUT_OOS_OPEN"

# The gamma column rename mirrors agents/auditor/ipca_differential/runner.load_dev_signals, so the
# returned signals frame is byte-format-identical to load_dev_inputs (canonical name gamma_illiq).
_GAMMA_RENAME = {"gamma_raw": "gamma_illiq_raw", "gamma_corr": "gamma_illiq_corr"}

# The per-paper baseline PROFILE families (bbw_2019 / jostova_2013) that the drf/mom6
# as-published cells need (via monthly_panel_profiles.parquet + profiles_signals.parquet) are
# built here: assemble_profile_monthly/assemble_profile_signals reproduce the recorded dev
# monthly_panel_profiles + profiles_signals exactly (verified by selfcheck_on_dev), and
# load_holdout_inputs seeds + merges them, so the drf/mom6 as-published cells resolve. The
# runner's pre-open guard refuses BEFORE opening the holdout if this is ever set False.
PROFILE_FAMILIES_SUPPORTED = True

# FISD layer (size, rating, investment_grade, maturity, exit_reason). True: assemble_maximal
# attaches FISD via `_merge_fisd_grid` — built OVER THE PANEL'S OWN grid from the shared data/fisd/
# reference DB (grid-driven build_ratings_monthly clamps to the grid's max month + backward as-of +
# inline leakage assert), so holdout months get their real as-of rating (drf/lrf control) and static
# size, not the dev-file NaN. selfcheck_on_dev reproduces the recorded dev size/rating/universe/
# exit_reason exactly (0-diff, leakage 0). The runner's pre-open guard refuses while this is False.
FISD_HOLDOUT_LAYER_SUPPORTED = True

# TOTAL-RETURN companion layer (default-flat). True:
# assemble_total_return_maximal derives the seeded total-return base panel from the SAME seeded pieces
# via the audited accrual transform (with FISD default_date, so defaulted bonds trade flat), and
# selfcheck_on_dev proves it reproduces the recorded dev monthly_panel_total_return_default_flat
# (raw/corr) + monthly_panel_profiles_total_return_default_flat (profiles) EXACTLY. The runner's
# pre-open guard refuses BEFORE opening the holdout while this is False.
TOTAL_RETURN_LAYER_SUPPORTED = True


class HoldoutGateError(RuntimeError):
    """Raised when the holdout inventory build is invoked without the explicit two-part
    operator confirmation. The default, closed state — the holdout is not read."""


# --------------------------------------------------------------------------------------------
# Faithful assembly (dev-validated by selfcheck_on_dev)
# --------------------------------------------------------------------------------------------

def _merge_fisd_grid(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach FISD (size, rating, investment_grade, maturity, exit_reason) built OVER THE PANEL'S
    OWN (cusip,date) grid from the shared data/fisd/ reference DB — so HOLDOUT months get their
    backward-as-of rating (from the grid-driven build_ratings_monthly, which clamps events to the
    grid's own max month) instead of the DEV-only pre-built file's NaN. Reuses the audited
    build_static/build_ratings_monthly verbatim; only merge_fisd's ~20-line column assembly is
    mirrored (pinned exactly by selfcheck_on_dev). An inline as-of leakage assert guarantees no
    look-ahead on the holdout (a rating dated after its own month)."""
    from agents.quant.library.fisd_reference import build_ratings_monthly, build_static, load_config
    cfg = load_config()
    size_proxy = cfg["amount_outstanding"]["size_proxy"]
    static = build_static(cfg)
    issue_to_cusip = static.dropna(subset=["issue_id"]).set_index("issue_id")["cusip"]
    issue_to_cusip = issue_to_cusip[~issue_to_cusip.index.duplicated(keep="first")]
    grid = panel[["cusip", "date"]].copy()
    grid["cusip"] = grid["cusip"].astype("string")
    ratings = build_ratings_monthly(cfg, grid, issue_to_cusip)
    sel, dt = pd.to_datetime(ratings["_sel_rating_date"]), pd.to_datetime(ratings["date"])
    leak = int((sel.notna() & (sel > dt)).sum())
    if leak:
        raise RuntimeError(f"FISD as-of leakage: {leak} rows carry a rating dated after their month")

    static_sel = static[["cusip", "universe_eligible", size_proxy, "maturity",
                         "default_date", "defeased_date"]].copy()
    static_sel["cusip"] = static_sel["cusip"].astype(str)
    ratings = ratings[["cusip", "date", "rating_numeric", "investment_grade"]].rename(
        columns={"rating_numeric": "rating"})
    ratings["cusip"] = ratings["cusip"].astype(str)

    out = panel.copy()
    out["cusip"] = out["cusip"].astype(str)
    out = out.merge(static_sel, on="cusip", how="left")
    out = out.merge(ratings, on=["cusip", "date"], how="left")
    out["size"] = out[size_proxy]
    out["universe_eligible"] = out["universe_eligible"].fillna(False).astype(bool)
    out["time_to_maturity"] = (out["maturity"] - out["date"]).dt.days / 365.25
    exit_date = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    exit_type = pd.Series(pd.NA, index=out.index, dtype="object")
    for label, col in (("defaulted", "default_date"), ("defeased", "defeased_date"),
                       ("matured", "maturity")):
        d = out[col]
        better = d.notna() & (exit_date.isna() | (d < exit_date))
        exit_date = exit_date.mask(better, d)
        exit_type = exit_type.mask(better, label)
    out["exit_reason"] = exit_type.where(out["date"] >= exit_date, other=pd.NA)
    return out


def assemble_maximal(raw_daily_path: Path, corr_daily_path: Path) -> pd.DataFrame:
    """Build the maximal monthly panel from two daily paths — reproduces build_panel's assembly
    (aggregate_family per family → outer join → per-family ret/xret adjacency → rf merge →
    merge_fisd), the only difference being configurable input paths. Seam-safe: callers pass a
    CONCATENATED (seed+holdout) daily so ret is computed once over the continuous price series."""
    raw_df = _B.aggregate_family(raw_daily_path, "raw")
    corr_df = _B.aggregate_family(corr_daily_path, "corr")
    panel = pd.merge(raw_df, corr_df, on=["cusip_id", "year_month"], how="outer") \
        .sort_values(["cusip_id", "year_month"]).reset_index(drop=True)

    panel["_ym_pd"] = pd.PeriodIndex(panel["year_month"], freq="M")
    for family in ("raw", "corr"):
        price_col = f"price_eom_{family}"
        panel[f"_lag_{family}"] = panel.groupby("cusip_id")[price_col].shift(1)
        panel[f"_prev_ym_{family}"] = panel.groupby("cusip_id")["_ym_pd"].shift(1)
        panel[f"ret_{family}"] = (panel[price_col] - panel[f"_lag_{family}"]) / panel[f"_lag_{family}"]
        gap = (panel["_ym_pd"] - panel[f"_prev_ym_{family}"]).map(
            lambda x: x.n if pd.notna(x) else float("nan"))
        panel.loc[(gap != 1) | gap.isna(), f"ret_{family}"] = float("nan")
        panel.drop(columns=[f"_lag_{family}", f"_prev_ym_{family}"], inplace=True)
    panel.drop(columns=["_ym_pd"], inplace=True)

    rf = pd.read_parquet(_B.RF_FILE)
    panel = panel.merge(rf, on="year_month", how="left", validate="m:1")
    panel["xret_raw"] = panel["ret_raw"] - panel["rf_monthly"]
    panel["xret_corr"] = panel["ret_corr"] - panel["rf_monthly"]
    panel["cusip"] = panel["cusip_id"]
    panel["date"] = (pd.PeriodIndex(panel["year_month"], freq="M").to_timestamp(how="end").normalize()
                     + pd.offsets.MonthEnd(0))
    panel = _merge_fisd_grid(panel)   # grid-driven FISD so holdout months get as-of ratings (not NaN)
    out_cols = ["cusip", "date", "size", "universe_eligible", "rating", "investment_grade",
                "maturity", "time_to_maturity", "price_eom_raw", "price_eom_corr",
                "ret_raw", "ret_corr", "xret_raw", "xret_corr", "n_trades_raw", "n_trades_corr",
                "total_vol_raw", "total_vol_corr", "last_trade_date_raw", "last_trade_date_corr",
                "rf_monthly", "exit_reason"]
    return panel[out_cols].reset_index(drop=True)


def assemble_signals(maximal: pd.DataFrame, raw_daily_path: Path, corr_daily_path: Path) -> pd.DataFrame:
    """Build the four signals over the maximal panel + daily layer and merge into the exact
    load_dev_inputs signals format (var_5pct/mom6/bond_vol from maximal; gamma from daily; gamma
    renamed to the canonical gamma_illiq_*). Each computation reuses the signal builder verbatim."""
    vcfg = _V.load_config()
    var5 = _V.compute_dual_family(maximal, vcfg["window"], vcfg["min_obs"], vcfg["rank"], vcfg["multiplier"])
    mcfg = _M.load_config()
    mom6 = _M.compute_dual_family(maximal, mcfg["formation_months"], mcfg["min_obs"])
    bcfg = _BV.load_config()
    bond_vol = _BV.compute_dual_family(maximal, bcfg["window"], bcfg["min_obs"])

    gcfg = _G.load_config()
    gk = dict(min_pairs=int(gcfg["min_pairs"]), max_gap_bdays=int(gcfg["max_gap_bdays"]),
              sign_multiplier=float(gcfg["sign_multiplier"]), cov_ddof=int(gcfg["cov_ddof"]),
              strict=gcfg.get("bpw_strict"))
    raw_daily = pd.read_parquet(raw_daily_path, columns=["cusip_id", "trd_exctn_dt", "price_vwap"])
    corr_daily = pd.read_parquet(corr_daily_path, columns=["cusip_id", "trd_exctn_dt", "price_vwap"])
    g_raw = _G.compute_gamma(raw_daily, **gk).rename(columns={"gamma": "gamma_raw"})
    g_corr = _G.compute_gamma(corr_daily, **gk).rename(columns={"gamma": "gamma_corr"})
    gamma = g_raw.merge(g_corr, on=["cusip", "date"], how="outer").rename(columns=_GAMMA_RENAME)

    merged = var5
    for frame in (mom6, gamma, bond_vol):
        merged = merged.merge(frame, on=["cusip", "date"], how="outer")
    return merged


def assemble_profile_monthly(profile_daily_by_pid: dict) -> pd.DataFrame:
    """The per-paper profile monthly panel (reproduces monthly_panel_profiles.parquet): for each pid
    aggregate its dedup-ON daily → monthly with ret/xret via the profile builder's own worker, then
    merge. Columns: cusip, date, {price_eom,ret,xret,n_trades,total_vol,last_trade_date}_<pid>."""
    rf = pd.read_parquet(_B.RF_FILE)
    merged: pd.DataFrame | None = None
    for pid, daily in profile_daily_by_pid.items():
        fam = _PM._monthly_for(daily, pid, rf).drop(columns=["rf_monthly"])
        merged = fam if merged is None else merged.merge(fam, on=["cusip_id", "year_month"], how="outer")
    merged["cusip"] = merged["cusip_id"]
    merged["date"] = _PM._date_col(merged["year_month"])
    fam_cols = [c for c in merged.columns if any(c.startswith(f"{b}_") for b in
                ("price_eom", "ret", "xret", "n_trades", "total_vol", "last_trade_date"))]
    return merged[["cusip", "date", *sorted(fam_cols)]].sort_values(["cusip", "date"]).reset_index(drop=True)


def assemble_profile_signals(profile_monthly: pd.DataFrame, profile_daily_by_pid: dict) -> pd.DataFrame:
    """The per-paper profile signals (reproduces profiles_signals.parquet): var_5pct/bond_vol/mom6
    from the profile monthly ret/xret, gamma_illiq from the profile daily — each reusing the
    production signal builder's compute core. Mirrors build_profile_signals.build()."""
    v5c, bvc, m6c, gic = _V.load_config(), _BV.load_config(), _M.load_config(), _G.load_config()
    merged: pd.DataFrame | None = None
    for pid in profile_daily_by_pid:
        ret_col, xret_col = f"ret_{pid}", f"xret_{pid}"
        base = profile_monthly[["cusip", "date", ret_col, xret_col]]
        v = _V.compute_var_5pct(base[["cusip", "date", ret_col]].rename(columns={ret_col: "ret"}),
                                int(v5c["window"]), int(v5c["min_obs"]), int(v5c["rank"]),
                                float(v5c["multiplier"])).rename(columns={"var_5pct": f"var_5pct_{pid}"})
        bvol = _BV.compute_bond_vol(base[["cusip", "date", xret_col]].rename(columns={xret_col: "ret"}),
                                    int(bvc["window"]), int(bvc["min_obs"])).rename(
                                        columns={"bond_vol": f"bond_vol_{pid}"})
        mm = _M.compute_mom6_signal(base[["cusip", "date", ret_col]].rename(columns={ret_col: "ret"}),
                                    int(m6c["formation_months"]), int(m6c["min_obs"])).rename(
                                        columns={"mom6": f"mom6_{pid}"})
        daily = pd.read_parquet(profile_daily_by_pid[pid],
                                columns=["cusip_id", "trd_exctn_dt", "price_vwap"])
        g = _G.compute_gamma(daily, min_pairs=int(gic["min_pairs"]), max_gap_bdays=int(gic["max_gap_bdays"]),
                             sign_multiplier=float(gic["sign_multiplier"]), cov_ddof=int(gic["cov_ddof"]),
                             strict=gic.get("bpw_strict")).rename(columns={"gamma": f"gamma_illiq_{pid}"})
        pf = v
        for other in (bvol, mm, g):
            pf = pf.merge(other, on=["cusip", "date"], how="outer")
        merged = pf if merged is None else merged.merge(pf, on=["cusip", "date"], how="outer")
    return merged.sort_values(["cusip", "date"]).reset_index(drop=True)


FISD_STATIC = DEV_ROOT / "fisd" / "fisd_reference_static.parquet"
# The recorded DEVELOPMENT total-return artefacts the seeded total-return build must reproduce: the
# default-flat substrate (a defaulted bond trades flat, AI=C=0 from its default month).
DEV_TR_PANEL = DEV_ROOT / "monthly_panel_total_return_default_flat.parquet"
DEV_TR_PROFILES = DEV_ROOT / "monthly_panel_profiles_total_return_default_flat.parquet"
_FISD_TR_COLUMNS = ["cusip", "coupon", "interest_frequency", "day_count_basis", "maturity",
                    "default_date"]


def _fisd_total_return_schedule() -> pd.DataFrame:
    """The FISD coupon schedule + ``default_date`` that both total-return builders consume (the
    columns ``build_total_return_panel.main`` / ``build_profile_total_return_panel.main`` read to
    produce the recorded default-flat panels). ``default_date`` licenses the trade-flat cutoff.
    The static file is a per-cusip reference, not a dated panel: it equals the shared
    ``fisd_reference.build_static`` output row-for-row and carries default dates past the development
    end, so defaults inside the holdout window are covered without reading ``/data/holdout/``."""
    return pd.read_parquet(require_licensed_input(FISD_STATIC, "FISD static table"),
                           columns=_FISD_TR_COLUMNS)


def assemble_total_return_maximal(base_maximal: pd.DataFrame,
                                  profile_monthly: pd.DataFrame | None) -> pd.DataFrame:
    """The TOTAL-RETURN base maximal for the seeded panel: apply the audited accrual transform to
    the seeded (dev+holdout) BASE maximal (raw/corr → total return via
    ``build_total_return_panel.to_total_return``), then LEFT-merge the seeded profile families
    converted by ``build_profile_total_return_panel.build_total_return_profiles`` — exactly the two
    base panels the dev-side total-return audit consumes (monthly_panel_total_return_default_flat +
    monthly_panel_profiles_total_return_default_flat), reconstructed on the seeded grid. Signals are
    unchanged (the clean sort variable); only the return leg becomes total. No re-implemented accrual:
    the same kernels that produced the recorded dev artefacts, so the seeded DEV half reproduces them."""
    import build_profile_total_return_panel as _PTR
    import build_total_return_panel as _TR
    fisd = _fisd_total_return_schedule()
    tr = _TR.to_total_return(base_maximal, fisd)
    if profile_monthly is not None:
        rf = pd.read_parquet(require_licensed_input(_B.RF_FILE, "risk-free rate series"))
        tr_profiles, _diag = _PTR.build_total_return_profiles(profile_monthly, fisd, rf)
        tr = tr.merge(tr_profiles, on=["cusip", "date"], how="left")
    return tr


def build_inventory_from_daily(raw_daily_path: Path, corr_daily_path: Path,
                               profile_daily_by_pid: dict | None = None, *,
                               with_total_return: bool = False):
    """(maximal, signals) from a raw + corr daily layer — the full dev-validated assembly. When
    ``profile_daily_by_pid`` is given, the per-paper profile families are assembled and merged onto
    maximal (LEFT) and signals (OUTER), exactly as load_dev_inputs/load_dev_signals attach the
    recorded monthly_panel_profiles / profiles_signals — so the drf/mom6 as-published cells resolve.

    When ``with_total_return`` is set, ALSO builds the seeded total-return base maximal (from the
    SAME seeded pieces, before the clean profile merge) and returns ``(maximal, signals, maximal_tr)``
    — so the descriptive runner reports the clean and total-return substrates from one holdout open."""
    maximal = assemble_maximal(raw_daily_path, corr_daily_path)
    signals = assemble_signals(maximal, raw_daily_path, corr_daily_path)
    profile_monthly = None
    if profile_daily_by_pid:
        profile_monthly = assemble_profile_monthly(profile_daily_by_pid)
        profile_signals = assemble_profile_signals(profile_monthly, profile_daily_by_pid)
    maximal_tr = assemble_total_return_maximal(maximal, profile_monthly) if with_total_return else None
    if profile_daily_by_pid:
        maximal = maximal.merge(profile_monthly, on=["cusip", "date"], how="left")
        signals = signals.merge(profile_signals, on=["cusip", "date"], how="outer")
    if with_total_return:
        return maximal, signals, maximal_tr
    return maximal, signals


# --------------------------------------------------------------------------------------------
# Seeded concat + the GATED holdout entry
# --------------------------------------------------------------------------------------------

def _assert_daily_side(df: pd.DataFrame, what: str, *, before: bool) -> None:
    """Seam-integrity guard: the dev daily must end strictly BEFORE the holdout floor and the
    holdout daily must start AT/AFTER it. A mis-split (a boundary month present on both sides)
    would let aggregate_family sum both sides' trades into one VWAP, silently corrupting
    price_eom/ret at the seam. Uses HOLDOUT_FLOOR so a split error fails loud, never silently."""
    if "trd_exctn_dt" not in df.columns:
        raise ValueError(f"{what}: missing 'trd_exctn_dt' for the seam integrity check")
    dt = pd.to_datetime(df["trd_exctn_dt"])
    if before and pd.Timestamp(dt.max()) >= HOLDOUT_FLOOR:
        raise RuntimeError(f"{what}: latest trade {dt.max()} is at/after the holdout floor "
                           f"{HOLDOUT_FLOOR.date()} — dev/holdout mis-split, would corrupt the seam")
    if not before and pd.Timestamp(dt.min()) < HOLDOUT_FLOOR:
        raise RuntimeError(f"{what}: earliest trade {dt.min()} is before the holdout floor "
                           f"{HOLDOUT_FLOOR.date()} — dev/holdout mis-split, would corrupt the seam")


def _concat_daily(dev_daily_path: Path, holdout_daily_path: Path, dest: Path) -> Path:
    """Concatenate the full dev daily layer with the holdout daily layer (row union) and write to
    ``dest``. Seeding = all of development, so every rolling window is warm at 2022-01. The seam is
    asserted first: dev strictly < 2022-01, holdout >= 2022-01 (no boundary-month double-count)."""
    dev = pd.read_parquet(dev_daily_path)
    hold = pd.read_parquet(holdout_daily_path)
    _assert_daily_side(dev, f"dev daily ({dev_daily_path.name})", before=True)
    _assert_daily_side(hold, f"holdout daily ({holdout_daily_path.name})", before=False)
    out = pd.concat([dev, hold], ignore_index=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest)
    return dest


def _require_gate(gate: str | None) -> None:
    if gate != _GATE_TOKEN or os.environ.get(_GATE_ENV) != "1":
        raise HoldoutGateError(
            "Holdout inventory build REFUSED: this reads /data/holdout/ and is the single "
            f"confirmed open. It requires gate={_GATE_TOKEN!r} AND env "
            f"{_GATE_ENV}=1 — both are unset. Build-before-open stops here; the open is a "
            "separate, explicit action.")


def load_holdout_inputs(*, holdout_daily_dir: Path, gate: str | None = None,
                        tmp_dir: Path | None = None, registry: dict | None = None,
                        with_total_return: bool = False):
    """GATED. Build the seeded (maximal, signals, registry) inventory over dev+holdout. REFUSES
    unless the explicit gate is set. When open: reads the holdout daily layer
    (`trace_daily_raw` + `trace_daily_corr_filtered`, produced by the gated raw-cleaning
    prerequisite), concatenates with the full dev daily, and runs the dev-validated assembly.

    When ``with_total_return`` is set, returns ``(maximal, maximal_tr, signals, registry)`` — the
    total-return base maximal is derived in-memory from the SAME seeded inventory (no second holdout
    read), so the descriptive runner emits both substrates from the single gated open."""
    _require_gate(gate)   # closed by default — nothing past here runs without the gated open
    import shutil

    from agents.auditor.ipca_differential.runner import load_registry

    hold_dir = Path(holdout_daily_dir)
    # Holdout-scoped, clearly-labelled temp dir (gitignored under results/, never committed),
    # removed in the finally below. A hard process kill mid-run can leave the seeded parquet
    # behind; clear it manually if a --real run is terminated abnormally.
    tmp = Path(tmp_dir) if tmp_dir else (
        _REPO_ROOT / "results" / "scientist" / "holdout_oos_real" / "_seeded_tmp")
    try:
        raw_cat = _concat_daily(DEV_ROOT / RAW_DAILY_NAME, hold_dir / RAW_DAILY_NAME,
                                tmp / "seeded_daily_raw.parquet")
        corr_cat = _concat_daily(DEV_ROOT / CORR_DAILY_NAME, hold_dir / CORR_DAILY_NAME,
                                 tmp / "seeded_daily_corr.parquet")
        # Per-paper profile daily (drf/mom6 as-published), seeded the same way.
        profile_cat = {
            pid: _concat_daily(DEV_ROOT / name, hold_dir / name, tmp / f"seeded_{name}")
            for pid, name in _PROFILE_DAILY_NAMES.items()
        }
        inv = build_inventory_from_daily(raw_cat, corr_cat, profile_daily_by_pid=profile_cat,
                                         with_total_return=with_total_return)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    reg = registry if registry is not None else load_registry()
    if with_total_return:
        maximal, signals, maximal_tr = inv
        return maximal, maximal_tr, signals, reg
    maximal, signals = inv
    return maximal, signals, reg


# --------------------------------------------------------------------------------------------
# Dev self-check (validation; no holdout read)
# --------------------------------------------------------------------------------------------

def _max_abs_diff(rebuilt: pd.DataFrame, committed: pd.DataFrame, cols: list[str]) -> dict:
    r = rebuilt.set_index(["cusip", "date"]).sort_index()
    c = committed.set_index(["cusip", "date"]).sort_index()
    common = r.index.intersection(c.index)
    out = {"index_equal": bool(r.index.equals(c.index)),
           "rebuilt_only": int(len(r.index.difference(c.index))),
           "committed_only": int(len(c.index.difference(r.index)))}
    r, c = r.loc[common], c.loc[common]
    for col in cols:
        # na_value=np.nan so a nullable column (e.g. investment_grade, a pandas boolean with pd.NA)
        # casts cleanly instead of raising; plain float/bool columns are unaffected.
        a = r[col].to_numpy(dtype=float, na_value=np.nan)
        b = c[col].to_numpy(dtype=float, na_value=np.nan)
        both = np.isfinite(a) & np.isfinite(b)   # exclude inf too, so inf-inf never warns/hides a diff
        out[col] = {"max_abs": float(np.nanmax(np.abs(a[both] - b[both]))) if both.any() else 0.0,
                    "nan_mismatch": int((np.isnan(a) ^ np.isnan(b)).sum())}
    return out


def _dev_profile_daily() -> dict:
    return {pid: DEV_ROOT / name for pid, name in _PROFILE_DAILY_NAMES.items()}


def selfcheck_on_dev(*, atol: float = 1e-9) -> dict:
    """Rebuild the recorded dev maximal, the four base signals, AND the per-paper profile monthly
    panel + profile signals from the dev daily layers via this module's assembly, and assert
    reproduction within ``atol`` (max abs diff per checked column). Dev-only; proves every layer the holdout build reuses (incl. the drf/mom6
    as-published profile families) is faithful. Raises on any divergence."""
    raw_daily, corr_daily = DEV_ROOT / RAW_DAILY_NAME, DEV_ROOT / CORR_DAILY_NAME
    profile_daily = _dev_profile_daily()
    maximal = assemble_maximal(raw_daily, corr_daily)
    signals = assemble_signals(maximal, raw_daily, corr_daily)
    profile_monthly = assemble_profile_monthly(profile_daily)
    profile_signals = assemble_profile_signals(profile_monthly, profile_daily)

    sig_dir = DEV_ROOT / "signals"
    recorded_max = pd.read_parquet(DEV_ROOT / "monthly_panel_maximal.parquet")
    checks = {}
    # price/ret/xret + the FISD columns the anchors depend on: size (VW weight, all anchors),
    # rating (drf/lrf bivariate control), universe_eligible (view filter). bool casts to 0/1.
    checks["maximal"] = _max_abs_diff(
        maximal, recorded_max,
        ["price_eom_raw", "price_eom_corr", "ret_raw", "ret_corr", "xret_raw", "xret_corr",
         "size", "rating", "investment_grade", "universe_eligible"])
    # exit_reason is object (matured/defaulted/defeased/NA — survivorship filter): string equality.
    _rex = maximal.set_index(["cusip", "date"]).sort_index()["exit_reason"].fillna("__NA__")
    _cex = recorded_max.set_index(["cusip", "date"]).sort_index()["exit_reason"].fillna("__NA__")
    checks["exit_reason_equal"] = bool(_rex.index.equals(_cex.index)
                                       and (_rex.to_numpy() == _cex.to_numpy()).all())
    for name, cols in (("var_5pct", ["var_5pct_raw", "var_5pct_corr"]),
                       ("mom6", ["mom6_raw", "mom6_corr"]),
                       ("bond_vol", ["bond_vol_raw", "bond_vol_corr"])):
        checks[name] = _max_abs_diff(signals, pd.read_parquet(sig_dir / f"{name}.parquet"), cols)
    gamma_recorded = pd.read_parquet(sig_dir / "gamma_illiq.parquet").rename(columns=_GAMMA_RENAME)
    checks["gamma_illiq"] = _max_abs_diff(signals, gamma_recorded,
                                          ["gamma_illiq_raw", "gamma_illiq_corr"])
    # Per-paper profile families (the drf/mom6 as-published layer).
    checks["profile_monthly"] = _max_abs_diff(
        profile_monthly, pd.read_parquet(DEV_ROOT / "monthly_panel_profiles.parquet"),
        [f"{b}_{pid}" for b in ("price_eom", "ret", "xret") for pid in PROFILE_IDS])
    checks["profile_signals"] = _max_abs_diff(
        profile_signals, pd.read_parquet(sig_dir / "profiles_signals.parquet"),
        [f"{s}_{pid}" for s in ("var_5pct", "mom6", "bond_vol", "gamma_illiq") for pid in PROFILE_IDS])
    # Total-return companion: validate the two building blocks assemble_total_return_maximal
    # composes, each against its recorded dev artefact on its OWN grid (strict, apples-to-apples,
    # exactly as the clean maximal + standalone profile_monthly are checked above). The LEFT-merge
    # that composes them onto the maximal grid is the SAME pattern validated for clean above.
    import build_profile_total_return_panel as _PTR
    import build_total_return_panel as _TR
    # Default-flat substrate: the schedule carries default_date, and the targets are
    # the recorded *_default_flat dev panels — the same artefacts the total-return audit consumes.
    _fisd_sched = _fisd_total_return_schedule()
    tr_base = _TR.to_total_return(maximal, _fisd_sched)
    tr_profiles, _ = _PTR.build_total_return_profiles(
        profile_monthly, _fisd_sched,
        pd.read_parquet(require_licensed_input(_B.RF_FILE, "risk-free rate series")))
    checks["total_return_maximal"] = _max_abs_diff(
        tr_base, pd.read_parquet(require_licensed_input(DEV_TR_PANEL, "total-return development panel")),
        ["ret_raw", "ret_corr", "xret_raw", "xret_corr"])
    checks["total_return_profiles"] = _max_abs_diff(
        tr_profiles, pd.read_parquet(require_licensed_input(DEV_TR_PROFILES, "total-return profile panel")),
        [f"{b}_{pid}" for b in ("ret", "xret") for pid in PROFILE_IDS])

    def _passes(res: dict, *, strict_index: bool) -> bool:
        # Every recorded key must be reproduced with exact values. Strict-index artefacts (maximal,
        # the standalone profile frames) must match the recorded index exactly; the merged base
        # signals frame is legitimately a superset (outer merge over signals with different key
        # coverage — sparse gamma), so rebuilt-only keys are allowed but never a missing recorded key.
        if res["committed_only"] > 0:
            return False
        if strict_index and (not res["index_equal"] or res["rebuilt_only"] > 0):
            return False
        return all(not (isinstance(v, dict) and (v["max_abs"] > atol or v["nan_mismatch"] > 0))
                   for v in res.values())

    exact = (_passes(checks["maximal"], strict_index=True)
             and checks["exit_reason_equal"]
             and _passes(checks["profile_monthly"], strict_index=True)
             and _passes(checks["profile_signals"], strict_index=True)
             and _passes(checks["total_return_maximal"], strict_index=True)
             and _passes(checks["total_return_profiles"], strict_index=True)
             and all(_passes(checks[n], strict_index=False)
                     for n in ("var_5pct", "mom6", "bond_vol", "gamma_illiq")))
    if not exact:
        raise RuntimeError(f"holdout_inventory dev self-check DIVERGED: {checks}")
    return {"exact": True, "atol": atol, "checks": checks}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seeded holdout inventory builder (build-before-open)")
    parser.add_argument("--selfcheck", action="store_true",
                        help="rebuild + verify the dev maximal/signals from the dev daily layer "
                             "(dev-only; the holdout is never read)")
    args = parser.parse_args(argv)
    if args.selfcheck:
        res = selfcheck_on_dev()
        print("holdout_inventory dev self-check: EXACT reproduction of maximal (incl FISD size/"
              f"rating/universe/exit_reason) + 4 signals + profile families (atol {res['atol']:.0e})")
        for name, chk in res["checks"].items():
            if not isinstance(chk, dict):   # scalar checks (e.g. exit_reason_equal: bool)
                print(f"  {name:16s} = {chk}")
                continue
            worst = max((v["max_abs"] for v in chk.values() if isinstance(v, dict)), default=0.0)
            print(f"  {name:16s} index_equal={chk['index_equal']} worst_max|Δ| {worst:.1e}")
        return 0
    print(
        "holdout_inventory real build: REFUSED. Building the seeded inventory reads "
        "data/holdout/ and is the single confirmed open. Use --selfcheck for the dev-only "
        "validation; the open is a separate explicit action.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
