"""Concrete PanelBuilders for one-shot holdout stage 1 (spec §2) — dev-pseudo (runnable) + holdout (gated).

A ``PanelBuilder`` (see ``stage1_build.PanelBuilder``) is ``(*, seed_start, window)
-> {name: DataFrame}`` returning the one-shot holdout inventory as in-memory frames. Two concrete builders:

  * ``dev_pseudo_builder`` — reads the EXISTING development artefacts (monthly panels, signals,
    factors) and slices them to ``[seed_start .. window.end]``; the recursive-OOS IPCA factors and
    the FISD ratings are (re)constructed over that slice via the frozen library workers. The
    development-pseudo-window (2018-01..2021-09) is fully covered by the stored dev artefacts, so
    this is runnable in the ``--rehearsal`` (development data only; the gate never opens).

  * ``holdout_builder`` — the real run's builder. The holdout months (2022-01..) do not exist
    in any stored artefact, so it must clean the holdout raw and RE-BUILD every rolling construction
    over the seeded development+holdout span. That path reads ``data/holdout/``; it runs only behind
    the single-access gate (the factory refuses otherwise) and never fabricates a holdout read.

FIREWALL: this module imports only ``agents.quant`` / the ``scripts`` builders / stored parquets — never
``agents.auditor`` — so the one-shot holdout no-lattice source scan still passes. Heavy library imports
are lazy (inside functions) so importing this module is cheap and cannot fail at import time.

VALIDATION: the ``--rehearsal`` run validates the dev-pseudo builder end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .windows import Window

REPO_ROOT = Path(__file__).resolve().parents[4]
DEV_ROOT = REPO_ROOT / "data" / "development"

# The one-shot holdout inventory the holdout evaluation consumes downstream. Names are stable keys in the
# returned mapping and in the stage-1 quarantine + manifest.
INVENTORY = (
    "monthly_panel_corrected",
    "monthly_panel_uncorrected",
    "signals_var_5pct",
    "signals_gamma_illiq",
    "signals_mom6",
    "signals_bond_vol",
    "factors_mktb",
    "factors_bbw",          # drf / crf / lrf (+ mktb) composite factor frame
    "factors_str",
    "factors_mom6",
    "ipca_oos_factors",     # recursive-OOS IPCA factors (re-computed; not a stored artefact)
    "fisd_ratings",         # as-of monthly ratings over the window grid (+ leakage marker)
)


class OneshotHoldoutBuilderGated(RuntimeError):
    """The holdout builder was invoked outside the gated real run. It reads data/holdout and
    re-builds every construction; it never runs in development and never fabricates a holdout read."""


# --- FISD zero-leakage callback (mirrors build_fisd_reference.main's inline predicate) ------------

def zero_leakage_check(fisd_ratings: pd.DataFrame, window: Window) -> None:
    """Assert the as-of ratings carry no future-dated selection (the backward merge_asof is sound).

    Mirrors build_fisd_reference.main's inline check: any row whose selected rating event date is
    LATER than the row's own month is leakage. Fail-loud — a look-ahead rating on the holdout would
    silently leak the future into the evaluation. Threaded into one-shot holdout via OneshotHoldoutConfig.zero_leakage_check.
    """
    if "_sel_rating_date" not in fisd_ratings.columns or "date" not in fisd_ratings.columns:
        raise ValueError("fisd_ratings must carry '_sel_rating_date' and 'date' for the leakage check")
    sel = pd.to_datetime(fisd_ratings["_sel_rating_date"])
    dt = pd.to_datetime(fisd_ratings["date"])
    leakage = int((sel.notna() & (sel > dt)).sum())
    if leakage != 0:
        raise AssertionError(
            f"FISD as-of leakage on the holdout window {window.start}..{window.end}: "
            f"{leakage} rows carry a rating event dated after their own month"
        )


# --- helpers -------------------------------------------------------------------------------------

def _slice_months(df: pd.DataFrame, start: str, end: str, date_col: str = "date") -> pd.DataFrame:
    """Slice a frame with a monthly date column to the inclusive [start, end] month range."""
    if date_col not in df.columns:
        raise ValueError(f"expected a '{date_col}' column to slice; got {list(df.columns)[:6]}…")
    idx = pd.to_datetime(df[date_col]).dt.to_period("M")
    lo, hi = pd.Period(start, "M"), pd.Period(end, "M")
    return df[(idx >= lo) & (idx <= hi)].reset_index(drop=True)


def _load_sliced(path: Path, start: str, end: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"expected development artefact not found: {path}")
    return _slice_months(pd.read_parquet(path), start, end)


# --- dev-pseudo builder (runnable in --rehearsal; development data only) --------------------------

def dev_pseudo_builder(dev_root: Path | None = None):
    """Return a PanelBuilder that assembles the one-shot holdout inventory from the STORED development artefacts,
    sliced to [seed_start .. window.end]. Used by ``--rehearsal`` on the dev pseudo-window; every
    read is under data/development. The gate never opens for this builder."""
    root = Path(dev_root) if dev_root is not None else DEV_ROOT

    def _build(*, seed_start: str, window: Window):
        lo, hi = seed_start, window.end
        inv: dict[str, pd.DataFrame] = {}

        # Monthly panels + signals + factors: the pseudo-window is covered by the stored artefacts,
        # so a slice yields series already warm from the first evaluation month (seeding by
        # construction). The parquets are keyed cusip+date (panels/signals) or date (factors).
        inv["monthly_panel_corrected"] = _load_sliced(root / "monthly_panel_corrected.parquet", lo, hi)
        inv["monthly_panel_uncorrected"] = _load_sliced(root / "monthly_panel_uncorrected.parquet", lo, hi)
        inv["signals_var_5pct"] = _load_sliced(root / "signals" / "var_5pct.parquet", lo, hi)
        inv["signals_gamma_illiq"] = _load_sliced(root / "signals" / "gamma_illiq.parquet", lo, hi)
        inv["signals_mom6"] = _load_sliced(root / "signals" / "mom6.parquet", lo, hi)
        inv["signals_bond_vol"] = _load_sliced(root / "signals" / "bond_vol.parquet", lo, hi)
        inv["factors_mktb"] = _load_sliced(root / "factors" / "mktb.parquet", lo, hi)
        inv["factors_bbw"] = _load_sliced(root / "factors" / "bbw_factors.parquet", lo, hi)
        inv["factors_str"] = _load_sliced(root / "factors" / "str.parquet", lo, hi)
        inv["factors_mom6"] = _load_sliced(root / "factors" / "mom6.parquet", lo, hi)

        # IPCA recursive-OOS factors are NOT stored — reconstruct them from the (sliced) feed via the
        # frozen estimator. The rehearsal checks shapes/burn-in numerically.
        inv["ipca_oos_factors"] = _ipca_oos_from_feed(root / "ipca_panel_corr.parquet", lo, hi)

        # FISD as-of ratings over the window grid (backward merge_asof; leakage marker retained).
        inv["fisd_ratings"] = _fisd_ratings_over_window(root, window)
        return inv

    return _build


def _ipca_oos_from_feed(feed_path: Path, start: str, end: str) -> pd.DataFrame:
    """Recursive-OOS IPCA factors over [start, end] via the CANONICAL chain the shakedown itself
    uses: ``load_feed`` (the adapter) → ``build_sufficient_stats`` → ``fit_ipca_recursive``. ``K``
    and the OOS burn-in are read from the frozen IPCA config (``model.K`` / ``window.
    oos_burn_in_months``), never hard-coded — one source of truth."""
    if not feed_path.exists():
        raise FileNotFoundError(f"IPCA feed not found: {feed_path}")
    # Lazy imports: heavy, and keep this module import-cheap + firewall-clean (no agents.auditor).
    import numpy as np
    import yaml

    from agents.quant.library.ipca import build_sufficient_stats, fit_ipca_recursive
    from agents.quant.library.ipca_feed import load_feed

    ipca_cfg = yaml.safe_load((REPO_ROOT / "agents" / "quant" / "library" / "configs" / "kpp_ipca.yaml").read_text())
    k = int(ipca_cfg["model"]["K"])
    burn_in = int(ipca_cfg["window"]["oos_burn_in_months"])

    z, r, months, _asof, _vol = load_feed(feed_path)        # the shakedown's own loader (single source)
    stats = build_sufficient_stats(z, r, months)
    rec = fit_ipca_recursive(stats, K=k, burn_in=burn_in)
    # RecursiveResult: rec.oos_months (J,) are the predicted-month ordinals (months[burn_in:]) and
    # rec.f_oos is (K, J). Align f_oos.T to those months — not to the full feed months.
    oos_months = pd.PeriodIndex(
        [pd.Period(ordinal=int(m), freq="M") for m in rec.oos_months], freq="M",
    )
    frame = pd.DataFrame(np.asarray(rec.f_oos).T, index=oos_months.to_timestamp("M"))
    frame.columns = [f"ipca_f{i + 1}" for i in range(frame.shape[1])]
    frame = frame.reset_index().rename(columns={"index": "date"})
    return _slice_months(frame, start, end)


def _fisd_ratings_over_window(dev_root: Path, window: Window, grid: pd.DataFrame | None = None) -> pd.DataFrame:
    """As-of monthly FISD ratings over the window grid, BUILT FRESH via the frozen backward
    merge_asof worker (``build_ratings_monthly``) so ``_sel_rating_date`` — the leakage marker — is
    REAL. The stored ``fisd_ratings_monthly.parquet`` does not persist that marker, so slicing it and
    injecting ``NaT`` would make ``zero_leakage_check`` vacuous (a guard that validates nothing is
    worse than none). Fail-loud if the worker does not emit the marker."""
    from agents.quant.library.fisd_reference import build_ratings_monthly, build_static, load_config

    cfg = load_config()
    static = build_static(cfg)
    issue_to_cusip = static.dropna(subset=["issue_id"]).set_index("issue_id")["cusip"]
    issue_to_cusip = issue_to_cusip[~issue_to_cusip.index.duplicated(keep="first")]

    if grid is None:                                   # dev pseudo-window: the stored development grid
        grid = pd.read_parquet(dev_root / "monthly_panel_maximal.parquet", columns=["cusip", "date"])
        grid["cusip"] = grid["cusip"].astype("string")
        grid = _slice_months(grid, window.start, window.end)

    ratings = build_ratings_monthly(cfg, grid, issue_to_cusip)     # carries a real _sel_rating_date
    if "_sel_rating_date" not in ratings.columns:
        raise ValueError(
            "build_ratings_monthly did not emit '_sel_rating_date'; the FISD leakage guard cannot run"
        )
    return ratings


# --- holdout builder (GATED — the real run only; never runs in development) ------------------

_CORR_VIEW_COLS = ["cusip", "date", "size", "rf_monthly", "rating", "investment_grade", "maturity",
                   "time_to_maturity", "price_eom", "ret", "xret", "n_trades", "total_vol", "last_trade_date",
                   "var_5pct"]
_SIGNAL_FILES = {"signals_var_5pct": ("var_5pct_raw", "var_5pct_corr"), "signals_mom6": ("mom6_raw", "mom6_corr"),
                 "signals_bond_vol": ("bond_vol_raw", "bond_vol_corr"), "signals_gamma_illiq": ("gamma_raw", "gamma_corr")}
_GAMMA_BACK = {"gamma_illiq_raw": "gamma_raw", "gamma_illiq_corr": "gamma_corr"}


def _month_end_index(year_month) -> pd.DatetimeIndex:
    return pd.PeriodIndex(year_month, freq="M").to_timestamp(how="end").normalize()


def fetch_baa_aaa_spread_full() -> pd.Series:
    """The BAA-AAA spread (percentage points) over the FULL FRED history, month-end indexed — the same
    two series and arithmetic as ``scripts/download_baa_aaa_spread.py`` without its development cap."""
    from scripts.download_baa_aaa_spread import _monthly_series, load_config

    cfg = load_config()
    baa = _monthly_series(cfg["baa_yield_url"], "baa")
    aaa = _monthly_series(cfg["aaa_yield_url"], "aaa")
    joined = pd.concat([baa, aaa], axis=1).dropna()
    s = (joined["baa"] - joined["aaa"]).rename("baa_aaa_spread")
    s.index = _month_end_index(s.index)
    return s.sort_index()


def seeded_macro_series(dev_root: Path, holdout_end: str, fetch=fetch_baa_aaa_spread_full) -> tuple[pd.Series, dict]:
    """Development months come from the STORED development artefact (the exact inputs the development run
    used); holdout-era months come from the fetched series. Any development-month difference between the two
    is recorded, never silently adopted."""
    from shared.licensed_inputs import require_licensed_input

    dev = pd.read_parquet(require_licensed_input(dev_root / "baa_aaa_spread.parquet",
                                                 "macro conditioning series"))
    val = next(c for c in dev.columns if c != "year_month")
    dev_s = pd.Series(dev[val].to_numpy(dtype=float), index=_month_end_index(dev["year_month"]),
                      name="baa_aaa_spread").sort_index()
    full = fetch()
    hi = pd.Period(holdout_end, "M").to_timestamp(how="end").normalize()
    hold_s = full[(full.index > dev_s.index.max()) & (full.index <= hi)]
    common = full.reindex(dev_s.index).dropna()
    diff = (common - dev_s.reindex(common.index)).abs()
    record = {"dev_months": int(len(dev_s)), "dev_last": str(dev_s.index.max().date()), "holdout_months": int(len(hold_s)),
              "holdout_last": None if hold_s.empty else str(hold_s.index.max().date()),
              "fetched_vs_stored_dev_max_abs_diff": float(diff.max()) if len(diff) else None,
              "fetched_vs_stored_dev_n_diff_gt_1e-9": int((diff > 1e-9).sum()) if len(diff) else None}
    return pd.concat([dev_s, hold_s]).sort_index(), record


def validate_seeded_feed(feed: pd.DataFrame, reg: dict, family: str, *, last_month: str) -> None:
    """The frozen feed validation (``ipca_feed.validate_feed``: the receipt check on every month's Z/R/asof)
    with the §7 wall relocated to ``last_month`` — the registered evaluation window's end. The registry's
    ``meta.train_end`` wall exists so that no development-time feed can carry a post-development month; the
    seeded holdout feed carries them BY DESIGN, behind the single-access gate, and must still refuse anything
    past the registered window. Raises ``ipca.ContractViolation`` on any violation."""
    import numpy as np
    from agents.quant.library.ipca import assert_within_wall, validate_panel
    from agents.quant.library.ipca_feed import INSTRUMENTS, L

    z_cols = [f"z_{c}" for c in INSTRUMENTS]
    Z, R, months, asof = [], [], [], []
    for month, grp in feed.groupby("month", sort=True):
        Z.append(np.column_stack([grp[z_cols].to_numpy(dtype=float), np.ones(len(grp))]))
        R.append(grp["R"].to_numpy(dtype=float))
        months.append(int(month))
        asof.append(int(grp["asof"].iloc[0]))
    validate_panel(Z, R, np.asarray(months), L=L, family=family, characteristic_asof=np.asarray(asof))
    assert_within_wall(np.asarray(months), train_end=pd.Period(last_month, "M").ordinal)


def build_seeded_inventory(maximal: pd.DataFrame, signals: pd.DataFrame, *, dev_root: Path, seed_start: str,
                           window: Window, feed_path: Path, macro_fetch=None, holder: dict | None = None,
                           macro_series: pd.Series | None = None) -> dict:
    """The one-shot holdout inventory from an in-memory SEEDED ``(maximal, signals)`` pair in the ``load_dev_inputs``
    format (profile-family columns included) — pure construction, no holdout read, so it runs identically on the
    development frames. The same workers as development: endpoint views, the four signal
    files, the factor builders' ``run_family`` / ``compute_dual_family``, the frozen IPCA feed → recursive-OOS chain,
    the as-of FISD ratings over the seeded grid, and the seeded BAA-AAA regime series. Frames are sliced to
    ``[seed_start .. window.end]``; ``holder`` (if given) receives the FULL seeded frames for the stage-2 derivation."""
    from functools import reduce

    import scripts.build_bbw_factors as BB
    import scripts.build_mktb as MK
    import scripts.build_mom6 as M6
    import scripts.build_str as ST
    from agents.quant.library.ipca_feed import build_ipca_feed
    from agents.quant.library.run_config import corrected, uncorrected
    from agents.quant.library.views import view
    from scripts.build_ipca_panel import load_registry

    lo, hi = seed_start, window.end
    inv: dict[str, pd.DataFrame] = {}
    base = signals.rename(columns=_GAMMA_BACK)                  # the dev parquet / builder column names
    var5 = base[["cusip", "date", "var_5pct_raw", "var_5pct_corr"]]
    bbw_signals = base[["cusip", "date", "var_5pct_raw", "var_5pct_corr", "gamma_raw", "gamma_corr"]]
    mom6_sig = base[["cusip", "date", "mom6_raw", "mom6_corr"]]

    for name, cfg in (("monthly_panel_corrected", corrected()), ("monthly_panel_uncorrected", uncorrected())):
        v = view(maximal, cfg, signals=var5)
        inv[name] = _slice_months(v[[c for c in _CORR_VIEW_COLS if c in v.columns]], lo, hi)
    for name, cols in _SIGNAL_FILES.items():
        inv[name] = _slice_months(base[["cusip", "date", *cols]], lo, hi)

    frames = []
    for fam in ("raw", "corr"):
        monthly, _summ = BB.run_family(maximal, bbw_signals, fam)
        frames.extend(monthly[n] for n in BB.FACTORS + ["crf"])
    bbw_full = reduce(lambda a, b: a.merge(b, on="date", how="outer"), frames).sort_values("date").reset_index(drop=True)
    mktb_full = MK.compute_dual_family(maximal)
    inv["factors_bbw"] = _slice_months(bbw_full, lo, hi)
    inv["factors_mktb"] = _slice_months(mktb_full, lo, hi)
    st = {fam: ST.run_family(maximal, fam)["monthly"] for fam in ("raw", "corr")}
    inv["factors_str"] = _slice_months(st["raw"].merge(st["corr"], on="date", how="outer").sort_values("date")
                                       .reset_index(drop=True), lo, hi)
    m6cfg = M6.load_config()
    m6 = {fam: M6.run_family(maximal, mom6_sig, fam, m6cfg)["monthly"] for fam in ("raw", "corr")}
    inv["factors_mom6"] = _slice_months(m6["raw"].merge(m6["corr"], on="date", how="outer").sort_values("date")
                                        .reset_index(drop=True), lo, hi)

    reg = load_registry()
    xret, vol_col = reg["return_column"]["corr"], reg["scaler"]["column"]["corr"]
    merged = (maximal[["cusip", "date", xret, "rating", "time_to_maturity"]]
              .merge(base[["cusip", "date", "mom6_corr", "var_5pct_corr", "gamma_corr", vol_col]],
                     on=["cusip", "date"], how="left")
              .rename(columns={xret: "str_reversal", "mom6_corr": "mom6", "var_5pct_corr": "var_5pct",
                               "gamma_corr": "gamma_illiq", vol_col: "bond_vol"}))
    feed, _counts = build_ipca_feed(merged, reg, "corr", validate=False)
    validate_seeded_feed(feed, reg, "corr", last_month=window.end)
    Path(feed_path).parent.mkdir(parents=True, exist_ok=True)
    feed.to_parquet(feed_path)
    inv["ipca_oos_factors"] = _ipca_oos_from_feed(Path(feed_path), lo, hi)

    grid = _slice_months(maximal[["cusip", "date"]].assign(cusip=lambda d: d["cusip"].astype("string")),
                         window.start, window.end)
    inv["fisd_ratings"] = _fisd_ratings_over_window(dev_root, window, grid=grid)

    if macro_series is not None:                     # an injected regime series (instead of the fetched one)
        macro, macro_record = macro_series.sort_index(), {"injected": True, "months": int(len(macro_series))}
    else:
        macro, macro_record = seeded_macro_series(dev_root, window.end, fetch=macro_fetch or fetch_baa_aaa_spread_full)
    inv["macro_baa_aaa_spread"] = _slice_months(macro.rename("baa_aaa_spread").rename_axis("date").reset_index(), lo, hi)
    if holder is not None:
        holder.update(maximal=maximal, signals=signals, macros={"baa_aaa_spread": macro},
                      full_factors_bbw=bbw_full, full_factors_mktb=mktb_full, macro_record=macro_record)
    return inv


def holdout_builder(dev_root: Path | None = None, *, gate: str | None = None, holdout_daily_dir: Path | None = None,
                    macro_fetch=None, holder: dict | None = None):
    """Return the REAL PanelBuilder. GATED: the factory refuses (``OneshotHoldoutBuilderGated``) unless the holdout
    inventory's gate (token argument + env) is set, and the builder re-checks it before reading.

    Recipe (the spec's §8, on top of the dev-validated seeded inventory shared with the descriptive holdout
    runner): ``scripts.holdout_inventory.load_holdout_inputs`` concatenates the FULL development daily layer with
    the holdout daily layer and runs the dev-validated monthly assembly + signal builders, so every rolling
    construction is warm at the first holdout month (seeding by construction); ``selfcheck_on_dev`` must
    reproduce the stored development artefacts exactly first (fail-closed). Every further construction is
    ``build_seeded_inventory`` — which runs identically on development frames. One holdout read serves
    both stages: ``holder`` receives the full seeded frames for ``scripts/oneshot_holdout_survivor_inputs``."""
    root = Path(dev_root) if dev_root is not None else DEV_ROOT
    from scripts import holdout_inventory as HI   # the dev-validated seeded inventory (scripts namespace)

    try:
        HI._require_gate(gate)
    except HI.HoldoutGateError as exc:
        raise OneshotHoldoutBuilderGated(str(exc)) from exc
    hold_dir = Path(holdout_daily_dir) if holdout_daily_dir is not None else REPO_ROOT / "data" / "holdout"

    def _build(*, seed_start: str, window: Window):
        HI._require_gate(gate)                                   # defence in depth: re-check at read time
        HI.selfcheck_on_dev()                                    # fail-closed dev reproduction before the open
        maximal, signals, _reg = HI.load_holdout_inputs(holdout_daily_dir=hold_dir, gate=gate)
        feed_path = (holder or {}).get("feed_path") or (REPO_ROOT / "results" / "scientist" / "oneshot_quarantine"
                                                          / "_seeded_ipca_feed_corr.parquet")
        return build_seeded_inventory(maximal, signals, dev_root=root, seed_start=seed_start, window=window,
                                      feed_path=feed_path, macro_fetch=macro_fetch, holder=holder)

    return _build


BUILDERS = {
    "dev_pseudo": dev_pseudo_builder,
    "holdout": holdout_builder,
}
