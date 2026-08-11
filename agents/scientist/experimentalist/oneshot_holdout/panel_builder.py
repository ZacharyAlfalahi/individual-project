"""Concrete PanelBuilders for one-shot holdout stage 1 (spec §2) — dev-pseudo (runnable) + holdout (gated).

A ``PanelBuilder`` (see ``stage1_build.PanelBuilder``) is ``(*, seed_start, window, sub_window)
-> {name: DataFrame}`` returning the one-shot holdout inventory as in-memory frames. Two concrete builders:

  * ``dev_pseudo_builder`` — reads the EXISTING development artefacts (monthly panels, signals,
    factors) and slices them to ``[seed_start .. window.end]``; the recursive-OOS IPCA factors and
    the FISD ratings are (re)constructed over that slice via the frozen library workers. The
    development-pseudo-window (2018-01..2021-09) is fully covered by the stored dev artefacts, so
    this is runnable in the closeout ``--rehearsal`` (development data only; the gate never opens).

  * ``holdout_builder`` — the REAL one-shot's builder. The holdout months (2022-01..) do not exist
    in any stored artefact, so it must clean the holdout raw and RE-BUILD every rolling construction
    over the seeded development+holdout span. That path reads ``data/holdout/`` and is executed only
    behind the single-access gate; it is specified precisely below, and it REFUSES to run
    in development (it never fabricates a holdout read).

FIREWALL: this module imports only ``agents.quant`` / stored parquets — never ``agents.auditor`` —
so the one-shot holdout no-lattice source scan still passes. Heavy library imports are lazy (inside functions) so
importing this module is cheap and cannot fail at import time.

VALIDATION STATUS (2026-08-10): the dev-pseudo builder is wired against the frozen library workers
but has NOT been executed here (nothing has been run here). The closeout ``--rehearsal``
run is what validates it end-to-end; seams that need that run to confirm carry a VERIFY-ON-RUN note.
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


# --- FISD zero-leakage callback (packaged from build_fisd_reference.main's inline predicate) -----

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


# --- dev-pseudo builder (runnable in closeout --rehearsal; development data only) -----------------

def dev_pseudo_builder(dev_root: Path | None = None):
    """Return a PanelBuilder that assembles the one-shot holdout inventory from the STORED development artefacts,
    sliced to [seed_start .. window.end]. Used by ``--rehearsal`` on the dev pseudo-window; every
    read is under data/development. The gate never opens for this builder."""
    root = Path(dev_root) if dev_root is not None else DEV_ROOT

    def _build(*, seed_start: str, window: Window, sub_window: Window):
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
        # frozen estimator. VERIFY-ON-RUN: the feed→matrices→SuffStats→fit_ipca_recursive assembly is
        # wired to the library workers below; the closeout rehearsal confirms shapes/burn-in numerically.
        inv["ipca_oos_factors"] = _ipca_oos_from_feed(root / "ipca_panel_corr.parquet", lo, hi)

        # FISD as-of ratings over the window grid (backward merge_asof; leakage marker retained).
        inv["fisd_ratings"] = _fisd_ratings_over_window(root, window)
        return inv

    return _build


def _ipca_oos_from_feed(feed_path: Path, start: str, end: str) -> pd.DataFrame:
    """Recursive-OOS IPCA factors over [start, end] via the CANONICAL chain the shakedown itself
    uses: ``load_feed`` (the adapter) → ``build_sufficient_stats`` → ``fit_ipca_recursive``. ``K``
    and the OOS burn-in are read from the frozen IPCA config (``model.K`` / ``window.
    oos_burn_in_months``), never hard-coded — one source of truth, no VERIFY-ON-RUN guess."""
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


def _fisd_ratings_over_window(dev_root: Path, window: Window) -> pd.DataFrame:
    """As-of monthly FISD ratings over the window grid, BUILT FRESH via the frozen backward
    merge_asof worker (``build_ratings_monthly``) so ``_sel_rating_date`` — the leakage marker — is
    REAL. The stored ``fisd_ratings_monthly.parquet`` does not persist that marker, so slicing it and
    injecting ``NaT`` would make ``zero_leakage_check`` vacuous (a guard that validates nothing is
    worse than none). Fail-loud if the worker still does not emit the marker."""
    from agents.quant.library.fisd_reference import build_ratings_monthly, build_static, load_config

    cfg = load_config()
    static = build_static(cfg)
    issue_to_cusip = static.dropna(subset=["issue_id"]).set_index("issue_id")["cusip"]
    issue_to_cusip = issue_to_cusip[~issue_to_cusip.index.duplicated(keep="first")]

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

def holdout_builder(dev_root: Path | None = None):
    """Return the REAL PanelBuilder. It cleans the holdout raw and RE-BUILDS every rolling
    construction over the seeded development+holdout span, then extracts the window. It reads
    data/holdout and therefore runs ONLY behind the single-access gate in the real run.

    It refuses to execute here. The precise recipe (the worker chain it must run) is:

      1. Clean the holdout partition (reads data/holdout raw):
           preprocess_trace.run_pandas(cfg)  → dev/holdout split + raise-guard (holdout_end_year)
           apply_decimal_shift.process_partition(hold_in, hold_out, cfg)
           bounce_back_filter.process_partition(...)  → trace_clean_corr (holdout)
           apply_distressed_filters.process_partition(...)  → trace_daily_corr_filtered (holdout)
           build_daily_panel.aggregate_daily(...)  (raw + corr families, holdout)
      2. Assemble the SEEDED monthly panel = development maximal [seed_start..2021-12]
         CONCATENATED with the freshly-built holdout maximal [2022-01..window.end], so every
         rolling window is warm at 2022-01 (build_monthly_panel.aggregate_family + merge_fisd).
      3. Re-build the constructions over that seeded panel (the SAME workers as dev):
           signals: build_var_5pct.compute_dual_family / build_mom6_signal.compute_dual_family /
                    build_bond_vol.compute_dual_family / build_gamma_illiq.compute_gamma
           factors: build_bbw_factors.run_family / build_str.run_family / build_mom6.run_family /
                    build_mktb.compute_dual_family
           ipca:    build_ipca_feed → feed_matrices → build_sufficient_stats →
                    fit_ipca_recursive(burn_in=36)
           fisd:    asof_monthly_rating(events, grid=window months, agencies, ig_threshold)
      4. Extract the window months; return the inventory. zero_leakage_check runs on fisd_ratings.

    Open wiring points to close during the gated run (documented, not silently assumed):
      * survivor / benchmark derivation: the G6 survivor strategy returns and the BBW-4 / DFPS-4
        benchmark factors handed to stage 2 are extracted from THIS inventory (currently OneshotHoldoutConfig
        carries them; the real path derives them here);
      * scripts.* importability: the compute_* workers live in scripts/, not the library — the gated
        run imports them via the repo-root path the scripts already insert, or they are lifted into
        agents/quant/library first (preferred).
    """
    raise OneshotHoldoutBuilderGated(
        "holdout_builder reads data/holdout and re-builds every construction; it runs only behind "
        "the single-access gate in the real run (an explicit go-ahead + G6 survivors + P3 + E9)."
    )


BUILDERS = {
    "dev_pseudo": dev_pseudo_builder,
    "holdout": holdout_builder,
}
