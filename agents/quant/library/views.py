"""
views.py — pure-function materialisation of an engine-shape panel from the
maximal monthly panel, driven by a RunConfig.

The Phase 2 canonical interface: every characteristic-sort run flows through
`view(maximal_panel, config, signals, ...)` rather than ad-hoc family-selection
adapters. Same RunConfig hash → byte-identical output (the spec's purity
invariant).

Behaviour summary:
  1. Family selection per `config.panel_view.price_family`. Maximal-panel
     columns `*_<family>` are renamed to their engine-contract names
     (e.g. `ret_corr → ret`).
  2. Stale-price mask per A3 when `config.panel_view.stale_mask`. Masks
     `price_eom(i,t)` iff `month_end(t) − last_trade_date(i,t) > θ`,
     propagates the mask forward to `ret(t+1)` and `xret(t+1)`.
  3. Terminal rows kept/dropped per `config.panel_view.include_terminal_rows`.
     Pre-FISD the `exit_reason` column is NaN everywhere so this is a
     documented no-op.
  4. Signal resolution per A9: signals are family-indexed, the view selects
     the matching `*_<family>` column. A mixed-family request (signal has
     only the wrong family) raises.

The function is pure: no mutation of inputs, no global state, deterministic
column ordering. Two calls with the same (maximal_panel, config, signals,
stale_threshold_days) produce identical output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import yaml

from .run_config import RunConfig


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"


# ---------------------------------------------------------------------------
# Threshold loading
# ---------------------------------------------------------------------------

def _load_stale_theta_days() -> int:
    """Default theta_days from thresholds.yaml:signals.stale_price.theta_days.
    Callers can override via the `stale_threshold_days` argument to view()."""
    with open(THRESHOLDS_FILE) as f:
        cfg = yaml.safe_load(f)
    block = cfg.get("signals", {}).get("stale_price")
    if block is None:
        raise KeyError(
            "thresholds.yaml missing signals.stale_price block; "
            "needed for the A3 stale-price mask default"
        )
    return int(block["theta_days"])


# ---------------------------------------------------------------------------
# Family-column selection
# ---------------------------------------------------------------------------

# Columns on the maximal panel that are family-indexed (suffix _raw / _corr).
# Everything else is either pass-through metadata or the primary key.
_FAMILY_INDEXED_PANEL_COLUMNS = (
    "price_eom",
    "ret",
    "xret",
    "n_trades",
    "total_vol",
    "last_trade_date",
)

# Columns that pass through unchanged.
_SHARED_PANEL_COLUMNS = (
    "cusip",
    "date",
    "size",
    "rf_monthly",
    "exit_reason",
)


def _select_family_columns(maximal: pd.DataFrame, family: str) -> pd.DataFrame:
    """Take a single family's view of the maximal panel.

    `*_<family>` columns are renamed to their unsuffixed engine-contract
    names; shared metadata columns pass through; `*_<other_family>` columns
    are dropped.
    """
    out_cols: dict[str, pd.Series] = {}
    for col in _SHARED_PANEL_COLUMNS:
        if col in maximal.columns:
            out_cols[col] = maximal[col]
    for base in _FAMILY_INDEXED_PANEL_COLUMNS:
        family_col = f"{base}_{family}"
        if family_col not in maximal.columns:
            raise KeyError(
                f"maximal panel missing required column '{family_col}' for "
                f"price_family={family!r}"
            )
        out_cols[base] = maximal[family_col]
    return pd.DataFrame(out_cols)


# ---------------------------------------------------------------------------
# A3 stale-price mask
# ---------------------------------------------------------------------------

def _apply_stale_mask(panel: pd.DataFrame, theta_days: int) -> pd.DataFrame:
    """Mask price_eom(i,t) iff month_end(t) − last_trade_date(i,t) > θ days,
    and propagate the mask to ret(t+1), xret(t+1) per A3.2.

    This is distinct from (and coexists with) the ret-adjacency rule
    already baked into ret_raw/ret_corr — A3.4.
    """
    if theta_days < 0:
        raise ValueError(f"theta_days must be >= 0; got {theta_days}")

    panel = panel.sort_values(["cusip", "date"], kind="mergesort").reset_index(drop=True)

    # Compute gap in calendar days between the row's date (month-end) and
    # its last_trade_date_<family>. Rows with NaT last_trade_date are
    # treated as fully stale (NaT subtraction → NaT → not > θ); they get
    # no row-level mask but their price is already NaN so downstream is safe.
    last_td = pd.to_datetime(panel["last_trade_date"])
    row_dates = pd.to_datetime(panel["date"])
    gap_days = (row_dates - last_td).dt.days

    # A masked row: a real gap > θ. NaT → NaN → False; not flagged here.
    self_stale = (gap_days > theta_days).fillna(False).astype(bool)

    # Mask price_eom(t), ret(t), xret(t) — this row's own observations.
    panel = panel.copy()
    panel.loc[self_stale, "price_eom"] = float("nan")
    panel.loc[self_stale, "ret"] = float("nan")
    panel.loc[self_stale, "xret"] = float("nan")

    # Propagation: ret(t+1) depends on price(t), so if price(t) was masked
    # the NEXT row for the same cusip must also have its ret + xret masked.
    panel["_stale_t"] = self_stale.values
    panel["_stale_prev"] = (
        panel.groupby("cusip")["_stale_t"].shift(1, fill_value=False)
    )
    propagate_mask = panel["_stale_prev"]
    panel.loc[propagate_mask, "ret"] = float("nan")
    panel.loc[propagate_mask, "xret"] = float("nan")
    panel = panel.drop(columns=["_stale_t", "_stale_prev"])
    return panel


# ---------------------------------------------------------------------------
# A9 signal resolution
# ---------------------------------------------------------------------------

def _resolve_signals_family(signals: pd.DataFrame, family: str) -> pd.DataFrame:
    """Select the requested family's columns from a signals DataFrame.

    For each "base name" in signals (e.g. `var_5pct`):
      - If `<base>_<family>` is present, include and rename to `<base>`.
      - Else if `<base>_<other_family>` is present (chimera case), RAISE
        per A9.
    Columns without a family suffix and the (cusip, date) keys pass through.
    """
    other = "corr" if family == "raw" else "raw"
    suffix_self = f"_{family}"
    suffix_other = f"_{other}"

    bases: dict[str, dict[str, str]] = {}  # base_name -> {family: column_name}
    pass_through: list[str] = []
    for col in signals.columns:
        if col in ("cusip", "date"):
            pass_through.append(col)
        elif col.endswith(suffix_self):
            base = col[: -len(suffix_self)]
            bases.setdefault(base, {})[family] = col
        elif col.endswith(suffix_other):
            base = col[: -len(suffix_other)]
            bases.setdefault(base, {})[other] = col
        else:
            pass_through.append(col)

    # A9 enforcement: any base where the wrong family is present BUT the
    # right family is missing is a chimeric request.
    for base, fams in bases.items():
        if family not in fams and other in fams:
            raise ValueError(
                f"A9: signal '{base}_{other}' provided but '{base}_{family}' "
                f"missing; cannot resolve for price_family={family!r}. Sorting "
                f"a {family}-family panel on a {other}-family signal is a "
                f"chimera that exists at no lattice point in the registry."
            )

    # Build output column list: pass-through then renamed family columns.
    rename_map: dict[str, str] = {}
    out_cols = list(pass_through)
    for base, fams in bases.items():
        if family in fams:
            out_cols.append(fams[family])
            rename_map[fams[family]] = base
    return signals[out_cols].rename(columns=rename_map)


# ---------------------------------------------------------------------------
# Public: view()
# ---------------------------------------------------------------------------

def view(
    maximal_panel: pd.DataFrame,
    config: RunConfig,
    signals: Optional[pd.DataFrame] = None,
    stale_threshold_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    Materialise an engine-shape panel from the maximal panel + optional
    signals, applying the config's view-level transformations.

    Parameters
    ----------
    maximal_panel : pd.DataFrame
        The maximal monthly panel with dual column families. Must contain
        `cusip`, `date`, `size`, `rf_monthly`, `exit_reason`, plus
        `<base>_raw` / `<base>_corr` for each family-indexed column
        (price_eom, ret, xret, n_trades, total_vol, last_trade_date).
    config : RunConfig
        The run configuration. view() consumes ONLY `config.panel_view`
        (family selection, stale mask, terminal rows); the construction and
        evaluation blocks apply downstream at the engine/rulebook level.
        Cache keys or provenance stamps for view() output must therefore
        use `RunConfig.panel_view_hash()`, not the full `hash()`.
    signals : pd.DataFrame, optional
        Family-indexed signals to attach. Resolved per A9: the matching
        family's columns are renamed to their unsuffixed names; cross-family
        requests raise.
    stale_threshold_days : int, optional
        Override the default θ from thresholds.yaml. Useful for tests.

    Returns
    -------
    pd.DataFrame with engine-contract columns: cusip, date, size, ret, xret,
    price_eom, last_trade_date, n_trades, total_vol, rf_monthly, plus any
    family-resolved signal columns. Deterministic row ordering (sorted by
    cusip, date) and column ordering.
    """
    if not isinstance(maximal_panel, pd.DataFrame):
        raise TypeError("maximal_panel must be a pandas DataFrame")
    if not isinstance(config, RunConfig):
        raise TypeError("config must be a RunConfig")

    family = config.panel_view.price_family

    # 1. Family selection.
    panel = _select_family_columns(maximal_panel, family)

    # 2. Stale mask.
    if config.panel_view.stale_mask:
        if stale_threshold_days is None:
            stale_threshold_days = _load_stale_theta_days()
        panel = _apply_stale_mask(panel, stale_threshold_days)

    # 3. Terminal rows.
    # Pre-FISD exit_reason is NaN everywhere so the include-terminal toggle
    # has no observable effect. When FISD lands and exit_reason carries
    # default / maturity / other terminal codes, dropping `exit_reason != NaN`
    # excludes those rows.
    if "exit_reason" in panel.columns and not config.panel_view.include_terminal_rows:
        panel = panel[panel["exit_reason"].isna()].copy()

    # exit_reason itself is metadata, not an engine input — drop from output.
    if "exit_reason" in panel.columns:
        panel = panel.drop(columns=["exit_reason"])

    # 4. Signals.
    if signals is not None:
        if not isinstance(signals, pd.DataFrame):
            raise TypeError("signals must be a pandas DataFrame when provided")
        resolved = _resolve_signals_family(signals, family)
        dup = resolved.duplicated(subset=["cusip", "date"])
        if dup.any():
            raise ValueError(
                f"signals contain {int(dup.sum())} duplicate (cusip, date) "
                f"rows; a left-merge would silently multiply panel rows"
            )
        panel = panel.merge(resolved, on=["cusip", "date"], how="left")

    # 5. Deterministic ordering — by (cusip, date) and a canonical column order.
    panel = panel.sort_values(["cusip", "date"], kind="mergesort").reset_index(drop=True)
    return panel
