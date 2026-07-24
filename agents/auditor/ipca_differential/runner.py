"""
runner.py — real dev-panel execution driver for the IPCA differential (§12.7).

Loads the development panel (2002-2021; NEVER holdout), merges the signal parquets into the
dual-family frame ``view()`` expects, and drives the runnable (bias, anchor) pairs through
``run_differential``. The 6 construction-toggle pairs are typed-refused (FEED_OFF_STATE_UNDEFINED),
never run. All inputs come from ``data/development/`` — the holdout is untouched.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from agents.auditor.thresholds import (
    IPCAExecutionPairs,
    load_ipca_execution_pairs,
)

from .differential import run_differential

REPO_ROOT = Path(__file__).resolve().parents[3]
DEV = REPO_ROOT / "data" / "development"
REGISTRY = REPO_ROOT / "agents" / "quant" / "library" / "configs" / "ipca_instruments.yaml"
MAXIMAL_PANEL = DEV / "monthly_panel_maximal.parquet"

# The four IPCA signal parquets and their dual-family columns. gamma_illiq.parquet stores its
# columns as gamma_raw / gamma_corr, but the canonical INSTRUMENT name is gamma_illiq — so we rename
# on load, and view()'s A9 family resolution then yields the column `gamma_illiq` that to_merged
# expects (fixing the gamma vs gamma_illiq mismatch at the single point where the name is known).
_SIGNAL_COLUMNS: dict[str, tuple[str, str]] = {
    "mom6.parquet": ("mom6_raw", "mom6_corr"),
    "var_5pct.parquet": ("var_5pct_raw", "var_5pct_corr"),
    "gamma_illiq.parquet": ("gamma_raw", "gamma_corr"),
    "bond_vol.parquet": ("bond_vol_raw", "bond_vol_corr"),
}
_GAMMA_RENAME = {"gamma_raw": "gamma_illiq_raw", "gamma_corr": "gamma_illiq_corr"}


def load_dev_signals(dev: Path = DEV) -> pd.DataFrame:
    """Load + outer-merge the 4 dev signal parquets on (cusip, date) into the dual-family frame
    view() resolves. The gamma columns are renamed to the canonical gamma_illiq_* instrument name."""
    merged: pd.DataFrame | None = None
    for fname, cols in _SIGNAL_COLUMNS.items():
        df = pd.read_parquet(dev / "signals" / fname, columns=["cusip", "date", *cols])
        if fname == "gamma_illiq.parquet":
            df = df.rename(columns=_GAMMA_RENAME)
        merged = df if merged is None else merged.merge(df, on=["cusip", "date"], how="outer")
    assert merged is not None
    return merged


def load_registry(path: Path = REGISTRY) -> dict:
    """The ipca_instruments.yaml registry (meta.train_end, scaler.floor) as a dict."""
    return yaml.safe_load(Path(path).read_text())


def load_dev_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(maximal_panel, merged_signals, registry) for the dev window. Holdout is never read."""
    maximal = pd.read_parquet(MAXIMAL_PANEL)
    return maximal, load_dev_signals(), load_registry()


def refused_record(bias: str, anchor: str, pairs: IPCAExecutionPairs) -> dict:
    """The typed-refusal record for a construction-toggle pair (§12.7). No cell is computed."""
    return {
        "bias": bias,
        "anchor": anchor,
        "status": "refused",
        "reason": pairs.refused.reason,
        "note": pairs.refused.note,
    }


def run_runnable_pairs(
    maximal: pd.DataFrame,
    signals: pd.DataFrame,
    reg: dict,
    *,
    subset: list[tuple[str, str]] | None = None,
    pairs: IPCAExecutionPairs | None = None,
    thresholds_path=None,
    bootstrap_seed: int = 20260612,
) -> list:
    """Run the runnable (bias, anchor) pairs through ``run_differential`` (with §5.3 bootstrap).
    ``subset`` restricts to specific pairs (e.g. the smoke). Returns the IPCADifferentialResult list."""
    pairs = pairs or load_ipca_execution_pairs(thresholds_path)
    todo = subset if subset is not None else pairs.runnable_pairs()
    runnable = set(pairs.runnable_pairs())
    results = []
    for bias, anchor in todo:
        if (bias, anchor) not in runnable:
            raise ValueError(f"pair ({bias}, {anchor}) is not in the registered runnable set")
        results.append(
            run_differential(
                bias, anchor, maximal, signals, reg,
                thresholds_path=thresholds_path, bootstrap_seed=bootstrap_seed,
            )
        )
    return results
