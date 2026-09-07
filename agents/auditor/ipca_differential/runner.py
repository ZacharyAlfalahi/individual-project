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

from shared.licensed_inputs import require_licensed_input
from agents.auditor.thresholds import (
    IPCAExecutionPairs,
    load_ipca_bootstrap_config,
    load_ipca_execution_pairs,
    load_ipca_lambda,
    load_ipca_projection_gate,
    load_ipca_reporting,
    load_ipca_stability_config,
)

from .differential import _anchor_series, differential_from_feeds, run_differential
from .panels import build_cell_feed, panel_states
from .stability import StabilityDiagnostic, stability_diagnostic

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
    # Per-paper baseline profile signal variants (FL-D21a), if built: the 8
    # `<signal>_<pid>` columns join the dual-family frame so view()'s (now
    # family-general) A9 resolver can select a profile OFF-arm family. ADDITIVE —
    # absent file => raw/corr behaviour is byte-identical.
    prof = dev / "signals" / "profiles_signals.parquet"
    if prof.exists():
        pdf = pd.read_parquet(prof)
        merged = merged.merge(pdf, on=["cusip", "date"], how="outer")
    return merged


def load_registry(path: Path = REGISTRY) -> dict:
    """The ipca_instruments.yaml registry (meta.train_end, scaler.floor) as a dict."""
    return yaml.safe_load(Path(path).read_text())


def load_dev_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(maximal_panel, merged_signals, registry) for the dev window. Holdout is never read.

    The per-paper baseline profile family columns (FL-D21a), if built, are
    LEFT-JOINED onto the maximal panel here so it carries `*_bbw_2019` /
    `*_jostova_2013` alongside `*_raw` / `*_corr`. ADDITIVE — the committed
    `monthly_panel_maximal.parquet` file is never modified, and an absent
    profiles file leaves the panel byte-identical (raw/corr behaviour unchanged)."""
    maximal = pd.read_parquet(require_licensed_input(MAXIMAL_PANEL, "maximal development panel"))
    prof = DEV / "monthly_panel_profiles.parquet"
    if prof.exists():
        pdf = pd.read_parquet(prof)
        maximal = maximal.merge(pdf, on=["cusip", "date"], how="left")
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


def run_pair_full(
    bias: str,
    anchor: str,
    maximal: pd.DataFrame,
    signals: pd.DataFrame,
    reg: dict,
    *,
    thresholds_path=None,
    bootstrap_seed: int = 20260612,
    stability_seed: int = 20260612,
):
    """Build one runnable pair ONCE (panels → feeds → anchor) and return both the 2x2 differential
    (with §5.3 bootstrap) and the §5.4 coupling stability diagnostic, reusing the same feeds — so the
    stability refits are not paid twice for feed construction."""
    lam = load_ipca_lambda(thresholds_path)
    gate = load_ipca_projection_gate(thresholds_path)
    bootstrap = load_ipca_bootstrap_config(thresholds_path)
    stab_cfg = load_ipca_stability_config(thresholds_path)
    is_focal = load_ipca_reporting(thresholds_path).focal_pairs.get(bias) == anchor

    p_n, p_b = panel_states(bias, maximal, signals)
    family_b = "raw" if bias == "meas_err" else "corr"
    feed_n = build_cell_feed(p_n, reg, "corr", recompute_signals=True, thresholds_path=thresholds_path)
    feed_b = build_cell_feed(p_b, reg, family_b, recompute_signals=True, thresholds_path=thresholds_path)
    anchor_series = _anchor_series(p_n, anchor, thresholds_path=thresholds_path)

    result = differential_from_feeds(
        bias, anchor, feed_n, feed_b, anchor_series, lam, gate,
        is_focal=is_focal, bootstrap=bootstrap, bootstrap_seed=bootstrap_seed,
    )
    stab: StabilityDiagnostic = stability_diagnostic(
        bias, anchor, feed_n, feed_b, anchor_series, lam, gate, stab_cfg,
        i_obs=result.interaction_bracket_raw.value, seed=stability_seed,
    )
    return result, stab
