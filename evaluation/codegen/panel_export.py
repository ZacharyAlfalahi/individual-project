"""Export the corr-family engine-shape panel for the P1/P2 codegen sandbox (WS-C).

The sandbox feeds LLM-generated strategy code a single ``PANEL_PATH`` parquet. That panel MUST be
byte-faithful to the panel the frozen oracle builders (``build_{str,mom6,bbw_factors}.py``) consumed
-- otherwise even *correct* generated code would score RUNS_WRONG against the oracle. All three
builders share ONE recipe:

    view(maximal,
         RunConfig(panel_view=PanelViewConfig(price_family="corr", stale_mask=False,
                                              include_terminal_rows=False)),
         signals=merge(var_5pct, gamma_illiq, mom6))

at construction ``signal_lag=0``. This module replicates that recipe exactly and projects to the
frozen ``prompts/panel_schema.md`` columns:

    [cusip, date, ret, xret, size, rating, var_5pct, gamma, mom6]

The oracle series (``data/development/factors/*_corr``) are the corr family, so the exported panel
is the corr family. ``str`` sorts on the prior-month reversal (from ``xret``), ``drf`` on
``var_5pct``, ``lrf`` on ``gamma``, ``crf`` on ``rating``, ``mom6`` on ``mom6`` -- one panel covers
all five.

DEV ONLY: every input is under ``data/development/``; the holdout is never opened (asserted).
Provenance for the panel source mirrors the builders: ``$BBW_ANCHOR_PANEL`` override, else the
§2.1 total-return panel when present, else the maximal panel.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd

from shared.licensed_inputs import require_licensed_input

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEV = _REPO_ROOT / "data" / "development"
_TOTAL = _DEV / "monthly_panel_total_return.parquet"
_CLEAN = _DEV / "monthly_panel_maximal.parquet"
_SIGNAL_DIR = _DEV / "signals"

#: Where the exported codegen panel lands (gitignored data partition, dev only).
CODEGEN_PANEL = _DEV / "codegen" / "engine_panel_corr.parquet"

#: The frozen panel_schema.md column contract, in order.
SCHEMA_COLUMNS: tuple[str, ...] = (
    "cusip", "date", "ret", "xret", "size", "rating", "var_5pct", "gamma", "mom6",
)


def _maximal_panel_path() -> Path:
    """The maximal-panel source, resolved exactly as the oracle builders resolve it
    (``$BBW_ANCHOR_PANEL`` override -> total-return panel if present -> maximal panel)."""
    override = os.environ.get("BBW_ANCHOR_PANEL")
    if override:
        return Path(override)
    return _TOTAL if _TOTAL.exists() else _CLEAN


def _load_signals() -> pd.DataFrame:
    """The three precomputed signal parquets merged on (cusip, date), family-indexed
    (``*_raw`` / ``*_corr``). ``view()`` resolves the requested family to unsuffixed names."""
    frames = []
    for name in ("var_5pct", "gamma_illiq", "mom6"):
        path = _SIGNAL_DIR / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"signal parquet missing: {path} — run scripts/build_{name}.py "
                "(dev window; holdout never)")
        frames.append(pd.read_parquet(path))
    signals = frames[0]
    for extra in frames[1:]:
        signals = signals.merge(extra, on=["cusip", "date"], how="outer")
    return signals


def build_codegen_panel() -> pd.DataFrame:
    """Materialise the corr-family engine panel via the canonical ``view()`` interface —
    the exact input the frozen oracle builders consumed — projected to SCHEMA_COLUMNS.

    Fail-loud on a missing schema column (a silently absent ``rating`` / signal would make the
    sandbox panel diverge from the oracle panel and mis-score every candidate)."""
    from agents.quant.library.run_config import (  # lazy: keeps import light for tests
        ConstructionConfig,
        EvaluationConfig,
        PanelViewConfig,
        RunConfig,
    )
    from agents.quant.library.views import view

    panel_path = _maximal_panel_path()
    if "holdout" in panel_path.parts:
        raise RuntimeError(f"codegen panel source touches the holdout partition: {panel_path}")
    maximal = pd.read_parquet(require_licensed_input(panel_path, "development panel"))
    signals = _load_signals()

    cfg = RunConfig(
        panel_view=PanelViewConfig(
            price_family="corr", stale_mask=False, include_terminal_rows=False),
        construction=ConstructionConfig(signal_lag=0, expost_trim="none"),
        evaluation=EvaluationConfig(),
    )
    panel = (
        view(maximal, cfg, signals=signals)
        .drop_duplicates(subset=["cusip", "date"])
        .reset_index(drop=True)
    )

    # `rating` is family-agnostic (bbw_factors selects it post-view). If view() did not carry it,
    # merge it from the maximal panel — never invent it.
    if "rating" not in panel.columns:
        if "rating" not in maximal.columns:
            raise KeyError("neither the engine view nor the maximal panel carries `rating`")
        panel = panel.merge(
            maximal[["cusip", "date", "rating"]].drop_duplicates(subset=["cusip", "date"]),
            on=["cusip", "date"], how="left")

    missing = [c for c in SCHEMA_COLUMNS if c not in panel.columns]
    if missing:
        raise KeyError(
            f"engine panel is missing schema columns {missing}; have {sorted(panel.columns)}")
    return (
        panel[list(SCHEMA_COLUMNS)]
        .sort_values(["cusip", "date"])
        .reset_index(drop=True)
    )


def export_codegen_panel(out_path: Path = CODEGEN_PANEL) -> tuple[Path, str]:
    """Write the codegen panel and return (path, sha256). Deterministic; overwrite-safe."""
    panel = build_codegen_panel()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    return out_path, digest


if __name__ == "__main__":
    path, digest = export_codegen_panel()
    frame = pd.read_parquet(path)
    print(f"exported codegen panel: {path}")
    print(f"  rows={len(frame):,}  bonds={frame['cusip'].nunique():,}  "
          f"months={frame['date'].nunique()}  sha256={digest[:16]}…")
    print(f"  date range: {frame['date'].min()}  ..  {frame['date'].max()}")
    print("  non-null signal coverage:")
    for col in ("var_5pct", "gamma", "mom6", "rating"):
        print(f"    {col}: {frame[col].notna().mean():.1%}")
