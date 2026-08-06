"""P1 codegen ablation (WS-C) — the oracle registry.

The "oracles" are the hand-built, audited anchor factor series the generated
code is scored against: the three anchors (drf, str, mom6) plus the crf
composite, all at the corrected family, as materialised by the frozen builders
into ``data/development/factors/``. crf carries no standalone builder — the
``compose_crf`` output in ``bbw_factors.parquet`` byte-agrees with the
authoritative runner ``equal_average`` path on the frozen window (documented
in ``agents/quant/library/bbw_factors.py``), so it is the sanctioned source.

Rebuild order when the parquets are absent (dev window only, holdout never):
  build_var_5pct -> build_gamma_illiq -> build_mom6_signal -> build_mom6
  -> build_str -> build_bbw_factors  (emits drf_* AND crf_*)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]

# strategy -> (factor parquet under the repo root, oracle column).
ORACLES: dict[str, tuple[str, str]] = {
    "drf": ("data/development/factors/bbw_factors.parquet", "drf_corr"),
    "crf": ("data/development/factors/bbw_factors.parquet", "crf_corr"),
    "str": ("data/development/factors/str.parquet", "str_corr"),
    "mom6": ("data/development/factors/mom6.parquet", "mom6_corr"),
}


def oracle_path(strategy: str) -> Path:
    if strategy not in ORACLES:
        raise KeyError(f"unknown oracle strategy {strategy!r}; expected one of {sorted(ORACLES)}")
    rel, _ = ORACLES[strategy]
    path = _REPO_ROOT / rel
    if "holdout" in path.parts:
        raise RuntimeError(f"oracle path {path} touches the holdout partition")
    return path


def load_oracle_series(strategy: str) -> pd.Series:
    """The oracle monthly return series, date-indexed, warm-up NaNs dropped.
    Fail-loud when the parquet or column is absent — a silently empty oracle
    would score every candidate WONT_RUN and read as a codegen failure."""
    rel, column = ORACLES[strategy]
    path = oracle_path(strategy)
    if not path.exists():
        raise FileNotFoundError(
            f"oracle parquet missing: {path} — run the builder chain documented in "
            "evaluation/codegen/oracles.py"
        )
    frame = pd.read_parquet(path)
    if column not in frame.columns:
        raise KeyError(f"{path} has no column {column!r} (columns: {list(frame.columns)})")
    series = pd.Series(
        frame[column].values, index=pd.DatetimeIndex(frame["date"]), name="portfolio_return"
    ).dropna()
    if series.empty:
        raise ValueError(f"oracle series {strategy!r} is empty after dropping warm-up NaNs")
    return series.sort_index()


def export_oracle_csv(strategy: str, path: Path) -> Path:
    """Write the oracle in the sandbox output contract's exact shape
    (columns ``date, portfolio_return``) — the round-trip input for the
    oracle-vs-oracle sanity check."""
    series = load_oracle_series(strategy)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({
        "date": series.index.strftime("%Y-%m-%d"),
        "portfolio_return": series.values,
    })
    frame.to_csv(path, index=False)
    return path
