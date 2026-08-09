"""Stage 1 — build the holdout-side panel, seeded from development (one-shot holdout §2).

Every rolling construction in the one-shot holdout inventory seeds from development data starting at the
derived ``seed_start`` (§2 / ``seed_start.py``), so each series is non-NaN from the first
evaluation month. Outputs land in a QUARANTINE directory with a per-artefact SHA-256;
nothing under ``data/`` or ``docs/`` is touched.

The actual panel construction is injected as a ``PanelBuilder`` — the frozen cleaning/panel
pipeline over ``data/holdout/`` raw for the real run (behind the single-access gate), a
development-pseudo-window builder for ``--rehearsal``, or a synthetic builder in tests. This
module owns the seeding, quarantine, hashing, the non-NaN seeding validation, and the
zero-leakage assertion for the holdout FISD ratings — never a hard-coded ``data/holdout`` read.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import pandas as pd

from .seed_start import derive_seed_start
from .windows import Window

# A PanelBuilder receives the derived seed_start and both windows and returns the one-shot holdout-inventory
# artefacts as {name: DataFrame} (both price families, signals, factors, IPCA OOS, FISD ratings).
PanelBuilder = Callable[..., Mapping[str, pd.DataFrame]]


@dataclass(frozen=True)
class Stage1Artefact:
    name: str
    path: Path
    sha256: str
    n_rows: int
    non_nan_at_first_month: bool


@dataclass(frozen=True)
class Stage1Result:
    quarantine_dir: Path
    seed_start: str
    artefacts: tuple[Stage1Artefact, ...]

    def artefact_hashes(self) -> dict[str, str]:
        return {a.name: a.sha256 for a in self.artefacts}

    def all_seeded_green(self) -> bool:
        return all(a.non_nan_at_first_month for a in self.artefacts)


def _first_month_non_nan(df: pd.DataFrame, first_month: str) -> bool:
    """True iff the artefact has at least one non-NaN value dated in ``first_month`` (the
    seeding validation: a rolling construction must be warm by the first evaluation month)."""
    if "date" in df.columns:
        idx = pd.to_datetime(df["date"])
        value_cols = [c for c in df.columns if c != "date"]
    else:
        idx = pd.to_datetime(df.index)
        value_cols = list(df.columns)
    target = pd.Period(first_month, "M")
    rows = df.loc[idx.dt.to_period("M") == target]
    if rows.empty:
        return False
    return bool(rows[value_cols].notna().any().any()) if value_cols else True


def build_holdout_panel(
    *,
    window: Window,
    sub_window: Window,
    quarantine_dir: str | Path,
    panel_builder: PanelBuilder,
    thresholds_path: str | Path | None = None,
    margin_months: int = 3,
    fisd_ratings_name: str = "fisd_ratings",
    zero_leakage_check: Callable[[pd.DataFrame, Window], None] | None = None,
) -> Stage1Result:
    """Derive the seed, run the injected builder, write artefacts to quarantine with hashes,
    and validate each is non-NaN at the first evaluation month. Returns a Stage1Result.

    ``quarantine_dir`` must be outside ``data/`` and ``docs/`` (asserted). ``panel_builder`` is
    the only thing that reads panel inputs — for the real run it reads ``data/holdout/`` behind
    the gate; here it is injected, keeping this module firewall-clean and testable.
    """
    qdir = Path(quarantine_dir)
    parts = qdir.resolve().parts
    if "data" in parts or "docs" in parts:
        raise ValueError(f"quarantine_dir must not be under data/ or docs/; got {qdir}")

    seed_start, _sources = derive_seed_start(
        window.start, thresholds_path=thresholds_path, margin_months=margin_months,
    )

    artefacts_data = panel_builder(seed_start=seed_start, window=window, sub_window=sub_window)
    if not artefacts_data:
        raise ValueError("panel_builder returned no artefacts")

    qdir.mkdir(parents=True, exist_ok=True)
    artefacts: list[Stage1Artefact] = []
    for name, df in artefacts_data.items():
        path = qdir / f"{name}.parquet"
        df.to_parquet(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        artefacts.append(Stage1Artefact(
            name=name, path=path, sha256=sha, n_rows=len(df),
            non_nan_at_first_month=_first_month_non_nan(df, window.start),
        ))

    # Zero-leakage assertion for the holdout FISD ratings (backward merge_asof discipline).
    if zero_leakage_check is not None and fisd_ratings_name in artefacts_data:
        zero_leakage_check(artefacts_data[fisd_ratings_name], window)

    return Stage1Result(quarantine_dir=qdir, seed_start=seed_start, artefacts=tuple(artefacts))
