"""Basis-selectable development inputs: one loader for every experiment that runs on BOTH return bases.

The two bases:

  * ``total_return`` — the default-flat total-return panels (a defaulted bond trades flat: AI = C = 0
    from its default month, the ``default`` cutoff of ``accrual.accrued_and_coupon``):
    ``monthly_panel_total_return_default_flat`` LEFT-joined with
    ``monthly_panel_profiles_total_return_default_flat``.
  * ``clean`` — the clean-price panels via the auditor's dev loader (``load_dev_inputs``):
    ``monthly_panel_maximal`` LEFT-joined with ``monthly_panel_profiles``.

Signals (the sort variables var_5pct / mom6 / gamma / bond_vol) are the clean-price signal files on BOTH bases —
only the return leg changes, exactly as in ``scripts/run_full_audit_total_return.py`` and the holdout runner.
Short-term reversal is the exception: it sorts on the panel's own prior-month ``xret``, so on ``total_return``
its signal is the total-return prior-month excess return. Development data only: the loaded
dates are asserted to end before the holdout floor (2022-01); ``/data/holdout/`` is never touched.

Every basis-specific output lives under ``RESULTS_ROOT`` (one place), split by basis; ``basis_dir`` builds
those paths. Callers use their module defaults when no basis is given, so default-location artefacts are
never overwritten.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from shared.licensed_inputs import require_licensed_input

REPO_ROOT = Path(__file__).resolve().parents[1]
DEV = REPO_ROOT / "data" / "development"
RESULTS_ROOT = REPO_ROOT / "results" / "consistent_basis"
HOLDOUT_FLOOR = pd.Timestamp("2022-01-01")

BASES = ("total_return", "clean")
PANELS = {
    "total_return": (DEV / "monthly_panel_total_return_default_flat.parquet",
                     DEV / "monthly_panel_profiles_total_return_default_flat.parquet"),
    "clean": (DEV / "monthly_panel_maximal.parquet",
              DEV / "monthly_panel_profiles.parquet"),
}


class BasisError(ValueError):
    """An unknown basis name, or a loaded panel that reads into the holdout."""


def check_basis(basis: str) -> str:
    if basis not in BASES:
        raise BasisError(f"basis must be one of {BASES}; got {basis!r}")
    return basis


def basis_dir(basis: str, *parts: str) -> Path:
    """``RESULTS_ROOT/<basis>/<parts...>`` — the one place for a basis-specific output (not created here)."""
    return RESULTS_ROOT.joinpath(check_basis(basis), *parts)


def shared_dir(*parts: str) -> Path:
    """``RESULTS_ROOT/shared/<parts...>`` — outputs that compare both bases in one artefact."""
    return RESULTS_ROOT.joinpath("shared", *parts)


def factors_dir(basis: str) -> Path:
    """Where a basis run writes (and its consumers read) the basis-specific factor series."""
    return basis_dir(basis, "factors")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class BasisProvenance:
    basis: str
    panels: dict = field(default_factory=dict)   # repo-relative path -> sha256
    signals: str = ("clean-price signal files (agents/auditor/ipca_differential/runner.load_dev_signals: var_5pct, "
                    "mom6, gamma, bond_vol); str sorts on the panel's own prior-month xret (total return on the "
                    "total_return basis)")

    def to_dict(self) -> dict:
        return {"basis": self.basis, "panels": dict(self.panels), "signals": self.signals}


def basis_provenance(basis: str) -> BasisProvenance:
    panel, profiles = PANELS[check_basis(basis)]
    return BasisProvenance(basis, {
        str(p.relative_to(REPO_ROOT)): sha256(require_licensed_input(p, "development panel"))
        for p in (panel, profiles)})


def assert_dev_only(dates, what: str) -> None:
    s = pd.to_datetime(pd.Series(dates)).dropna()
    if s.empty:
        raise BasisError(f"{what}: no dated rows")
    if pd.Timestamp(s.max()) >= HOLDOUT_FLOOR:
        raise BasisError(f"{what}: reads into the holdout (max {s.max()} >= {HOLDOUT_FLOOR.date()})")


def load_basis_maximal(basis: str) -> pd.DataFrame:
    """The maximal panel (profiles LEFT-joined) for one basis — the ``maximal`` half of ``load_basis_inputs``."""
    check_basis(basis)
    if basis == "clean":
        from agents.auditor.ipca_differential.runner import load_dev_inputs
        maximal, _signals, _reg = load_dev_inputs()
    else:
        panel, profiles = PANELS["total_return"]
        maximal = pd.read_parquet(
            require_licensed_input(panel, "default-flat total-return development panel")
        ).merge(
            pd.read_parquet(require_licensed_input(profiles, "default-flat total-return profiles panel")),
            on=["cusip", "date"], how="left",
        )
    assert_dev_only(maximal["date"], f"{basis} maximal panel")
    return maximal


def load_basis_inputs(basis: str):
    """``(maximal, signals, registry)`` for one basis — a drop-in for ``load_dev_inputs()``."""
    from agents.auditor.ipca_differential.runner import load_dev_signals, load_registry
    maximal = load_basis_maximal(basis)
    signals = load_dev_signals()
    if "date" in getattr(signals, "columns", []):
        assert_dev_only(signals["date"], "dev signals")
    return maximal, signals, load_registry()
