"""Seed-start derivation (one-shot holdout §2).

Every rolling construction in the one-shot holdout inventory seeds from DEVELOPMENT data starting at

    seed_start = window_start − (max module burn-in + margin)

where each burn-in is read from that module's OWN config (never hard-coded), so a config
change moves the seed mechanically. The non-NaN burn-in of a rolling monthly signal is its
``min_obs``; the recursive-OOS IPCA factors need ``oos_burn_in_months``. Seeding reads
development only — no firewall tension. ``seed_start`` is asserted ≥ development start; a
value before it is a fail-loud misconfiguration, never silently clipped.

Judgment call (documented, 2026-08-08): the seed uses the MAXIMUM burn-in across the
inventory (currently the IPCA OOS burn-in, 36 → seed_start ≈ 2018-10). Seeding earlier only
reads more development data and strictly guarantees every series is non-NaN from the first
evaluation month; it is the safe reading of the spec's "max module burn-in + margin".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .windows import add_months, months_inclusive

REPO_ROOT = Path(__file__).resolve().parents[4]
_THRESHOLDS = REPO_ROOT / "docs" / "thresholds.yaml"

# The TRACE Enhanced development panel start — the floor seed_start may never precede.
DEVELOPMENT_START = "2002-07"
_DEFAULT_MARGIN_MONTHS = 3


class SeedStartError(RuntimeError):
    """seed_start could not be derived, or would precede the development start. Fail-loud —
    a clipped or defaulted seed would silently shorten a rolling window's warm-up."""


@dataclass(frozen=True)
class BurnInSource:
    label: str
    months: int
    source: str


def _load(path: str | Path | None) -> dict:
    p = Path(path) if path is not None else _THRESHOLDS
    try:
        data = yaml.safe_load(open(p))
    except FileNotFoundError as exc:
        raise SeedStartError(f"thresholds file not found: {p}") from exc
    if not isinstance(data, dict):
        raise SeedStartError(f"thresholds file empty/malformed: {p}")
    return data


def _get(data: dict, dotted: str):
    node = data
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            raise SeedStartError(f"required config key '{dotted}' missing in thresholds.yaml")
        node = node[key]
    return node


def _get_optional(data: dict, dotted: str):
    node = data
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def collect_module_burn_ins(thresholds_path: str | Path | None = None) -> list[BurnInSource]:
    """Read the one-shot holdout-inventory modules' burn-ins from their own config keys."""
    data = _load(thresholds_path)
    sources = [
        BurnInSource("var_5pct", int(_get(data, "signals.var_5pct.min_obs")), "signals.var_5pct.min_obs"),
        BurnInSource("mom6", int(_get(data, "signals.mom6.min_obs")), "signals.mom6.min_obs"),
        BurnInSource("bond_vol", int(_get(data, "signals.bond_vol.min_obs")), "signals.bond_vol.min_obs"),
    ]
    ipca_burn = _get_optional(data, "ipca.oos_burn_in_months")
    if ipca_burn is None:
        sources.append(BurnInSource(
            "ipca_oos", 36,
            "agents/quant/library/ipca.py recursive_oos(burn_in=36) default (no thresholds override)",
        ))
    else:
        sources.append(BurnInSource("ipca_oos", int(ipca_burn), "ipca.oos_burn_in_months"))
    return sources


def derive_seed_start(
    window_start: str,
    *,
    thresholds_path: str | Path | None = None,
    margin_months: int = _DEFAULT_MARGIN_MONTHS,
    development_start: str = DEVELOPMENT_START,
) -> tuple[str, list[BurnInSource]]:
    """Return ``(seed_start, sources)``. Fail-loud if seed_start precedes ``development_start``."""
    if margin_months < 0:
        raise SeedStartError(f"margin_months must be >= 0; got {margin_months}")
    sources = collect_module_burn_ins(thresholds_path)
    max_burn = max(s.months for s in sources)
    seed_start = add_months(window_start, -(max_burn + margin_months))
    if months_inclusive(development_start, seed_start) < 1:
        raise SeedStartError(
            f"seed_start {seed_start} precedes development start {development_start} "
            f"(max burn-in {max_burn} + margin {margin_months} back from {window_start})"
        )
    return seed_start, sources
