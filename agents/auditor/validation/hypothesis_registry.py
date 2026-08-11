"""
Factor-level hypothesis registry + runtime gate (spec v4 Part F).

"No number is computed on a factor until a dated, committed hypothesis exists for
it." This loads `config/hypothesis_registry.yaml` (append-only, one row per
factor, seeded from auditor_design.md §13.2) and exposes:

  * ``FactorHypothesis`` — the frozen per-factor expectation.
  * ``load_hypothesis_registry`` — fail-loud loader (mirrors
    ``anchor_triangulation.load_anchor_gate``'s D-1..D-4 guards; adds `sign_only`
    / `near_zero` magnitude modes for factors with no external band).
  * ``require_factor_registered`` — the RUNTIME GATE: raises
    ``FactorHypothesisAbsent`` (a DsrPreRegistrationAbsent-style refusal) when a
    factor's CONFIRMATORY OUTCOME is requested but its row is absent/unlocked.
    Scoped to confirmatory outcomes ONLY — it must never be called to guard row
    counts, invariants, synthetic fixtures, execution diagnostics, or PILOT
    outputs (those are not confirmatory factor outcomes).

APPEND-ONLY: edits to an existing row require a new dated entry with
justification (enforced by review + the register_commit provenance, not by code).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
_REGISTRY_PATH = REPO_ROOT / "config" / "hypothesis_registry.yaml"
_SECTION = "spec v4 Part F (factor-level hypothesis registry)"

# Magnitude modes for a factor with no defensible external band.
_SIGN_ONLY = "sign_only"
_NEAR_ZERO = "near_zero"


class HypothesisRegistryError(RuntimeError):
    """A malformed / missing field in config/hypothesis_registry.yaml."""


class FactorHypothesisAbsent(RuntimeError):
    """The runtime gate (Part F): a CONFIRMATORY OUTCOME was requested for a factor
    whose registry row is absent or not lockable — refuse rather than computing a
    number against no committed hypothesis (mirrors DsrPreRegistrationAbsent).
    Scoped to confirmatory outcomes only."""

    def __init__(self, factor_id: str, reason: str) -> None:
        self.factor_id = factor_id
        super().__init__(
            f"factor {factor_id!r} has no committed confirmatory hypothesis "
            f"({reason}) — register + git-tag config/hypothesis_registry.yaml "
            f"before computing its confirmatory outcome (Part F)."
        )


@dataclass(frozen=True)
class FactorHypothesis:
    factor_id: str
    dominant_bias: str
    expected_sign: int                          # +1 / -1 / 0 (negative control)
    magnitude_mode: str                          # 'band' | 'sign_only' | 'near_zero'
    expected_magnitude_range: tuple[float, float] | None  # decimal monthly, unsigned; None if not 'band'
    is_locked: bool
    status: str
    source: str
    registered_commit: str

    @property
    def is_confirmatory(self) -> bool:
        """A locked, non-pilot factor enters the confirmatory family. Pilots
        (str) and outputs labelled pilot/exploratory do NOT (D-A33)."""
        return self.is_locked and self.status != "pilot"


def load_hypothesis_registry(path: str | Path | None = None) -> dict[str, FactorHypothesis]:
    """Load the append-only factor registry fail-loud. Enforces AT LOAD the guards a
    malformed pre-registration would slip past (mirrors load_anchor_gate)."""
    p = Path(path) if path else _REGISTRY_PATH
    if not p.exists():
        raise HypothesisRegistryError(f"registry file absent: {p}")
    doc = yaml.safe_load(p.read_text())
    factors = doc.get("factors") if isinstance(doc, dict) else None
    if not isinstance(factors, dict) or not factors:
        raise HypothesisRegistryError(f"{p} has no non-empty 'factors' map")

    out: dict[str, FactorHypothesis] = {}
    for fid, row in factors.items():
        stem = f"factors.{fid}"
        if not isinstance(row, dict):
            raise HypothesisRegistryError(f"{stem} must be a mapping")

        def req(key):
            if key not in row:
                raise HypothesisRegistryError(f"{stem}.{key} is required")
            return row[key]

        sign = req("expected_sign")
        if sign not in (-1, 0, 1):
            raise HypothesisRegistryError(f"{stem}.expected_sign must be -1/0/+1; got {sign!r}")

        is_locked = req("is_locked")
        if not isinstance(is_locked, bool):
            raise HypothesisRegistryError(f"{stem}.is_locked must be bool (D-3, required)")

        # Magnitude: either a band [lo,hi] (unsigned ascending decimal) or a mode token.
        band = row.get("expected_magnitude_range")
        mode_token = row.get("magnitude")
        if band is not None:
            if not (isinstance(band, (list, tuple)) and len(band) == 2):
                raise HypothesisRegistryError(f"{stem}.expected_magnitude_range must be [lo, hi]")
            lo, hi = float(band[0]), float(band[1])
            if not (0.0 <= lo <= hi):
                raise HypothesisRegistryError(
                    f"{stem}.expected_magnitude_range must be unsigned ascending 0<=lo<=hi "
                    f"(decimal monthly; direction in expected_sign); got [{lo}, {hi}]"
                )
            # A LOCKED band gate must be FALSIFIABLE. Falsifiability comes from a
            # definite sign (±1) AND a finite upper bound (hi>lo): an effect of the
            # wrong sign, or |effect|>hi, fails — so lo=0 is fine (e.g. mom6's
            # DRR-published band [0, 0.0030], where the ex-ante endpoint is ≈0).
            # It is unfalsifiable only when the sign is UNDEFINED (0) AND lo=0,
            # where any small effect passes; a control uses `near_zero` instead.
            if is_locked and sign == 0 and lo == 0.0:
                raise HypothesisRegistryError(
                    f"{stem}: a LOCKED sign-0 band starting at lo=0 is unfalsifiable; "
                    f"use magnitude: near_zero for a control, or set a definite sign"
                )
            if is_locked and not (hi > lo) and not (lo > 0.0):
                raise HypothesisRegistryError(
                    f"{stem}: a LOCKED band needs either lo>0 or hi>lo to be falsifiable; "
                    f"got a degenerate [{lo}, {hi}]"
                )
            magnitude_mode, mag_range = "band", (lo, hi)
        elif mode_token in (_SIGN_ONLY, _NEAR_ZERO):
            magnitude_mode, mag_range = mode_token, None
        else:
            raise HypothesisRegistryError(
                f"{stem}: needs expected_magnitude_range [lo,hi] OR magnitude: "
                f"sign_only|near_zero; got magnitude={mode_token!r}"
            )

        out[fid] = FactorHypothesis(
            factor_id=fid,
            dominant_bias=str(req("dominant_bias")),
            expected_sign=int(sign),
            magnitude_mode=magnitude_mode,
            expected_magnitude_range=mag_range,
            is_locked=bool(is_locked),
            status=str(req("status")),
            source=str(req("source")),
            registered_commit=str(req("registered_commit")),
        )
    return out


def require_factor_registered(
    factor_id: str, *, path: str | Path | None = None
) -> FactorHypothesis:
    """RUNTIME GATE (Part F): return the factor's committed hypothesis, or REFUSE
    (`FactorHypothesisAbsent`) if its row is absent. Call this ONLY before computing
    a CONFIRMATORY OUTCOME for the factor — never for row counts / invariants /
    synthetic fixtures / execution diagnostics / PILOT outputs."""
    try:
        registry = load_hypothesis_registry(path)
    except HypothesisRegistryError as exc:
        raise FactorHypothesisAbsent(factor_id, f"registry unloadable: {exc}") from exc
    hyp = registry.get(factor_id)
    if hyp is None:
        raise FactorHypothesisAbsent(factor_id, "no row in config/hypothesis_registry.yaml")
    return hyp
