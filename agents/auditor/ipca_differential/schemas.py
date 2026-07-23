"""
schemas.py — the serialized output types for the IPCA differential (spec §2.2, §2.3, §5.3).

Three disciplines are enforced structurally, not by convention:

  * **Orientation + field names (§2.2, D-A44).** Every contrast is in *correction orientation* and
    carries the ``_corr`` suffix; ``interaction_bracket_raw`` (the raw bracket I) and
    ``doe_interaction_effect`` (I/2) are DISTINCT fields that never share a name.
  * **No naked effects (§5.3 output rule, amendment).** An ``EffectEstimate`` cannot be constructed
    — and therefore cannot serialize — without BOTH an interval slot carrying a status label AND
    the exact conditioning caveat. Even while intervals are deferred, no effect can
    escape "naked". The output rule: *no reported effect estimate ships without an interval and an
    explicit conditioning statement.*
  * **Algebraic identity asserted at serialization (§2.3, the Theorem row).** ``IPCADifferentialResult``
    checks, at construction, that the bracket equals the difference-in-differences of the four
    realised cells and of the two margin pairs, and that the DOE effect is exactly I/2 and the total
    decomposes — so a wiring error cannot serialize.

The word "primary" is deliberately absent from every field name — the whole extension is
exploratory (§3.2, D-A50).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

# The §5.3 conditioning statement carried by every effect estimate (normative wording).
CONDITIONING_CAVEAT = (
    "conditional on the two realised fitted states; the extension does not estimate the additional "
    "uncertainty from re-fitting those states under repeated samples — an exploratory conditional "
    "estimate, not a fully uncertainty-integrated population parameter"
)

# The §2.3 single-freezing caveat (claim hierarchy), carried verbatim on every result.
SINGLE_FREEZING_CAVEAT = (
    "the difference between the frozen and re-estimated arms is the effect of estimator adaptation "
    "plus any interaction between adaptation and biased data; a single freezing does not identify "
    "these components separately — the 2x2 table is what does"
)

# Interval-status vocabulary. "deferred_inc3": the conditional bootstrap (§5.3) was not requested.
# "computed": a bootstrap interval is attached. "refused": a bootstrap was requested but the common
# support could not support it (block length vs support, §6.2) — an honest refusal, not a fake CI.
INTERVAL_DEFERRED = "deferred_inc3"
INTERVAL_COMPUTED = "computed"
INTERVAL_REFUSED = "refused"

_ALGEBRA_TOL = 1e-9


class IPCASchemaError(ValueError):
    """A serialized output violated the orientation / no-naked-effect / algebraic-identity contract."""


@dataclass(frozen=True)
class EffectEstimate:
    """One effect estimate. Structurally cannot be 'naked' (§5.3): it must carry an interval slot
    with a status label AND a non-empty conditioning statement."""

    name: str                          # e.g. "data_margin_theta_n_corr"
    value: float
    orientation: str                   # "corr" — correction orientation, always
    interval_status: str               # INTERVAL_DEFERRED | INTERVAL_COMPUTED
    interval: tuple[float, float] | None
    conditioning: str

    def __post_init__(self) -> None:
        if self.orientation != "corr":
            raise IPCASchemaError(
                f"effect {self.name!r}: orientation must be 'corr' (correction orientation); "
                f"got {self.orientation!r}"
            )
        if not self.name.endswith("_corr"):
            raise IPCASchemaError(f"effect name {self.name!r} must carry the _corr suffix (§2.2)")
        if self.interval_status not in (INTERVAL_DEFERRED, INTERVAL_COMPUTED, INTERVAL_REFUSED):
            raise IPCASchemaError(
                f"effect {self.name!r}: interval_status must be one of {INTERVAL_DEFERRED!r}, "
                f"{INTERVAL_COMPUTED!r}, {INTERVAL_REFUSED!r}; got {self.interval_status!r}"
            )
        if self.interval_status == INTERVAL_COMPUTED and self.interval is None:
            raise IPCASchemaError(
                f"effect {self.name!r}: interval_status is 'computed' but no interval was supplied"
            )
        if self.interval_status != INTERVAL_COMPUTED and self.interval is not None:
            raise IPCASchemaError(
                f"effect {self.name!r}: a non-'computed' status must not carry an interval"
            )
        if not self.conditioning or not self.conditioning.strip():
            raise IPCASchemaError(
                f"effect {self.name!r}: an effect estimate must carry a conditioning statement (§5.3)"
            )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "orientation": self.orientation,
            "interval_status": self.interval_status,
            "interval": list(self.interval) if self.interval is not None else None,
            "conditioning": self.conditioning,
        }


def deferred_effect(name: str, value: float) -> EffectEstimate:
    """A correction-oriented effect whose interval is deferred (no bootstrap requested), carrying
    the standard conditioning caveat."""
    return EffectEstimate(
        name=name, value=float(value), orientation="corr",
        interval_status=INTERVAL_DEFERRED, interval=None, conditioning=CONDITIONING_CAVEAT,
    )


def computed_effect(name: str, value: float, interval: tuple[float, float]) -> EffectEstimate:
    """A correction-oriented effect with a computed conditional bootstrap interval (§5.3)."""
    return EffectEstimate(
        name=name, value=float(value), orientation="corr",
        interval_status=INTERVAL_COMPUTED, interval=(float(interval[0]), float(interval[1])),
        conditioning=CONDITIONING_CAVEAT,
    )


def refused_effect(name: str, value: float) -> EffectEstimate:
    """A correction-oriented effect whose bootstrap interval was refused by the common support
    (block length incompatible, §6.2) — honest, not a fabricated CI."""
    return EffectEstimate(
        name=name, value=float(value), orientation="corr",
        interval_status=INTERVAL_REFUSED, interval=None, conditioning=CONDITIONING_CAVEAT,
    )


@dataclass(frozen=True)
class IPCADifferentialCell:
    """One realised cell Y_{P,Θ}: a panel state × a fitted-state provenance."""

    panel_state: str                   # "P_N" | "P_b"
    fitted_state: str                  # "Theta_N" | "Theta_b"
    label: str                         # "Y_NN" | "Y_Nb" | "Y_bN" | "Y_bb"
    value: float
    frozen_state_hash: str             # the fitted state's content hash (artefact identity)
    projection_diagnostics: Mapping[str, object]

    def to_dict(self) -> dict:
        return {
            "panel_state": self.panel_state,
            "fitted_state": self.fitted_state,
            "label": self.label,
            "value": self.value,
            "frozen_state_hash": self.frozen_state_hash,
            "projection_diagnostics": dict(self.projection_diagnostics),
        }


@dataclass(frozen=True)
class IPCADifferentialResult:
    """The frozen reporting set for one (bias, anchor) pair: four cells, the two data margins, the
    bracket (raw + DOE), the stored-but-not-headlined estimator margins and total, and the
    descriptive arm-matched diagonal. The algebraic identity is asserted at construction."""

    bias: str
    anchor: str
    is_focal: bool
    cells: tuple[IPCADifferentialCell, ...]                 # exactly four: Y_NN, Y_Nb, Y_bN, Y_bb
    data_margin_theta_n: EffectEstimate                     # Δ_data|Θ_N (headlined)
    data_margin_theta_b: EffectEstimate                     # Δ_data|Θ_b (headlined)
    interaction_bracket_raw: EffectEstimate                 # I (the primary interaction quantity)
    doe_interaction_effect: EffectEstimate                  # I/2 (for the Auditor's DOE tables)
    est_margin_p_n: EffectEstimate                          # Δ_est|P_N (stored, not headlined)
    est_margin_p_b: EffectEstimate                          # Δ_est|P_b (stored, not headlined)
    total: EffectEstimate                                   # Δ_total (stored, not headlined)
    secondary_endtoend: Mapping[str, float] | None          # arm-matched diagonal (one line) or None
    common_support_n_months: int
    claim_caveat: str = SINGLE_FREEZING_CAVEAT

    def __post_init__(self) -> None:
        if len(self.cells) != 4:
            raise IPCASchemaError(f"expected 4 cells, got {len(self.cells)}")
        by_label = {c.label: c.value for c in self.cells}
        for lbl in ("Y_NN", "Y_Nb", "Y_bN", "Y_bb"):
            if lbl not in by_label:
                raise IPCASchemaError(f"missing cell {lbl!r}")
        y_nn, y_nb, y_bn, y_bb = (by_label["Y_NN"], by_label["Y_Nb"],
                                  by_label["Y_bN"], by_label["Y_bb"])

        # Field-name discipline (§2.2): the two interaction fields never share a name.
        if self.interaction_bracket_raw.name == self.doe_interaction_effect.name:
            raise IPCASchemaError("interaction_bracket_raw and doe_interaction_effect share a name")

        # Algebraic identity (§2.3, the Theorem row) — asserted at serialization.
        bracket = y_nn - y_bn - y_nb + y_bb
        checks = {
            "bracket == DiD of cells": abs(self.interaction_bracket_raw.value - bracket),
            "bracket == d_data_tn - d_data_tb":
                abs(self.interaction_bracket_raw.value
                    - (self.data_margin_theta_n.value - self.data_margin_theta_b.value)),
            "bracket == d_est_pn - d_est_pb":
                abs(self.interaction_bracket_raw.value
                    - (self.est_margin_p_n.value - self.est_margin_p_b.value)),
            "doe == I/2":
                abs(self.doe_interaction_effect.value - self.interaction_bracket_raw.value / 2.0),
            "data_margin_theta_n == Y_NN - Y_bN": abs(self.data_margin_theta_n.value - (y_nn - y_bn)),
            "data_margin_theta_b == Y_Nb - Y_bb": abs(self.data_margin_theta_b.value - (y_nb - y_bb)),
            "total == Y_NN - Y_bb": abs(self.total.value - (y_nn - y_bb)),
            "total == d_data_tb + d_est_pn":
                abs(self.total.value - (self.data_margin_theta_b.value + self.est_margin_p_n.value)),
        }
        # NaN-tolerant: a refused/empty cell yields NaN; skip identities that involve NaN.
        for name, resid in checks.items():
            if resid == resid and resid > _ALGEBRA_TOL:   # resid==resid filters NaN
                raise IPCASchemaError(f"algebraic identity violated: {name} (residual {resid:.2e})")

    def to_dict(self) -> dict:
        return {
            "bias": self.bias,
            "anchor": self.anchor,
            "is_focal": self.is_focal,
            "cells": [c.to_dict() for c in self.cells],
            "data_margin_theta_n_corr": self.data_margin_theta_n.to_dict(),
            "data_margin_theta_b_corr": self.data_margin_theta_b.to_dict(),
            "interaction_bracket_raw": self.interaction_bracket_raw.to_dict(),
            "doe_interaction_effect": self.doe_interaction_effect.to_dict(),
            "est_margin_p_n_corr": self.est_margin_p_n.to_dict(),
            "est_margin_p_b_corr": self.est_margin_p_b.to_dict(),
            "total_corr": self.total.to_dict(),
            "secondary_endtoend": dict(self.secondary_endtoend) if self.secondary_endtoend else None,
            "common_support_n_months": self.common_support_n_months,
            "claim_caveat": self.claim_caveat,
        }
