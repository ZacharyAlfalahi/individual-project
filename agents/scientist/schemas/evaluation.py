"""
evaluation.py — `Booleans`, `Measurements`, `EvaluationRecord` (spec §5.4).

One immutable `EvaluationRecord` per proposal. Its `final_outcome` is DERIVED from the twelve
booleans by `derive_outcome` (Appendix B) and exposed as a read-only property, so it can never
be assigned by a human or a model (INVARIANT 4 / D5). `measurements` are populated
progressively by gates G3–G6; every sub-field is optional and a gate that never ran leaves its
field None — a proposal that failed an earlier gate has NO performance computed (prohibition 9).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .outcomes import Outcome, RefusalCode

# The twelve booleans, in gate order G0->G6. `derive_outcome` reads them in this precedence.
BOOLEAN_FIELDS: tuple[str, ...] = (
    "schema_valid",
    "mechanism_authorised",
    "template_supported",
    "toggles_preserved",
    "inputs_available",
    "not_duplicate",
    "compiled",
    "execution_verified",
    "audit_clean",
    "bh_survived",
    "cpcv_qualified",
    "holdout_evaluated",
)


@dataclass(frozen=True)
class Booleans:
    """The twelve named gate booleans (§5.4). Order mirrors the gate stack G0->G6; consumed in
    that precedence by `derive_outcome` (first-match-wins). Note `holdout_evaluated` records
    whether the holdout was actually evaluated and is NOT read by `derive_outcome` — the
    HOLDOUT_EVALUATED outcome means "advanced to holdout" and follows from cpcv_qualified."""

    schema_valid: bool
    mechanism_authorised: bool
    template_supported: bool
    toggles_preserved: bool
    inputs_available: bool
    not_duplicate: bool
    compiled: bool
    execution_verified: bool
    audit_clean: bool
    bh_survived: bool
    cpcv_qualified: bool
    holdout_evaluated: bool

    def __post_init__(self) -> None:
        for name in BOOLEAN_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise TypeError(f"Booleans.{name} must be a bool; got {value!r}")

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in BOOLEAN_FIELDS}


@dataclass(frozen=True)
class GrossMeasurements:
    """Gross (pre-cost) primary-inference measurements (§5.4). All optional — populated at G3+."""

    mean_return: float | None = None
    sharpe: float | None = None
    sharpe_ci95: tuple[float, float] | None = None
    alpha_bbw4: float | None = None
    t_stat: float | None = None
    p_raw: float | None = None
    p_bh: float | None = None
    bh_rejected: bool | None = None    # two-sided BH rejection (SC-SCI-8) — beside the sign-aware
                                       #   `bh_survived` boolean; a rejected-but-wrong-signed
                                       #   candidate is bh_rejected=True yet bh_survived=False.

    def to_dict(self) -> dict:
        return {
            "mean_return": self.mean_return,
            "sharpe": self.sharpe,
            "sharpe_ci95": list(self.sharpe_ci95) if self.sharpe_ci95 is not None else None,
            "alpha_bbw4": self.alpha_bbw4,
            "t_stat": self.t_stat,
            "p_raw": self.p_raw,
            "p_bh": self.p_bh,
            "bh_rejected": self.bh_rejected,
        }


@dataclass(frozen=True)
class NetMeasurements:
    """Net-of-cost measurements — reported BESIDE every gross figure, never instead (§10.2)."""

    sharpe_net: float | None = None
    mean_return_net: float | None = None

    def to_dict(self) -> dict:
        return {"sharpe_net": self.sharpe_net, "mean_return_net": self.mean_return_net}


@dataclass(frozen=True)
class Measurements:
    """The measurements block (§5.4). A lean, all-optional container populated progressively by
    gates G3–G6; the structured sub-blocks (crowding, cpcv, paired/descriptive comparisons) are
    immutable mappings the later gates fill in. Nothing here is read by `derive_outcome` — the
    outcome derives from booleans only."""

    gross: GrossMeasurements | None = None
    net: NetMeasurements | None = None
    turnover_monthly_mean: float | None = None
    crowding: Mapping[str, float] | None = None
    cpcv: Mapping[str, float] | None = None
    deflated_sharpe: float | None = None
    paired_vs_corrected_parent: Mapping[str, float] | None = None
    descriptive_vs_as_published: Mapping[str, float] | None = None

    def to_dict(self) -> dict:
        return {
            "gross": self.gross.to_dict() if self.gross is not None else None,
            "net": self.net.to_dict() if self.net is not None else None,
            "turnover_monthly_mean": self.turnover_monthly_mean,
            "crowding": dict(self.crowding) if self.crowding is not None else None,
            "cpcv": dict(self.cpcv) if self.cpcv is not None else None,
            "deflated_sharpe": self.deflated_sharpe,
            "paired_vs_corrected_parent": (
                dict(self.paired_vs_corrected_parent)
                if self.paired_vs_corrected_parent is not None
                else None
            ),
            "descriptive_vs_as_published": (
                dict(self.descriptive_vs_as_published)
                if self.descriptive_vs_as_published is not None
                else None
            ),
        }


@dataclass(frozen=True)
class EvaluationRecord:
    """One immutable evaluation record per proposal (§5.4). `final_outcome` is a DERIVED,
    read-only property — there is deliberately no way to assign it (INVARIANT 4 / D5)."""

    proposal_id: str
    booleans: Booleans
    refusal_code: RefusalCode | None = None
    measurements: Measurements = field(default_factory=Measurements)

    @property
    def final_outcome(self) -> Outcome:
        """DERIVED from `booleans` via Appendix B; never assigned (INVARIANT 4 / D5). The
        import is deferred to call time to keep the schemas package free of a load-order cycle
        with the experimentalist (which imports these schemas)."""
        from ..experimentalist.outcome import derive_outcome

        return derive_outcome(self.booleans)

    def to_dict(self) -> dict:
        return {
            "proposal_id": self.proposal_id,
            "booleans": self.booleans.to_dict(),
            "refusal_code": self.refusal_code.value if self.refusal_code is not None else None,
            "measurements": self.measurements.to_dict(),
            "final_outcome": self.final_outcome.value,  # DERIVED — see property
        }
