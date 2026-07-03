"""
ConfigRefusal -- the structured, first-class "this strategy cannot be
configured / run" outcome.

It is NOT an exception: it is a *returned* value, recorded and aggregated
downstream for RQ2 coverage. Distinct from a ``ValueError`` /
``QuantConfigError`` raised on malformed agent output (an extraction bug): a
refusal means a legitimate strategy the engine stack cannot represent.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .provenance import Evidence


class RefusalCode(str, Enum):
    MISSING_BINDING = "MISSING_BINDING"                    # a required signal has no panel column
    OUT_OF_ENUM_WEIGHTING = "OUT_OF_ENUM_WEIGHTING"        # weighting the config cannot represent
    UNSUPPORTED_TRIM_VARIANT = "UNSUPPORTED_TRIM_VARIANT"  # trim the engine cannot express
    UNSUPPORTED_COMBINATION = "UNSUPPORTED_COMBINATION"    # e.g. control + holding_period > 1
    # ASSUMPTION_MISMATCH is reserved for the future assumptions ledger (out of
    # scope for this component).


@dataclass(frozen=True)
class ConfigRefusal:
    strategy_id: str
    code: RefusalCode
    field: str | None
    detail: str
    evidence: Evidence | None = None

    def to_dict(self) -> dict:
        """JSON/YAML-safe dict for RQ2 aggregation (tuples handled by
        ``Evidence.to_dict``)."""
        out: dict = {
            "strategy_id": self.strategy_id,
            "code": self.code.value,
            "field": self.field,
            "detail": self.detail,
        }
        if self.evidence is not None:
            out["evidence"] = self.evidence.to_dict()
        return out
