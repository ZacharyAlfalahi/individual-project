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
    # --- additive members (D29). These
    #     are ADAPTER-side refusal codes: the factory above never emits them (it
    #     owns only the four engine-representability refusals). Adding them is an
    #     explicit, bounded reopening of frozen code -- additive enum members only;
    #     no existing code path, value, or refusal behaviour changes; guarded by
    #     the factory suite passing unmodified plus test_refusal_code_additive.py
    #     (existing refusal fixtures serialise byte-identically). -----------------
    ASSUMPTION_MISMATCH = "ASSUMPTION_MISMATCH"    # promoted from comment (D29); reserved for the ledger check
    REVIEW_REQUIRED = "REVIEW_REQUIRED"            # adapter: a field needs the manual-review lane
                                                   # (dual-model disagreement / quote-match failure /
                                                   # single response / a flagged STATED value)
    REFUSED_ON_SILENCE = "REFUSED_ON_SILENCE"      # adapter: paper silent on a load-bearing field (D26)
    UNSUPPORTED_COMBINER = "UNSUPPORTED_COMBINER"  # adapter: combiner='other' (D28, adapter-owned)


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
