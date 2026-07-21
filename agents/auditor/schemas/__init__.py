"""Auditor schemas subpackage — the frozen data contracts."""

from __future__ import annotations

from .audit_core import (
    AuditCore,
    AuditScope,
    InvarianceResult,
    SupportInfo,
    conditioning_statement_for,
)
from .decomposition import (
    CornerMarginals,
    SaturatedBasis,
    ShapleyResult,
    subset_label,
)
from .lattice_types import (
    METRIC_NAMES,
    CellReturns,
    LatticeResult,
    MetricSet,
)
from .toggle import (
    RUNNABLE_REASONS,
    TOGGLE_AXES,
    TOGGLE_IDS,
    ToggleAxis,
    ToggleFacts,
    ToggleId,
    ToggleState,
)

__all__ = [
    "AuditCore",
    "AuditScope",
    "InvarianceResult",
    "SupportInfo",
    "conditioning_statement_for",
    "CornerMarginals",
    "SaturatedBasis",
    "ShapleyResult",
    "subset_label",
    "METRIC_NAMES",
    "CellReturns",
    "LatticeResult",
    "MetricSet",
    "RUNNABLE_REASONS",
    "TOGGLE_AXES",
    "TOGGLE_IDS",
    "ToggleAxis",
    "ToggleFacts",
    "ToggleId",
    "ToggleState",
]
