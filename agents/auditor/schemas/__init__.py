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
    BiasClassPartition,
    CornerMarginals,
    SaturatedBasis,
    ShapleyResult,
    label_to_subset,
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
    TOGGLE_BIAS_CLASS,
    TOGGLE_IDS,
    BiasClass,
    DummyReason,
    ToggleAxis,
    ToggleFacts,
    ToggleId,
    ToggleState,
    classify_dummy,
    construction_toggles,
    data_quality_toggles,
)

__all__ = [
    "AuditCore",
    "AuditScope",
    "InvarianceResult",
    "SupportInfo",
    "conditioning_statement_for",
    "BiasClassPartition",
    "CornerMarginals",
    "SaturatedBasis",
    "ShapleyResult",
    "label_to_subset",
    "subset_label",
    "METRIC_NAMES",
    "CellReturns",
    "LatticeResult",
    "MetricSet",
    "RUNNABLE_REASONS",
    "TOGGLE_AXES",
    "TOGGLE_BIAS_CLASS",
    "TOGGLE_IDS",
    "BiasClass",
    "DummyReason",
    "ToggleAxis",
    "ToggleFacts",
    "ToggleId",
    "ToggleState",
    "classify_dummy",
    "construction_toggles",
    "data_quality_toggles",
]
