"""
The deterministic adapter (D25-D28): StrategySpec (paper language) -> per-leg
``build_quant_config`` calls + a combiner instruction.

The one component that speaks engine coordinates (D3); the Librarian never does.
Pure + deterministic (no LLM): vocabulary + structure + signal-resolution
translations, each one documented hop from a paper quote (D24). Representability
refusals stay the factory's; the adapter owns silence routing (D26), the leg loop /
combiner (D28), and run-to-completion refusal collection (D29).
"""

from __future__ import annotations

from .adapt import adapt_spec
from .authorisation import (
    AuthorisationRecords,
    BindingSubstitution,
    FieldOverride,
    load_authorisation_records,
)
from .result import AdaptResult, CombinerInstruction, LegCall
from .transform_table import load_transform_table

__all__ = [
    "adapt_spec",
    "AdaptResult",
    "LegCall",
    "CombinerInstruction",
    "AuthorisationRecords",
    "BindingSubstitution",
    "FieldOverride",
    "load_authorisation_records",
    "load_transform_table",
]
