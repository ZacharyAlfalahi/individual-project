"""
QuantConfig -- typed, provenance-carrying configuration layer for the frozen
characteristic-sort engine (``agents/quant/library``). It sits ABOVE the library
(the "LLM configures, never modifies" boundary): the library is imported and
configured here, never the reverse.
"""

from __future__ import annotations

from .provenance import (
    Binding,
    BindingTag,
    Evidence,
    Inherited,
    InheritedTag,
    Locator,
    ProvenanceError,
)
from .quant_config import QuantConfig, QuantConfigError, build_quant_config, to_rulebook
from .ledger_check import (
    LedgerCheckError,
    LedgerCheckRow,
    LedgerCheckTable,
    check_assumptions,
    load_ledger_check_table,
)
from .refusal import ConfigRefusal, RefusalCode
from .runner import StrategyResult, run_from_config, run_strategy
from .trim_rule import TrimMethod, TrimRule

__all__ = [
    "Binding",
    "BindingTag",
    "Evidence",
    "Inherited",
    "InheritedTag",
    "Locator",
    "ProvenanceError",
    "TrimMethod",
    "TrimRule",
    "ConfigRefusal",
    "RefusalCode",
    "QuantConfig",
    "QuantConfigError",
    "build_quant_config",
    "to_rulebook",
    "run_from_config",
    "run_strategy",
    "StrategyResult",
    "check_assumptions",
    "load_ledger_check_table",
    "LedgerCheckTable",
    "LedgerCheckRow",
    "LedgerCheckError",
]
