"""
shared/evaluation — evaluation-layer diagnostics shared by the Auditor and the
Scientist's Experimentalist (Scientist spec §10.2 / deviation D4). Not an agent.

Two generations of diagnostics live here:

  * The Layer-1 crowding survivor diagnostic (`crowding`, `crowding_l2`) — the flat-dict
    factor-spanning measurement already wired into the Scientist's G4. Unchanged.

  * The typed shared economic diagnostics (spec §3-§10; supersedes the Devil's Advocate
    agent, D4): `contracts` (typed results + refusal codes), `alignment`, `spanning`
    (a typed superset of `crowding` — a characterisation test pins their equivalence),
    `costs`, `regimes`, `reporting_rules`, and the `orchestrator.evaluate` entry point.
    No function here returns a pass/fail verdict — the callers may not branch on a
    diagnostic (D-E7).
"""

# --- Layer-1 crowding (unchanged) ---
from .crowding import crowding_diagnostic, load_crowding_factor_bundle
from .thresholds import CrowdingConfig, CrowdingThresholdError, load_crowding_config

# --- Typed shared economic diagnostics ---
from .contracts import (
    BreakEvenStatus,
    CostResult,
    CostScenarioResult,
    CostUnit,
    EstimatorProvenance,
    EvaluationScope,
    PairedDifference,
    PositionArtefactCapability,
    PredictionAssessment,
    RefusalCode,
    RegimeResult,
    RegimeRole,
    SampleAlignment,
    SampleWindow,
    SharedEvaluationResult,
    SpanningResult,
    TurnoverMethod,
    WeightingScheme,
)
from .alignment import align_monthly, paired_difference
from .spanning import spanning_regression
from .costs import (
    break_even_alpha,
    break_even_grid,
    break_even_mean,
    cost_adjusted_series,
    cost_result_from_turnover,
    evaluate_costs,
    lambda_break_even,
    probe_position_artefacts,
)
from .regimes import (
    evaluate_regimes,
    expanding_past_only_median,
    prediction_contrast,
)
from .reporting_rules import (
    FORBIDDEN_STRINGS,
    cost_sentence,
    hit_count_sentence,
    regime_sentence,
    spanning_sentence,
)
from .orchestrator import CODE_VERSION, evaluate
from .thresholds import (
    CostScenarioSpec,
    CostsConfig,
    RegimesConfig,
    SharedEvalThresholdError,
    load_costs_config,
    load_regimes_config,
)

__all__ = [
    # Layer-1 crowding
    "crowding_diagnostic",
    "load_crowding_factor_bundle",
    "CrowdingConfig",
    "CrowdingThresholdError",
    "load_crowding_config",
    # entry points
    "evaluate",
    "spanning_regression",
    "evaluate_costs",
    "cost_result_from_turnover",
    "evaluate_regimes",
    "align_monthly",
    "paired_difference",
    "prediction_contrast",
    "expanding_past_only_median",
    "probe_position_artefacts",
    # cost math
    "cost_adjusted_series",
    "break_even_mean",
    "break_even_alpha",
    "lambda_break_even",
    "break_even_grid",
    # sentences
    "spanning_sentence",
    "cost_sentence",
    "regime_sentence",
    "hit_count_sentence",
    "FORBIDDEN_STRINGS",
    # result types
    "SharedEvaluationResult",
    "SpanningResult",
    "CostResult",
    "CostScenarioResult",
    "RegimeResult",
    "PredictionAssessment",
    "SampleAlignment",
    "SampleWindow",
    "PairedDifference",
    "EstimatorProvenance",
    "PositionArtefactCapability",
    # enums
    "RefusalCode",
    "EvaluationScope",
    "TurnoverMethod",
    "WeightingScheme",
    "CostUnit",
    "BreakEvenStatus",
    "RegimeRole",
    # configs / loaders
    "CostsConfig",
    "RegimesConfig",
    "CostScenarioSpec",
    "SharedEvalThresholdError",
    "load_costs_config",
    "load_regimes_config",
    "CODE_VERSION",
]
