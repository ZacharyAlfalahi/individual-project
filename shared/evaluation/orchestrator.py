"""
orchestrator.py — single entry point for the shared economic diagnostics
(shared/evaluation, spec §10).

Pure, I/O-free composition of spanning + costs + regimes + alignment into one
`SharedEvaluationResult`. Fails loud on a missing config (the sub-loaders raise a
KeyError subclass rather than defaulting); stamps `config_hash` and `code_version` on
every result. Blocked sub-results carry REAL typed refusals (the cost turnover stub, a
not-applicable regime), so the end-to-end typed flow is exercised now — before any
Auditor/Scientist wiring — and the eventual plug-ins are drop-in.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from . import costs as costs_mod
from . import regimes as regimes_mod
from . import spanning as spanning_mod
from .alignment import align_monthly
from .contracts import EvaluationScope, SharedEvaluationResult, WeightingScheme
from .thresholds import (
    CostsConfig,
    CrowdingConfig,
    RegimesConfig,
    load_costs_config,
    load_crowding_config,
    load_regimes_config,
)

CODE_VERSION = "shared_evaluation@0.1.0"


def _config_hash(
    crowding: CrowdingConfig,
    costs: CostsConfig,
    regimes: RegimesConfig,
    stage: str,
    scope: EvaluationScope,
) -> str:
    payload = json.dumps(
        {
            "crowding_factor_set": list(crowding.factor_set),
            "crowding_hac_lag_rule": crowding.hac_lag_rule,
            "crowding_min_obs": crowding.min_obs,
            "costs_scenarios": [
                [s.scenario_id, s.ig_bps, s.hy_bps, s.unit.value, s.usable]
                for s in costs.scenarios
            ],
            "costs_break_even_denom": costs.break_even_alpha_denominator,
            "regimes_evaluation_median": regimes.evaluation_median,
            "regimes_min_obs_conditional": regimes.min_obs_conditional,
            "stage": stage,
            "scope": scope.value,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def evaluate(
    candidate_returns: pd.Series,
    *,
    stage: str,
    candidate_id: str,
    parent_id: str | None = None,
    parent_returns: pd.Series | None = None,
    spread: pd.Series | None = None,
    run_artefact=None,
    crowding_config: CrowdingConfig | None = None,
    crowding_factors: pd.DataFrame | None = None,
    costs_config: CostsConfig | None = None,
    regimes_config: RegimesConfig | None = None,
    predicted_sign: int | None = None,
    mechanically_implied: bool = False,
    weighting_scheme: WeightingScheme = WeightingScheme.PAR,
    scope: EvaluationScope = EvaluationScope.SUPPLEMENTARY,
) -> SharedEvaluationResult:
    """Compose the three diagnostics for one candidate.

    `candidate_returns` is a monthly (month-end) return Series. `parent_returns` /
    `spread` / `run_artefact` enable the alignment, regime, and cost blocks respectively;
    omitting any yields a typed not-applicable / refusal sub-result, never a crash.
    Configs default to the pre-registered fail-loud loaders when omitted.
    """
    crowding_config = crowding_config or load_crowding_config()
    costs_config = costs_config or load_costs_config()
    regimes_config = regimes_config or load_regimes_config()

    spanning = spanning_mod.spanning_regression(
        candidate_returns, config=crowding_config, factors=crowding_factors, scope=scope
    )

    # Costs: turnover is not sourceable from the current artefact -> typed refusal (P4).
    costs = costs_mod.evaluate_costs(run_artefact, weighting_scheme=weighting_scheme, scope=scope)

    # Regimes: evaluation-regime decomposition + prediction contrast when a spread series
    # is supplied; otherwise a real not-applicable result.
    if spread is not None:
        regimes = regimes_mod.evaluate_regimes(
            candidate_returns,
            spread,
            parent_returns=parent_returns,
            config=regimes_config,
            predicted_sign=predicted_sign,
            mechanically_implied=mechanically_implied,
            scope=scope,
        )
    else:
        regimes = regimes_mod.not_applicable_result(
            config=regimes_config, scope=scope, mechanically_implied=mechanically_implied
        )

    # Alignment: month-level pairing of the candidate against the corrected parent.
    alignment = (
        align_monthly(candidate_returns, parent_returns, a_label=candidate_id, b_label=parent_id or "parent")
        if parent_returns is not None
        else None
    )

    return SharedEvaluationResult(
        candidate_id=candidate_id,
        parent_id=parent_id,
        stage=stage,
        alignment=alignment,
        spanning=spanning,
        costs=costs,
        regimes=regimes,
        config_hash=_config_hash(crowding_config, costs_config, regimes_config, stage, scope),
        code_version=CODE_VERSION,
    )
