"""
thresholds.py — typed, FAIL-LOUD loader for the Layer-1 crowding diagnostic's
pre-registration constants (docs/thresholds.yaml `crowding:` block).

Mirrors agents/auditor/thresholds.py: the crowding factor set, HAC lag rule, and
minimum-overlap floor are pre-registered by the researcher and committed to
docs/thresholds.yaml, NOT invented in code. Every loader raises
`CrowdingThresholdError` (a KeyError subclass) rather than defaulting — a silent
default would launder a post-hoc constant, the exact sin the pipeline exists to
expose. For testability `load_crowding_config` accepts an explicit `path`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from .contracts import CostUnit

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"

# The two HAC lag rules the diagnostic understands. `floor_t_pow_0.25` is the
# pre-registered default (the same rule the primary alpha-vs-BBW-4 test uses);
# `newey_west_auto` is the characteristic-sort engine's native
# floor(4*(T/100)^(2/9)), retained only as an alternative. Any other value is
# rejected — the rule is a pre-registered choice, never guessed.
VALID_HAC_RULES = ("floor_t_pow_0.25", "newey_west_auto")


class CrowdingThresholdError(KeyError):
    """A required crowding pre-registration constant is absent from or malformed in
    docs/thresholds.yaml. Raised rather than defaulted — a silent default would
    launder a post-hoc constant."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            f"crowding pre-registration constant {detail} in {THRESHOLDS_FILE.name}. "
            f"It must be pre-registered under the top-level `crowding:` block — the "
            f"diagnostic never defaults it."
        )


@dataclass(frozen=True)
class CrowdingConfig:
    """Layer-1 crowding constants (docs/thresholds.yaml `crowding:` block).

    `hac_lag_rule` is kept as the RULE NAME, not a pre-resolved lag count: the
    `floor_t_pow_0.25` rule depends on the realised regression sample T, which is
    only known at call time (after the candidate/factor overlap is formed). The
    diagnostic resolves (rule, T) -> nw_lags in `crowding.py`.
    """

    factor_set: tuple[str, ...]
    hac_lag_rule: str
    min_obs: int
    bundles: Mapping[str, tuple[str, str]]   # logical name -> (parquet path, _corr column)


def _bundles_from(raw: object) -> dict[str, tuple[str, str]]:
    if not isinstance(raw, dict) or not raw:
        raise CrowdingThresholdError("`crowding.bundles` (missing or not a non-empty mapping)")
    out: dict[str, tuple[str, str]] = {}
    for name, spec in raw.items():
        if not isinstance(spec, dict) or "path" not in spec or "column" not in spec:
            raise CrowdingThresholdError(
                f"`crowding.bundles.{name}` (must be a mapping with `path` and `column`)"
            )
        path, column = spec["path"], spec["column"]
        if not isinstance(path, str) or not path.strip():
            raise CrowdingThresholdError(f"`crowding.bundles.{name}.path` (not a non-empty string)")
        if not isinstance(column, str) or not column.strip():
            raise CrowdingThresholdError(f"`crowding.bundles.{name}.column` (not a non-empty string)")
        out[str(name)] = (path, column)
    return out


def load_crowding_config(path: str | Path | None = None) -> CrowdingConfig:
    """Read the `crowding:` block fail-loud. Raises `CrowdingThresholdError` on any
    missing or malformed key."""
    p = Path(path) if path is not None else THRESHOLDS_FILE
    with open(p) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise CrowdingThresholdError("(docs/thresholds.yaml is empty or malformed)")
    block = data.get("crowding")
    if not isinstance(block, dict):
        raise CrowdingThresholdError("`crowding` (the whole crowding: block is missing)")

    factor_set = block.get("factor_set")
    if (
        not isinstance(factor_set, list)
        or not factor_set
        or not all(isinstance(x, str) and x.strip() for x in factor_set)
    ):
        raise CrowdingThresholdError("`crowding.factor_set` (not a non-empty list of names)")

    hac_rule = block.get("hac_lag_rule")
    if hac_rule not in VALID_HAC_RULES:
        raise CrowdingThresholdError(
            f"`crowding.hac_lag_rule` (must be one of {VALID_HAC_RULES}; got {hac_rule!r})"
        )

    min_obs = block.get("min_obs")
    if isinstance(min_obs, bool) or not isinstance(min_obs, int) or min_obs < 1:
        raise CrowdingThresholdError(f"`crowding.min_obs` (must be a positive int; got {min_obs!r})")

    bundles = _bundles_from(block.get("bundles"))

    # Every factor in the set must have a bundle source (else the assembly would
    # silently drop a pre-registered factor).
    missing = [f for f in factor_set if f not in bundles]
    if missing:
        raise CrowdingThresholdError(f"`crowding.bundles` is missing sources for factor(s) {missing}")

    return CrowdingConfig(
        factor_set=tuple(factor_set),
        hac_lag_rule=hac_rule,
        min_obs=int(min_obs),
        bundles=bundles,
    )


# ===========================================================================
# Cost / regime pre-registration constants (docs/thresholds.yaml
# `shared_evaluation:` block — a 14th additive sibling block, invisible to the
# existing loaders, exactly as the `reporter:` block established). These loaders are
# ADDITIVE: CrowdingConfig / load_crowding_config above are untouched. Spanning
# constants stay in the `crowding:` block; this block adds only the cost scenarios
# and the regime evaluation/contrast constants (spec §6, §7).
# ===========================================================================

EXTENSION_1_FILE = REPO_ROOT / "docs" / "extension_1_config.yaml"

_UNRESOLVED_SENTINEL = "UNRESOLVED_pending_citation"


class SharedEvalThresholdError(KeyError):
    """A required cost/regime pre-registration constant is absent from or malformed in
    docs/thresholds.yaml (`shared_evaluation:` block). Raised rather than defaulted — a
    silent default would launder a post-hoc constant."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            f"shared_evaluation pre-registration constant {detail} in "
            f"{THRESHOLDS_FILE.name}. It must be pre-registered under the top-level "
            f"`shared_evaluation:` block — the diagnostic never defaults it."
        )


def _cost_unit(raw: object, where: str) -> CostUnit:
    if raw == "one_way":
        return CostUnit.ONE_WAY
    if raw == "round_trip":
        return CostUnit.ROUND_TRIP
    raise SharedEvalThresholdError(f"`{where}` (must be 'one_way' or 'round_trip'; got {raw!r})")


def _number(raw: object, where: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise SharedEvalThresholdError(f"`{where}` (must be a number; got {raw!r})")
    return float(raw)


@dataclass(frozen=True)
class CostScenarioSpec:
    """One registered cost scenario (D-E12). `usable` is derived: a scenario whose
    source is empty or the UNRESOLVED sentinel is NOT usable and is never emitted as a
    cost figure until a real citation lands (P5)."""

    scenario_id: str
    ig_bps: float
    hy_bps: float
    unit: CostUnit
    source: str
    usable: bool


@dataclass(frozen=True)
class CostsConfig:
    scenarios: tuple[CostScenarioSpec, ...]
    break_even_alpha_denominator: str
    assumed_turnover_grid: tuple[float, ...]
    gross_exposure_convention: int

    def scenario(self, scenario_id: str) -> CostScenarioSpec:
        for s in self.scenarios:
            if s.scenario_id == scenario_id:
                return s
        raise SharedEvalThresholdError(f"`shared_evaluation.costs.scenarios.{scenario_id}` (not registered)")


@dataclass(frozen=True)
class RegimesConfig:
    evaluation_median: float            # frozen dev median, RESOLVED from extension_1
    evaluation_median_source: str       # "<file>#<dotted key>", recorded for provenance
    macro_data_contract_id: str
    min_obs_conditional: int


def _resolve_dotted(doc: object, dotted: str, where: str):
    cur = doc
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise SharedEvalThresholdError(f"`{where}` reference {dotted!r} is unresolved")
        cur = cur[part]
    return cur


def _shared_eval_block(path: str | Path | None) -> dict:
    p = Path(path) if path is not None else THRESHOLDS_FILE
    with open(p) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SharedEvalThresholdError("(docs/thresholds.yaml is empty or malformed)")
    block = data.get("shared_evaluation")
    if not isinstance(block, dict):
        raise SharedEvalThresholdError("`shared_evaluation` (the whole block is missing)")
    return block


def load_costs_config(path: str | Path | None = None) -> CostsConfig:
    """Read the `shared_evaluation.costs` block fail-loud."""
    costs = _shared_eval_block(path).get("costs")
    if not isinstance(costs, dict):
        raise SharedEvalThresholdError("`shared_evaluation.costs` (missing or not a mapping)")

    raw_scenarios = costs.get("scenarios")
    if not isinstance(raw_scenarios, dict) or not raw_scenarios:
        raise SharedEvalThresholdError("`shared_evaluation.costs.scenarios` (missing or empty mapping)")
    scenarios: list[CostScenarioSpec] = []
    for sid, spec in raw_scenarios.items():
        if not isinstance(spec, dict):
            raise SharedEvalThresholdError(f"`shared_evaluation.costs.scenarios.{sid}` (not a mapping)")
        where = f"shared_evaluation.costs.scenarios.{sid}"
        source = spec.get("source")
        if not isinstance(source, str) or not source.strip():
            raise SharedEvalThresholdError(f"`{where}.source` (must be a non-empty string)")
        usable = source.strip() != _UNRESOLVED_SENTINEL
        scenarios.append(
            CostScenarioSpec(
                scenario_id=str(sid),
                ig_bps=_number(spec.get("ig_bps"), f"{where}.ig_bps"),
                hy_bps=_number(spec.get("hy_bps"), f"{where}.hy_bps"),
                unit=_cost_unit(spec.get("unit"), f"{where}.unit"),
                source=source.strip(),
                usable=usable,
            )
        )

    be = costs.get("break_even")
    if not isinstance(be, dict):
        raise SharedEvalThresholdError("`shared_evaluation.costs.break_even` (missing or not a mapping)")
    denom = be.get("alpha_denominator")
    if not isinstance(denom, str) or not denom.strip():
        raise SharedEvalThresholdError("`shared_evaluation.costs.break_even.alpha_denominator` (non-empty string)")
    grid_raw = be.get("assumed_turnover_grid")
    if not isinstance(grid_raw, list) or not grid_raw or not all(
        (not isinstance(x, bool)) and isinstance(x, (int, float)) and x > 0 for x in grid_raw
    ):
        raise SharedEvalThresholdError(
            "`shared_evaluation.costs.break_even.assumed_turnover_grid` (non-empty list of positive numbers)"
        )

    gec = costs.get("gross_exposure_convention")
    if isinstance(gec, bool) or gec not in (1, 2):
        raise SharedEvalThresholdError("`shared_evaluation.costs.gross_exposure_convention` (must be 1 or 2)")

    return CostsConfig(
        scenarios=tuple(scenarios),
        break_even_alpha_denominator=denom.strip(),
        assumed_turnover_grid=tuple(float(x) for x in grid_raw),
        gross_exposure_convention=int(gec),
    )


def load_regimes_config(
    path: str | Path | None = None, *, extension_1_path: str | Path | None = None
) -> RegimesConfig:
    """Read the `shared_evaluation.regimes` block fail-loud, RESOLVING the evaluation
    median from extension_1_config.yaml (referenced, never restated — spec §7 / D-E14)."""
    regimes = _shared_eval_block(path).get("regimes")
    if not isinstance(regimes, dict):
        raise SharedEvalThresholdError("`shared_evaluation.regimes` (missing or not a mapping)")

    macro_id = regimes.get("macro_data_contract_id")
    if not isinstance(macro_id, str) or not macro_id.strip():
        raise SharedEvalThresholdError("`shared_evaluation.regimes.macro_data_contract_id` (non-empty string)")

    moc = regimes.get("min_obs_conditional")
    if isinstance(moc, bool) or not isinstance(moc, int) or moc < 1:
        raise SharedEvalThresholdError("`shared_evaluation.regimes.min_obs_conditional` (positive int)")

    src = regimes.get("evaluation_median_source")
    if not isinstance(src, dict) or "file" not in src or "key" not in src:
        raise SharedEvalThresholdError(
            "`shared_evaluation.regimes.evaluation_median_source` (mapping with `file` and `key`)"
        )
    ext_path = Path(extension_1_path) if extension_1_path is not None else REPO_ROOT / str(src["file"])
    try:
        with open(ext_path) as f:
            ext_doc = yaml.safe_load(f)
    except FileNotFoundError as exc:
        raise SharedEvalThresholdError(
            f"`shared_evaluation.regimes.evaluation_median_source.file` points at "
            f"{src['file']!r}, which does not exist"
        ) from exc
    median = _resolve_dotted(ext_doc, str(src["key"]), "shared_evaluation.regimes.evaluation_median_source.key")
    if isinstance(median, bool) or not isinstance(median, (int, float)):
        raise SharedEvalThresholdError(
            f"resolved evaluation median from {src['file']}#{src['key']} is not a number (got {median!r})"
        )

    return RegimesConfig(
        evaluation_median=float(median),
        evaluation_median_source=f"{src['file']}#{src['key']}",
        macro_data_contract_id=macro_id.strip(),
        min_obs_conditional=int(moc),
    )


# ---------------------------------------------------------------------------
# SC-SCI-13 holdout block-bootstrap diagnostic constants (docs/thresholds.yaml
# `auditor.bootstrap` block). B and B_min are REUSED from the Auditor's bootstrap
# (single source of truth); only the labelled-diagnostic block-length pair is new.
# Only the one-shot holdout caller (agents/scientist/experimentalist/oneshot_holdout/stage2_evaluate.py,
# built + wired, not yet executed) reads this; unit tests pass literals.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HoldoutBootstrapDiagnosticConfig:
    """Constants for the SC-SCI-13 labelled-diagnostic holdout bootstrap. Read fail-loud;
    never defaulted (a silent default would let a pre-registered constant drift)."""

    block_lengths: tuple[int, ...]   # auditor.bootstrap.holdout_diagnostic_block_lengths_months
    n_replicates: int                # REUSED: auditor.bootstrap.n_replicates
    min_effective_blocks: int        # REUSED: auditor.bootstrap.min_effective_blocks


def load_holdout_bootstrap_diagnostic_config(
    path: str | Path | None = None,
) -> HoldoutBootstrapDiagnosticConfig:
    """Read the SC-SCI-13 holdout-diagnostic constants from the `auditor.bootstrap` block
    fail-loud. B (`n_replicates`) and B_min (`min_effective_blocks`) are reused from that
    same block so there is one source of truth for the pre-registered bootstrap constants."""
    p = Path(path) if path is not None else THRESHOLDS_FILE
    with open(p) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SharedEvalThresholdError("(docs/thresholds.yaml is empty or malformed)")
    boot = _resolve_dotted(data, "auditor.bootstrap", "auditor.bootstrap")
    if not isinstance(boot, dict):
        raise SharedEvalThresholdError("`auditor.bootstrap` (missing or not a mapping)")

    lengths = boot.get("holdout_diagnostic_block_lengths_months")
    if (
        not isinstance(lengths, list)
        or not lengths
        or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in lengths)
    ):
        raise SharedEvalThresholdError(
            "`auditor.bootstrap.holdout_diagnostic_block_lengths_months` (non-empty list of positive ints)"
        )
    for key in ("n_replicates", "min_effective_blocks"):
        val = boot.get(key)
        if isinstance(val, bool) or not isinstance(val, int) or val < 1:
            raise SharedEvalThresholdError(f"`auditor.bootstrap.{key}` (positive int)")

    return HoldoutBootstrapDiagnosticConfig(
        block_lengths=tuple(int(x) for x in lengths),
        n_replicates=int(boot["n_replicates"]),
        min_effective_blocks=int(boot["min_effective_blocks"]),
    )
