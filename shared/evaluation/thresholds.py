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
