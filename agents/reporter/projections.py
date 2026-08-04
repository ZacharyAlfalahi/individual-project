"""projections.py — canonical projections for types with no usable to_dict (§8.4, C6).

From PERSISTED artefacts the Reporter resolves json_pointers directly against the parsed dicts:
the quant `<anchor>.json` is already a summary-bearing dict, and the audit `<anchor>_core.json`
is `AuditCore.to_dict()`. Projections exist for the other case — a claim whose source is an
IN-MEMORY typed object with no JSON-serialisable serialisation: `StrategyResult` (embeds a
`pd.DataFrame`) and `QuantConfig` (only a provenance-stripping `to_rulebook`).

Each projection SELECTS declared fields into a stable JSON shape and computes NOTHING — a test
asserts the module contains no arithmetic. `StrategyResult.monthly_returns` is deliberately never
projected (Scope B is text; §1.2). Projections are versioned by id and accept either the typed
object or its already-serialised dict, so the same locator works in fixtures and from disk.
"""

from __future__ import annotations

from typing import Callable

STRATEGY_RESULT_PROJECTION_ID = "strategy_result_v1"
QUANT_CONFIG_PROJECTION_ID = "quant_config_v1"

# Whitelisted, JSON-safe display fields. `monthly_returns` is intentionally excluded.
_STRATEGY_RESULT_FIELDS = (
    "strategy_label",
    "summary",
    "variant",
    "n_legs",
    "combiner",
    "status",
    "anchor",
)
_QUANT_CONFIG_FIELDS = (
    "strategy_id",
    "groups",
    "weighting",
    "signal_lag",
    "min_bonds",
    "long_group",
    "short_group",
    "control_groups",
    "holding_period",
)


class UnknownProjectionError(KeyError):
    """A `projection_pointer` names a projection id that does not exist. Fail-closed."""


def _get(source: object, key: str) -> object:
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


def _unwrap(value: object) -> object:
    """An `Inherited[T]` carries its scalar on `.value`; a raw value is returned unchanged."""
    return getattr(value, "value", value)


def project_strategy_result(source: object) -> dict:
    """Select display fields from a `StrategyResult` (or its persisted dict). The
    `pd.DataFrame` `monthly_returns` is never included."""
    out: dict = {}
    for field in _STRATEGY_RESULT_FIELDS:
        value = _get(source, field)
        if value is not None:
            out[field] = value
    return out


def project_quant_config(source: object) -> dict:
    """Select the provenance-stripped scalar of each declared `QuantConfig` field, reading the
    `Inherited` wrapper's `.value`. Not a claim source (§6) — display metadata only."""
    out: dict = {}
    for field in _QUANT_CONFIG_FIELDS:
        value = _get(source, field)
        if value is None:
            continue
        out[field] = _unwrap(value)
    return out


PROJECTIONS: dict[str, Callable[[object], dict]] = {
    STRATEGY_RESULT_PROJECTION_ID: project_strategy_result,
    QUANT_CONFIG_PROJECTION_ID: project_quant_config,
}


def resolve_projection(projection_id: str, source: object) -> dict:
    """Produce the projection dict for `projection_id` from `source`. Unknown id fails closed."""
    try:
        projector = PROJECTIONS[projection_id]
    except KeyError:
        raise UnknownProjectionError(projection_id) from None
    return projector(source)
