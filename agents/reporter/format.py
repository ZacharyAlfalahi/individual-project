"""format.py — the single place display formatting and format-only conversions live (§6).

The Reporter computes NO research quantity. The only arithmetic permitted anywhere in the
renderer path is here: decimal->percent (x100) and decimal->bps (x10000) unit conversions,
display rounding, and `-0.0 -> 0.0` normalisation. Everything else must come from upstream
typed output or not be rendered. `fmt` is the ONE rounding/normalisation function (§6), so
the AST lint (§8.1) exempts this module while forbidding numeric literals and arithmetic in
renderer template functions.

A NaN metric renders as the literal token ``nan`` — which the numeric verifier's number
regex deliberately does not match, so it is unverifiable-by-omission, never fabricated (§9).
"""

from __future__ import annotations

import math

from shared.reporting.canonical import NanValue
from shared.reporting.claims import Unit

# Format-only unit scales, applied to a decimal source value.
_SCALE_BY_UNIT: dict[Unit, float] = {
    Unit.PERCENT: 100.0,
    Unit.BPS: 10000.0,
}


def scale_for_unit(unit: Unit) -> float:
    """The format-only multiplier that takes a decimal source value to `unit`'s display
    scale (1.0 for units that are not a rescaling of decimal)."""
    return _SCALE_BY_UNIT.get(unit, 1.0)


def to_percent(decimal_value: float) -> float:
    """Decimal -> percent (format-only, x100)."""
    return decimal_value * 100.0


def to_bps(decimal_value: float) -> float:
    """Decimal -> basis points (format-only, x10000)."""
    return decimal_value * 10000.0


def _is_nan(value: object) -> bool:
    return isinstance(value, NanValue) or (
        isinstance(value, float) and math.isnan(value)
    )


def fmt(raw_value: object, *, precision: int | None = None, scale: float = 1.0) -> str:
    """Format `raw_value` for display. `scale` is a format-only unit multiplier (see
    `scale_for_unit`); `precision` is decimal places (None -> shortest round-trip repr).

    Rules: a NaN renders as ``nan``; ``-0.0`` is normalised to ``0.0``; booleans are
    rejected (a boolean is not a display number); a string is returned verbatim (already
    display text, e.g. a categorical label) with no scaling."""
    if _is_nan(raw_value):
        return "nan"
    if isinstance(raw_value, bool):
        raise TypeError("fmt does not format booleans as display numbers")
    if isinstance(raw_value, str):
        return raw_value
    if not isinstance(raw_value, (int, float)):
        raise TypeError(f"fmt cannot format {type(raw_value).__name__}")
    if precision is not None and precision < 0:
        raise ValueError(f"precision must be non-negative; got {precision}")

    # Preserve int identity when there is no rescaling, so a count renders "3", not "3.0".
    scaled = raw_value if scale == 1.0 else raw_value * scale
    if precision is None:
        scaled = 0.0 if scaled == 0.0 else scaled
        return repr(scaled) if isinstance(scaled, float) else str(scaled)
    rounded = round(float(scaled), precision)
    if rounded == 0.0:  # collapses -0.0 as well
        rounded = 0.0
    return f"{rounded:.{precision}f}"
