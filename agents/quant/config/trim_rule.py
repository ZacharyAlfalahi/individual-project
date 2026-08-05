"""
TrimRule -- typed view of the engine's post-realisation return trim.

Supports the trim variants the frozen engine implements (see
``characteristic_sort._apply_defaults`` / ``run_characteristic_sort``):

  * ``bounds_type='absolute'`` -- lo/hi are absolute return bounds (the original,
    unchanged behaviour for every existing strategy).
  * ``bounds_type='percentile'`` -- lo/hi are percentile LEVELS in (0, 1); the
    engine resolves them to absolute bounds on the FULL-SAMPLE series it trims,
    using ``percentile_method`` (spec E). This expresses a rule a paper actually
    stated (Jostova fn.16: "eliminate returns above the 99.5th percentile") as a
    data-dependent procedure recomputed per lattice cell, not a fixed cutoff.

``percentile_method`` is REQUIRED for percentile bounds and has no default here:
the engine must not inherit a library default (spec E binding condition 2); the
value is pre-registered in ``thresholds.yaml`` (bias_toggles.lab_filter.
percentile_interpolation) and threaded in by the config layer.

``sample`` remains pinned to ``full_sample`` (the only implemented sample).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TrimMethod = Literal["none", "truncate", "winsorise"]
BoundsType = Literal["absolute", "percentile"]

_SUPPORTED_METHODS: tuple[str, ...] = ("none", "truncate", "winsorise")
_SUPPORTED_BOUNDS: tuple[str, ...] = ("absolute", "percentile")
# numpy/pandas quantile interpolation options (the free parameter that must be
# pinned in thresholds.yaml, not defaulted here).
_SUPPORTED_PERCENTILE_METHODS: tuple[str, ...] = (
    "linear", "lower", "higher", "nearest", "midpoint",
)


@dataclass(frozen=True)
class TrimRule:
    method: TrimMethod = "none"
    lo: float | None = None
    hi: float | None = None
    # target / sample are pinned to the only values the engine implements.
    target: Literal["return"] = "return"
    bounds_type: BoundsType = "absolute"
    sample: Literal["full_sample"] = "full_sample"
    # Only meaningful (and required) when bounds_type == 'percentile'.
    percentile_method: str | None = None

    def __post_init__(self) -> None:
        if self.method not in _SUPPORTED_METHODS:
            raise ValueError(
                f"TrimRule.method must be none/truncate/winsorise; got {self.method!r}"
            )
        if self.target != "return":
            raise ValueError(f"TrimRule.target must be 'return'; got {self.target!r}")
        if self.bounds_type not in _SUPPORTED_BOUNDS:
            raise ValueError(
                f"TrimRule.bounds_type must be absolute/percentile; got {self.bounds_type!r}"
            )
        if self.sample != "full_sample":
            raise ValueError(
                f"TrimRule.sample must be 'full_sample'; got {self.sample!r}"
            )
        for name, bound in (("lo", self.lo), ("hi", self.hi)):
            if bound is not None and (
                not isinstance(bound, (int, float)) or isinstance(bound, bool)
            ):
                raise TypeError(f"TrimRule.{name} must be a number or None; got {bound!r}")
        if self.method != "none" and self.lo is None and self.hi is None:
            raise ValueError(
                f"TrimRule.method={self.method!r} requires at least one of lo/hi"
            )
        if self.bounds_type == "percentile":
            # No library default: the method is pre-registered in thresholds.yaml.
            if self.percentile_method is None:
                raise ValueError(
                    "TrimRule.bounds_type='percentile' requires an explicit "
                    "percentile_method (pre-registered in thresholds.yaml; the "
                    "engine must not inherit a default — spec E condition 2)"
                )
            if self.percentile_method not in _SUPPORTED_PERCENTILE_METHODS:
                raise ValueError(
                    f"TrimRule.percentile_method must be one of "
                    f"{_SUPPORTED_PERCENTILE_METHODS}; got {self.percentile_method!r}"
                )
            for name, bound in (("lo", self.lo), ("hi", self.hi)):
                if bound is not None and not (0.0 < bound < 1.0):
                    raise ValueError(
                        f"TrimRule.{name}={bound!r} must be a percentile LEVEL in "
                        f"(0, 1) when bounds_type='percentile'"
                    )
        elif self.percentile_method is not None:
            raise ValueError(
                "TrimRule.percentile_method is only valid with "
                "bounds_type='percentile'"
            )

    def to_engine_dict(self) -> dict:
        """The nested ``trim_rule`` dict the engine consumes. Minimal for
        ``none`` (matching the engine's own default), full otherwise."""
        if self.method == "none":
            return {"method": "none"}
        bounds: dict = {"type": self.bounds_type}
        if self.lo is not None:
            bounds["lo"] = self.lo
        if self.hi is not None:
            bounds["hi"] = self.hi
        if self.bounds_type == "percentile":
            bounds["percentile_method"] = self.percentile_method
        return {
            "method": self.method,
            "target": self.target,
            "bounds": bounds,
            "sample": self.sample,
        }
