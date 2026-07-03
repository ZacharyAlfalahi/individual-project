"""
TrimRule -- typed view of the engine's post-realisation return trim.

Represents ONLY the trim variants the frozen engine supports (see
``characteristic_sort._apply_defaults``). Unsupported-but-well-formed variants
(percentile bounds, by-month sample, non-return target) are deliberately NOT
representable here: the QuantConfig factory refuses them
(``UNSUPPORTED_TRIM_VARIANT``) *before* a ``TrimRule`` is ever constructed.

``TrimRule.__post_init__`` therefore raises only on MALFORMED supported input
(e.g. a ``truncate`` with neither bound set), which is an agent-extraction bug,
not an unsupportable strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TrimMethod = Literal["none", "truncate", "winsorise"]

_SUPPORTED_METHODS: tuple[str, ...] = ("none", "truncate", "winsorise")


@dataclass(frozen=True)
class TrimRule:
    method: TrimMethod = "none"
    lo: float | None = None
    hi: float | None = None
    # target / bounds_type / sample are pinned to the only values the engine
    # implements; kept as explicit fields so a wrong value is caught here rather
    # than silently dropped. The factory refuses unsupported values before this
    # object is built.
    target: Literal["return"] = "return"
    bounds_type: Literal["absolute"] = "absolute"
    sample: Literal["full_sample"] = "full_sample"

    def __post_init__(self) -> None:
        if self.method not in _SUPPORTED_METHODS:
            raise ValueError(
                f"TrimRule.method must be none/truncate/winsorise; got {self.method!r}"
            )
        if self.target != "return":
            raise ValueError(f"TrimRule.target must be 'return'; got {self.target!r}")
        if self.bounds_type != "absolute":
            raise ValueError(
                f"TrimRule.bounds_type must be 'absolute'; got {self.bounds_type!r}"
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
        return {
            "method": self.method,
            "target": self.target,
            "bounds": bounds,
            "sample": self.sample,
        }
