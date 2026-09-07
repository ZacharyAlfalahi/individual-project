"""
Typed, FAIL-LOUD loader for the relocator-diagnostic thresholds.

Reads ``docs/thresholds.yaml`` → ``librarian.relocator_diagnostic``. Raises
``RelocatorThresholdError`` on any missing or malformed key rather than
defaulting — a silent default would launder a post-hoc constant, the exact sin
the instrument exists to avoid.

Two loaders implement the two-stage freeze:

* ``load_relocator_calibration_config`` — everything EXCEPT ``accept_bar`` (which
  may still be ``null`` before calibration). Used by the calibration sweep,
  which is the step that *produces* the bar.
* ``load_relocator_config`` — includes ``accept_bar`` and RAISES while it is
  ``null``/absent, so the re-scoring pass structurally cannot run before its
  protocol's bar is frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"

# The only registered metric. A different string in the yaml means the block
# and the instrument have drifted apart — refuse to run.
EXPECTED_METRIC = "sequencematcher_ratio_l1"


class RelocatorThresholdError(KeyError):
    """A required relocator-diagnostic threshold is missing or malformed."""


@dataclass(frozen=True)
class RelocatorCalibrationConfig:
    metric: str
    min_quote_chars: int
    min_anchor_chars: int
    bar_grid: tuple[float, ...]
    recall_floor_light: float
    recall_floor_moderate: float


@dataclass(frozen=True)
class RelocatorConfig:
    metric: str
    accept_bar: float
    min_quote_chars: int
    min_anchor_chars: int


def _read_block(path: str | Path | None) -> dict:
    p = Path(path) if path is not None else THRESHOLDS_FILE
    if not p.exists():
        raise RelocatorThresholdError(f"thresholds file not found at {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict) or "librarian" not in raw:
        raise RelocatorThresholdError(f"no 'librarian' block in {p}")
    block = raw["librarian"].get("relocator_diagnostic")
    if not isinstance(block, dict):
        raise RelocatorThresholdError(
            f"no 'librarian.relocator_diagnostic' block in {p} (relocator not registered?)"
        )
    return block


def _require_metric(block: dict) -> str:
    metric = block.get("metric")
    if metric != EXPECTED_METRIC:
        raise RelocatorThresholdError(
            f"librarian.relocator_diagnostic.metric must be {EXPECTED_METRIC!r}; got {metric!r}"
        )
    return metric


def _require_positive_int(block: dict, key: str) -> int:
    v = block.get(key)
    if not isinstance(v, int) or isinstance(v, bool) or v < 1:
        raise RelocatorThresholdError(
            f"librarian.relocator_diagnostic.{key} must be an int >= 1; got {v!r}"
        )
    return v


def _require_unit_interval(block: dict, key: str) -> float:
    v = block.get(key)
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not (0.0 < float(v) <= 1.0):
        raise RelocatorThresholdError(
            f"librarian.relocator_diagnostic.{key} must be a number in (0, 1]; got {v!r}"
        )
    return float(v)


def _bar_grid(block: dict) -> tuple[float, ...]:
    g = block.get("bar_grid_hundredths")
    if not isinstance(g, dict) or any(k not in g for k in ("start", "stop", "step")):
        raise RelocatorThresholdError(
            "librarian.relocator_diagnostic.bar_grid_hundredths must be a mapping "
            f"with start/stop/step; got {g!r}"
        )
    start, stop, step = g["start"], g["stop"], g["step"]
    for name, v in (("start", start), ("stop", stop), ("step", step)):
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            raise RelocatorThresholdError(
                f"bar_grid_hundredths.{name} must be an int >= 1; got {v!r}"
            )
    if not (0 < start <= stop <= 100):
        raise RelocatorThresholdError(
            f"bar_grid_hundredths requires 0 < start <= stop <= 100; got start={start}, stop={stop}"
        )
    # Integer hundredths, floated once — no accumulation drift.
    return tuple(b / 100.0 for b in range(start, stop + 1, step))


def load_relocator_calibration_config(path: str | Path | None = None) -> RelocatorCalibrationConfig:
    block = _read_block(path)
    return RelocatorCalibrationConfig(
        metric=_require_metric(block),
        min_quote_chars=_require_positive_int(block, "min_quote_chars"),
        min_anchor_chars=_require_positive_int(block, "min_anchor_chars"),
        bar_grid=_bar_grid(block),
        recall_floor_light=_require_unit_interval(block, "recall_floor_light"),
        recall_floor_moderate=_require_unit_interval(block, "recall_floor_moderate"),
    )


def load_relocator_config(path: str | Path | None = None, *,
                          protocol: str = "qr1") -> RelocatorConfig:
    """``protocol`` selects which calibration's bar governs: ``"qr1"`` reads
    ``accept_bar`` (null forever — the strict all-cross-paper negative universe
    admits no zero-false-accept bar on this corpus), ``"qr2"`` reads
    ``qr2.accept_bar`` (the corrected universe: same-author and exact-present
    negative pairs excluded). Raises while the selected bar is null — the
    corresponding re-scoring pass may not run uncalibrated."""
    block = _read_block(path)
    if protocol == "qr1":
        bar = block.get("accept_bar")
    elif protocol == "qr2":
        qr2 = block.get("qr2")
        if not isinstance(qr2, dict):
            raise RelocatorThresholdError(
                "librarian.relocator_diagnostic.qr2 block absent (corrected-universe protocol not registered?)"
            )
        bar = qr2.get("accept_bar")
    else:
        raise RelocatorThresholdError(f"unknown relocator protocol {protocol!r}")
    if bar is None:
        raise RelocatorThresholdError(
            f"the {protocol} accept_bar is null/absent: its calibration has not been "
            "frozen — the corresponding re-scoring pass may not run before calibration"
        )
    if not isinstance(bar, (int, float)) or isinstance(bar, bool) or not (0.0 < float(bar) <= 1.0):
        raise RelocatorThresholdError(
            f"librarian.relocator_diagnostic.accept_bar must be a number in (0, 1]; got {bar!r}"
        )
    return RelocatorConfig(
        metric=_require_metric(block),
        accept_bar=float(bar),
        min_quote_chars=_require_positive_int(block, "min_quote_chars"),
        min_anchor_chars=_require_positive_int(block, "min_anchor_chars"),
    )
