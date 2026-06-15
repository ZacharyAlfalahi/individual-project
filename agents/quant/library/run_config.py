"""
RunConfig — the canonical run identity for the bias-toggle registry.

A single immutable object that drives every characteristic-sort run via the
view layer. Mirrors the YAML structure from the registry spec §3:

  run_config:
    panel_view:
      price_family: raw | corr             # meas_err OFF | ON
      stale_mask: false | true             # stale_price OFF | ON
      include_terminal_rows: false | true  # survivorship OFF | ON (no-op pre-FISD)
    construction:
      signal_lag: 0 | >=1                  # lib_gap OFF | ON
      expost_trim: as_published | none     # lab_trim OFF | ON
    evaluation:
      mt_flag: informational               # never gates differentials per A7/D7

Polarity convention is uniform: OFF = as-published / biased; ON = corrected.

`RunConfig.hash()` is the Phase 2 cache-invariant identity. Two configs with
the same hash() MUST drive views.py to produce byte-identical output (the
spec's purity invariant). frozen=True dataclasses make hash stability
non-negotiable — mutation would break the contract.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal, Union

import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PanelViewConfig:
    """Panel-layer toggles. OFF = as-published; ON = corrected.

    Attributes
    ----------
    price_family : 'raw' | 'corr'
        meas_err toggle. OFF (= 'raw') reads the raw column family (no
        decimal-shift, no bounce-back, no distressed filters). ON
        (= 'corr') reads the corrected family.
    stale_mask : bool
        stale_price toggle. True masks bond-months where
        month_end − last_trade_date > θ per A3 (θ from thresholds.yaml).
    include_terminal_rows : bool
        survivorship toggle. FISD-gated; pre-FISD the panel's
        exit_reason column is NaN everywhere so this toggle is a
        documented no-op.
    """
    price_family: Literal["raw", "corr"]
    stale_mask: bool
    include_terminal_rows: bool

    def __post_init__(self):
        if self.price_family not in ("raw", "corr"):
            raise ValueError(
                f"price_family must be 'raw' or 'corr'; got {self.price_family!r}"
            )
        if not isinstance(self.stale_mask, bool):
            raise TypeError(f"stale_mask must be bool; got {type(self.stale_mask).__name__}")
        if not isinstance(self.include_terminal_rows, bool):
            raise TypeError(
                f"include_terminal_rows must be bool; "
                f"got {type(self.include_terminal_rows).__name__}"
            )


@dataclass(frozen=True)
class ConstructionConfig:
    """Construction-layer toggles. OFF = as-published; ON = corrected.

    Attributes
    ----------
    signal_lag : int
        lib_gap toggle. 0 (OFF, has LIB) or >=1 (ON, gap inserted).
    expost_trim : 'as_published' | 'none'
        lab_trim toggle. 'as_published' (OFF) applies the paper's
        published ex-post return trim per A2's per-paper YAML schema;
        'none' (ON) skips the trim entirely.
    """
    signal_lag: int
    expost_trim: Literal["as_published", "none"]

    def __post_init__(self):
        if not isinstance(self.signal_lag, int) or isinstance(self.signal_lag, bool):
            raise TypeError(f"signal_lag must be int; got {type(self.signal_lag).__name__}")
        if self.signal_lag < 0:
            raise ValueError(f"signal_lag must be >= 0; got {self.signal_lag}")
        if self.expost_trim not in ("as_published", "none"):
            raise ValueError(
                f"expost_trim must be 'as_published' or 'none'; "
                f"got {self.expost_trim!r}"
            )


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation-layer informational flags.

    Attributes
    ----------
    mt_flag : 'informational'
        Multiple testing is a property of inference over the family of
        strategies; it has no per-strategy on/off run pair, so it never
        gates a differential. Forcing it into the toggle machinery would
        fabricate a differential that doesn't exist (per A7/D7). The
        only accepted value is 'informational'.
    """
    mt_flag: Literal["informational"] = "informational"

    def __post_init__(self):
        if self.mt_flag != "informational":
            raise ValueError(
                f"mt_flag must be 'informational' per A7/D7; got {self.mt_flag!r}"
            )


# ---------------------------------------------------------------------------
# Top-level RunConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    panel_view: PanelViewConfig
    construction: ConstructionConfig
    evaluation: EvaluationConfig

    # ---- Parsing -----------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict) -> "RunConfig":
        """Parse from a nested dict. Accepts either the inner config
        (`{"panel_view": ..., ...}`) or the outer wrapper
        (`{"run_config": {"panel_view": ..., ...}}`)."""
        if not isinstance(data, dict):
            raise TypeError(f"from_dict expects a dict; got {type(data).__name__}")
        block = data["run_config"] if "run_config" in data else data
        try:
            return cls(
                panel_view=PanelViewConfig(**block["panel_view"]),
                construction=ConstructionConfig(**block["construction"]),
                evaluation=EvaluationConfig(
                    **block.get("evaluation", {"mt_flag": "informational"})
                ),
            )
        except KeyError as exc:
            raise KeyError(f"RunConfig missing required key: {exc}") from None

    @classmethod
    def from_yaml(cls, text: str) -> "RunConfig":
        """Parse from a YAML string."""
        data = yaml.safe_load(text)
        return cls.from_dict(data)

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> "RunConfig":
        """Read a YAML file from disk and parse."""
        return cls.from_yaml(Path(path).read_text())

    # ---- Serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        """Return the run_config-wrapped dict form (round-trips with
        from_dict)."""
        return {"run_config": asdict(self)}

    def to_yaml(self) -> str:
        """Serialise to canonical YAML — sorted keys, no flow style. The
        hash() invariant depends on this being deterministic across runs."""
        return yaml.safe_dump(
            self.to_dict(), sort_keys=True, default_flow_style=False
        )

    # ---- Identity ----------------------------------------------------------

    def hash(self) -> str:
        """Stable SHA-256 over the canonical YAML representation.

        Two configs with the same hash() drive views.py to byte-identical
        output. This is the Phase 2 cache-invariant identity per the
        registry spec §3.
        """
        return hashlib.sha256(self.to_yaml().encode("utf-8")).hexdigest()

    def panel_view_hash(self) -> str:
        """Stable SHA-256 over the panel_view block only.

        views.view() consumes ONLY panel_view — construction and
        evaluation toggles apply downstream at the engine/rulebook level.
        Artefacts produced by view() must therefore be identified by this
        hash, not hash(): stamping the full-config hash on a panel would
        assert construction toggles the panel does not contain.
        """
        block = yaml.safe_dump(
            {"panel_view": asdict(self.panel_view)},
            sort_keys=True, default_flow_style=False,
        )
        return hashlib.sha256(block.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Convenience constructors for the two endpoint views
# ---------------------------------------------------------------------------

def uncorrected() -> RunConfig:
    """The all-OFF view — as-published baseline.

      raw family + no stale mask + no terminal rows
      signal_lag = 0 + expost_trim = as_published

    Used to materialise `monthly_panel_uncorrected.parquet`.
    """
    return RunConfig(
        panel_view=PanelViewConfig(
            price_family="raw",
            stale_mask=False,
            include_terminal_rows=False,
        ),
        construction=ConstructionConfig(
            signal_lag=0,
            expost_trim="as_published",
        ),
        evaluation=EvaluationConfig(),
    )


def corrected() -> RunConfig:
    """The all-ON view — corrected endpoint.

      corr family + stale mask on + terminal rows KEPT (survivorship on)
      signal_lag = 1 + expost_trim = none

    `include_terminal_rows=True` KEEPS the rows the panel's `exit_reason`
    marks terminal (matured | defaulted | defeased). Excluding them is the
    survivorship bias (the dead bonds' final crater returns vanish), so the
    correction is to keep them — now possible because FISD populates
    `exit_reason`. (FISD cannot date calls, so called bonds are not flagged —
    a documented limitation.)

    Used to materialise `monthly_panel_corrected.parquet`.
    """
    return RunConfig(
        panel_view=PanelViewConfig(
            price_family="corr",
            stale_mask=True,
            include_terminal_rows=True,
        ),
        construction=ConstructionConfig(
            signal_lag=1,
            expost_trim="none",
        ),
        evaluation=EvaluationConfig(),
    )
