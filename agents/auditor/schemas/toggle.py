"""
toggle.py — the canonical bias-toggle identity and the gold-spec INPUT facts.

The five registered toggles (design §3.1), their fixed canonical order, their
uniform polarity (OFF = as-published / biased; ON = corrected), and how each maps
onto a `RunConfig` field. This mapping is the single source of truth consumed by
the lattice builder and the cell runner — nothing else may hard-code a
toggle->RunConfig correspondence.

`ToggleFacts` is the pre-flight INPUT shape (§3.3): per-toggle `runnable`,
`runnable_reason`, `fixed_state`, `expect_no_op`. It deliberately has NO
`is_no_op` field — that is a post-run RESULT, and it lives only in the output
schema (schemas.audit_core), so it is structurally impossible to declare a no-op
in a spec file (§3.3: "the trap closes itself").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Toggle identity
# ---------------------------------------------------------------------------

ToggleId = Literal["meas_err", "stale_price", "survivorship", "lib_gap", "lab_trim"]
ToggleState = Literal["OFF", "ON"]

# Canonical order — used for bit-labelling the 2^k lattice and for building the
# deterministic conditioning signature. NEVER reorder: cell bit positions and
# stored decompositions are keyed on it.
TOGGLE_IDS: tuple[ToggleId, ...] = (
    "meas_err",
    "stale_price",
    "survivorship",
    "lib_gap",
    "lab_trim",
)

# runnable_reason vocabulary (§3.3). `UNIVERSE_INCOMPATIBLE` is deliberately
# ABSENT — it was removed in v1.5 (D-A30): "no events in the sample" is a MEASURED
# no-op (runnable: true, expect_no_op: true), not an unmeasurable toggle.
RUNNABLE_REASONS: tuple[str, ...] = (
    "MISSING_EXIT_DATA",         # survivorship: FISD exit_reason unresolved
    "PAPER_RULE_NOT_STATED",     # lab_trim: the paper never states its trim -> OFF undefined
    "PROVENANCE_INSUFFICIENT",   # a required Librarian field is not STATED and no rule applies
    "ENGINE_UNSUPPORTED",        # the construction lies outside the engine's supported class
)


@dataclass(frozen=True)
class ToggleAxis:
    """How one toggle maps onto a RunConfig field. `block` is the RunConfig
    sub-config attribute; `field` is the attribute on it; `off`/`on` are the
    concrete values for the two states. For lib_gap the ON value is the
    *default* corrected lag — a per-strategy override is threaded separately by
    the lattice builder (the mom6 skip subtlety, §3.5 / O-A2)."""

    block: Literal["panel_view", "construction"]
    field: str
    off: object
    on: object


# The one and only toggle->RunConfig correspondence (registry spec §3; verified
# against run_config.uncorrected()/corrected()).
TOGGLE_AXES: dict[ToggleId, ToggleAxis] = {
    "meas_err": ToggleAxis("panel_view", "price_family", off="raw", on="corr"),
    "stale_price": ToggleAxis("panel_view", "stale_mask", off=False, on=True),
    "survivorship": ToggleAxis(
        "panel_view", "include_terminal_rows", off=False, on=True
    ),
    "lib_gap": ToggleAxis("construction", "signal_lag", off=0, on=1),
    "lab_trim": ToggleAxis(
        "construction", "expost_trim", off="as_published", on="none"
    ),
}


# ---------------------------------------------------------------------------
# Pre-flight INPUT facts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToggleFacts:
    """Pre-flight facts about ONE toggle for ONE strategy (gold-spec input, §3.3).

    Fields
    ------
    toggle_id : ToggleId
    runnable : bool
        PRE-FLIGHT executability. false => the toggle is dropped from the lattice.
    runnable_reason : str | None
        Required iff runnable is False; must be one of RUNNABLE_REASONS.
    fixed_state : 'OFF' | 'ON' | None
        The state a NON-runnable toggle is held at while the reduced lattice runs
        (§3.4). None when even the OFF state cannot be reconstructed (=> the
        strategy audit is REFUSED). Must be None for a runnable toggle — a
        runnable toggle is varied, not held.
    expect_no_op : bool
        HYPOTHESIS: do we predict the toggle does nothing? Tested post-run by the
        invariance assertion (§3.5); it PRUNES NOTHING.

    There is NO `is_no_op` field: that is a measured RESULT (schemas.audit_core),
    never an input.
    """

    toggle_id: ToggleId
    runnable: bool
    runnable_reason: str | None = None
    fixed_state: ToggleState | None = None
    expect_no_op: bool = False

    def __post_init__(self) -> None:
        if self.toggle_id not in TOGGLE_IDS:
            raise ValueError(
                f"toggle_id must be one of {TOGGLE_IDS}; got {self.toggle_id!r}"
            )
        if not isinstance(self.runnable, bool):
            raise TypeError("runnable must be a bool")
        if not isinstance(self.expect_no_op, bool):
            raise TypeError("expect_no_op must be a bool")

        if self.runnable:
            # A runnable toggle is varied across the lattice, so it is neither
            # held at a fixed state nor does it carry a non-runnable reason.
            if self.runnable_reason is not None:
                raise ValueError(
                    "runnable toggle must not carry a runnable_reason "
                    f"(got {self.runnable_reason!r})"
                )
            if self.fixed_state is not None:
                raise ValueError(
                    "runnable toggle must not carry a fixed_state "
                    f"(got {self.fixed_state!r}); a runnable toggle is varied, not held"
                )
        else:
            # Non-runnable: a reason is mandatory (§3.3). fixed_state may be
            # OFF/ON (a defensible held state => PARTIAL) or None (no defensible
            # endpoint => REFUSED); the strategy-level derivation is in preflight.
            if self.runnable_reason is None:
                raise ValueError(
                    "non-runnable toggle requires a runnable_reason "
                    f"(one of {RUNNABLE_REASONS})"
                )
            if self.runnable_reason not in RUNNABLE_REASONS:
                raise ValueError(
                    f"runnable_reason must be one of {RUNNABLE_REASONS}; "
                    f"got {self.runnable_reason!r}"
                )

        if self.fixed_state is not None and self.fixed_state not in ("OFF", "ON"):
            raise ValueError(
                f"fixed_state must be 'OFF', 'ON' or None; got {self.fixed_state!r}"
            )
