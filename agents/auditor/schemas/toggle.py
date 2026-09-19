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
# Bias class (ADR bias_class_taxonomy §5.1)
# ---------------------------------------------------------------------------

# The two kinds of intervention a toggle represents. `meas_err` is a
# DATA-QUALITY CORRECTION (how the underlying transaction data was cleaned); the
# other four are METHODOLOGICAL-CONSTRUCTION choices (how the strategy was built).
# Summing an attribution across the two classes yields a number with no referent,
# so the headline is partitioned by class (see checks/algebra.bias_class_partition).
#
# This is a GLOBAL property of the intervention, fixed per toggle — NOT a property
# of any paper. It is deliberately distinct from the provenance vocabulary
# (STATED/INFERRED/DESIGN/UNKNOWN, which lives on extracted fields) and from
# RUNNABLE_REASONS: a `bias_class` never appears on an extracted field (ADR §7). A
# third class (e.g. a data-provider difference) is a new enum value and NOTHING in
# the design space changes — the whole point of keeping it in the interpretation
# layer (ADR §5.1 extensibility test).
BiasClass = Literal["data_quality_correction", "methodological_construction"]

TOGGLE_BIAS_CLASS: dict[ToggleId, BiasClass] = {
    "meas_err": "data_quality_correction",
    "stale_price": "methodological_construction",
    "survivorship": "methodological_construction",
    "lib_gap": "methodological_construction",
    "lab_trim": "methodological_construction",
}

# Fail-loud at import: every registered toggle must carry a bias_class, so the
# partition can never silently drop a coordinate.
_missing_bias_class = set(TOGGLE_IDS) - set(TOGGLE_BIAS_CLASS)
if _missing_bias_class:
    raise RuntimeError(
        f"TOGGLE_BIAS_CLASS is missing a bias_class for {sorted(_missing_bias_class)}"
    )
del _missing_bias_class


def construction_toggles() -> tuple[ToggleId, ...]:
    """The methodological-construction toggles, in canonical order. The single
    source of truth for the partition's method bucket — never hard-code the list."""
    return tuple(
        t for t in TOGGLE_IDS
        if TOGGLE_BIAS_CLASS[t] == "methodological_construction"
    )


def data_quality_toggles() -> tuple[ToggleId, ...]:
    """The data-quality-correction toggles, in canonical order (as configured: meas_err)."""
    return tuple(
        t for t in TOGGLE_IDS
        if TOGGLE_BIAS_CLASS[t] == "data_quality_correction"
    )


# ---------------------------------------------------------------------------
# Dummy reason (ADR bias_class_taxonomy §5.4)
# ---------------------------------------------------------------------------

# WHY a toggle's coordinate is absent or degenerate for a given strategy. Three
# distinct states that must never be collapsed (ADR §5.4):
#
#   not_applicable       — no estimand exists: the two states are identical BY
#                          DEFINITION for this strategy (the paper made no such
#                          choice). The coordinate is EXCLUDED and NO effect is
#                          reported — never rendered as 0/≈0. Depends on what the
#                          source paper did, so it is DECLARED per strategy
#                          (ToggleFacts.not_applicable), not derivable.
#   no_treatment_support — the intervention is implemented and ran, but no
#                          observations satisfy it in this sample (a MEASURED
#                          no-op: runnable=True, is_no_op measured post-run).
#   input_unavailable    — required fields/data are missing, so the intervention
#                          cannot be constructed (runnable=False + a RUNNABLE_REASON).
#
# An empirical null (the intervention operated on real treated rows and the
# estimated effect is ~0) is NOT a dummy at all — it is a result, and carries no
# DummyReason. This vocabulary is separate from RUNNABLE_REASONS and from the
# provenance tags (ADR §7).
DummyReason = Literal["not_applicable", "no_treatment_support", "input_unavailable"]


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
    not_applicable : bool
        The THIRD pre-flight disposition (ADR §5.4): no estimand exists for this
        strategy — the two states are identical BY DEFINITION (the paper made no
        such choice). Like a non-runnable toggle it is EXCLUDED from the lattice,
        but it is NOT a construct failure: it carries a per-paper justification
        (`not_applicable_reason`), holds no `fixed_state`, and does NOT downgrade the
        audit to PARTIAL (see preflight.derive_scope). Must be DECLARED per strategy
        — it depends on what the source paper did and is not derivable.
    not_applicable_reason : str | None
        Required iff `not_applicable` is True: the per-paper justification. This is
        NOT one of `RUNNABLE_REASONS` and NOT a provenance tag (ADR §7).

    There is NO `is_no_op` field: that is a measured RESULT (schemas.audit_core),
    never an input.
    """

    toggle_id: ToggleId
    runnable: bool
    runnable_reason: str | None = None
    fixed_state: ToggleState | None = None
    expect_no_op: bool = False
    not_applicable: bool = False
    not_applicable_reason: str | None = None

    def __post_init__(self) -> None:
        if self.toggle_id not in TOGGLE_IDS:
            raise ValueError(
                f"toggle_id must be one of {TOGGLE_IDS}; got {self.toggle_id!r}"
            )
        if not isinstance(self.runnable, bool):
            raise TypeError("runnable must be a bool")
        if not isinstance(self.expect_no_op, bool):
            raise TypeError("expect_no_op must be a bool")
        if not isinstance(self.not_applicable, bool):
            raise TypeError("not_applicable must be a bool")

        if self.runnable:
            # (a) runnable — varied across the lattice: neither held nor absent,
            # and it cannot simultaneously be "no estimand".
            if self.not_applicable or self.not_applicable_reason is not None:
                raise ValueError(
                    "a runnable toggle cannot be not_applicable; it is varied, "
                    "not absent"
                )
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
            # (a runnable toggle MAY carry expect_no_op — the measured-no-op path.)
        elif self.not_applicable:
            # (b) not_applicable — no estimand exists: excluded, not held, not a
            # construct failure. Carries a per-paper justification only.
            if self.not_applicable_reason is None:
                raise ValueError(
                    "a not_applicable toggle requires a not_applicable_reason "
                    "(the per-paper justification)"
                )
            if self.runnable_reason is not None:
                raise ValueError(
                    "not_applicable is a no-estimand disposition, not a construct "
                    "failure; it must not carry a runnable_reason"
                )
            if self.fixed_state is not None:
                raise ValueError(
                    "a not_applicable toggle is excluded, not held; fixed_state "
                    f"must be None (got {self.fixed_state!r})"
                )
            if self.expect_no_op:
                raise ValueError(
                    "not_applicable has no estimand to run, so expect_no_op is "
                    "meaningless; leave it False"
                )
        else:
            # (c) non-runnable but applicable — a construct failure
            # (input_unavailable): a RUNNABLE_REASON is mandatory (§3.3). fixed_state
            # may be OFF/ON (held => PARTIAL) or None (=> REFUSED); derived in preflight.
            if self.not_applicable_reason is not None:
                raise ValueError(
                    "not_applicable_reason is only valid on a not_applicable toggle"
                )
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


def classify_dummy(
    facts: "ToggleFacts", *, is_no_op: bool | None = None
) -> DummyReason | None:
    """Classify WHY a toggle's coordinate is absent or degenerate for a strategy
    (ADR §5.4), or None when it is NOT a dummy at all.

      not_applicable       — declared: no estimand exists (not derivable, §9.5).
      input_unavailable    — non-runnable + a RUNNABLE_REASON: cannot construct.
      no_treatment_support — runnable but MEASURED as a no-op in this sample.
      None                 — runnable with a real effect, OR an empirical null on
                             real treated rows (a RESULT, not a dummy).

    `is_no_op` is the measured post-run invariance flag (InvarianceResult.is_no_op)
    for a runnable toggle; None if not yet measured. Passing the bool rather than the
    whole InvarianceResult keeps this in the toggle module, decoupled from the output
    schema."""
    if facts.not_applicable:
        return "not_applicable"
    if not facts.runnable:
        return "input_unavailable"
    if is_no_op:
        return "no_treatment_support"
    return None
