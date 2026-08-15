"""
G2 round-trip harness (evaluation contract §7 gate 2).

Drives the decisive G2 gate: for each anchor, load the hand-authored gold spec,
compile it through the deterministic adapter/factory with the (hash-verified)
standing-substitutions register, and prove the produced rulebook byte-equals the
independently hand-authored golden production rulebook -- ``R_produced ==
A(R_paper, H)`` -- while emitting the authorised-diff register.

The harness REQUIRES an explicit, hash-verified standing-substitutions file
(``load_verified_standing_subs``): it must not fall back to the empty default, so
the contract §6 temporal rule (the register's hash predates + matches the run) is
not bypassable.

str + drf assert now; mom6 is gated on the JNPS canonical-text freeze (provisional
locators) -- the harness is anchor-agnostic and activates on freeze with no change.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agents.librarian.adapter import adapt_spec  # noqa: E402
from agents.librarian.registries.standing_substitutions import (  # noqa: E402
    STANDING_SUBS_V1_SHA256,
    StandingSubstitutionTable,
    load_standing_substitutions,
)
from agents.quant.config import QuantConfig, to_rulebook  # noqa: E402
from agents.quant.library.bbw_factors import factor_rulebook  # noqa: E402
from evaluation.gold_specs.gold_loader import load_gold_spec  # noqa: E402
from evaluation.harness.canonical_yaml import assert_rulebook_byte_equal  # noqa: E402

from build_str import str_rulebook  # noqa: E402  (scripts/)
from build_mom6 import mom6_rulebook  # noqa: E402  (scripts/)


class HarnessError(RuntimeError):
    """The G2 harness could not run a clean round trip -- a missing/tampered
    standing file, an anchor that did not compile, etc. Surfaced loudly."""


# mom6's base golden config (the mom6_rulebook input): deciles, EW, Jostova skip=1.
_MOM6_CFG = {"n_groups": 10, "weighting": "equal", "skip_months": 1}


def expected_rulebook(anchor_id: str) -> dict:
    """The golden ``R_paper`` -- the independently hand-authored production rulebook."""
    if anchor_id == "str":
        return str_rulebook()
    if anchor_id == "drf":
        return factor_rulebook("drf")
    if anchor_id == "mom6":
        return mom6_rulebook(_MOM6_CFG)
    raise HarnessError(f"unknown anchor_id {anchor_id!r}")


def load_verified_standing_subs(path=None) -> StandingSubstitutionTable:
    """Load the standing-substitutions file and ASSERT its recorded sha256 (contract
    §6 temporal rule). The harness must NOT fall back to the empty default -- an
    absent/tampered file fails ``verify_hash`` and raises here."""
    table = load_standing_substitutions(path)
    if not table.verify_hash(STANDING_SUBS_V1_SHA256):
        raise HarnessError(
            "standing-substitutions file is absent or its byte-hash does not match the "
            "recorded STANDING_SUBS_V1_SHA256 -- refusing to run the G2 gate on an "
            "unverified register (contract §6 temporal rule: the hash must predate + "
            "match the run)"
        )
    return table


def adapt_gold(anchor_id: str, standing_subs: StandingSubstitutionTable):
    """Load the anchor gold + compile it with the verified standing register."""
    return adapt_spec(load_gold_spec(anchor_id), standing_subs=standing_subs)


def produced_rulebook(adapt_result) -> dict:
    """The compiler's rulebook for a single-leg anchor: ``to_rulebook`` of leg 0's
    ``QuantConfig``. Raises if the leg did not compile to a QuantConfig."""
    lc = adapt_result.leg_calls[0]  # anchors are single-leg (D19/D30)
    if not isinstance(lc.result, QuantConfig):
        raise HarnessError(
            f"leg 0 did not compile to a QuantConfig (got {type(lc.result).__name__}); "
            "cannot form a produced rulebook"
        )
    return to_rulebook(lc.result)


def combiner_dict(adapt_result) -> dict:
    return adapt_result.combiner.to_dict()


# --- multi-leg composite G2 (CRF: the first multi-leg equal_average anchor) ----
#
# A composite rulebook is {"legs": {control_col: leg_rulebook, ...}, "combiner": {...}}.
# Legs are keyed by their DISTINGUISHING control column (order-invariant), so the
# gate proves byte-equality per leg AND the combiner instruction.

_CRF_COMPONENTS = ("crf_var", "crf_illiq", "crf_rev")


def expected_composite_rulebook(anchor_id: str) -> dict:
    """The golden composite ``R_paper`` for a multi-leg equal_average anchor: each
    component's independently hand-authored golden rulebook, keyed by its control
    column, plus the equal_average combiner instruction. CRF only, today."""
    if anchor_id != "crf":
        raise HarnessError(f"no composite golden rulebook for anchor_id {anchor_id!r}")
    legs: dict[str, dict] = {}
    for name in _CRF_COMPONENTS:
        rb = factor_rulebook(name)
        control = rb["control"]
        if control in legs:
            raise HarnessError(
                f"two CRF golden components share control column {control!r}; the "
                "composite legs are keyed by control and would collapse to one"
            )
        legs[control] = rb
    return {"legs": legs, "combiner": {"kind": "equal_average", "divisor": "available"}}


def produced_rulebook_composite(adapt_result) -> dict:
    """The compiler's composite rulebook for a multi-leg anchor: every leg's
    ``to_rulebook`` keyed by its control column, plus ``combiner.to_dict()``. Raises
    if any leg did not compile, has no control column, or two legs collide on one."""
    legs: dict[str, dict] = {}
    for i, lc in enumerate(adapt_result.leg_calls):
        if not isinstance(lc.result, QuantConfig):
            raise HarnessError(
                f"leg {i} did not compile to a QuantConfig (got {type(lc.result).__name__}); "
                "cannot form a produced composite rulebook"
            )
        rb = to_rulebook(lc.result)
        control = rb.get("control")
        if control is None:
            raise HarnessError(f"leg {i} produced a rulebook with no control column")
        if control in legs:
            raise HarnessError(
                f"two legs share control column {control!r}; cannot key the composite"
            )
        legs[control] = rb
    return {"legs": legs, "combiner": adapt_result.combiner.to_dict()}


def assert_composite_rulebook_byte_equal(produced: dict, expected: dict) -> None:
    """Assert a multi-leg composite matches PER LEG **and** the combiner -- the
    headline new G2 capability. Legs are keyed by control column (order-invariant);
    each leg is D30 byte-equal (reusing ``assert_rulebook_byte_equal``), and the
    combiner dict matches exactly. A missing/extra leg or a mismatched control column
    (e.g. the crf_rev rev->xret reconciliation regressing) surfaces loudly here."""
    prod_legs, exp_legs = produced["legs"], expected["legs"]
    if set(prod_legs) != set(exp_legs):
        raise AssertionError(
            f"composite legs differ: produced controls {sorted(prod_legs)} vs "
            f"expected {sorted(exp_legs)}"
        )
    for control in exp_legs:
        assert_rulebook_byte_equal(prod_legs[control], exp_legs[control])
    if produced["combiner"] != expected["combiner"]:
        raise AssertionError(
            f"combiner differs: produced {produced['combiner']} vs "
            f"expected {expected['combiner']}"
        )


# --- gate 1-2 verdict: RQ2 anchor fidelity (contract §7 gates 1-2) -----------
#
# Under the v1.6 re-scope (2026-08-14) the RQ2 anchor fidelity PASS CONDITION is
# gates 1-2 -- construction invariants + rulebook byte-equality -- NOT the §7
# gate-5 bias-mechanism differential (retained as a reported RQ3-facing
# diagnostic). Gate 2 is the byte-equal rulebook (produced == A(R_paper, H)); the
# gate-1 holding_period invariant is checked EXPLICITLY here because to_rulebook
# omits holding_period (an overlap argument, not a rulebook key), so byte-equality
# alone would not cover it.

# Independently hand-authored expected holding period per anchor (from the papers,
# not the gold): str/drf/crf hold one month (monthly re-formation; the golds state
# holding UNKNOWN -> engine default 1, which the papers entail); mom6 holds 6
# (JNPS "held over months t+1 to t+6", gold_mom6_jnps_2013.md STATED).
_EXPECTED_HOLDING_PERIOD = {"str": 1, "drf": 1, "mom6": 6, "crf": 1}


def _assert_byte_equal(anchor_id: str, adapt_result) -> None:
    """Raise on rulebook mismatch (single-leg or the crf composite)."""
    if anchor_id == "crf":
        assert_composite_rulebook_byte_equal(
            produced_rulebook_composite(adapt_result),
            expected_composite_rulebook(anchor_id))
    else:
        assert_rulebook_byte_equal(
            produced_rulebook(adapt_result), expected_rulebook(anchor_id))


def _produced_holding_periods(adapt_result) -> list[int]:
    """The compiled holding_period on every leg (anchors: 1 leg; crf: 3)."""
    out: list[int] = []
    for i, lc in enumerate(adapt_result.leg_calls):
        if not isinstance(lc.result, QuantConfig):
            raise HarnessError(
                f"leg {i} did not compile to a QuantConfig; no holding_period")
        out.append(lc.result.holding_period.value)
    return out


def gate12_verdict(anchor_id: str, standing_subs: StandingSubstitutionTable) -> dict:
    """The RQ2 anchor fidelity verdict (contract §7 gates 1-2), as a report dict.

    Adapts the gold ONCE and reports both components:
      * ``rulebook_byte_equal`` (gate 2) -- the produced rulebook byte-equals the
        independently hand-authored golden rulebook;
      * ``holding_period_match`` (the gate-1 invariant to_rulebook omits) -- the
        compiled holding_period equals the hand-authored expected on every leg.
    ``pass`` is their conjunction. Never raises: a refusal / compile failure /
    mismatch sets the relevant flag False and records ``error``.
    """
    expected_holding = _EXPECTED_HOLDING_PERIOD.get(anchor_id)
    if expected_holding is None:
        raise HarnessError(f"no expected holding_period for anchor_id {anchor_id!r}")
    out: dict = {
        "rulebook_byte_equal": False,
        "holding_period_match": False,
        "expected_holding_period": expected_holding,
        "produced_holding_period": None,
        "pass": False,
        "error": None,
    }
    try:
        r = adapt_gold(anchor_id, standing_subs)
        if r.refused:
            out["error"] = "adapter refused to compile the gold"
            return out
        try:
            _assert_byte_equal(anchor_id, r)
            out["rulebook_byte_equal"] = True
        except AssertionError as exc:
            out["error"] = f"rulebook mismatch: {exc}"
        holds = _produced_holding_periods(r)
        out["produced_holding_period"] = holds
        out["holding_period_match"] = all(h == expected_holding for h in holds)
        out["pass"] = out["rulebook_byte_equal"] and out["holding_period_match"]
    except HarnessError as exc:
        out["error"] = str(exc)
    return out


def g2_pass(anchor_id: str, standing_subs: StandingSubstitutionTable) -> bool:
    """Gate-2 rulebook byte-equality as a boolean (contract §7 gate 2)."""
    return gate12_verdict(anchor_id, standing_subs)["rulebook_byte_equal"]


# --- the authorised-diff register (contract §7.2 register rows) --------------

@dataclass(frozen=True)
class RegisterRow:
    """One authorised paper<->engine difference (contract §7.2)."""

    anchor_id: str
    field: str
    paper_value: object
    engine_value: object
    provenance: str
    authorisation_id: str
    authorisation_class: str  # "standing" | "override"
    rationale: str
    source_decision: str
    variant_effect: bool
    fidelity_aggregate_exclusion: bool


def _find_sub(standing_subs: StandingSubstitutionTable, sub_id: str):
    for s in standing_subs.substitutions:
        if s.id == sub_id:
            return s
    return None


# The par-weighting convention's paper<->engine framing for an anchor whose gold
# already STATES par (drf): the divergence is the size BASIS, not a field substitution.
_PAR_CONVENTION_PAPER = "amount_outstanding (market value of amount outstanding)"
_PAR_CONVENTION_ENGINE = "par (offering_amt proxy)"


def emit_register(anchor_id, adapt_result, standing_subs, rulebook=None) -> list[RegisterRow]:
    """The authorised-diff register for one anchor. Sources (deduped one row per
    (anchor, convention)):
      (i)   standing subs the ADAPTER applied (str: weighting_base market_value->par);
      (ii)  standing CONVENTIONS bearing on the produced rulebook even when the gold
            already states the engine value (drf: by_size == the par-proxy for amount
            outstanding) -- so every value-weighted anchor carries the par row;
      (iii) per-strategy overrides (variant=true) -- none for the base anchors.
    Standing rows: variant_effect=False, excluded=False. Override rows: True/True."""
    rows: list[RegisterRow] = []
    seen: set[tuple[str, str]] = set()

    # (i) adapter-applied standing substitutions.
    for ap in adapt_result.standing_subs_applied:
        sub = _find_sub(standing_subs, ap.substitution_id)
        rows.append(RegisterRow(
            anchor_id=anchor_id, field=ap.field,
            paper_value=ap.paper_value, engine_value=ap.engine_value,
            provenance="DESIGN", authorisation_id=ap.substitution_id,
            authorisation_class="standing",
            rationale=(sub.note if sub else ""),
            source_decision=(sub.source_decision if sub else ""),
            variant_effect=False, fidelity_aggregate_exclusion=False,
        ))
        seen.add((anchor_id, ap.substitution_id))

    # (ii) par-weighting convention on any by_size anchor not already covered.
    if rulebook is not None and rulebook.get("weighting") == "by_size":
        if (anchor_id, "par_weighting_v1") not in seen:
            sub = _find_sub(standing_subs, "par_weighting_v1")
            if sub is not None:
                rows.append(RegisterRow(
                    anchor_id=anchor_id, field="weighting_base",
                    paper_value=_PAR_CONVENTION_PAPER, engine_value=_PAR_CONVENTION_ENGINE,
                    provenance="DESIGN", authorisation_id="par_weighting_v1",
                    authorisation_class="standing",
                    rationale=sub.note, source_decision=sub.source_decision,
                    variant_effect=False, fidelity_aggregate_exclusion=False,
                ))
                seen.add((anchor_id, "par_weighting_v1"))

    # (iii) per-strategy overrides (D23 variant). None for the base anchors; recorded
    # here for completeness so a future override lands as a variant-excluded row.
    if adapt_result.variant:
        rows.append(RegisterRow(
            anchor_id=anchor_id, field="(override)", paper_value=None, engine_value=None,
            provenance="DESIGN", authorisation_id="(per-strategy override)",
            authorisation_class="override", rationale="per-strategy override (D27)",
            source_decision="authorisation_records.yaml",
            variant_effect=True, fidelity_aggregate_exclusion=True,
        ))
    return rows
