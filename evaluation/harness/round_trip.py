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
