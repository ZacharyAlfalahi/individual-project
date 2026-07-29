"""
proposal.py — `ExtensionProposal`, THE decode boundary (spec §5.2, R6).

This is the ONE place Pydantic v2 is used in the whole Scientist (R6). A validation failure
here is *data* — a counted invalid proposal (§8.2: "invalid proposals are counted, never
regenerated"; prohibition 6: no retry-until-valid) — not a crash. Everything else in the
Scientist is a frozen dataclass. The models are frozen (immutable after construction) and
reject unknown fields (`extra='forbid'`), so a malformed generation is caught at the boundary
rather than surfacing as a mystery three gates later.

INVARIANT 1: the proposal carries mechanism/template refs, the typed config delta, the
prediction, and generation provenance — and NO realised performance statistic.
`retrieval_similarity` is logged AS a similarity and consumed by no gate (INVARIANT 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, ConfigDict, ValidationError

# Frozen + reject-unknown-keys: the decode boundary is strict on structure.
_FROZEN = ConfigDict(frozen=True, extra="forbid")


class ProposalSource(str, Enum):
    """The three ablation-ladder rungs (§8.1). One interface, three implementations."""

    LLM_RESEARCHER = "llm_researcher"
    RETRIEVAL_ONLY = "retrieval_only"
    RANDOM_ELIGIBLE = "random_eligible"


class ConfigDelta(BaseModel):
    """The typed config delta (§5.2). Models the `lagged_binary_regime_interaction` template —
    the only template with a worked delta in v1. Additional templates (§7) extend this with
    their own permitted fields as they are built; per-FIELD domain validation lives at G0
    (`FIELD_OUT_OF_DOMAIN`) against the template registry, NOT here. Fields are optional so a
    partial delta still decodes structurally; `extra='forbid'` rejects unknown keys, so a
    proposal aimed at a not-yet-built template decodes to a counted invalid — correct, since
    that template does not exist yet."""

    model_config = _FROZEN
    conditioning_variable: str | None = None
    conditioning_lag_months: int | None = None
    interaction_form: str | None = None


class Generation(BaseModel):
    """Generation provenance (§5.2 / §8.2). `prompt_version` and `library_version` are sha
    hashes folded into the run manifest. `retrieval_similarity` is LOGGED as a similarity and
    is consumed by no gate (INVARIANT 4) — never treated as a probability or a score."""

    model_config = _FROZEN
    source: ProposalSource
    seed: int
    model: str
    prompt_version: str
    library_version: str
    generated_at: str  # iso8601
    retrieval_rank: int | None = None
    retrieval_similarity: float | None = None  # similarity ONLY — never consumed by a gate


class ExtensionProposal(BaseModel):
    """A candidate strategy derived from the corrected run by an authorised template (§5.2).
    Decoded from structured LLM (or baseline) output at the boundary via `decode_proposal`."""

    model_config = _FROZEN
    proposal_id: str
    case_id: str
    parent_strategy_id: str
    mechanism_ref: str
    template_ref: str
    rationale: str
    config_delta: ConfigDelta
    prediction: str
    required_inputs: tuple[str, ...]
    generation: Generation


@dataclass(frozen=True)
class ProposalDecode:
    """The result of decoding ONE raw proposal dict at the boundary (R6). When `ok` is False,
    `error` carries the validation message AS DATA — the caller COUNTS the invalid proposal
    (§8.2) and never retries (prohibition 6). Consumer: the Researcher generation loop and the
    agent-quality funnel (invalid-proposal rate)."""

    ok: bool
    proposal: ExtensionProposal | None
    error: str | None


def decode_proposal(raw: dict) -> ProposalDecode:
    """Decode one raw (structured-decoding) proposal dict into an `ExtensionProposal`.

    Validation failure is DATA, not an exception (R6): returns
    `ProposalDecode(ok=False, error=...)` so the Researcher can count the invalid without a
    try/except at each call site — and without a retry loop (§8.2 / prohibition 6)."""
    try:
        proposal = ExtensionProposal.model_validate(raw)
    except ValidationError as exc:
        return ProposalDecode(ok=False, proposal=None, error=str(exc))
    return ProposalDecode(ok=True, proposal=proposal, error=None)
