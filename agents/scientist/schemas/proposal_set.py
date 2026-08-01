"""
proposal_set.py — `ProposalSet`, frozen before execution (spec §5.3).

All proposals for ONE case, from ONE source, at ONE seed, plus generation metadata and a
sha256 content hash. Persisting EVERY seed's set (not just seed 0) is what makes it
demonstrable that seed 0 was not selected retrospectively (§5.3 / R5, the on-disk artefact
log). The hash is computed by `create` and the frozen dataclass only stores it, so a stored
set cannot silently drift from its fingerprint.

The content hash covers the content-defining fields (case, source, seed, model, prompt/library
versions, and the proposals) but NOT `generated_at`: the fingerprint must be reproducible — an
identical set generated at a different wall-clock time is the same content and must hash
identically. `generated_at` remains stored (part of the immutable record), just outside the
hash.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .proposal import ExtensionProposal, ProposalSource


def _source_value(source: object) -> str:
    return source.value if isinstance(source, ProposalSource) else str(source)


def _canonical_hash(payload: dict) -> str:
    """sha256 over a canonicalised (sorted-key, tight-separator) JSON encoding."""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _proposal_for_hash(p: ExtensionProposal) -> dict:
    """A proposal dump for the CONTENT HASH, with the per-proposal wall-clock stamp
    (`generation.generated_at`) removed. The fingerprint must be reproducible (§5.3 / R5): the
    top-level `generated_at` is already excluded, and the nested per-proposal one must be too, or
    an identical set generated at a different time would hash differently."""
    dump = p.model_dump(mode="json")
    generation = dump.get("generation")
    if isinstance(generation, dict):
        generation.pop("generated_at", None)
    return dump


@dataclass(frozen=True)
class ProposalSet:
    case_id: str
    source: ProposalSource
    seed: int
    proposals: tuple[ExtensionProposal, ...]
    model: str
    prompt_version: str
    library_version: str
    generated_at: str
    content_hash: str

    @classmethod
    def create(
        cls,
        *,
        case_id: str,
        source: ProposalSource,
        seed: int,
        proposals: tuple[ExtensionProposal, ...],
        model: str,
        prompt_version: str,
        library_version: str,
        generated_at: str,
    ) -> "ProposalSet":
        """Assemble a ProposalSet and compute its content hash. `proposals` is coerced to a
        tuple so the frozen set is genuinely immutable."""
        proposals = tuple(proposals)
        payload = {
            "case_id": case_id,
            "source": _source_value(source),
            "seed": seed,
            "model": model,
            "prompt_version": prompt_version,
            "library_version": library_version,
            "proposals": [_proposal_for_hash(p) for p in proposals],
        }
        return cls(
            case_id=case_id,
            source=source,
            seed=seed,
            proposals=proposals,
            model=model,
            prompt_version=prompt_version,
            library_version=library_version,
            generated_at=generated_at,
            content_hash=_canonical_hash(payload),
        )

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "source": _source_value(self.source),
            "seed": self.seed,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "library_version": self.library_version,
            "generated_at": self.generated_at,
            "content_hash": self.content_hash,
            "proposals": [p.model_dump(mode="json") for p in self.proposals],
        }
