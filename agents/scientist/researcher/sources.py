"""ProposalSource rungs (spec §8.1) — the deterministic ablation ladder, plus the generation
loop (§8.2). Build the protocol first, rung 3 (llm) last.

  rung 1  random_eligible  — random.sample over the eligible proposal space (isolates: does the
                             library + deterministic filter alone produce anything?).
  rung 2  retrieval_only   — brute-force EXACT cosine over MiniLM embeddings, top-m (isolates:
                             does semantic retrieval add value over arbitrary eligible choice?).
                             R4: exact cosine over <=25 vectors, no index.faiss (ANN is not
                             bit-reproducible; exact ranking is). The embedder is INJECTED so the
                             logic is testable without the sentence-transformers dependency.
  rung 3  llm_researcher   — DEFERRED (not built in this cycle).

Generation rules (§8.2 / prohibition 6): EXACTLY m candidates per case per seed; invalid and
duplicate proposals are COUNTED, never regenerated (no retry loop); persist ALL seeds (R5). The
valid-unique count (<= m) is the reported RQ4 generation-quality metric — never topped up to m.

All rungs emit the identical ExtensionProposal schema and pass the identical Experimentalist.
INVARIANT 1: proposals carry refs + typed delta + a magnitude-free rationale/prediction only.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from ..schemas.equivalence import equivalence_key as _equivalence_key
from ..schemas.proposal import ExtensionProposal, decode_proposal
from ..schemas.proposal_set import ProposalSet
from .context_builder import TOGGLE_DEFINITIONS
from .eligibility import EligibilityResult
from .library import MechanismLibrary


@dataclass(frozen=True, order=True)
class ProposalSpec:
    """One executable point in the proposal space: mechanism x template x (variable, lag, form)."""
    mechanism_id: str
    template_id: str
    conditioning_variable: str
    conditioning_lag_months: int
    interaction_form: str


# ---- proposal space enumeration -----------------------------------------------------------

def enumerate_proposal_space(
    eligible_results, library: MechanismLibrary
) -> list[ProposalSpec]:
    """Every valid ProposalSpec across eligible mechanisms: for each mechanism's reachable
    (template, variable), cross the template's permitted lags and forms. Sorted → deterministic."""
    space: list[ProposalSpec] = []
    for r in eligible_results:
        if not r.eligible:
            continue
        # ConfigDelta conditions on ONE variable; a multi-required-input mechanism would need the
        # inputs crossed. None exist in the library today — fail loud rather than silently emit a
        # proposal that satisfies only one input (see review m5).
        if len(r.reachable) != 1:
            raise NotImplementedError(
                f"{r.mechanism_id}: multi-required-input mechanisms are not modeled "
                f"(saw families {sorted(r.reachable)})"
            )
        for _fam, options in r.reachable.items():
            for tid, var in options:
                t = library.templates[tid]
                for lag in t["permitted_fields"]["conditioning_lag_months"]["allowed"]:
                    for form in t["permitted_fields"]["interaction_form"]["allowed"]:
                        space.append(ProposalSpec(r.mechanism_id, tid, var, int(lag), form))
    return sorted(set(space))


def _first_config(r: EligibilityResult, library: MechanismLibrary) -> ProposalSpec:
    """The deterministic 'default template' config for a mechanism (§8.1 retrieval_only): the
    first reachable (template, variable) in canonical order, first lag, first form."""
    if len(r.reachable) != 1:
        raise NotImplementedError(
            f"{r.mechanism_id}: multi-required-input mechanisms are not modeled "
            f"(saw families {sorted(r.reachable)})"
        )
    for _fam, options in sorted(r.reachable.items()):
        if options:
            tid, var = sorted(options)[0]
            t = library.templates[tid]
            lag = sorted(t["permitted_fields"]["conditioning_lag_months"]["allowed"])[0]
            form = sorted(t["permitted_fields"]["interaction_form"]["allowed"])[0]
            return ProposalSpec(r.mechanism_id, tid, var, int(lag), form)
    raise ValueError(f"{r.mechanism_id} eligible but has no reachable config")


# ---- raw-dict construction (goes through decode_proposal) ----------------------------------

def _proposal_id(case_id: str, spec: ProposalSpec, source: str, seed: int) -> str:
    payload = "|".join([case_id, source, str(seed), spec.mechanism_id, spec.template_id,
                        spec.conditioning_variable, str(spec.conditioning_lag_months),
                        spec.interaction_form])
    return "prop_" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def _spec_to_raw(
    spec: ProposalSpec, case, library: MechanismLibrary, *, source: str, seed: int, model: str,
    prompt_version: str, generated_at: str, rank: int | None = None, sim: float | None = None,
) -> dict:
    mech = library.mechanism(spec.mechanism_id)
    return {
        "proposal_id": _proposal_id(case.case_id, spec, source, seed),
        "case_id": case.case_id,
        "parent_strategy_id": case.strategy_id,
        "mechanism_ref": spec.mechanism_id,
        "template_ref": spec.template_id,
        "rationale": mech["claim"].strip(),                       # magnitude-free
        "config_delta": {
            "conditioning_variable": spec.conditioning_variable,
            "conditioning_lag_months": spec.conditioning_lag_months,
            "interaction_form": spec.interaction_form,
        },
        "prediction": (                                           # magnitude-free (no numbers)
            f"Conditioning on {spec.conditioning_variable} (lag {spec.conditioning_lag_months}) "
            f"via {spec.template_id} yields incremental risk-adjusted performance per "
            f"{spec.mechanism_id}."
        ),
        "required_inputs": [spec.conditioning_variable],
        "generation": {
            "source": source, "seed": seed, "model": model, "prompt_version": prompt_version,
            "library_version": library.version_hash, "generated_at": generated_at,
            "retrieval_rank": rank, "retrieval_similarity": sim,
        },
    }


# ---- the ProposalSource protocol + the two deterministic rungs -----------------------------

class ProposalSource(Protocol):
    source_name: str

    def candidates(self, case, eligible_results, library, *, seed: int, m: int, model: str,
                   prompt_version: str, generated_at: str) -> list[dict]: ...


class RandomEligibleSource:
    """Rung 1 — seeded random.sample over the eligible proposal space."""
    source_name = "random_eligible"

    def candidates(self, case, eligible_results, library, *, seed, m, model, prompt_version,
                   generated_at) -> list[dict]:
        space = enumerate_proposal_space(eligible_results, library)
        rng = random.Random(seed)
        chosen = rng.sample(space, min(m, len(space)))            # distinct; seeded
        return [_spec_to_raw(s, case, library, source=self.source_name, seed=seed, model=model,
                             prompt_version=prompt_version, generated_at=generated_at)
                for s in chosen]


class RetrievalOnlySource:
    """Rung 2 — brute-force exact cosine over injected embeddings, top-m mechanisms (R4)."""
    source_name = "retrieval_only"

    def __init__(self, embed: Callable[[list[str]], np.ndarray]):
        self.embed = embed

    def candidates(self, case, eligible_results, library, *, seed, m, model, prompt_version,
                   generated_at) -> list[dict]:
        elig = [r for r in eligible_results if r.eligible]
        if not elig:
            return []
        query = _case_query_text(case)
        mech_texts = [_mech_text(library.mechanism(r.mechanism_id)) for r in elig]
        vecs = self.embed([query] + mech_texts)
        sims = _cosine(np.asarray(vecs[0]), np.asarray(vecs[1:]))
        order = np.argsort(-sims, kind="stable")                 # descending, ties by index
        out = []
        for rank, idx in enumerate(order[:m]):
            r = elig[int(idx)]
            spec = _first_config(r, library)
            out.append(_spec_to_raw(spec, case, library, source=self.source_name, seed=seed,
                                    model=model, prompt_version=prompt_version,
                                    generated_at=generated_at, rank=rank, sim=float(sims[idx])))
        return out


def _case_query_text(case) -> str:
    toggles = "; ".join(TOGGLE_DEFINITIONS.get(t, t) for t in case.applicable_toggles)
    return (f"Strategy {case.strategy_id}; correction-sensitive checks "
            f"{', '.join(case.failed_check_ids)}; {toggles}")


def _mech_text(mech: dict) -> str:
    return f"{mech['title']}. {mech['claim'].strip()}"


def _cosine(q: np.ndarray, M: np.ndarray) -> np.ndarray:
    qn = q / (np.linalg.norm(q) + 1e-12)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    return Mn @ qn


def minilm_embedder() -> Callable[[list[str]], np.ndarray]:
    """Production embedder: lazy MiniLM (all-MiniLM-L6-v2). Raises a clear error if
    sentence-transformers is not installed — a DEFERRED dependency (like the rung-3 model access),
    NOT needed for the retrieval logic, which is tested with an injected deterministic stub."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - exercised only when the dep is absent
        raise RuntimeError(
            "retrieval_only's MiniLM embedder needs `sentence-transformers` (deferred dep). "
            "Install it, or inject a custom embedder into RetrievalOnlySource."
        ) from exc
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return lambda texts: np.asarray(model.encode(list(texts)))


# ---- the generation loop (§8.2) -----------------------------------------------------------

@dataclass(frozen=True)
class GenerationResult:
    proposal_set: ProposalSet
    n_requested: int          # candidates the source emitted (<= m)
    n_invalid: int            # decode failures — counted, NOT regenerated
    n_duplicate: int          # equivalent-key collisions — counted, NOT regenerated

    @property
    def n_valid_unique(self) -> int:
        return len(self.proposal_set.proposals)


def run_generation(
    source: ProposalSource, case, eligible_results, library, *, seed: int, m: int, model: str,
    prompt_version: str, generated_at: str,
) -> GenerationResult:
    """Decode each candidate ONCE; count invalid + duplicate; never regenerate. The surviving
    valid-unique proposals form the seed's ProposalSet (R5: every seed persisted upstream)."""
    raws = source.candidates(case, eligible_results, library, seed=seed, m=m, model=model,
                             prompt_version=prompt_version, generated_at=generated_at)
    seen: set = set()
    valid: list[ExtensionProposal] = []
    n_invalid = n_dup = 0
    for raw in raws:
        dec = decode_proposal(raw)
        if not dec.ok:
            n_invalid += 1
            continue
        key = _equivalence_key(dec.proposal)
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        valid.append(dec.proposal)
    ps = ProposalSet.create(
        case_id=case.case_id, source=source.source_name, seed=seed, proposals=valid,
        model=model, prompt_version=prompt_version, library_version=library.version_hash,
        generated_at=generated_at,
    )
    return GenerationResult(ps, n_requested=len(raws), n_invalid=n_invalid, n_duplicate=n_dup)


def run_all_seeds(
    source: ProposalSource, case, eligible_results, library, *, k: int, m: int, model: str,
    prompt_version: str, generated_at: str,
) -> list[GenerationResult]:
    """R5 — run and persist ALL k seeds (0..k-1), so it is demonstrable that seed 0 was not
    selected retrospectively. Seed 0's ProposalSet is the one advanced to the Experimentalist."""
    return [
        run_generation(source, case, eligible_results, library, seed=s, m=m, model=model,
                       prompt_version=prompt_version, generated_at=generated_at)
        for s in range(k)
    ]
