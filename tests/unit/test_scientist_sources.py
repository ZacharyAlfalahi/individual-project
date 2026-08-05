"""
Researcher rungs 1-2 (random_eligible, retrieval_only) + the generation loop (§8.2):
determinism, exactly-m candidates, invalid/duplicate COUNTED-never-regenerated, cosine ranking,
persist-all-seeds, and the wall (emitted proposals carry no magnitudes).
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.scientist.researcher import sources as S  # noqa: E402
from agents.scientist.researcher.eligibility import evaluate  # noqa: E402
from agents.scientist.researcher.library import load_library  # noqa: E402
from agents.scientist.schemas.case import DevelopmentWindow, HoldoutStatus, ScientistCase  # noqa: E402

LIB = load_library()
AVAILABLE = {
    "baa_aaa_spread", "vix", "term_spread",
    "rating", "investment_grade", "var_5pct", "gamma_illiq", "bond_vol", "size", "time_to_maturity",
}
GEN_AT = "2026-08-01T00:00:00Z"
_BANNED = {"sharpe", "alpha", "t_stat", "p_value", "p_raw", "p_bh", "effect", "doe",
           "bh_adjusted_p", "mean_return"}


def _elig(strategy_family="CHARACTERISTIC_SORT", holding_period=1):
    return [evaluate(m, strategy_family=strategy_family, holding_period=holding_period,
                     templates=LIB.templates, variable_families=LIB.variable_families,
                     available_variables=AVAILABLE)
            for m in LIB.mechanisms]


def _case(strategy_id="str"):
    return ScientistCase(
        case_id=f"case_{strategy_id}", strategy_id=strategy_id,
        corrected_quant_config_ref="qc://c", corrected_run_ref="run://c", audit_report_ref="a://c",
        failed_check_ids=("lib_gap",), failed_check_verdicts={"lib_gap": "FAIL"},
        applicable_toggles=("lib_gap", "lab_trim"),
        development_window=DevelopmentWindow(start="2004-08", end="2021-12"),
        holdout_status=HoldoutStatus(accessible=False),
    )


def _keyword_embed(texts):
    kws = ["illiquidity", "momentum", "credit", "volatility", "characteristic", "macro",
           "rating", "exposure", "premium", "regime"]
    return np.array([[1.0 if k in t.lower() else 0.0 for k in kws] for t in texts])


# ---- rung 1: random_eligible --------------------------------------------------------------

def test_random_eligible_is_deterministic_in_seed():
    src, case, elig = S.RandomEligibleSource(), _case(), _elig()
    a = S.run_generation(src, case, elig, LIB, seed=0, m=6, model="deterministic",
                         prompt_version="v0", generated_at=GEN_AT)
    b = S.run_generation(src, case, elig, LIB, seed=0, m=6, model="deterministic",
                         prompt_version="v0", generated_at=GEN_AT)
    def ids(r):
        return [p.proposal_id for p in r.proposal_set.proposals]
    assert ids(a) == ids(b)                                   # same seed -> identical
    assert a.proposal_set.content_hash == b.proposal_set.content_hash


def test_random_eligible_seed_changes_selection():
    src, case, elig = S.RandomEligibleSource(), _case(), _elig()
    s0 = {p.proposal_id for p in S.run_generation(src, case, elig, LIB, seed=0, m=6,
          model="d", prompt_version="v0", generated_at=GEN_AT).proposal_set.proposals}
    s1 = {p.proposal_id for p in S.run_generation(src, case, elig, LIB, seed=1, m=6,
          model="d", prompt_version="v0", generated_at=GEN_AT).proposal_set.proposals}
    assert s0 != s1                                           # different seed -> different draw


def test_random_eligible_exactly_m_distinct():
    src, case, elig = S.RandomEligibleSource(), _case(), _elig()
    r = S.run_generation(src, case, elig, LIB, seed=3, m=6, model="d", prompt_version="v0",
                         generated_at=GEN_AT)
    assert r.n_valid_unique == 6
    keys = {S._equivalence_key(p) for p in r.proposal_set.proposals}
    assert len(keys) == 6                                     # all distinct by equivalence key


# ---- rung 2: retrieval_only ---------------------------------------------------------------

def test_retrieval_topm_ranked_and_deterministic():
    src = S.RetrievalOnlySource(embed=_keyword_embed)
    case, elig = _case(), _elig()
    a = src.candidates(case, elig, LIB, seed=0, m=5, model="minilm-stub", prompt_version="v0",
                       generated_at=GEN_AT)
    b = src.candidates(case, elig, LIB, seed=0, m=5, model="minilm-stub", prompt_version="v0",
                       generated_at=GEN_AT)
    assert [p["proposal_id"] for p in a] == [p["proposal_id"] for p in b]      # deterministic
    assert len(a) == 5
    ranks = [p["generation"]["retrieval_rank"] for p in a]
    sims = [p["generation"]["retrieval_similarity"] for p in a]
    assert ranks == [0, 1, 2, 3, 4]
    assert all(sims[i] >= sims[i + 1] for i in range(len(sims) - 1))           # non-increasing


def test_retrieval_ranks_by_cosine_semantically():
    # A query mentioning illiquidity should score an illiquidity mechanism above a macro one.
    case = _case(strategy_id="illiquidity")                  # query text contains 'illiquidity'
    q = S._case_query_text(case)
    texts = [S._mech_text(LIB.mechanism("mech_008")), S._mech_text(LIB.mechanism("mech_005"))]
    vecs = _keyword_embed([q] + texts)
    sims = S._cosine(np.asarray(vecs[0]), np.asarray(vecs[1:]))
    assert sims[0] > sims[1]                                  # mech_008 (illiq) > mech_005 (macro)


# ---- generation loop: count, never regenerate ---------------------------------------------

class _FixedSource:
    source_name = "random_eligible"

    def __init__(self, raws):
        self._raws = raws

    def candidates(self, *a, **k):
        return list(self._raws)


def test_invalid_and_duplicate_are_counted_never_regenerated():
    case, elig = _case(), _elig()
    good = S.RandomEligibleSource().candidates(case, elig, LIB, seed=0, m=2, model="d",
                                               prompt_version="v0", generated_at=GEN_AT)
    invalid = dict(good[0])
    invalid.pop("prediction")                                 # missing required field -> invalid
    duplicate = dict(good[0])                                 # same equivalence key as good[0]
    src = _FixedSource([good[0], invalid, duplicate, good[1]])
    r = S.run_generation(src, case, elig, LIB, seed=0, m=4, model="d", prompt_version="v0",
                         generated_at=GEN_AT)
    assert r.n_requested == 4
    assert r.n_invalid == 1
    assert r.n_duplicate == 1
    assert r.n_valid_unique == 2                              # NOT topped back up to 4


# ---- persist all seeds (R5) ---------------------------------------------------------------

def test_content_hash_reproducible_across_generated_at():
    # M3: identical content at a different wall-clock time must hash identically — the per-proposal
    # generation.generated_at must be excluded from the content hash, not just the top-level one.
    src, case, elig = S.RandomEligibleSource(), _case(), _elig()
    a = S.run_generation(src, case, elig, LIB, seed=0, m=6, model="d", prompt_version="v0",
                         generated_at="2026-08-01T00:00:00Z")
    b = S.run_generation(src, case, elig, LIB, seed=0, m=6, model="d", prompt_version="v0",
                         generated_at="2027-01-01T12:34:56Z")
    assert a.proposal_set.content_hash == b.proposal_set.content_hash
    assert a.proposal_set.generated_at != b.proposal_set.generated_at   # stamp still differs


def test_run_all_seeds_persists_every_seed():
    results = S.run_all_seeds(S.RandomEligibleSource(), _case(), _elig(), LIB, k=5, m=6,
                              model="d", prompt_version="v0", generated_at=GEN_AT)
    assert [r.proposal_set.seed for r in results] == [0, 1, 2, 3, 4]


# ---- the wall: emitted proposals carry no magnitudes --------------------------------------

def test_emitted_proposals_have_no_magnitude_keys():
    r = S.run_generation(S.RandomEligibleSource(), _case(), _elig(), LIB, seed=0, m=6,
                         model="d", prompt_version="v0", generated_at=GEN_AT)

    def keys(o):
        if isinstance(o, dict):
            for k, v in o.items():
                yield str(k)
                yield from keys(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                yield from keys(v)

    assert set(keys(r.proposal_set.to_dict())).isdisjoint(_BANNED)
