"""Tests for scripts/run_rq4_exhaustive_benchmark.py — the exhaustive canonical-mechanism
benchmark (descriptive diagnostic).

The load-bearing guarantees: (1) the canonical mapping is the frozen §8.1 rule — deterministic,
exactly one proposal per eligible mechanism, typed provenance — checked against the real frozen
mechanism library with a stub case (the mapping reads only case_id/strategy_id, so no run
artifact is needed); (2) the LLM replay can never make a live vendor call; (3) the script forms
no significance family and touches no G4/G5/G6 path (structural: the machinery is never
imported); (4) rank semantics.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_rq4_exhaustive_benchmark as B  # noqa: E402
import run_rq4_funnel as F  # noqa: E402


def _stub_case():
    """The canonical-mapping path reads only case_id + strategy_id (via _spec_to_raw)."""
    return types.SimpleNamespace(case_id="case_str_stub", strategy_id="str")


@pytest.fixture(scope="module")
def elig_library():
    library = F.load_library()
    # Data-free availability: the union of the library's own variable families. Faithful, not
    # hypothetical — the library is authored backwards from the census, so every variable listed
    # in variable_families.yaml is present in the real data by construction.
    available = {v for vs in library.variable_families.values() for v in vs}
    elig = [F.evaluate(mm, strategy_family=F._STRATEGY_FAMILY, holding_period=1,
                       templates=library.templates, variable_families=library.variable_families,
                       available_variables=available) for mm in library.mechanisms]
    return elig, library


def test_canonical_mapping_deterministic_one_per_eligible(elig_library):
    elig, library = elig_library
    case = _stub_case()
    a = B.canonical_proposals(case, elig, library)
    b = B.canonical_proposals(case, elig, library)
    n_eligible = sum(1 for r in elig if r.eligible)
    assert len(a) == n_eligible > 0
    assert [p.proposal_id for p in a] == [p.proposal_id for p in b]  # deterministic
    assert len({p.mechanism_ref for p in a}) == n_eligible           # one per mechanism
    assert all(p.generation.source.value == "exhaustive_canonical" for p in a)


def test_cache_only_client_raises():
    with pytest.raises(RuntimeError, match="live calls are forbidden"):
        B.CacheOnlyClient("some-model").generate("prompt", seed=0)


def test_phase_f_model_names_resolve_from_config():
    names = B.phase_f_model_names()
    assert len(names) == 2 and all(isinstance(n, str) and n for n in names)


def test_rank_within_order_and_ties():
    rows = [
        {"status": "evaluated", "alpha_t": 2.0},
        {"status": "evaluated", "alpha_t": 1.0},
        {"status": "evaluated", "alpha_t": 1.0},
        {"status": "refused", "refusal_code": "X"},          # never counted
    ]
    assert B.rank_within(rows, 3.0) == 1
    assert B.rank_within(rows, 2.0) == 1                      # ties rank equal-best
    assert B.rank_within(rows, 1.0) == 2
    assert B.rank_within(rows, 0.0) == 4


def test_no_significance_family_and_no_g5_structurally():
    # The guarantee is that the script never IMPORTS the BH/G4/G5/rehearsal machinery — a
    # significance family or holdout nomination is then impossible, not merely avoided.
    src = (REPO_ROOT / "scripts" / "run_rq4_exhaustive_benchmark.py").read_text(encoding="utf-8")
    import_lines = [ln for ln in src.splitlines()
                    if ln.lstrip().startswith(("import ", "from "))]
    for forbidden in ("run_fdr", "select_g5", "robustness_g4", "_run_rehearsal", "shared.stats",
                      "selector", "robustness", "holdout"):
        offending = [ln for ln in import_lines if forbidden in ln]
        assert not offending, f"{forbidden!r} imported: {offending}"
    for name in ("run_fdr", "select_g5", "robustness_g4", "_run_rehearsal"):
        assert not hasattr(B, name)
