"""Rung 3 (llm_researcher): the content-addressed cache, structured-output decoding through the
generation loop (invalid/duplicate counted, never regenerated), cache hits, the wall on the
prompt, and the deferred-client guard."""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.scientist.researcher.cache import ResponseCache  # noqa: E402
from agents.scientist.researcher.eligibility import evaluate  # noqa: E402
from agents.scientist.researcher.library import load_library  # noqa: E402
from agents.scientist.researcher.llm_source import (  # noqa: E402
    DeferredModelClient,
    LLMResearcherSource,
    build_prompt,
)
from agents.scientist.researcher.sources import run_generation  # noqa: E402
from agents.scientist.schemas.case import DevelopmentWindow, HoldoutStatus, ScientistCase  # noqa: E402

LIB = load_library()
AVAILABLE = {"baa_aaa_spread", "vix", "term_spread", "rating", "investment_grade", "var_5pct",
             "gamma_illiq", "bond_vol", "size", "time_to_maturity"}
CASE = ScientistCase(
    case_id="c", strategy_id="str", corrected_quant_config_ref="qc", corrected_run_ref="r",
    audit_report_ref="a", failed_check_ids=("lib_gap",), failed_check_verdicts={"lib_gap": "FAIL"},
    applicable_toggles=("lib_gap", "lab_trim"),
    development_window=DevelopmentWindow(start="2004-08", end="2021-12"),
    holdout_status=HoldoutStatus(accessible=False))
ELIG = [evaluate(m, strategy_family="CHARACTERISTIC_SORT", holding_period=1, templates=LIB.templates,
                 variable_families=LIB.variable_families, available_variables=AVAILABLE)
        for m in LIB.mechanisms]

_VALID = {"mechanism_ref": "mech_003", "template_ref": "lagged_binary_regime_interaction_v1",
          "conditioning_variable": "baa_aaa_spread", "conditioning_lag_months": 1,
          "interaction_form": "binary_above_historical_median", "rationale": "x", "prediction": "y"}


class _StubClient:
    name = "stub"

    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate(self, prompt, *, seed):
        self.calls.append(seed)
        return self.response


def _source(response, cache=None):
    return LLMResearcherSource(_StubClient(response), cache=cache)


# ---- cache --------------------------------------------------------------------------------

def test_cache_round_trip_and_key(tmp_path):
    c = ResponseCache(tmp_path)
    assert c.get("p", "m", 0) is None
    c.put("p", "m", 0, "resp")
    assert c.get("p", "m", 0) == "resp"
    assert c.key("p", "m", 0) == c.key("p", "m", 0)            # deterministic
    assert c.key("p", "m", 0) != c.key("p", "m", 1)            # seed is part of the key


# ---- generation through the loop ----------------------------------------------------------

def test_valid_invalid_duplicate_are_counted():
    response = json.dumps([_VALID, {}, dict(_VALID)])          # valid, malformed, duplicate
    r = run_generation(_source(response), CASE, ELIG, LIB, seed=0, m=3, model="stub",
                       prompt_version="v1", generated_at="2026-08-02T00:00:00Z")
    assert r.n_valid_unique == 1 and r.n_invalid == 1 and r.n_duplicate == 1


def test_cache_hit_avoids_a_second_model_call(tmp_path):
    src = _source(json.dumps([_VALID]), cache=ResponseCache(tmp_path))
    kw = dict(seed=0, m=1, model="stub", prompt_version="v1", generated_at="t")
    src.candidates(CASE, ELIG, LIB, **kw)
    src.candidates(CASE, ELIG, LIB, **kw)                      # second call -> cache hit
    assert src.client.calls == [0]                            # model invoked exactly once


def test_unparseable_response_yields_no_candidates():
    assert _source("not json").candidates(CASE, ELIG, LIB, seed=0, m=3, model="stub",
                                          prompt_version="v1", generated_at="t") == []


# ---- the wall + deferred client -----------------------------------------------------------

def test_prompt_is_magnitude_free():
    from agents.scientist.researcher.context_builder import build_context
    prompt = build_prompt(build_context(CASE, ELIG, LIB), m=6).lower()
    for banned in ("sharpe", "alpha", "t_stat", "p_value", "bh_adjusted", "mean_return"):
        assert banned not in prompt


def test_deferred_client_raises():
    with pytest.raises(RuntimeError):
        DeferredModelClient().generate("p", seed=0)
