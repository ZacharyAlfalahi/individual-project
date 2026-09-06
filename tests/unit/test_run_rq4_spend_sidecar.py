"""RQ4 spend sidecar — the retroactive reconstruction from the response-only cache. Offline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_rq4_spend_sidecar as SC  # noqa: E402

_PRICES = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "gemini-3.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-3.1-flash-lite": {"input": 0.0, "output": 0.0},
}


def _rec(model, response):
    return {"model": model, "seed": 0, "key": "k", "response": response}


def test_reconstruct_counts_paid_and_free_correctly():
    records = [
        _rec("claude-sonnet-4-6", "x" * 4000),   # ~1000 output tokens
        _rec("gemini-3.5-flash", "y" * 4000),
        _rec("gemini-3.1-flash-lite", "z" * 4000),  # free
    ]
    r = SC.reconstruct_spend(records, _PRICES)
    assert r["paid_calls"] == 2                                   # the two Phase-F models
    assert r["models"]["gemini-3.1-flash-lite"]["paid"] is False
    assert r["models"]["gemini-3.1-flash-lite"]["output_cost_usd"] == 0.0
    # claude output cost = 1000 tok * $15/1M = $0.015
    assert r["models"]["claude-sonnet-4-6"]["output_cost_usd"] == pytest.approx(0.015, abs=1e-4)


def test_input_tokens_flagged_not_captured_and_upper_bounded():
    r = SC.reconstruct_spend([_rec("claude-sonnet-4-6", "x" * 4000)], _PRICES)
    m = r["models"]["claude-sonnet-4-6"]
    assert m["input_tokens_captured"] is False
    assert m["input_upper_bound_tokens"] == SC._INPUT_UPPER_BOUND_TOKENS   # 1 call * bound
    assert r["governance"]["usage_metered_at_runtime"] is False
    assert r["governance"]["reconstruction"] is True


def test_unknown_model_fails_loud():
    with pytest.raises(SC.RQ4SidecarError, match="no pre-registered price"):
        SC.reconstruct_spend([_rec("gpt-4o", "x")], _PRICES)


def test_load_prices_fails_loud_on_missing_block(tmp_path):
    import yaml
    bad = tmp_path / "th.yaml"
    bad.write_text(yaml.safe_dump({"p1_codegen": {}}))
    with pytest.raises(SC.RQ4SidecarError):
        SC.load_prices(bad)


def test_main_writes_zero_spend_sidecar(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "a.json").write_text(json.dumps(_rec("gemini-3.5-flash", "y" * 8000)))
    out = tmp_path / "sidecar.json"
    assert SC.main(["--cache", str(cache), "--out", str(out)]) == 0
    r = json.loads(out.read_text())
    assert r["provenance"]["dev_only"] is True and r["provenance"]["model_calls"] == 0
    assert r["totals"]["total_estimate_upper_bound_usd"] >= r["totals"]["output_cost_usd_metered_from_cache"]


def test_main_missing_cache_fails_loud(tmp_path):
    with pytest.raises(SC.RQ4SidecarError, match="no cache records"):
        SC.main(["--cache", str(tmp_path / "nope"), "--out", str(tmp_path / "o.json")])
