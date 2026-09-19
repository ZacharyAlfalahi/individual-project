"""WS-D — the RQ4 development-funnel orchestrator. Offline core-wiring: case (from the corrected
str AuditReport) → eligibility → random+retrieval generation → G0-G5 → reporting. No LLM client,
no network. Skips if the corrected audit report / dev panel are absent."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from agents.scientist.researcher.phase_d_client import build_phase_f_clients  # noqa: E402
from agents.scientist.schemas.outcomes import REFUSAL_CODES  # noqa: E402


def _funnel():
    return __import__("run_rq4_funnel")


def test_build_case_str_enters():
    f = _funnel()
    if not f._STR_AUDIT.exists():
        pytest.skip("corrected str AuditReport absent (local pipeline output, not shipped with the repository)")
    case, params = f.build_case()
    assert case.strategy_id == "str"
    assert case.failed_check_ids == ("lib_gap",)          # str enters on lib_gap (θ=0.001, q=0.10)
    assert params.theta == 0.001 and params.q == 0.10


def test_corrected_parent_direction_is_positive():
    f = _funnel()
    try:
        panel, base_rulebook, parent, direction = f.corrected_str_parent()
    except FileNotFoundError:
        pytest.skip("dev panel absent")
    assert base_rulebook["signal_lag"] == 1               # lib_gap ON = corrected
    assert direction == 1 and float(parent.mean()) > 0    # realised corrected premium is positive


def test_funnel_chains_offline_random_and_retrieval():
    f = _funnel()
    if not f._STR_AUDIT.exists():
        pytest.skip("corrected str AuditReport absent (local pipeline output, not shipped with the repository)")
    try:
        r = f.run_rq4_funnel(phase="dev", llm_clients=[], k=1, m=2, run_rehearsal=False)
    except FileNotFoundError:
        pytest.skip("dev market data absent")
    assert r["entered"] is True and r["reportable"] is False
    assert set(r["sources"]) == {"random_eligible", "retrieval_only"}
    for _src, rep in r["reports"].items():
        # every stage present, funnel monotone, refusal profile carries all 13 codes
        for stage in ("generated", "valid", "compiled", "executed", "audit_clean",
                      "bh_survivor", "cpcv_qualified", "holdout_evaluated"):
            assert stage in rep["economic_funnel"]
        assert set(rep["refusal_profile"]) == set(REFUSAL_CODES)
        assert isinstance(rep["outcome_funnel"], dict)
        assert isinstance(rep["advanced"], list)


def test_rehearsal_skipped_when_zero_survivors_is_terminal():
    f = _funnel()
    if not f._STR_AUDIT.exists():
        pytest.skip("corrected str AuditReport absent (local pipeline output, not shipped with the repository)")
    try:
        r = f.run_rq4_funnel(phase="dev", llm_clients=[], k=1, m=2, run_rehearsal=True)
    except FileNotFoundError:
        pytest.skip("dev market data absent")
    # random+retrieval alone are unlikely to advance; if none advance, the holdout is NOT opened
    if r["rehearsal"]["advanced_total"] == 0:
        assert r["rehearsal"]["ran"] is False
        assert "not opened" in r["rehearsal"]["note"]


def test_build_phase_f_clients_fails_loud_without_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        build_phase_f_clients()


# ---- --embedder flag: offline stub (default, byte-identical) vs registered MiniLM ----------

_FAKE_HEADER = {"embedder": "minilm", "model_name": "all-MiniLM-L6-v2",
                "sentence_transformers_version": "0.test", "torch_version": "0.test",
                "note": "fake header (test)"}


def test_resolve_embedder_offline_selects_stub_nonreportable():
    f = _funnel()
    embed, label, reportable, header = f.resolve_embedder("offline")
    assert label == "minilm_offline" and reportable is False and header is None
    texts = ["credit risk premium", "bond illiquidity regime"]
    ref = f.deterministic_embedder()
    assert np.array_equal(embed(texts), ref(texts))       # the SHA-256 stub, bit-for-bit


def test_resolve_embedder_offline_honours_injection():
    f = _funnel()
    def fake(texts):
        return np.zeros((len(texts), 3))
    embed, label, reportable, header = f.resolve_embedder("offline", fake)
    assert embed is fake and label == "minilm_offline" and reportable is False


def test_resolve_embedder_minilm_routes_through_sources_and_is_reportable(monkeypatch):
    f = _funnel()
    def fake(texts):
        return np.ones((len(texts), 4))
    monkeypatch.setattr(f.S, "minilm_embedder", lambda: fake)
    monkeypatch.setattr(f, "minilm_header", lambda: dict(_FAKE_HEADER))
    embed, label, reportable, header = f.resolve_embedder("minilm")
    assert embed is fake                                  # the real embedder PATH, not the stub
    assert label == "minilm" and reportable is True
    assert header["embedder"] == "minilm"


def test_resolve_embedder_unknown_mode_raises():
    f = _funnel()
    with pytest.raises(ValueError):
        f.resolve_embedder("bag_of_words")


def test_minilm_mode_fails_loud_without_sentence_transformers(monkeypatch):
    # sys.modules[name] = None halts the import -> minilm_embedder raises RuntimeError.
    # NO fallback to the stub is possible: resolve_embedder("minilm") never touches it.
    f = _funnel()
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(RuntimeError, match="sentence-transformers"):
        f.resolve_embedder("minilm")


def test_minilm_header_records_versions_when_installed():
    import importlib.util
    if importlib.util.find_spec("sentence_transformers") is None:
        pytest.skip("sentence-transformers not installed")
    hdr = _funnel().minilm_header()
    assert hdr["model_name"] == "all-MiniLM-L6-v2"
    assert hdr["sentence_transformers_version"] and hdr["torch_version"]


def test_minilm_output_name_never_clobbers_recorded_artifacts():
    f = _funnel()
    recorded = {"rq4_funnel_reported.json", "rq4_funnel_dev.json",
                "rq4_funnel.json", "rq4_funnel.md"}
    for phase in ("dev", "reported"):
        name = f.output_filename(phase, "minilm")
        assert name not in recorded and "_minilm" in name and name.endswith(".json")
    # offline keeps the historical names — the byte-identical regression bar
    assert f.output_filename("dev", "offline") == "rq4_funnel_dev.json"
    assert f.output_filename("reported", "offline") == "rq4_funnel_reported.json"


def test_cli_embedder_flag_default_offline_and_choices():
    ap = _funnel().build_arg_parser()
    assert ap.parse_args([]).embedder == "offline"
    assert ap.parse_args(["--embedder", "minilm"]).embedder == "minilm"
    with pytest.raises(SystemExit):
        ap.parse_args(["--embedder", "bag_of_words"])


def test_main_passes_embedder_mode_through(monkeypatch):
    f = _funnel()
    seen = {}
    def fake_run(**kw):
        seen.update(kw)
        return {"entered": False, "note": "stubbed"}
    monkeypatch.setattr(f, "run_rq4_funnel", fake_run)
    monkeypatch.setattr(f, "minilm_header", lambda: dict(_FAKE_HEADER))
    assert f.main(["--embedder", "minilm", "--no-rehearsal"]) == 0
    assert seen["embedder_mode"] == "minilm"
    seen.clear()
    assert f.main(["--no-rehearsal"]) == 0
    assert seen["embedder_mode"] == "offline"             # default = historical behaviour


def test_funnel_minilm_annotates_retrieval_arm_offline_stays_clean(monkeypatch):
    """Funnel-level: minilm mode stamps the embedder header + retrieval-arm reportable=True;
    offline mode emits NO new keys (the byte-identical bar). Fake embedder — no model/network."""
    f = _funnel()
    if not f._STR_AUDIT.exists():
        pytest.skip("corrected str AuditReport absent (local pipeline output, not shipped with the repository)")
    monkeypatch.setattr(f.S, "minilm_embedder", lambda: f.deterministic_embedder())
    monkeypatch.setattr(f, "minilm_header", lambda: dict(_FAKE_HEADER))
    try:
        r_min = f.run_rq4_funnel(phase="dev", llm_clients=[], embedder_mode="minilm",
                                 k=1, m=2, run_rehearsal=False)
        r_off = f.run_rq4_funnel(phase="dev", llm_clients=[], k=1, m=2, run_rehearsal=False)
    except FileNotFoundError:
        pytest.skip("dev market data absent")
    assert r_min["embedder"] == _FAKE_HEADER
    assert r_min["reports"]["retrieval_only"]["reportable"] is True
    assert r_min["reports"]["retrieval_only"]["embedder_note"] == _FAKE_HEADER["note"]
    assert "reportable" not in r_min["reports"]["random_eligible"]     # only the retrieval arm
    assert "embedder" not in r_off                                     # offline: no new top-level key
    assert "reportable" not in r_off["reports"]["retrieval_only"]      # offline: no new per-arm keys
    assert "embedder_note" not in r_off["reports"]["retrieval_only"]
