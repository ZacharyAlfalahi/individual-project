"""Corpus confirmatory FDR driver (scripts/run_corpus_confirmatory_fdr.py).

Synthetic reports pin membership (pilot + negative control excluded), the union family size, the
pooled step-up against a hand-computed answer, and every fail-loud cross-check. Tripwires pin
the recorded artefacts when they are present.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import run_corpus_confirmatory_fdr as C
from agents.auditor.validation.hypothesis_registry import FactorHypothesis

_COORDS = ["meas_err", "stale_price", "survivorship", "lib_gap", "lab_trim",
           "meas_err×stale_price", "lib_gap×lab_trim", "meas_err×survivorship"]
_TOGGLES = ["meas_err", "stale_price", "survivorship", "lib_gap", "lab_trim"]


def _hyp(fid: str, status: str, locked: bool = True) -> FactorHypothesis:
    return FactorHypothesis(fid, "x", 1, "sign_only", None, locked, status, "s", "c")


def _registry() -> dict:
    return {"str": _hyp("str", "pilot", locked=False), "mom6": _hyp("mom6", "locked"),
            "drf": _hyp("drf", "locked"), "traded_liquidity": _hyp("traded_liquidity", "negative_control")}


def _report(pvalues: list[float], *, q: float = 0.1, tag: str = "t") -> dict:
    return {
        "runnable_toggles": list(_TOGGLES),
        "pre_registration_tag": tag,
        "inference": {c: {"p_value": p} for c, p in zip(_COORDS, pvalues)},
        "fdr": {"q": q, "scope": "within_strategy",
                "decisions": {c: {"p_value": p, "adjusted_p": p, "rejected": False}
                              for c, p in zip(_COORDS, pvalues)}},
    }


def test_membership_excludes_pilot_and_negative_control():
    members, excluded = C.confirmatory_members(_registry())
    assert members == ["mom6", "drf"]
    assert set(excluded) == {"str", "traded_liquidity"}


def test_pooled_family_matches_hand_computed_step_up():
    # 16 p-values: one strong (0.001) + 15 at 0.5. BH: rank1 -> 0.001*16 = 0.016 (rejected at
    # 0.10); the rest -> 0.5 (step-up from the top: 0.5*16/16).
    mom6 = _report([0.001] + [0.5] * 7)
    drf = _report([0.5] * 8)
    out = C.build_corpus_confirmatory({"mom6": mom6, "drf": drf}, _registry(), 0.1)
    fdr = out["fdr"]
    assert fdr["scope"] == "corpus_confirmatory"
    assert fdr["n_family"] == 16 and fdr["n_rejected"] == 1
    assert fdr["decisions"]["mom6::meas_err"]["adjusted_p"] == pytest.approx(0.016)
    assert fdr["decisions"]["drf::lab_trim"]["adjusted_p"] == pytest.approx(0.5)
    assert out["min_adjusted_p_pooled"] == {"mom6": pytest.approx(0.016), "drf": pytest.approx(0.5)}


def test_pooled_adjustment_can_fall_below_within_strategy():
    # Pooling is NOT a conservative bound on the within-strategy adjustment: when the other
    # anchor's p-values are all small, an item's pooled rank rises faster than m doubles.
    # mom6::meas_err p=0.04 is 0.04*8/1 = 0.32 within mom6, but rank 9 of 16 pooled -> 16/9*0.04.
    from agents.auditor.checks.fdr import benjamini_hochberg
    mom6 = _report([0.04] + [0.9] * 7)
    drf = _report([0.001] * 8)
    for rep in (mom6, drf):
        within = benjamini_hochberg({c: v["p_value"] for c, v in rep["inference"].items()}, 0.1)
        for c, d in within.items():
            rep["fdr"]["decisions"][c]["adjusted_p"] = d.adjusted_p
    out = C.build_corpus_confirmatory({"mom6": mom6, "drf": drf}, _registry(), 0.1)
    row = next(r for r in out["comparison"] if r["strategy"] == "mom6" and r["coordinate"] == "meas_err")
    assert row["adjusted_p_within_strategy"] == pytest.approx(0.32)
    assert row["adjusted_p_pooled"] == pytest.approx(16 / 9 * 0.04)
    assert row["rejected_pooled"] and not row["rejected_within_strategy"]


def test_refuses_inference_p_disagreeing_with_fdr_block():
    bad = _report([0.5] * 8)
    bad["fdr"]["decisions"]["lib_gap"]["p_value"] = 0.49
    with pytest.raises(C.CorpusFdrError, match="lib_gap"):
        C.build_corpus_confirmatory({"mom6": _report([0.5] * 8), "drf": bad}, _registry(), 0.1)


def test_refuses_missing_confirmatory_coordinate():
    bad = _report([0.5] * 8)
    del bad["inference"]["meas_err×survivorship"]
    with pytest.raises(C.CorpusFdrError, match="confirmatory coordinates"):
        C.build_corpus_confirmatory({"mom6": _report([0.5] * 8), "drf": bad}, _registry(), 0.1)


def test_refuses_q_mismatch_and_mixed_tags_and_missing_anchor():
    reg = _registry()
    with pytest.raises(C.CorpusFdrError, match="q"):
        C.build_corpus_confirmatory({"mom6": _report([0.5] * 8, q=0.05), "drf": _report([0.5] * 8)}, reg, 0.1)
    with pytest.raises(C.CorpusFdrError, match="tags"):
        C.build_corpus_confirmatory({"mom6": _report([0.5] * 8, tag="a"), "drf": _report([0.5] * 8, tag="b")}, reg, 0.1)
    with pytest.raises(C.CorpusFdrError, match="drf"):
        C.build_corpus_confirmatory({"mom6": _report([0.5] * 8)}, reg, 0.1)


_ARTIFACT = Path(__file__).resolve().parents[2] / "results" / "auditor" / "corpus_confirmatory_fdr.json"


@pytest.mark.skipif(not _ARTIFACT.is_file(),
                    reason="recorded corpus FDR artefact (local pipeline output) not shipped with "
                           "the repository")
def test_recorded_artifact_tripwire():
    a = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    assert a["membership"]["strategies"] == ["mom6", "drf"]
    assert a["fdr"]["scope"] == "corpus_confirmatory"
    assert a["fdr"]["q"] == 0.1
    assert a["fdr"]["n_family"] == 16
    assert a["fdr"]["n_rejected"] == 0
    assert a["min_adjusted_p_pooled"]["drf"] == pytest.approx(0.38556, abs=1e-4)
    assert a["min_adjusted_p_pooled"]["mom6"] == pytest.approx(0.38556, abs=1e-4)
    assert {v["basis"] for v in a["source_run"].values()} == {"clean"}


_ARTIFACT_TR = _ARTIFACT.parent / "corpus_confirmatory_fdr_total_return.json"


@pytest.mark.skipif(not _ARTIFACT_TR.is_file(),
                    reason="recorded total-return corpus FDR artefact (local pipeline output) not "
                           "shipped with the repository")
def test_recorded_total_return_artifact_tripwire():
    a = json.loads(_ARTIFACT_TR.read_text(encoding="utf-8"))
    assert a["membership"]["strategies"] == ["mom6", "drf"]
    assert {v["basis"] for v in a["source_run"].values()} == {"total_return"}
    assert a["fdr"]["n_family"] == 16 and a["fdr"]["n_rejected"] == 0
    assert a["min_adjusted_p_pooled"]["drf"] == pytest.approx(0.19306, abs=1e-4)
    assert a["min_adjusted_p_pooled"]["mom6"] == pytest.approx(0.19306, abs=1e-4)


def test_main_refuses_mixed_bases(tmp_path):
    clean = tmp_path / "clean"
    tr = tmp_path / "tr"
    for d, basis in ((clean, None), (tr, "total_return")):
        d.mkdir()
        (d / "run_log.json").write_text(json.dumps({"basis": basis} if basis else {}))
    src = Path(__file__).resolve().parents[2] / "results" / "auditor" / "recorded"
    if not (src / "mom6_report.json").is_file():
        pytest.skip("recorded audit reports (local pipeline output) not shipped with the repository")
    (clean / "mom6_report.json").write_text((src / "mom6_report.json").read_text())
    (tr / "drf_report.json").write_text((src / "drf_report.json").read_text())
    with pytest.raises(C.CorpusFdrError, match="mix return bases"):
        C.main(["--report", f"mom6={clean / 'mom6_report.json'}",
                "--report", f"drf={tr / 'drf_report.json'}", "--out", str(tmp_path / "o.json")])
