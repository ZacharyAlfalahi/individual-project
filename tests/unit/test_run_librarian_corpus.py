"""B4: the corpus batch driver (skip-and-continue, typed-exit recording, the T5
exit-map contract) + the run_librarian manifest now emitted on the 2/4 exits."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from agents.librarian.pipeline.real_client import RealClientError       # noqa: E402
from scripts import run_librarian                                        # noqa: E402
from scripts import run_librarian_corpus as corpus                       # noqa: E402


# ---------------------------------------------------------------------------
# Loop semantics (stubbed run_librarian.main).
# ---------------------------------------------------------------------------

def _stub_main(codes: dict[str, int | Exception]):
    def main(argv):
        key = argv[argv.index("--paper") + 1]
        out = Path(argv[argv.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        rc = codes[key]
        if isinstance(rc, Exception):
            raise rc
        return rc
    return main


def test_skip_and_continue_records_every_typed_exit(monkeypatch, tmp_path):
    monkeypatch.setattr(run_librarian, "main",
                        _stub_main({"bbw": 0, "jnps": 2, "drr": 4}))
    log = corpus.run_set(("bbw", "jnps", "drr"), tmp_path, ["--phase", "fake"])
    assert [log["papers"][k]["exit"] for k in ("bbw", "jnps", "drr")] == [0, 2, 4]
    assert log["counts"] == {"ok": 1, "review": 1, "paper_failed": 1}
    assert log["crashed"] is False


def test_crash_is_recorded_and_batch_continues(monkeypatch, tmp_path):
    monkeypatch.setattr(run_librarian, "main",
                        _stub_main({"bbw": 0, "jnps": ValueError("boom"), "drr": 3}))
    log = corpus.run_set(("bbw", "jnps", "drr"), tmp_path, [])
    assert log["papers"]["jnps"]["exit"] is None
    assert "CRASH" in log["papers"]["jnps"]["outcome"]
    assert log["papers"]["drr"]["exit"] == 3        # the batch went on
    assert log["crashed"] is True


def test_missing_key_is_fatal_for_all(monkeypatch, tmp_path):
    monkeypatch.setattr(run_librarian, "main",
                        _stub_main({"bbw": RealClientError("missing API key")}))
    with pytest.raises(SystemExit):
        corpus.run_set(("bbw", "jnps"), tmp_path, [])


def test_exit_map_uses_handles_and_t5_shape(monkeypatch, tmp_path):
    """main() writes the {handle: exit_code} JSON the T5 grader consumes."""
    monkeypatch.setattr(run_librarian, "main", _stub_main({"bbw2021": 2, "dfps": 0}))
    exit_map = tmp_path / "t5_exit.json"
    rc = corpus.main(["--set", "corpus", "--out-root", str(tmp_path / "out"),
                      "--exit-map", str(exit_map)])
    assert rc == 0
    m = json.loads(exit_map.read_text(encoding="utf-8"))
    assert m == {"bbw2021": 2, "dfps": 0}           # PAPERS key = default handle
    log = json.loads((tmp_path / "out" / "corpus_log.json").read_text(encoding="utf-8"))
    assert log["counts"] == {"review": 1, "ok": 1}


def test_sets_are_explicit_never_sorted_papers():
    """The T4(b) instrument must never ride a corpus set; anchors and corpus stay
    disjoint; every set member is a registered PAPERS key."""
    assert "synth" not in corpus.SETS["anchors"] + corpus.SETS["corpus"]
    assert set(corpus.SETS["anchors"]).isdisjoint(corpus.SETS["corpus"])
    for keys in corpus.SETS.values():
        for k in keys:
            assert k in run_librarian.PAPERS


# ---------------------------------------------------------------------------
# Integration: one real fake-phase paper through the loop.
# ---------------------------------------------------------------------------

def test_real_fake_run_through_the_loop(tmp_path):
    # The corpus driver reads the bbw canonical text and records a CRASH (not a raised
    # exception the report hook could catch) when it is absent, so guard explicitly: a
    # public clone does not ship evaluation/canonical_texts/ (copyrighted; see README).
    _bbw = Path(__file__).resolve().parents[2] / "evaluation" / "canonical_texts" / "bbw_2019.frozen.yaml"
    if not _bbw.exists():
        pytest.skip("requires the gitignored canonical text (evaluation/canonical_texts/; not shipped — see README)")
    log = corpus.run_set(("bbw",), tmp_path,
                         ["--phase", "fake", "--enumeration", "gold"])
    assert log["papers"]["bbw"]["exit"] == 0
    out = Path(log["papers"]["bbw"]["out"])
    assert (out / "spec_0.json").exists()
    assert (out / "run_manifest.json").exists()


# ---------------------------------------------------------------------------
# run_librarian: the manifest now lands on the review/failure exits too (B4).
# ---------------------------------------------------------------------------

def test_manifest_emitted_on_paper_failed_exit(monkeypatch, tmp_path):
    def boom(**kwargs):
        raise RealClientError("retries exhausted (429)")
    monkeypatch.setattr(run_librarian, "run_paper", boom)
    out = tmp_path / "out4"
    rc = run_librarian.main(["--paper", "bbw", "--phase", "fake", "--out", str(out)])
    assert rc == 4
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["manifest_schema"] == 2
    assert manifest["outputs"] == {}                # no spec emitted on a failed run


def test_manifest_emitted_on_review_exit(monkeypatch, tmp_path):
    from scripts.run_librarian import AssemblyIncomplete

    def boom(**kwargs):
        raise AssemblyIncomplete("sort signal unresolved")
    monkeypatch.setattr(run_librarian, "run_paper", boom)
    out = tmp_path / "out2"
    rc = run_librarian.main(["--paper", "bbw", "--phase", "fake", "--out", str(out)])
    assert rc == 2
    assert (out / "run_manifest.json").exists()


# ---------------------------------------------------------------------------
# B6: the T5 Arm-B reject set is registered and runnable.
# ---------------------------------------------------------------------------

def test_reject_set_registration_matches_the_prereg():
    """The rejects set derives from REJECT_HANDLE_OF; handles must match the
    ratified reject_set.yaml exactly (count_prereg 8), and every PAPERS entry
    must carry a canonical_text and NO gold_enum (live enumeration is the
    graded behaviour)."""
    import yaml
    rs = yaml.safe_load((_REPO_ROOT / "evaluation" / "adversarial" / "reject_set.yaml")
                        .read_text(encoding="utf-8"))
    prereg_handles = {p["handle"] for p in rs["papers"]}
    assert len(prereg_handles) == rs["count_prereg"] == 8
    assert set(corpus.REJECT_HANDLE_OF.values()) == prereg_handles

    for key in corpus.SETS["rejects"]:
        entry = run_librarian.PAPERS[key]
        assert "gold_enum" not in entry, f"{key}: a reject paper must never carry a gold"
        assert entry["canonical_text"].startswith("evaluation/adversarial/frozen/")


def test_reject_frozen_texts_load(tmp_path):
    """Machine-local (gitignored) frozen texts: when present they must satisfy
    require_frozen -- the same gate run_librarian applies."""
    from agents.librarian.config import load_canonical_text
    missing = []
    for key in corpus.SETS["rejects"]:
        p = _REPO_ROOT / run_librarian.PAPERS[key]["canonical_text"]
        if not p.exists():
            missing.append(key)
            continue
        ct = load_canonical_text(p)
        ct.require_frozen()
    if len(missing) == len(corpus.SETS["rejects"]):
        pytest.skip("reject frozen texts not on this machine (gitignored)")


# ---------------------------------------------------------------------------
# Code-review MAJOR (2026-09-02): re-running into a populated out_dir must be
# IDEMPOTENT -- the append-mode raw archive previously accumulated a second
# line per field and tripped the run_artefacts.check_raw integrity guard.
# ---------------------------------------------------------------------------

def test_rerun_into_same_out_dir_is_idempotent(tmp_path):
    """Two full runs into ONE out_dir: spec/trace/manifest present once, the
    raw-archive line counts still correspond to the trace, and load_run's
    integrity guard passes. (Fake phase writes no raw archive, so the guard is
    exercised via load_run with check_raw=True over whatever exists.)"""
    from evaluation.harness.run_artefacts import load_run

    out = tmp_path / "same_dir"
    assert run_librarian.main(["--paper", "bbw", "--phase", "fake", "--out", str(out)]) == 0
    assert run_librarian.main(["--paper", "bbw", "--phase", "fake", "--out", str(out)]) == 0

    specs = sorted(p.name for p in out.glob("spec_*.json"))
    # BBW emits 3 specs since the 2026-09-03 enum extension (DRF+CRF+LRF); the
    # invariant under test is NO ACCUMULATION across re-runs (3, never 6).
    assert specs == ["spec_0.json", "spec_1.json", "spec_2.json"]
    art = load_run(out, check_raw=True)             # integrity guard passes
    assert art.fields
    # (Spec bytes are NOT asserted identical: a re-run is a NEW counted run and
    # its provenance header rightly carries a fresh run_id/timestamp -- D31/I3.)


def test_stale_artefacts_cleared_before_a_failing_run(tmp_path, monkeypatch):
    """A re-run that FAILS must not leave the previous run's spec/trace lying
    beside a fresh failure manifest (a stale spec next to a paper_failed exit
    would look like output the failed run produced)."""
    out = tmp_path / "stale"
    assert run_librarian.main(["--paper", "bbw", "--phase", "fake", "--out", str(out)]) == 0
    assert (out / "spec_0.json").exists()

    def boom(**kwargs):
        raise RealClientError("retries exhausted (429)")
    monkeypatch.setattr(run_librarian, "run_paper", boom)
    assert run_librarian.main(["--paper", "bbw", "--phase", "fake", "--out", str(out)]) == 4
    emitted = {p.name for p in out.iterdir()}
    assert emitted <= {"run_manifest.json", "raw"}   # D31: no stale spec/trace survives


def test_missing_key_failure_preserves_prior_run_artefacts(tmp_path, monkeypatch):
    """Review footgun (2026-09-02): stale-clearing runs AFTER build_clients, so a
    run that cannot even build its clients (missing key, fatal by design) must
    leave a previous run's artefacts untouched."""
    out = tmp_path / "precious"
    assert run_librarian.main(["--paper", "bbw", "--phase", "fake", "--out", str(out)]) == 0
    assert (out / "spec_0.json").exists()

    monkeypatch.setattr(run_librarian, "_load_dotenv", lambda *_a, **_k: None)
    for var in ("GEMINI_API_KEY", "MISTRAL_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RealClientError):
        run_librarian.main(["--paper", "bbw", "--phase", "dev", "--out", str(out)])
    assert (out / "spec_0.json").exists()           # the paid artefacts survived
