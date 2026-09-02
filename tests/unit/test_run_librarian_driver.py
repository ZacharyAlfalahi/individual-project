"""
RQ1/RQ2 -- the Librarian driver (`scripts/run_librarian.py`).

Pins three driver behaviours WITHOUT a live model (Phase-F gated):

  (a) corpus-paper registration -- BBW 2021 + DFPS are registered as gold-enumeration
      papers (a `gold_enum` path, NO `constructions` fake seed), so the run feeds
      `load_gold_list` and bypasses the D20 dual-model gate.

  (b) the offline wiring proof -- `--phase fake` on an anchor paper enumerates via the
      gold list and returns 0.

  (c) run-to-completion on a client failure -- a `RealClientError` raised during
      extraction (e.g. 429 retries exhausted) is caught at the driver and routed to a
      typed `paper_failed` (exit 4), NOT a crash and NOT fabricated UNKNOWN fields
      (the `not_extracted` != UNKNOWN distinction; D31 no-partial; I3 no silent resume).
"""

from __future__ import annotations

from agents.librarian.pipeline import FakeModelClient
from agents.librarian.pipeline.real_client import RealClientError
from scripts import run_librarian
from scripts.run_librarian import PAPERS, main


def test_corpus_papers_registered_via_gold_enum():
    for key, pid in (("bbw2021", "BBW_2021"), ("dfps", "DFPS_2026")):
        assert key in PAPERS, f"{key} not registered"
        assert PAPERS[key]["paper_id"] == pid
        assert "gold_enum" in PAPERS[key], "corpus paper must carry a gold_enum path"
        # No fake seed: corpus papers run via gold enumeration only (--phase fake is the
        # BBW-2019-specific wiring proof, whose scripted quotes locate in BBW 2019 alone).
        assert "constructions" not in PAPERS[key]


def test_fake_phase_wiring_returns_zero(tmp_path):
    assert main(["--paper", "bbw", "--phase", "fake", "--out", str(tmp_path / "ok")]) == 0


def test_client_failure_routes_to_paper_failed_not_crash(tmp_path, monkeypatch, capsys):
    def boom(*_a, **_k):
        raise RealClientError(
            "gemini:x failed on field SORT_SIGNAL after 6 attempts: HTTPError: 429"
        )

    monkeypatch.setattr(run_librarian, "run_paper", boom)
    out = tmp_path / "boom"
    rc = main(["--paper", "bbw", "--phase", "fake", "--out", str(out)])

    assert rc == 4  # typed paper_failed, not a crash
    assert "paper_failed" in capsys.readouterr().out
    # D31: a client failure emits NO partial spec set. The WS-8 run-manifest
    # SIDECAR is allowed (B4): the calls already made are real spend, recorded
    # honestly -- it is an operational log, never a result artefact.
    emitted = {p.name for p in out.iterdir()}
    assert emitted <= {"run_manifest.json"}
    assert not any(n.startswith(("spec_", "trace_")) for n in emitted)


def test_missing_constructions_seed_does_not_crash(tmp_path):
    # A corpus paper carries no `constructions` fake seed; _fake_pair must fall back to
    # an empty seed (`.get(..., ())`), not KeyError. Under --enumeration live the empty
    # seed yields zero constructions -> REVIEW (exit 2), never a crash.
    rc = main(["--paper", "dfps", "--phase", "fake", "--enumeration", "live",
               "--out", str(tmp_path / "seed")])
    assert rc == 2


def test_enumeration_client_failure_routes_to_paper_failed(tmp_path, monkeypatch):
    # M1 regression: a client failure at the LIVE enumeration gate (where WS-3 observed
    # 429s) must degrade to paper_failed (exit 4), same as a per-field failure -- not crash.
    def boom(self, _ct):
        raise RealClientError("mistral:small enumeration: 429 retries exhausted")

    monkeypatch.setattr(FakeModelClient, "extract_enumeration", boom)
    rc = main(["--paper", "bbw", "--phase", "fake", "--enumeration", "live",
               "--out", str(tmp_path / "enum")])
    assert rc == 4
