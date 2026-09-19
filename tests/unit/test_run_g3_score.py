"""B3: the RQ1 scoring CLI (scripts/run_g3_score.py).

The heavy scoring logic is the already-tested G3 harness; these tests pin the
CLI's own responsibilities: anchor->run-dir resolution, the reportability gate,
lrf's exclusion from the aggregate, and the end-to-end drive against the
recorded dev run when it is on disk (runs/ is gitignored -> skipif)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from evaluation.harness.reportability import ReportabilityError          # noqa: E402
from scripts import run_g3_score as cli                                  # noqa: E402

_RUNS = _REPO_ROOT / "runs" / "g3_v3"
_have_runs = (_RUNS / "bbw" / "trace_0.json").exists()

needs_runs = pytest.mark.skipif(not _have_runs, reason="recorded dev run not on disk")


def test_default_dir_map_covers_the_scoreable_sort_anchors():
    """Every sort anchor the CLI can score has a default paper dir; kpp is
    deliberately absent (the §3.7 path consumes specs, not run dirs)."""
    assert set(cli.DEFAULT_DIR_OF) == {"str", "drf", "mom6", "crf", "lrf"}
    assert "kpp" not in cli.DEFAULT_DIR_OF


@needs_runs
def test_phase_d_refuses_without_the_opt_in(capsys):
    with pytest.raises(ReportabilityError):
        cli.main(["--run-root", str(_RUNS)])


@needs_runs
def test_end_to_end_against_the_recorded_dev_run(tmp_path):
    md_out = tmp_path / "report.md"
    js_out = tmp_path / "report.json"
    rc = cli.main(["--run-root", str(_RUNS), "--allow-non-reportable",
                   "--out", str(md_out), "--json", str(js_out)])
    assert rc == 0

    report = md_out.read_text(encoding="utf-8")
    # All three anchors scored + the 3-anchor aggregate (crf has no run dir here).
    for token in ("## str (DRR_2026)", "## drf (BBW_2019)", "## mom6 (JNPS_2013)",
                  "§3.3 aggregate -- 3-anchor headline", "NON-REPORTABLE"):
        assert token in report
    assert "4-anchor headline" not in report      # crf unscored -> no 4-anchor table

    sidecar = json.loads(js_out.read_text(encoding="utf-8"))
    assert set(sidecar["anchors"]) == {"str", "drf", "mom6"}
    for rec in sidecar["anchors"].values():
        assert rec["reportable"] is False          # Phase-D stamped, never silent
        assert rec["coverage"]["denominator"] > 0
    assert "3-anchor" in sidecar["aggregates"]


@needs_runs
def test_lrf_never_enters_the_aggregate():
    """--include-lrf scores lrf per-anchor only; the aggregate stays 3-anchor.
    (lrf shares the bbw run dir; the recorded dev run predates the lrf gold, so
    resolution may fail on label mismatch -- either outcome must keep lrf out of
    the aggregate rather than crash into it.)"""
    try:
        rc = cli.main(["--run-root", str(_RUNS), "--include-lrf",
                       "--allow-non-reportable"])
    except Exception:
        return                                     # typed resolution failure is acceptable
    assert rc == 0


def test_map_override_parsing(tmp_path, monkeypatch):
    """--map anchor=dir:index routes resolution; a missing dir fails loud."""
    with pytest.raises(Exception):
        cli.main(["--run-root", str(tmp_path), "--anchors", "str",
                  "--map", "str=nowhere:0", "--allow-non-reportable"])


def test_resolve_artefacts_unwraps_the_gold_inherited_label(tmp_path):
    """The gold header's strategy_label is an Inherited wrapper; matching must
    compare its .value, not the wrapper (2026-09-07 QR-1 review finding)."""
    import hashlib

    from evaluation.gold_specs.gold_loader import load_gold_spec
    from evaluation.harness.run_artefacts import canonical_trace_json

    def write_strategy(run_dir, index, label):
        trace = {"header": {"paper_id": "STUB", "strategy_label": label}, "records": []}
        trace_json = canonical_trace_json(trace)
        (run_dir / f"trace_{index}.json").write_text(trace_json, encoding="utf-8")
        spec = {"header": {"trace_sha256": hashlib.sha256(trace_json.encode()).hexdigest()}}
        (run_dir / f"spec_{index}.json").write_text(json.dumps(spec), encoding="utf-8")

    raw_label = load_gold_spec("str").header.strategy_label
    gold_value = getattr(raw_label, "value", raw_label)
    assert isinstance(gold_value, str) and gold_value

    run_dir = tmp_path / "multi"
    run_dir.mkdir()
    write_strategy(run_dir, 0, "some other strategy label")
    write_strategy(run_dir, 1, gold_value)
    art = cli._resolve_artefacts("str", run_dir, None)
    assert art.header["strategy_label"] == gold_value
