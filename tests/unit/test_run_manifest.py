"""Unit tests for the shared per-run execution manifest (WS-8 / O11)."""
from __future__ import annotations

import json
from pathlib import Path

from shared.reporting.run_manifest import (
    build_run_manifest,
    default_operational_profile,
    write_run_manifest,
)

REPO = Path(__file__).resolve().parents[2]


def test_manifest_shape_and_hash_passthrough():
    # A real config file (exists) hashes to 64 hex; a missing path hashes to None.
    m = build_run_manifest(
        run_id="run_test", driver="run_quant", timestamp="2026-08-10T00:00:00+00:00",
        inputs=["data/development/does_not_exist.parquet"],
        configs=["docs/thresholds.yaml"],
        outputs=[],
    )
    assert m["manifest_schema"] == 1
    assert m["run_id"] == "run_test" and m["driver"] == "run_quant"
    assert m["timestamp"] == "2026-08-10T00:00:00+00:00"            # passed in, never a clock
    assert m["inputs"]["data/development/does_not_exist.parquet"] is None
    cfg_hash = m["configs"]["docs/thresholds.yaml"]
    assert isinstance(cfg_hash, str) and len(cfg_hash) == 64
    assert m["operational_profile"] == default_operational_profile()
    assert set(m["code"]) == {"commit", "short", "dirty"}


def test_manifest_is_pure_given_timestamp():
    kw = dict(
        run_id="r", driver="d", timestamp="2026-08-10T00:00:00+00:00",
        configs=["docs/thresholds.yaml"],
    )
    # No clock read inside: identical inputs -> identical manifest (code hash stable per tree).
    assert build_run_manifest(**kw) == build_run_manifest(**kw)


def test_write_run_manifest_roundtrips(tmp_path):
    m = build_run_manifest(run_id="r", driver="d", timestamp="t")
    p = write_run_manifest(tmp_path, m)
    assert p.name == "run_manifest.json"
    assert json.loads(p.read_text()) == m
