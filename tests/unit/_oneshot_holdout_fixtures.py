"""Shared fixtures/builders for the one-shot holdout tests (helper module, not a conftest).

The valid checklist config reuses the REAL committed repo files (protocol, thresholds, release
tag) so the window/tag/fingerprint checks pass against genuine state, with temp files standing in
for the P3 / rehearsal artefacts. The synthetic panel builder + survivors let the full rehearsal
run on development-shaped data — nothing reads /data/holdout/.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from agents.scientist.experimentalist.oneshot_holdout.gate_checklist import ChecklistConfig
from agents.scientist.experimentalist.oneshot_holdout.windows import Window

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_PROTOCOL = REPO_ROOT / "docs" / "scientist_protocol.yaml"
REAL_THRESHOLDS = REPO_ROOT / "docs" / "thresholds.yaml"
RELEASE_TAG = "sc-sci-13-holdout-inference"


def real_thresholds_fingerprint() -> str:
    return hashlib.sha256(REAL_THRESHOLDS.read_bytes()).hexdigest()


def valid_checklist_cfg(tmp_path: Path, *, require_rehearsal: bool = False) -> ChecklistConfig:
    p3 = tmp_path / "p3_moderate_prior.json"
    p3.write_text('{"moderate_sigma": 0.006}')
    rehearsal_marker = tmp_path / "rehearsal_marker.jsonl"
    if require_rehearsal:
        rehearsal_marker.write_text('{"state": "REHEARSAL_GREEN", "ts": "t"}\n')
    from agents.scientist.experimentalist.oneshot_holdout.run_oneshot_holdout import _dir_code_hash
    return ChecklistConfig(
        release_tag=RELEASE_TAG,
        protocol_path=REAL_PROTOCOL,
        thresholds_path=REAL_THRESHOLDS,
        thresholds_fingerprint=real_thresholds_fingerprint(),
        frozen_script_hash=_dir_code_hash(),          # so the real-path test's gate can open (M1)
        p3_artefact_path=p3,
        e9_cost_model_id="dev_gross_returns_scoping_v1",
        manifest_writer_wired=True,
        rehearsal_marker_path=rehearsal_marker,
        require_rehearsal=require_rehearsal,
        repo_root=REPO_ROOT,
    )


def synthetic_panel_builder(inventory=("drf", "crf", "ipca_oos", "fisd_ratings")):
    """A PanelBuilder that returns one-shot holdout-inventory-shaped frames non-NaN from window.start,
    with a warm-up run from seed_start (mimicking a rolling construction seeded on development)."""

    def _build(*, seed_start: str, window: Window):
        idx = pd.period_range(seed_start, window.end, freq="M").to_timestamp("M")
        warm = pd.Period(window.start, "M").to_timestamp("M")
        rng = np.random.default_rng(11)
        out = {}
        for i, name in enumerate(inventory):
            vals = rng.normal(0, 0.02, len(idx)).astype(float)
            # Pre-warm-up (seeding period) is NaN; from the first evaluation month it is warm.
            vals[idx < warm] = np.nan
            out[name] = pd.DataFrame({"date": idx, name: vals})
        return out

    return _build


def synthetic_survivors(window: Window):
    from agents.scientist.experimentalist.oneshot_holdout.stage2_evaluate import SurvivorInput
    months = pd.period_range(window.start, window.end, freq="M").to_timestamp("M")
    rng = np.random.default_rng(5)
    surv = pd.Series(0.004 + rng.normal(0, 0.02, len(months)), index=months)
    par = pd.Series(0.001 + rng.normal(0, 0.02, len(months)), index=months)
    ext1 = pd.Series(0.002 + rng.normal(0, 0.02, len(months)), index=months)
    return [
        SurvivorInput("s1", surv, par),
        SurvivorInput("extension_1", ext1, par, is_extension_1=True),
    ]


def synthetic_benchmarks(window: Window):
    months = pd.period_range(window.start, window.end, freq="M").to_timestamp("M")
    rng = np.random.default_rng(9)
    frame = pd.DataFrame({
        "date": months,
        "mktb": rng.normal(0, 0.02, len(months)),
        "drf": rng.normal(0, 0.02, len(months)),
    })
    return {"bbw4": frame}
