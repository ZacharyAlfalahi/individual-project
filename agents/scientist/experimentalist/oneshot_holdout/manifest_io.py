"""Run-manifest persistence (one-shot holdout §3 / contract Extension 3).

The run manifest — timestamp, code / config / data / output hashes, window, seed, holdout_processed —
is a durable provenance record, written to disk, not a thrown-away dict. Kept in its own tiny module
so the gate checklist can verify a REAL writer exists (not merely a boolean) without importing the
orchestrator (which would be circular).
"""

from __future__ import annotations

import json
from pathlib import Path


def write_manifest(manifest: dict, run_dir: str | Path) -> Path:
    """Persist ``manifest`` as canonical JSON at ``run_dir/manifest.json`` and return the path."""
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return path
