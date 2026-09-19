"""Frozen SHA-256 pins for the core panel artefacts (built locally; the pins are committed).

The core monthly panel artefacts are large binary parquet files held out of
git (the whole ``/data/`` tree is ``.gitignore``d) but are stable, built-once
products of the raw cleaning chain: ``monthly_panel_maximal.parquet`` is the
source of truth, and the uncorrected (all-OFF, as-published) and corrected
(all-ON) endpoint views are exported from it by
``scripts/export_endpoint_views.py``. Their hashes are already recorded as
provenance in ``data/development/monthly_panel_endpoint_reports.json``; the
constants below turn that *recorded provenance* into a *guarded pin*,
recomputed against the on-disk bytes by ``tests/unit/test_panel_pins.py``.

RE-FREEZE DISCIPLINE
    Any deliberate rebuild of a core panel MUST re-pin the corresponding
    constant here in the SAME change, with the rebuild recorded in the
    provenance report. A stale pin fails CLOSED (the recompute test breaks) --
    the intended tripwire against a silent, unrecorded panel change.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# repo-relative path -> frozen sha256 of the pinned panel bytes.
# Values recomputed from the on-disk artefacts on 2026-09-10 and cross-checked
# against monthly_panel_endpoint_reports.json (see test_panel_pins.py).
FROZEN_PANEL_SHA256: dict[str, str] = {
    "data/development/monthly_panel_maximal.parquet": (
        "aef6ede3fe4b1dbc5f9adbbb70287d22c8027c5b077e0286b6edeff68e4e38f7"
    ),
    "data/development/monthly_panel_uncorrected.parquet": (
        "35871c91007cc5993899aaea6f556b98ef4d33ddf51f6ba1e9c6e5c1b18e7d08"
    ),
    "data/development/monthly_panel_corrected.parquet": (
        "ea51075bd754fd284de05ecf1aa781aaf937a5dfd3abc18b702cb9e5a4c864b9"
    ),
}
