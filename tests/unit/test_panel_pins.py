"""Guarded pin for the committed core panel artefacts.

Recomputes the SHA-256 of each core panel from its on-disk bytes and asserts
it still matches the frozen pin in ``shared/reporting/panel_pins.py``. The
panels are large gitignored artefacts, so each check skips (rather than fails)
when the file is absent locally -- the established idiom for tests over the
committed ``/data/development/`` artefacts (cf. test_ipca_feed_builder.py).
"""

import hashlib
import json
from pathlib import Path

import pytest

from shared.reporting.panel_pins import FROZEN_PANEL_SHA256, REPO_ROOT

_REPORT = REPO_ROOT / "data" / "development" / "monthly_panel_endpoint_reports.json"


def _recompute(path: Path) -> str:
    """Chunked SHA-256 of a file (panels are 50-150 MB)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.mark.parametrize("rel_path,pinned", sorted(FROZEN_PANEL_SHA256.items()))
def test_frozen_panel_sha_matches_on_disk(rel_path, pinned):
    path = REPO_ROOT / rel_path
    if not path.exists():
        pytest.skip(f"committed panel not present locally: {rel_path}")
    live = _recompute(path)
    assert live == pinned, (
        f"{rel_path} changed without re-pinning: on-disk sha256 {live} != "
        f"frozen pin {pinned}. If the panel was deliberately rebuilt, re-pin "
        f"shared/reporting/panel_pins.py in the SAME change and record the "
        f"rebuild in monthly_panel_endpoint_reports.json."
    )


def test_recorded_provenance_agrees_with_pins():
    """The pin and the provenance report must never silently diverge."""
    if not _REPORT.exists():
        pytest.skip("endpoint provenance report not present locally")
    d = json.loads(_REPORT.read_text())
    recorded = {
        d["source_panel"]["path"]: d["source_panel"]["sha256"],
        d["exports"]["uncorrected"]["path"]: d["exports"]["uncorrected"]["sha256"],
        d["exports"]["corrected"]["path"]: d["exports"]["corrected"]["sha256"],
    }
    for rel_path, pinned in FROZEN_PANEL_SHA256.items():
        assert recorded.get(rel_path) == pinned, (
            f"recorded provenance for {rel_path} ({recorded.get(rel_path)}) "
            f"disagrees with frozen pin ({pinned})"
        )
