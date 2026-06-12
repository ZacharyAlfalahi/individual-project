"""
Materialise the two endpoint views the registry spec mandates as deliverables:

  monthly_panel_uncorrected.parquet  ← all-OFF view (raw / no stale mask / no terminal rows)
  monthly_panel_corrected.parquet    ← all-ON-except-survivorship view (corr / stale mask on)

Both are EXPORTS of the maximal panel via `views.view()` — neither is a
source of truth. Re-running this script with the same inputs is idempotent:
the output sha256 hashes match across runs. Per the registry spec §3 D4,
this satisfies the two-parquet deliverable without making
the endpoint exports primary artefacts.

Each parquet's Arrow schema metadata carries:
  panel_kind = uncorrected | corrected
  primary_key = cusip
  registry_panel_view_hash = <hex digest of the panel_view block that produced it>
  source_panel_sha256 = <hex digest of the input maximal panel>

The parquet is stamped with panel_view_hash(), NOT the full-config hash():
view() consumes only the panel_view block, so the construction/evaluation
toggles (signal_lag, expost_trim) are not embodied in the exported panel —
stamping the full hash would over-claim. The JSON report records both the
full RunConfig YAML and both hashes for complete provenance.

Sibling report: data/development/monthly_panel_endpoint_reports.json
  - input panel sha256
  - two endpoint output sha256s
  - the RunConfig YAML for each export

Usage:
  python scripts/export_endpoint_views.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from agents.quant.library.run_config import RunConfig, uncorrected, corrected  # noqa: E402
from agents.quant.library.views import view  # noqa: E402


PANEL_FILE = REPO_ROOT / "data" / "development" / "monthly_panel_maximal.parquet"
SIGNALS_FILE = REPO_ROOT / "data" / "development" / "signals" / "var_5pct.parquet"
OUT_DIR = REPO_ROOT / "data" / "development"
UNCORR_OUT = OUT_DIR / "monthly_panel_uncorrected.parquet"
CORR_OUT = OUT_DIR / "monthly_panel_corrected.parquet"
REPORT_OUT = OUT_DIR / "monthly_panel_endpoint_reports.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_export(
    panel: pd.DataFrame,
    out_file: Path,
    panel_kind: str,
    config: RunConfig,
    source_panel_sha: str,
) -> str:
    """Write the view to parquet with registry-aware metadata. Returns the
    sha256 of the written file."""
    table = pa.Table.from_pandas(panel, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta.update({
        b"panel_kind":  panel_kind.encode("utf-8"),
        b"primary_key": b"cusip",
        b"registry_panel_view_hash": config.panel_view_hash().encode("utf-8"),
        b"source_panel_sha256":      source_panel_sha.encode("utf-8"),
    })
    table = table.replace_schema_metadata(meta)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_suffix(".parquet.tmp")
    if tmp.exists():
        tmp.unlink()
    pq.write_table(table, str(tmp))
    os.replace(tmp, out_file)
    return _sha256_file(out_file)


def main():
    if not PANEL_FILE.exists():
        print(f"ERROR: Required input not found: {PANEL_FILE}", file=sys.stderr)
        print("Run scripts/build_monthly_panel.py first.", file=sys.stderr)
        sys.exit(1)
    if not SIGNALS_FILE.exists():
        print(f"ERROR: Required input not found: {SIGNALS_FILE}", file=sys.stderr)
        print("Run scripts/build_var_5pct.py first.", file=sys.stderr)
        sys.exit(1)

    source_sha = _sha256_file(PANEL_FILE)
    print(f"Source panel: {PANEL_FILE.name} (sha256 {source_sha[:12]}...)")

    print("Loading maximal panel + signals ...")
    maximal = pd.read_parquet(PANEL_FILE)
    signals = pd.read_parquet(SIGNALS_FILE)

    # --- Uncorrected (all-OFF) -----------------------------------------------
    cfg_unc = uncorrected()
    print(f"\nExporting uncorrected view (config hash {cfg_unc.hash()[:12]}...)")
    panel_unc = view(maximal, cfg_unc, signals=signals)
    sha_unc = _write_export(panel_unc, UNCORR_OUT, "uncorrected", cfg_unc, source_sha)
    print(f"  {len(panel_unc):,} rows → {UNCORR_OUT.name} (sha256 {sha_unc[:12]}...)")

    # --- Corrected (all-ON-except-survivorship) ------------------------------
    cfg_corr = corrected()
    print(f"\nExporting corrected view (config hash {cfg_corr.hash()[:12]}...)")
    panel_corr = view(maximal, cfg_corr, signals=signals)
    sha_corr = _write_export(panel_corr, CORR_OUT, "corrected", cfg_corr, source_sha)
    print(f"  {len(panel_corr):,} rows → {CORR_OUT.name} (sha256 {sha_corr[:12]}...)")

    # --- Report --------------------------------------------------------------
    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_panel": {
            "path": str(PANEL_FILE.relative_to(REPO_ROOT)),
            "sha256": source_sha,
        },
        "source_signals": {
            "path": str(SIGNALS_FILE.relative_to(REPO_ROOT)),
            "sha256": _sha256_file(SIGNALS_FILE),
        },
        "exports": {
            "uncorrected": {
                "path": str(UNCORR_OUT.relative_to(REPO_ROOT)),
                "sha256": sha_unc,
                "rows": int(len(panel_unc)),
                "run_config_hash": cfg_unc.hash(),
                "panel_view_hash": cfg_unc.panel_view_hash(),
                "run_config_yaml": cfg_unc.to_yaml(),
            },
            "corrected": {
                "path": str(CORR_OUT.relative_to(REPO_ROOT)),
                "sha256": sha_corr,
                "rows": int(len(panel_corr)),
                "run_config_hash": cfg_corr.hash(),
                "panel_view_hash": cfg_corr.panel_view_hash(),
                "run_config_yaml": cfg_corr.to_yaml(),
            },
        },
        "note": (
            "Endpoint views are EXPORTS via views.view() from the maximal "
            "panel. Neither is a source of truth. Re-running this script "
            "with the same inputs produces identical output sha256s."
        ),
    }
    tmp = REPORT_OUT.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, REPORT_OUT)
    print(f"\nReport: {REPORT_OUT.name}")


if __name__ == "__main__":
    main()
