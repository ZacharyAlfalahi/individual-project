"""
Freeze a PDF into a canonical text (``evaluation/canonical_texts/<paper>.frozen.yaml``).

Reuses ``scripts/parser_bakeoff/build_canonical_text.build_canonical`` (PyMuPDF ->
L0 pages, watermark-stripped, determinism-gated) and stamps the FROZEN normalisation
recipe from ``config/canonical_text.yaml`` (ladder_level L1 + the v2 rules). The
pages are stored L0 (store-L0 / normalise-on-read); the recorded ``normalisation``
block is the ladder the locator normalises to at match time -- NOT the raw builder
default (``L0`` / ``rules: []``). Writes ``status: frozen``, matching the shape of the
existing ``drr_2026.frozen.yaml`` / ``bbw_2019.frozen.yaml``.

Usage:
  ./.venv/bin/python scripts/freeze_canonical_text.py \\
      "papers/pdf/Momentum in Corporate Bond Returns.pdf" \\
      evaluation/canonical_texts/jnps_2013.frozen.yaml \\
      --expect-sha 7e80f8cb919de4161822df63da1069310d8a113b0562ccd2d3d6420d4f707242
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.parser_bakeoff.build_canonical_text import build_canonical  # noqa: E402

_RECIPE = _REPO_ROOT / "config" / "canonical_text.yaml"


def freeze(pdf_path, out_path, *, expect_sha: str | None = None) -> dict:
    """Build the L0 canonical text, stamp the frozen L1 recipe + status, write it."""
    canon = build_canonical(pdf_path, strip_watermark=True)
    if expect_sha is not None and canon["source_sha256"] != expect_sha:
        raise SystemExit(
            f"source_sha256 mismatch for {pdf_path}: got {canon['source_sha256']}, "
            f"expected {expect_sha} -- refusing to freeze a different PDF"
        )
    recipe = yaml.safe_load(_RECIPE.read_text(encoding="utf-8"))["normalisation"]
    frozen = {
        "source_pdf": canon["source_pdf"],
        "source_sha256": canon["source_sha256"],
        "parser": canon["parser"],
        # Store-L0 / normalise-on-read: pages stay L0; the matcher normalises to this
        # ladder on read. Stamp the FROZEN recipe from config/canonical_text.yaml.
        "normalisation": {
            "ladder_level": recipe["ladder_level"],
            "rules": list(recipe["rules"]),
        },
        "page_canonicalisation": canon["page_canonicalisation"],
        "pages": canon["pages"],
        "status": "frozen",
    }
    Path(out_path).write_text(
        yaml.safe_dump(frozen, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return frozen


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze a PDF into a canonical-text yaml.")
    ap.add_argument("pdf")
    ap.add_argument("out")
    ap.add_argument("--expect-sha", default=None, help="assert the PDF's source_sha256")
    args = ap.parse_args()
    frozen = freeze(args.pdf, args.out, expect_sha=args.expect_sha)
    print(f"froze {args.pdf} -> {args.out}")
    print(f"  source_sha256: {frozen['source_sha256']}")
    print(f"  normalisation: {frozen['normalisation']['ladder_level']} "
          f"({len(frozen['normalisation']['rules'])} rules)")
    print(f"  pages: {len(frozen['pages'])}")


if __name__ == "__main__":
    main()
