"""Serialise / load a P2 ``CensusResult`` (WS-C).

``census.py`` is pure typed data (``CensusMember.to_dict()`` deliberately omits ``extracted_spec``).
P2 needs the spec on disk — the driver generates code from ``member.extracted_spec`` — so this module
round-trips the FULL member (spec included) to/from ``data/development/codegen/p2_census.json``.

The census this serialises is the scale-layer COVERAGE PARTITION, pre-registered at 32 = 27 implement
+ 5 equity-refuse (corpus_inventory §5, 2026-08-19). Auxiliary + 153-equity-family constructions are
pre-registered EXCLUSIONS recorded in ``metadata.exclusions``, never census members.

The ``extracted_spec`` values are now the REAL Phase-F Librarian emissions on disk
(``runs/corpus_corpus_report/``, phase=report): the five ``BBW_2021`` members carry their emitted
specs; the 27 ``DFPS_2026`` members carry no spec (that run exited to review with zero specs — D31
forbids partial emission) and are typed COUNTED eligibility exclusions. Eligibility exclusions are
still members, so the 32-id frozen order round-trips unchanged. ``metadata.spec_provenance`` records,
per member, the run dir / spec file / sha256 / manifest phase / code commit.
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.codegen.census import CensusMember, CensusResult


def save_census(result: CensusResult, path: Path, *, metadata: dict) -> Path:
    """Write the census (members incl. extracted_spec) + metadata as deterministic JSON."""
    payload = {
        "metadata": metadata,
        "members": [
            {**m.to_dict(), "extracted_spec": m.extracted_spec} for m in result.members
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def load_census(path: Path) -> CensusResult:
    """Reconstruct a ``CensusResult`` (members incl. ``extracted_spec``) from disk. The member
    invariants re-validate on construction, so a tampered file fails loud."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    members = tuple(
        CensusMember(
            paper_id=r["paper_id"],
            compilable=r["compilable"],
            refused=r["refused"],
            refusal_reason=r["refusal_reason"],
            text_quality_ok=r["text_quality_ok"],
            extracted_spec=r.get("extracted_spec"),
            exclusion_reason=r.get("exclusion_reason"),
        )
        for r in payload["members"]
    )
    return CensusResult(members)


def load_metadata(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8")).get("metadata", {})
