"""Phase-F specification provenance loader for the P2 coverage-boundary census (WS-C).

Loads reportable Phase-F Librarian specifications and matches them to census
members by byte-exact construction name. Each match carries the run directory,
specification path, SHA-256, manifest phase, and code commit.

Two responsibilities, kept separate so the matcher is fixture-testable with zero disk:

  * ``load_real_specs(run_dirs)`` — read every reportable ``spec_*.json`` under the
    given run directories, keyed by ``(paper_id.lower(), strategy_label.value)``.
    A run whose manifest ``operational_profile.phase != "report"`` is a BUILD
    ERROR, never a silent downgrade — this is what keeps a Phase-D artefact out of
    a reportable census. A directory with no spec files (e.g. a review exit)
    contributes nothing; the census overlay types the affected members as COUNTED
    eligibility exclusions.
  * ``match_specs_to_members(...)`` — join specs to member ids by byte-exact
    ``(paper, name)``. NO fuzzy matching anywhere: a label that matches a gold
    ``slug`` but not byte-exactly is a build error (near-miss guard), as is a spec
    that matches no member and two specs mapping to one member.

``slug`` is the single source of truth for the ``f"{paper}::{slug(name)}"`` member
id rule — ``scripts/build_p2_census`` imports it from here.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Default scale-layer Phase-F run directories (one per paper). BBW-2021 emitted
#: five specs (phase=report); DFPS-2026 exited to review with zero specs.
DEFAULT_RUN_DIRS: tuple[Path, ...] = (
    _REPO_ROOT / "runs" / "corpus_corpus_report" / "bbw2021",
    _REPO_ROOT / "runs" / "corpus_corpus_report" / "dfps",
)

_REPORT_PHASE = "report"


class P2SpecSourceError(ValueError):
    """A real-spec source is malformed or a spec cannot be matched to a census
    member — a build error surfaced loudly, never a silent drop or downgrade."""


def slug(text: str) -> str:
    """The canonical member-id slug (shared with ``scripts/build_p2_census``):
    lower-case, non-alphanumeric runs -> ``_``, collapsed and stripped."""
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", text.lower())).strip("_")


_SPEC_FILE_RE = re.compile(r"spec_\d+\.json")


def _declared_spec_digests(outputs: dict | None) -> dict[str, str] | None:
    """The manifest-declared ``spec_*.json`` outputs as ``{basename: sha256}``, or ``None`` when
    the manifest declares no ``outputs`` at all (an older schema we cannot integrity-check).
    Non-spec outputs (traces, raw) are ignored."""
    if outputs is None:
        return None
    return {Path(p).name: sha for p, sha in outputs.items()
            if _SPEC_FILE_RE.fullmatch(Path(p).name)}


def _spec_label(spec: dict, where: str) -> tuple[str, str]:
    """Extract ``(paper_id, strategy_label.value)`` from a spec header, fail-loud."""
    header = spec.get("header") or {}
    paper_id = header.get("paper_id")
    label_field = header.get("strategy_label")
    label = label_field.get("value") if isinstance(label_field, dict) else label_field
    if not isinstance(paper_id, str) or not isinstance(label, str):
        raise P2SpecSourceError(
            f"{where}: header.paper_id and header.strategy_label.value must both be "
            f"strings; got paper_id={paper_id!r}, label={label!r}"
        )
    return paper_id, label


def load_real_specs(
    run_dirs: Sequence[Path] = DEFAULT_RUN_DIRS,
) -> tuple[dict[tuple[str, str], dict], dict[tuple[str, str], dict]]:
    """Read every reportable spec emission under ``run_dirs``.

    Returns ``(specs_by_key, provenance_by_key)`` both keyed by
    ``(paper_id.lower(), strategy_label.value)``:
      * ``specs_by_key``  — the full spec JSON dict;
      * ``provenance_by_key`` — ``{run_dir, spec_file, spec_sha256, manifest_phase,
        code_commit}``.

    Raises ``P2SpecSourceError`` on a missing manifest, a non-``report`` phase, a
    malformed spec header, or two specs sharing a ``(paper, label)`` key.
    """
    specs_by_key: dict[tuple[str, str], dict] = {}
    provenance_by_key: dict[tuple[str, str], dict] = {}
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            raise P2SpecSourceError(f"{run_dir}: no run_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        phase = (manifest.get("operational_profile") or {}).get("phase")
        if phase != _REPORT_PHASE:
            raise P2SpecSourceError(
                f"{run_dir}: operational_profile.phase is {phase!r}, not {_REPORT_PHASE!r} "
                "— a non-report run cannot seed a reportable census (no silent downgrade)"
            )
        code_commit = (manifest.get("code") or {}).get("commit")

        # Provenance integrity. When the manifest declares its outputs, the spec files on disk
        # must EXACTLY match the declared spec set and each file's sha256 must match the
        # declared digest. A stray or partial spec_*.json is a build error (D31: a run emits a
        # full set or none) — never a silent disposition change. This is the guard that keeps a
        # spurious spec from flipping a member from eligibility-exclusion to routed (which the
        # id/order sha and the pre-overlay drift check would not catch); it is a check on the
        # frozen run artefacts, NOT an assertion on post-overlay census counts.
        declared_specs = _declared_spec_digests(manifest.get("outputs"))
        present = {p.name for p in run_dir.glob("spec_*.json")}
        if declared_specs is not None and present != set(declared_specs):
            raise P2SpecSourceError(
                f"{run_dir}: spec files on disk {sorted(present)} do not match the "
                f"manifest-declared outputs {sorted(declared_specs)} — a stray or missing "
                "spec breaks provenance integrity (D31: a run emits a full set or none)"
            )

        for spec_path in sorted(run_dir.glob("spec_*.json")):
            raw = spec_path.read_bytes()
            spec_sha = hashlib.sha256(raw).hexdigest()
            if declared_specs is not None and declared_specs.get(spec_path.name) not in (None, spec_sha):
                raise P2SpecSourceError(
                    f"{spec_path}: sha256 {spec_sha} != manifest-declared "
                    f"{declared_specs[spec_path.name]} — the spec was rewritten after the run"
                )
            spec = json.loads(raw.decode("utf-8"))
            paper_id, label = _spec_label(spec, spec_path.name)
            key = (paper_id.lower(), label)
            if key in specs_by_key:
                raise P2SpecSourceError(
                    f"two specs share (paper, label) {key!r} "
                    f"({provenance_by_key[key]['spec_file']} and {spec_path.name}) — "
                    "one spec per construction"
                )
            specs_by_key[key] = spec
            provenance_by_key[key] = {
                "run_dir": str(run_dir),
                "spec_file": spec_path.name,
                "spec_sha256": spec_sha,
                "manifest_phase": phase,
                "code_commit": code_commit,
            }
    return specs_by_key, provenance_by_key


def match_specs_to_members(
    specs_by_key: dict[tuple[str, str], dict],
    provenance_by_key: dict[tuple[str, str], dict],
    gold_rows: Iterable[dict],
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Join real specs to census member ids by BYTE-EXACT ``(paper, name)``.

    ``gold_rows``: the census MEMBER rows, each a dict with ``paper_id`` (the member
    id), ``paper`` (lower-case paper stem) and ``name`` (the gold construction name).

    Returns ``(by_member_id, provenance_by_member)`` over matched members only.
    Raises ``P2SpecSourceError`` on a spec that matches no member byte-exactly
    (``spec_unmatched_to_member``; a slug-only near-miss is called out explicitly)
    or two specs mapping to one member.
    """
    exact: dict[tuple[str, str], str] = {}
    slug_index: dict[tuple[str, str], str] = {}
    for r in gold_rows:
        exact[(r["paper"], r["name"])] = r["paper_id"]
        slug_index[(r["paper"], slug(r["name"]))] = r["paper_id"]

    by_member_id: dict[str, dict] = {}
    provenance_by_member: dict[str, dict] = {}
    for (paper, label), spec in specs_by_key.items():
        member_id = exact.get((paper, label))
        if member_id is None:
            near = slug_index.get((paper, slug(label)))
            if near is not None:
                raise P2SpecSourceError(
                    f"spec label {label!r} (paper {paper!r}) matches gold member "
                    f"{near!r} by slug but NOT byte-exactly — no fuzzy matching is "
                    "permitted; reconcile the label or the gold name"
                )
            raise P2SpecSourceError(
                f"spec label {label!r} (paper {paper!r}) matches no gold construction "
                "byte-exactly (spec_unmatched_to_member)"
            )
        if member_id in by_member_id:
            raise P2SpecSourceError(
                f"two specs map to the same member {member_id!r} — one spec per member"
            )
        by_member_id[member_id] = spec
        provenance_by_member[member_id] = provenance_by_key[(paper, label)]
    return by_member_id, provenance_by_member


def load_member_specs(
    gold_rows: Iterable[dict],
    run_dirs: Sequence[Path] = DEFAULT_RUN_DIRS,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Convenience: ``load_real_specs`` then ``match_specs_to_members``. Returns
    ``(by_member_id, provenance_by_member)`` — the two parallel per-member dicts the
    census overlay consumes."""
    specs_by_key, provenance_by_key = load_real_specs(run_dirs)
    return match_specs_to_members(specs_by_key, provenance_by_key, gold_rows)
