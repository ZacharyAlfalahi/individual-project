"""cli.py — Reporter command dispatch (docs/reporter/reporter_spec_v0.2.md §11): render / publish / check / scaffold.

Thin, deterministic entry points over the pipeline. `check` verifies in-memory and writes
nothing (CI mode); `render` writes note.md + claims.json into a scratch staging dir but nothing
published; `publish` renders + verifies every declared run and atomically swaps the published
directory; `scaffold` prints a filled pointer file (writes nothing).
"""

from __future__ import annotations

from pathlib import Path

from shared.reporting.canonical import canonical_json

from .bundle import ReportBundle, current_code_version, load_bundle
from .manifest import REPO_ROOT, ManifestError, load_manifest, scaffold_pointer
from .registry import publish
from .renderer import RenderedDocument, render_note
from .verify import assert_verified

# The standard filenames a scaffold looks for in a quant / audit run directory.
_QUANT_FILES = (("quant_result", "{s}.json"), ("quant_coverage", "coverage.json"),
                ("quant_run_log", "run_log.json"))
_AUDIT_FILES = (("audit_report", "{s}_core.json"), ("audit_run_log", "run_log.json"))


def _pointer_path(run_id: str, repo_root: Path) -> Path:
    return repo_root / "reporter" / "runs" / f"{run_id}.yaml"


def load_run_bundle(run_id: str, *, repo_root: Path = REPO_ROOT) -> ReportBundle:
    """Load and validate the pointer file for `run_id` and assemble its bundle."""
    pointer = _pointer_path(run_id, repo_root)
    if not pointer.exists():
        raise ManifestError(f"no pointer file for run {run_id!r} at {pointer}")
    manifest = load_manifest(pointer, repo_root=repo_root)
    return load_bundle(manifest, repo_root=repo_root, code_version=current_code_version())


def check_run(run_id: str, *, repo_root: Path = REPO_ROOT) -> RenderedDocument:
    """Render + verify one run in memory; write nothing (CI mode). Raises on any failure."""
    bundle = load_run_bundle(run_id, repo_root=repo_root)
    doc = render_note(bundle)
    assert_verified(doc, bundle)
    return doc


def render_run(
    run_id: str, *, repo_root: Path = REPO_ROOT, staging_root: Path | None = None
) -> Path:
    """Render + verify one run and write note.md + claims.json into a scratch staging dir
    (nothing published). Returns the staging directory."""
    bundle = load_run_bundle(run_id, repo_root=repo_root)
    doc = render_note(bundle)
    assert_verified(doc, bundle)
    staging = Path(staging_root) if staging_root is not None else (
        repo_root / "artifacts" / "reporter.render" / run_id
    )
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "note.md").write_text(doc.note_markdown, encoding="utf-8", newline="\n")
    claims_payload = {
        "reporter_run_id": bundle.reporter_run_id,
        "claims": [c.to_dict() for c in doc.claims],
        "evidence": [e.to_dict() for e in doc.evidence],
        "legal_state_hash": doc.legal_state_hash,
    }
    (staging / "claims.json").write_text(
        canonical_json(claims_payload) + "\n", encoding="utf-8", newline="\n"
    )
    return staging


def publish_all(
    *, repo_root: Path = REPO_ROOT, out_root: Path | None = None
) -> dict:
    """Render + verify every declared run (`reporter/runs/*.yaml`) and atomically publish."""
    runs_dir = repo_root / "reporter" / "runs"
    pointers = sorted(runs_dir.glob("*.yaml")) if runs_dir.exists() else []
    if not pointers:
        raise ManifestError(f"no pointer files found under {runs_dir}")
    bundles = [load_run_bundle(p.stem, repo_root=repo_root) for p in pointers]
    result = publish(
        bundles, out_root=out_root or (repo_root / "artifacts" / "reporter")
    )
    return {
        "n_runs": result.n_runs,
        "row_counts": result.row_counts,
        "out_root": str(result.out_root),
    }


def scaffold(
    *,
    reporter_run_id: str,
    paper_id: str,
    strategy_id: str,
    phase: str,
    spec: str,
    quant_dir: str | None = None,
    audit_dir: str | None = None,
    repo_root: Path = REPO_ROOT,
) -> str:
    """Build a filled pointer-file string from a spec path plus optional quant/audit run dirs,
    including only the artefacts that exist. Writes nothing (the caller prints it)."""
    artefacts: dict[str, str] = {"spec": _rel(spec, repo_root)}
    for base, table in ((quant_dir, _QUANT_FILES), (audit_dir, _AUDIT_FILES)):
        if not base:
            continue
        base_path = Path(base)
        abs_base = base_path if base_path.is_absolute() else (repo_root / base_path)
        for key, pattern in table:
            candidate = abs_base / pattern.format(s=strategy_id)
            if candidate.exists():
                artefacts[key] = _rel(str(candidate), repo_root)
    return scaffold_pointer(
        reporter_run_id=reporter_run_id,
        paper_id=paper_id,
        strategy_id=strategy_id,
        phase=phase,
        artefacts=artefacts,
        repo_root=repo_root,
    )


def _rel(path: str, repo_root: Path) -> str:
    p = Path(path)
    resolved = p if p.is_absolute() else (repo_root / p)
    return resolved.resolve().relative_to(repo_root.resolve()).as_posix()
