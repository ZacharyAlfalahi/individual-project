"""Mechanism library + template registry loaders (spec §6/§7).

Loads the manually-authored entries from prereg/mechanism_library/ and the templates from
prereg/templates/, plus the variable-family registry. Computes a SHA VERSION-LOCK over the
library content (spec §6: "SHA version-lock; hash recorded on every proposal") so a proposal can
record exactly which library it was drawn from and a changed library is detectable. Pure YAML —
no magnitudes, no data reads (the available-variable set is computed separately, see
`available_conditioning_variables`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from shared.licensed_inputs import require_licensed_input

REPO_ROOT = Path(__file__).resolve().parents[3]
LIB_DIR = REPO_ROOT / "prereg" / "mechanism_library"
TEMPLATES_DIR = REPO_ROOT / "prereg" / "templates"


@dataclass(frozen=True)
class MechanismLibrary:
    mechanisms: tuple[dict, ...]          # authored mechanism entries
    templates: dict                       # template_id -> template dict
    variable_families: dict               # family -> [conditioning variables]
    version_hash: str                     # sha256 over the content (the version-lock)

    def mechanism(self, mechanism_id: str) -> dict:
        for m in self.mechanisms:
            if m["mechanism_id"] == mechanism_id:
                return m
        raise KeyError(mechanism_id)


def _sha(payload) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_library(lib_dir: Path | str = LIB_DIR, templates_dir: Path | str = TEMPLATES_DIR) -> MechanismLibrary:
    lib_dir, templates_dir = Path(lib_dir), Path(templates_dir)
    mechanisms = tuple(yaml.safe_load(p.read_text()) for p in sorted(lib_dir.glob("mech_*.yaml")))
    # Fail loud rather than silently return an empty / malformed library (a zero-mechanism library
    # would make every strategy trivially fail the eligibility gate with no error).
    if not mechanisms:
        raise ValueError(f"no mechanism entries (mech_*.yaml) under {lib_dir}")
    for m in mechanisms:
        if not isinstance(m, dict) or "mechanism_id" not in m:
            raise ValueError(f"malformed mechanism entry (need a mapping with mechanism_id): {m!r}")
    templates = {}
    for p in sorted(templates_dir.glob("*.yaml")):
        t = yaml.safe_load(p.read_text())
        if not isinstance(t, dict) or "template_id" not in t:
            raise ValueError(f"malformed template (need a mapping with template_id): {p}")
        templates[t["template_id"]] = t
    if not templates:
        raise ValueError(f"no templates (*.yaml) under {templates_dir}")
    variable_families = yaml.safe_load((lib_dir / "variable_families.yaml").read_text())["variable_families"]
    version_hash = _sha([list(mechanisms), dict(sorted(templates.items())), variable_families])
    return MechanismLibrary(mechanisms, templates, variable_families, version_hash)


def available_conditioning_variables(repo_root: Path | str = REPO_ROOT) -> set[str]:
    """The conditioning variables actually present in the data — the eligibility filter's
    availability input. Union of monthly_panel_corrected columns, the signals/ file stems, and
    the macro series in reporting_delays.yaml. (Reads data + config; kept OUT of the pure filter.)"""
    import pandas as pd

    repo_root = Path(repo_root)
    panel = repo_root / "data" / "development" / "monthly_panel_corrected.parquet"
    signals = repo_root / "data" / "development" / "signals"
    delays = repo_root / "docs" / "reporting_delays.yaml"

    cols = set(pd.read_parquet(require_licensed_input(panel, "development panel")).columns)
    stems = {p.stem for p in signals.glob("*.parquet")}
    macro = set(yaml.safe_load(delays.read_text())["reporting_delays"])
    return cols | stems | macro
