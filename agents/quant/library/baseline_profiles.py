"""
baseline_profiles.py — the per-paper as-published cleaning-profile registry
(spec v4 B1/D1).

Loads ``config/baseline_profiles.yaml`` and computes a ``baseline_signature``
(sha256 over the canonical serialisation) recorded in every run's provenance, so a
differential traces to the baseline that produced it (B1). Two NEW profiles are
declared (``bbw_2019`` for drf/crf, ``jostova_2013`` for mom6); ``drr_2026`` is the
existing corrected panel (``price_family='corr'``), so it needs no entry.

The signature covers ordered cleaning ops + params + provenance tags, so a change
to any of them (or to a paper-source quote) changes the hash. Provenance tags are
STATED | INFERRED | ASSUMED (B1); a load-bearing ASSUMED step used for
paper-specific attribution must trigger typed refusal / a sensitivity envelope
(B3) — the caller checks ``profile_provenance`` for that; the loader only validates
the tag vocabulary.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PATH = REPO_ROOT / "config" / "baseline_profiles.yaml"

# The two NEW profiles. drr_2026 is the corrected panel (price_family='corr').
PROFILE_IDS: tuple[str, ...] = ("bbw_2019", "jostova_2013")
_VALID_PROVENANCE: tuple[str, ...] = ("STATED", "INFERRED", "ASSUMED")


def _load_all(path: str | Path | None = None) -> dict:
    p = Path(path) if path else _DEFAULT_PATH
    doc = yaml.safe_load(p.read_text())
    if not isinstance(doc, dict) or "profiles" not in doc:
        raise ValueError(f"{p} is not a baseline-profile registry (no 'profiles' key)")
    return doc["profiles"]


def load_profile(profile_id: str, *, path: str | Path | None = None) -> dict:
    """Return one profile's entry, validating its step provenance tags."""
    profiles = _load_all(path)
    if profile_id not in profiles:
        raise KeyError(
            f"unknown baseline profile {profile_id!r}; known: {sorted(profiles)}"
        )
    prof = profiles[profile_id]
    for step in prof.get("steps", []):
        prov = step.get("provenance")
        if prov not in _VALID_PROVENANCE:
            raise ValueError(
                f"profile {profile_id!r} step {step.get('op')!r}: provenance must be "
                f"one of {_VALID_PROVENANCE}; got {prov!r}"
            )
    return prof


def profile_provenance(profile_id: str, *, path: str | Path | None = None) -> set[str]:
    """The set of provenance tags across a profile's steps — the B3 guard input
    (a load-bearing ASSUMED/UNKNOWN step cannot support a paper-specific claim)."""
    return {s["provenance"] for s in load_profile(profile_id, path=path).get("steps", [])}


def baseline_signature(profile_id: str, *, path: str | Path | None = None) -> str:
    """sha256 over the canonical (sorted-key) YAML serialisation of the profile —
    the frozen baseline hash recorded in run provenance (spec B1)."""
    prof = load_profile(profile_id, path=path)
    canonical = yaml.safe_dump(prof, sort_keys=True, default_flow_style=False)
    return hashlib.sha256(canonical.encode()).hexdigest()
