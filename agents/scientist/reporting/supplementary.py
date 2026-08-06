"""G4 supplementary PBO — the measurement the pre-registration promised
(DEV-G4-PBO-1 resolution, E6, 2026-08-06).

The protocol pre-registered `pbo: {scope: realised_family_only, status:
supplementary_caveated}` with the in-file comment "coarse at m = 6; reported
with a caveat, never gating" — but the implemented G4 never computed it.
This module fulfils the promise INERTLY, under the three conditions:

  (i)  additive only — computed from the FINISHED `ExperimentReport` and the
       driver-held candidate returns, strictly after every gate decision;
       nothing here is importable from a gate module's execution path;
  (ii) provably inert — `tests/unit/test_g4_pbo_supplementary.py` pins that
       no module under `agents/scientist/experimentalist/` imports pbo, and
       that the report's records are byte-identical with and without this
       computation (it takes the report read-only and returns a separate
       block);
  (iii) had the wiring required touching any decision path, the ruling was
       to revert to disclosure-only — it did not: this lives in the
       `reporting` subpackage, sibling to (never inside) the gate stack.

The realised family = the audit-clean proposals whose candidate series
executed (the trials over which G3's joint BH actually operated). The
confirmatory report assembler places the returned block beside the G4
measurements, carrying the caveat verbatim.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import pandas as pd

from shared.stats import pbo_cscv

# The pre-registered caveat, quoted from docs/scientist_protocol.yaml `pbo:`.
PBO_CAVEAT = "coarse at m = 6; reported with a caveat, never gating"
PBO_SCOPE = "realised_family_only"


def _is_realised(record) -> bool:
    """Audit-clean membership read off the record's booleans — the family the
    joint BH ran over (compiled, execution-verified, audit-clean)."""
    b = record.booleans
    return bool(b.compiled and b.execution_verified and b.audit_clean)


def family_pbo_supplementary(
    records: Iterable,
    candidate_returns: Mapping[str, pd.Series],
    *,
    n_groups: int = 8,
    test_groups: int = 2,
    purge: int = 0,
    embargo: int = 1,
    months_per_year: int = 12,
) -> dict:
    """The family-level CSCV PBO block for one strategy's proposal family.

    ``records`` is `ExperimentReport.records` (read-only; never mutated);
    ``candidate_returns`` maps proposal_id -> monthly return series (held by
    the driver — the report deliberately retains no series). Returns a
    self-contained dict with a typed ``status``; NEVER raises for family-size
    or overlap shortfalls (a supplementary measurement must not be able to
    take down a run), and never gates anything.
    """
    geometry = {
        "n_groups": n_groups, "test_groups": test_groups,
        "purge": purge, "embargo": embargo,
    }
    base = {"scope": PBO_SCOPE, "caveat": PBO_CAVEAT, "geometry": geometry}

    realised = [r.proposal_id for r in records
                if _is_realised(r) and r.proposal_id in candidate_returns]
    if len(realised) < 2:
        return {**base, "status": "insufficient_family",
                "n_candidates": len(realised),
                "detail": "CSCV needs >= 2 realised candidates with returns"}

    frame = pd.concat(
        {pid: pd.Series(candidate_returns[pid]) for pid in realised},
        axis=1, join="inner",
    ).dropna()
    n_total = max(len(pd.Series(candidate_returns[pid]).dropna()) for pid in realised)
    months_dropped = int(n_total - len(frame))
    if len(frame) < n_groups:
        return {**base, "status": "insufficient_overlap",
                "n_candidates": len(realised), "n_obs": int(len(frame)),
                "months_dropped": months_dropped,
                "detail": f"overlapping months {len(frame)} < n_groups {n_groups}"}

    result = pbo_cscv(frame, n_groups=n_groups, test_groups=test_groups,
                      purge=purge, embargo=embargo, months_per_year=months_per_year)
    return {
        **base,
        "status": "computed",
        "n_candidates": result.n_candidates,
        "n_obs": result.n_obs,
        "n_folds": result.n_folds,
        "pbo": result.pbo,
        "median_oos_relative_rank": result.median_oos_relative_rank,
        "is_best_counts": dict(result.is_best_counts),
        "months_dropped": months_dropped,
    }
