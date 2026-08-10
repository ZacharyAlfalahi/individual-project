"""Completion / stage marker and failure protocol (one-shot holdout §1.1, §4).

An append-only JSONL marker records the one-shot's progress through
``STAGE1_STARTED → STAGE1_COMPLETE → STAGE2_STARTED → COMPLETE``. It holds state and
hashes only — never data — and is committed to the repo. The failure protocol is
pre-registered here so no judgement call happens mid-incident:

  * crash while only ``STAGE1_STARTED`` (no evaluation output exists) → ONE documented
    restart, logged;
  * ``STAGE2_STARTED`` or later, with any result artefact → NO rerun, ever;
  * ``COMPLETE`` → refuse re-invocation. There is no ``--force``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

STATES = ("STAGE1_STARTED", "STAGE1_COMPLETE", "STAGE2_STARTED", "COMPLETE")

# Actions the orchestrator may take on (re-)invocation.
FRESH_BUILD = "fresh_build"
RESTART_BUILD = "restart_build"
RESUME_EVALUATE = "resume_evaluate"


class MarkerError(RuntimeError):
    """Malformed marker, or an illegal transition."""


class RerunRefused(MarkerError):
    """Re-invocation refused by the failure protocol (§4) — the one-shot has progressed
    past the point where a rerun is permitted. No ``--force`` overrides this."""


def plan_next(records: list[dict]) -> str:
    """Decide the next action from the marker history, or refuse (§4). Pure function."""
    states = [r.get("state") for r in records]
    for s in states:
        if s not in STATES:
            raise MarkerError(f"unknown marker state {s!r}")
    if "COMPLETE" in states:
        raise RerunRefused("run already COMPLETE — no rerun, no --force (§1.1/§4)")
    if "STAGE2_STARTED" in states:
        raise RerunRefused("evaluation began (STAGE2_STARTED); a result artefact may exist — no rerun ever (§4)")
    if "STAGE1_COMPLETE" in states:
        return RESUME_EVALUATE
    n_stage1 = states.count("STAGE1_STARTED")
    if n_stage1 == 0:
        return FRESH_BUILD
    if n_stage1 == 1:
        return RESTART_BUILD                      # the single documented restart (§4)
    raise RerunRefused("the single documented restart has already been used (§4)")


@dataclass
class Marker:
    """Append-only view over one marker JSONL file."""

    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        out: list[dict] = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise MarkerError(f"corrupt marker line in {self.path}: {line!r}") from exc
        return out

    def states(self) -> list[str]:
        return [r.get("state") for r in self.records()]

    def latest_state(self) -> str | None:
        recs = self.records()
        return recs[-1].get("state") if recs else None

    def plan_next(self) -> str:
        return plan_next(self.records())

    def append(self, state: str, *, ts: str, extra: dict | None = None) -> None:
        if state not in STATES:
            raise MarkerError(f"cannot append unknown state {state!r}")
        record = {"state": state, "ts": ts}
        if extra:
            record.update(extra)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())          # durability: the no-rerun guarantee must survive a crash


# --- Rehearsal marker (§1.3 precondition / §5) --------------------------------------------

REHEARSAL_GREEN = "REHEARSAL_GREEN"


@dataclass
class RehearsalMarker:
    """Separate marker written green by ``--rehearsal``; the real run refuses without it (§1.3)."""

    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def is_green(self) -> bool:
        if not self.path.exists():
            return False
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MarkerError(f"corrupt rehearsal marker line in {self.path}: {line!r}") from exc
            if record.get("state") == REHEARSAL_GREEN:
                return True
        return False

    def write_green(self, *, ts: str, extra: dict | None = None) -> None:
        record = {"state": REHEARSAL_GREEN, "ts": ts}
        if extra:
            record.update(extra)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
