"""On-disk content-addressed cache for generative calls (R5). Keyed by (prompt hash, model, seed)
-> the raw response, one JSON file per key. This doubles as the artefact log evidencing that seed
0 was not selected retrospectively (every seed's call is persisted, immutably, before any
downstream selection). No Redis (R5) — a plain content-addressed directory is reproducible and
auditable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class ResponseCache:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(prompt: str, model: str, seed: int) -> str:
        return hashlib.sha256(f"{model}\x00{seed}\x00{prompt}".encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, prompt: str, model: str, seed: int) -> str | None:
        p = self._path(self.key(prompt, model, seed))
        if not p.exists():
            return None
        return json.loads(p.read_text())["response"]

    def put(self, prompt: str, model: str, seed: int, response: str) -> str:
        key = self.key(prompt, model, seed)
        self._path(key).write_text(json.dumps(
            {"model": model, "seed": seed, "prompt_sha256": key, "response": response},
            ensure_ascii=False))
        return key
