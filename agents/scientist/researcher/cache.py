"""On-disk content-addressed cache for generative calls (R5). Keyed by (prompt hash, model, seed)
-> the raw response, one JSON file per key. This doubles as the artefact log evidencing that seed
0 was not selected retrospectively (every seed's call is persisted, immutably, before any
downstream selection). No Redis (R5) — a plain content-addressed directory is reproducible and
auditable.
"""

from __future__ import annotations

import hashlib
import json
import os
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
        return json.loads(p.read_text(encoding="utf-8"))["response"]

    def put(self, prompt: str, model: str, seed: int, response: str) -> str:
        """Persist a response IMMUTABLY (an existing key is never overwritten — the artefact log is
        append-only). UTF-8 + atomic (temp-then-rename) so an interrupted run cannot leave a partial
        file that the next `get` chokes on."""
        key = self.key(prompt, model, seed)
        path = self._path(key)
        if path.exists():
            return key                                          # already logged — do not overwrite
        payload = json.dumps({"model": model, "seed": seed, "key": key, "response": response},
                             ensure_ascii=False)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)
        return key
