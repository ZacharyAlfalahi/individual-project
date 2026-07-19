#!/usr/bin/env python
"""
Probe Gemini SKUs for live availability + free-tier quota (L1 dev-model selection).

Model availability + free quotas drift, and this key's free tier has been unusually
restrictive (gemini-2.5-flash 404s, gemini-2.0-flash free-limit 0, gemini-3.5-flash
20/day). This makes ONE minimal generate_content call per candidate SKU -- using the
same config the RealModelClient uses (thinking_budget=0, JSON output, temp 0) -- and
classifies the outcome so we can pick a free dev model_a.

Read-only: one tiny call per SKU (each SKU has its own daily budget, so this never
exhausts any single model). Reads GEMINI_API_KEY from .env or the environment.

    ./.venv/bin/python scripts/probe_models.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Candidate Gemini "lite" SKUs (from a prior models.list()), cheapest/highest-RPD
# free tiers first. gemini-3.5-flash is included as the current baseline.
CANDIDATES = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.1-flash-lite-preview",
    "gemini-3.5-flash",          # current dev model_a (20/day) — baseline
]

_MINIMAL_PROMPT = 'Return exactly this JSON object and nothing else: {"ok": true}'


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def _quota_hint(msg: str) -> str:
    """Pull the free-tier limit(s) out of a 429 error message, if present."""
    limits = re.findall(r"limit:\s*(\d+)", msg)
    ids = re.findall(r"quotaId['\"]?[:=]\s*['\"]?([A-Za-z0-9\-]+)", msg)
    bits = []
    if limits:
        bits.append("limit=" + "/".join(sorted(set(limits))))
    if ids:
        bits.append("id=" + ",".join(sorted(set(ids))[:2]))
    return " ".join(bits)


def _probe_one(client, types, model_id: str) -> tuple[str, str]:
    """Return (verdict, detail). Tries with thinking_budget=0 (the client's config);
    if a SKU rejects thinking_config, retries without it and flags that."""

    def _call(with_thinking: bool):
        cfg_kwargs = dict(
            temperature=0,
            top_p=1,
            max_output_tokens=16,
            response_mime_type="application/json",
        )
        if with_thinking:
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        return client.models.generate_content(
            model=model_id, contents=_MINIMAL_PROMPT,
            config=types.GenerateContentConfig(**cfg_kwargs),
        )

    for with_thinking in (True, False):
        try:
            resp = _call(with_thinking)
            ver = getattr(resp, "model_version", None) or "?"
            note = "" if with_thinking else " (NO thinking_config — needs conditional)"
            return "OK", f"version={ver}{note}"
        except Exception as exc:  # noqa: BLE001 — vendor error surface is broad
            msg = str(exc)
            # Retry without thinking_config only if that's what it rejected.
            if with_thinking and "think" in msg.lower():
                continue
            status = getattr(exc, "code", None) or getattr(
                getattr(exc, "response", None), "status_code", None
            )
            if status == 404 or "NOT_FOUND" in msg:
                return "404 unavailable", msg.split(".")[0][:80]
            if status == 429 or "RESOURCE_EXHAUSTED" in msg:
                hint = _quota_hint(msg)
                verdict = "429 free-limit-0" if "limit=0" in hint else "429 rate/quota"
                return verdict, hint or "quota exceeded"
            return f"ERROR {type(exc).__name__}", msg[:80]
    return "ERROR", "unreachable"


def main() -> int:
    _load_dotenv(_REPO_ROOT / ".env")
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("GEMINI_API_KEY not set (see .env.example)", file=sys.stderr)
        return 1

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)

    print(f"{'MODEL':<32} {'VERDICT':<18} DETAIL")
    print("-" * 90)
    ok_models = []
    for model_id in CANDIDATES:
        verdict, detail = _probe_one(client, types, model_id)
        if verdict == "OK":
            ok_models.append(model_id)
        print(f"{model_id:<32} {verdict:<18} {detail}")

    print("-" * 90)
    if ok_models:
        print(f"WORKING (make one call, so quota not yet exhausted): {ok_models}")
        print("Pick the first for phase_d.model_a, then confirm free daily budget with a "
              "--limit 6 subset smoke.")
    else:
        print("No candidate returned OK — the free lite route has no viable SKU for this key.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
