"""
Real (vendor-API) ``ModelClient`` implementations for live dual-model extraction
(D33 / L1). This is the concrete client that sits behind the ``ModelClient``
Protocol -- the one piece the offline pipeline deliberately left unbuilt
(``model_client.py`` docstring: "vendors, SDKs, API keys, the network ... live
behind a real implementation of the protocol landed with the model-pair
experiment (D33)").

Design, honouring the frozen extraction contract:

  * **Per-field calls by default.** Every ``answer(FieldQuery, CanonicalText)``
    renders ONE frozen ``(template, schema, decoding)`` triple for the query's
    kind and makes ONE structured call. Each field carries its own prompt/schema
    provenance; per-field trace hashes stay exact. This is the reportable/default
    config -- NOT batch-by-block, and never batch-all/per-kind (which would
    silently change prompt/schema provenance).
  * **Per-(field, kind, text) cache.** ``fill_signal_ref`` re-asks the same query
    (signal_filler.py: "a real client is cached upstream"); the cache makes that
    reuse free AND deterministic. Keyed by the canonical text's ``source_sha256``.
  * **Temperature 0 + JSON output**, per the frozen decoding config (D9/D10/D33).
  * **Returned model-version logged per call** (free aliases drift): the raw
    response + the vendor's returned version string are archived to disk, and the
    version rides on each ``ModelAnswer.model_id``.

Prompt rendering: the templates carry ``{field}`` / ``{definition}`` / ``{menu}``
/ ``{range}`` / ``{registry_menu}`` / ``{decision}`` / ``{strategy_label}`` slots.
Menus + ranges come from ``data/domains.yaml`` (Part 2) and ``schema/fields.py``
(the two Part-1 menus); the registry menu from the Signal Concept Registry. The
per-field ``{definition}`` slot is filled from the frozen, hash-stamped
``data/prompts/definitions.yaml`` (loaded via ``registries.load_field_definitions``)
-- one gloss per routed field, byte-hashed like the silence-policy table, since
rendered prompt content affects extraction. A routed field with no definition
fails loud (never a silent ``_humanise`` fallback).

Vendors are lazy-imported inside each backend, so importing this module (and thus
running the offline unit tests) never requires an SDK to be installed.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from ..config.canonical_text import CanonicalText
from ..errors import LibrarianSchemaError
from ..registries import SignalConceptRegistry, load_field_definitions
from ..schema import fields as F
from .lister import CONSTRUCTION_CLASSES, Construction, GridInfo
from .model_client import FieldQuery, ModelAnswer
from .prompts import PromptManifest, load_prompt_manifest

_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
_DOMAINS_PATH = _DATA_ROOT / "domains.yaml"

# Part-1 menus live in schema/fields.py (they are NOT in domains.yaml, which is
# the Part-2 domain source). Mapped here so the prompt builder has one lookup.
_PART1_MENUS: dict[str, tuple[str, ...]] = {
    F.FORMATION_STRUCTURE: F.FORMATION_STRUCTURE_MENU,
    F.ASSET_CLASS: F.ASSET_CLASS_MENU,
}

# The D13 "decision this feeds" framing for the two Part-1 routing questions.
_PART1_DECISIONS: dict[str, str] = {
    F.FORMATION_STRUCTURE: (
        "how the strategy forms positions: sorted portfolios vs an estimated "
        "factor model vs a trained predictor (routes Part-2 extraction)."
    ),
    F.ASSET_CLASS: "which asset class the strategy trades (scopes the universe).",
}


class RealClientError(RuntimeError):
    """A live model call failed hard (auth/quota/network) after retries. Raised
    rather than returned so a dead API can never masquerade as a silent paper --
    an all-UNKNOWN spec from a broken key would be a silent, misleading result."""


# ---------------------------------------------------------------------------
# Prompt rendering.
# ---------------------------------------------------------------------------

def _humanise(field_name: str) -> str:
    return field_name.replace("_", " ")


def _render_menu(values: tuple[str, ...]) -> str:
    return "\n".join(f"  - {v}" for v in values)


def _render_range(dom: dict) -> str:
    lo, hi = dom.get("min"), dom.get("max")
    parts = ["an integer"]
    if lo is not None:
        parts.append(f">= {lo}")
    if hi is not None:
        parts.append(f"<= {hi}")
    elif lo is not None:
        parts.append("(no upper bound)")
    return " ".join(parts)


def _render_registry_menu(registry: SignalConceptRegistry) -> str:
    lines: list[str] = []
    for c in registry.concepts:
        aliases = ", ".join(c.aliases)
        lines.append(f"  - {c.concept_id} -- {c.definition.strip()} (aliases: {aliases})")
    return "\n".join(lines)


@dataclass(frozen=True)
class PromptBuilder:
    """Renders the full instruction prompt + JSON schema for one ``FieldQuery``,
    from the frozen templates + the domain/menu/registry sources. Loaded once and
    shared by both clients in a pair (immutable)."""

    manifest: PromptManifest
    registry: SignalConceptRegistry
    _templates: dict  # kind -> template text
    _schemas: dict    # kind -> schema dict
    _decoding: dict   # kind -> decoding dict
    _domains2: dict   # part2 field -> domain dict
    _registry_menu: str
    _definitions: dict  # field -> frozen {definition} gloss (definitions.yaml)
    _run_templates: dict  # run-template name -> template text (WS-3)
    _run_schemas: dict    # run-template name -> schema dict
    _run_decoding: dict   # run-template name -> decoding dict

    @classmethod
    def load(
        cls,
        registry: SignalConceptRegistry,
        manifest: PromptManifest | None = None,
    ) -> "PromptBuilder":
        manifest = manifest or load_prompt_manifest()
        templates: dict[str, str] = {}
        schemas: dict[str, dict] = {}
        decoding: dict[str, dict] = {}
        for kind, tpl in manifest.templates.items():
            templates[kind] = (_DATA_ROOT / tpl.prompt).read_text(encoding="utf-8")
            schemas[kind] = json.loads((_DATA_ROOT / tpl.schema).read_text(encoding="utf-8"))
            decoding[kind] = yaml.safe_load((_DATA_ROOT / tpl.decoding).read_text(encoding="utf-8"))
        # Run-templates (WS-3): whole-paper structured calls (e.g. enumeration),
        # loaded the same way but keyed by name and NOT routed through render().
        run_templates: dict[str, str] = {}
        run_schemas: dict[str, dict] = {}
        run_decoding: dict[str, dict] = {}
        for name, tpl in manifest.run_templates.items():
            run_templates[name] = (_DATA_ROOT / tpl.prompt).read_text(encoding="utf-8")
            run_schemas[name] = json.loads((_DATA_ROOT / tpl.schema).read_text(encoding="utf-8"))
            run_decoding[name] = yaml.safe_load((_DATA_ROOT / tpl.decoding).read_text(encoding="utf-8"))
        with _DOMAINS_PATH.open("r", encoding="utf-8") as fh:
            domains = yaml.safe_load(fh)
        field_defs = load_field_definitions()
        return cls(
            manifest=manifest,
            registry=registry,
            _templates=templates,
            _schemas=schemas,
            _decoding=decoding,
            _domains2=dict(domains.get("part2", {})),
            _registry_menu=_render_registry_menu(registry),
            _definitions=dict(field_defs.definitions),
            _run_templates=run_templates,
            _run_schemas=run_schemas,
            _run_decoding=run_decoding,
        )

    def _menu_for(self, field_name: str) -> str:
        if field_name in _PART1_MENUS:
            return _render_menu(_PART1_MENUS[field_name])
        dom = self._domains2.get(field_name)
        if dom and dom.get("kind") == "enum":
            return _render_menu(tuple(dom.get("values", ())))
        # No domain menu for a field routed here as an enum: fail loud rather than
        # ship a degenerate one-token prompt (fail-closed, matching the repo).
        raise LibrarianSchemaError(
            f"no enum menu for field {field_name!r} in domains.yaml or the Part-1 menus"
        )

    def _definition_for(self, field_name: str) -> str:
        """The frozen ``{definition}`` gloss for a field routed to a
        ``{definition}``-carrying template. Fail-closed (matching ``_menu_for``): a
        routed field with no authoritative gloss raises rather than falling back to
        an un-frozen ``_humanise`` gloss (rendered prompt content affects
        extraction, so an unstamped definition must never reach a live call)."""
        definition = self._definitions.get(field_name)
        if not definition:
            raise LibrarianSchemaError(
                f"no frozen definition for field {field_name!r} in definitions.yaml "
                "(a field routed to a {definition} template must have an authoritative gloss)"
            )
        return definition

    def schema_for(self, kind: str) -> dict:
        return self._schemas[kind]

    def max_tokens_for(self, kind: str) -> int:
        return int(self._decoding.get(kind, {}).get("max_output_tokens", 512))

    # -- run-template accessors (WS-3) --------------------------------------
    # A run-template is a whole-paper structured call (e.g. enumeration); it is
    # bound by name and NOT dispatched through render()/FIELD_KINDS.
    def run_prompt(self, name: str) -> str:
        return self._run_templates[name]

    def run_schema(self, name: str) -> dict:
        return self._run_schemas[name]

    def run_max_tokens(self, name: str) -> int:
        return int(self._run_decoding.get(name, {}).get("max_output_tokens", 1024))

    def render(self, query: FieldQuery, strategy_label: str) -> str:
        """The instruction block (template with slots filled). The paper text +
        the JSON-schema instruction are appended by the client.

        The ``{definition}`` slot (enum / int / date / paper_metric) is filled from
        the frozen, hash-stamped ``definitions.yaml`` via ``_definition_for`` --
        fail-loud on a routed field with no gloss. The kinds without a
        ``{definition}`` slot (part1_enum / signal_ref / method_summary) never
        touch it; part1_enum keeps its ``_humanise`` fallback for the ``{decision}``
        framing only (both Part-1 fields are covered by ``_PART1_DECISIONS``)."""
        kind = query.kind
        tpl = self._templates[kind]
        if kind in ("enum",):
            return tpl.format(
                field=query.field,
                definition=self._definition_for(query.field),
                menu=self._menu_for(query.field),
            )
        if kind == "int":
            dom = self._domains2.get(query.field, {})
            return tpl.format(
                field=query.field,
                definition=self._definition_for(query.field),
                range=_render_range(dom),
            )
        if kind == "part1_enum":
            decision = _PART1_DECISIONS.get(query.field, _humanise(query.field))
            return tpl.format(field=query.field, decision=decision, menu=self._menu_for(query.field))
        if kind == "signal_ref":
            return tpl.format(field=query.field, registry_menu=self._registry_menu)
        if kind == "method_summary":
            return tpl.format(strategy_label=strategy_label)
        if kind == "date":
            return tpl.format(field=query.field, definition=self._definition_for(query.field))
        if kind == "paper_metric":
            # The strategy label matters here: a paper reports many numbers, and
            # the field is "the headline figure THIS strategy claims" (D20 keys
            # RQ1 scoring by the label), not "a number from this paper".
            return tpl.format(
                field=query.field,
                definition=self._definition_for(query.field),
                strategy_label=strategy_label,
            )
        raise LibrarianSchemaError(f"RealModelClient cannot render unknown kind {kind!r}")


# ---------------------------------------------------------------------------
# Vendor backends (lazy-imported SDKs).
# ---------------------------------------------------------------------------

class _Backend(Protocol):
    def generate(self, prompt: str, max_output_tokens: int) -> tuple[str, str | None]:
        """Return (raw_text, returned_model_version). Raises on hard failure."""
        ...


@dataclass
class _GeminiBackend:
    model_id: str
    api_key: str
    temperature: float = 0.0

    def __post_init__(self) -> None:
        from google import genai  # lazy

        self._genai = genai
        self._client = genai.Client(api_key=self.api_key)

    def generate(self, prompt: str, max_output_tokens: int) -> tuple[str, str | None]:
        from google.genai import types  # lazy

        resp = self._client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                top_p=1,
                max_output_tokens=max_output_tokens,
                response_mime_type="application/json",
                # The pinned Gemini SKU (gemini-3.5-flash) is a THINKING model: with
                # thinking on, reasoning tokens share the max_output_tokens budget and can
                # starve/truncate the JSON (empty replies on tight budgets). Extraction is
                # a lookup, not a reasoning task -> disable thinking for a deterministic,
                # token-efficient (free-tier) call that spends the whole budget on output.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return resp.text or "", getattr(resp, "model_version", None)


@dataclass
class _MistralBackend:
    model_id: str
    api_key: str
    temperature: float = 0.0

    def __post_init__(self) -> None:
        try:  # lazy; v1.x exports Mistral at top level
            from mistralai import Mistral
        except ImportError:  # v2.x moved it under mistralai.client
            from mistralai.client import Mistral

        self._client = Mistral(api_key=self.api_key)

    def generate(self, prompt: str, max_output_tokens: int) -> tuple[str, str | None]:
        resp = self._client.chat.complete(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=max_output_tokens,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        # v2.x may return a list of content chunks instead of a plain string.
        if isinstance(content, list):
            content = "".join(
                getattr(chunk, "text", "") for chunk in content if getattr(chunk, "type", "text") == "text"
            )
        return content or "", getattr(resp, "model", None)


@dataclass
class _AnthropicBackend:
    model_id: str
    api_key: str
    temperature: float = 0.0

    def __post_init__(self) -> None:
        import anthropic  # lazy

        self._client = anthropic.Anthropic(api_key=self.api_key)

    def generate(self, prompt: str, max_output_tokens: int) -> tuple[str, str | None]:
        resp = self._client.messages.create(
            model=self.model_id,
            max_tokens=max_output_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return text, getattr(resp, "model", None)


_BACKENDS = {
    "gemini": _GeminiBackend,
    "mistral": _MistralBackend,
    "anthropic": _AnthropicBackend,
}


def make_backend(vendor: str, model_id: str, api_key: str, temperature: float = 0.0) -> _Backend:
    if vendor not in _BACKENDS:
        raise LibrarianSchemaError(
            f"unknown model vendor {vendor!r}; known: {sorted(_BACKENDS)}"
        )
    return _BACKENDS[vendor](model_id=model_id, api_key=api_key, temperature=temperature)


# 4xx client errors that never succeed on retry. 429 (rate limit) and 5xx are
# deliberately EXCLUDED -- those are retryable.
_NON_RETRYABLE_STATUS = frozenset({400, 401, 403, 404, 422})


def _is_non_retryable(exc: Exception) -> bool:
    """True for client errors a retry cannot fix (bad key, forbidden, not-found,
    bad request). SDK-agnostic: inspects a status code + the exception class name,
    so no vendor exception type needs importing. Unknown errors -> retryable."""
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return status in _NON_RETRYABLE_STATUS
    name = type(exc).__name__.lower()
    return any(
        tok in name
        for tok in ("authentication", "permission", "notfound", "badrequest", "unprocessable")
    )


def _retry_after_seconds(exc: Exception) -> float | None:
    """The server's retry delay (seconds) if exposed, else None. Checked in order:
    the ``Retry-After`` header (Mistral ``http_res`` / Gemini ``response``), then the
    error BODY -- Gemini puts its delay there, not in a header (a ``RetryInfo``
    ``retryDelay`` and a "Please retry in Ns" message)."""
    resp = getattr(exc, "http_res", None) or getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if headers:
        val = headers.get("Retry-After") or headers.get("retry-after")
        if val is not None:
            try:
                return max(0.0, float(val))
            except (TypeError, ValueError):
                pass
    # Gemini: delay is in the error body -- "retry in 57.1s" / "retryDelay': '57s'".
    m = re.search(r"retry(?:Delay['\"]?[:=]\s*['\"]?|\s+in\s+)(\d+(?:\.\d+)?)s", str(exc))
    if m:
        return max(0.0, float(m.group(1)))
    return None


# ---------------------------------------------------------------------------
# JSON parsing helpers.
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict | None:
    """Best-effort parse of a model's JSON reply. Strips ``` fences and slices to
    the outermost object. Returns None on failure (the caller records a format
    failure and treats the field as silent -- run-to-completion, D33 bucket 3)."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass
    lo, hi = t.find("{"), t.rfind("}")
    if 0 <= lo < hi:
        try:
            obj = json.loads(t[lo : hi + 1])
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


_PAPER_METRIC_KEYS: tuple[str, ...] = ("mean", "t_stat", "unit")

# The closed unit menu. Mirrors the `unit` enum in schemas/paper_metric.schema.json
# -- the JSON Schema is inlined into the prompt but NOT vendor-enforced, so this is
# the only place the menu is actually applied. A tripwire test asserts the two stay
# in step. An off-menu unit is not cosmetic: it mis-scales every downstream numeric
# comparison (pct_per_year vs pct_per_month is a 12x error).
_PAPER_METRIC_UNITS: frozenset[str] = frozenset(
    ("pct_per_month", "pct_per_year", "bps_per_month", "decimal_per_month")
)

# paper_facts dates are YYYY-MM (zero-padded month), per schemas/date.schema.json.
_DATE_RE = re.compile(r"^[0-9]{4}-(0[1-9]|1[0-2])$")


def _coerce_date(value: Any) -> str | None:
    """A YYYY-MM date string, or None if it is not one.

    The schema declares the pattern but nothing enforces it (schemas are inlined
    into the prompt, not vendor-enforced), so this is the enforcement point. A
    model that answers "July 2004" or a bare year has NOT answered this field:
    returning None degrades it to silent, which is the safe state -- otherwise two
    models both saying "July 2004" would agree, locate, and ship STATED with a
    value no gold can match."""
    if not isinstance(value, str):
        return None
    token = value.strip()
    return token if _DATE_RE.match(token) else None


def _coerce_paper_metric(value: Any) -> dict | None:
    """The composite claimed_headline_metric value, or None if it is not a
    complete {mean, t_stat, unit} triple.

    All-or-nothing by design: a mean without its t-statistic is not a headline
    claim, and shipping a half-populated dict would put a value into the D9 merge
    that no gold can ever match, and that the Reporter's numeric verifier would
    later have to special-case. Partial -> None -> the caller degrades to silent
    (the same rule as an unparseable int)."""
    if not isinstance(value, dict):
        return None
    out: dict = {}
    for key in _PAPER_METRIC_KEYS:
        if key not in value:
            return None
        v = value[key]
        if key == "unit":
            if not (isinstance(v, str) and v.strip() in _PAPER_METRIC_UNITS):
                return None
            out[key] = v.strip()
            continue
        # mean / t_stat: a number, or a numeric string the model quoted.
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            out[key] = float(v)
            continue
        if isinstance(v, str):
            try:
                out[key] = float(v.strip())
            except ValueError:
                return None
            continue
        return None
    return out


def _answer_from_parsed(field_name: str, kind: str, parsed: dict, model_id: str) -> ModelAnswer:
    """Map a parsed JSON reply to a ``ModelAnswer``, defensively. An ``answered``
    reply with no supporting quote is downgraded to silent -- the D9 gate has
    nothing to locate, and a value without evidence is worse than abstention."""
    answered = bool(parsed.get("answered"))
    if not answered:
        return ModelAnswer(field=field_name, answered=False, model_id=model_id)

    if kind == "method_summary":
        summary = parsed.get("summary")
        quotes = tuple(q for q in (parsed.get("quotes") or []) if isinstance(q, str) and q.strip())
        if not summary or not quotes:
            return ModelAnswer(field=field_name, answered=False, model_id=model_id)
        return ModelAnswer(
            field=field_name, answered=True, raw=summary, quotes=quotes, model_id=model_id
        )

    quote = parsed.get("quote")
    if kind == "signal_ref":
        raw = parsed.get("concept_id")
    elif kind == "int":
        raw = _coerce_int(parsed.get("value"))
    elif kind == "paper_metric":
        raw = _coerce_paper_metric(parsed.get("value"))
    elif kind == "date":
        raw = _coerce_date(parsed.get("value"))
    else:  # enum, part1_enum
        raw = parsed.get("value")

    # A missing value, an empty/whitespace-only string value, or a missing quote
    # all degrade to silent: a value without locatable evidence is worse than
    # abstention (and pollutes the D9 merge/trace with a meaningless token).
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return ModelAnswer(field=field_name, answered=False, model_id=model_id)
    if not (isinstance(quote, str) and quote.strip()):
        return ModelAnswer(field=field_name, answered=False, model_id=model_id)
    return ModelAnswer(field=field_name, answered=True, raw=raw, quote=quote, model_id=model_id)


def _constructions_from_parsed(parsed: dict) -> tuple[Construction, ...]:
    """Map a parsed enumeration reply to a tuple of UNLOCATED Constructions (D20).

    Defensive, like the per-field mappers: a malformed row (missing/blank name or
    quote, or a class not in ``CONSTRUCTION_CLASSES``) is SKIPPED, and any row that
    fails ``Construction``/``GridInfo`` validation (e.g. a grid with ``is_grid`` but
    no ``headline_cell``) is dropped rather than raised. A non-list ``constructions``
    payload (or a non-iterable ``cells_noted``) also degrades to empty rather than
    raising, so a hostile/hallucinated reply can never crash the live run. Locators
    are left ``None`` -- ``enumerate_constructions`` relocates each quote against the
    canonical text, so this producer never locates."""
    out: list[Construction] = []
    rows = parsed.get("constructions")
    if not isinstance(rows, list):
        return ()  # a non-list (or absent) 'constructions' -> nothing to map; never raise
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        quote = row.get("quote")
        cls = row.get("class")
        if not (isinstance(name, str) and name.strip()):
            continue
        if not (isinstance(quote, str) and quote.strip()):
            continue
        if cls not in CONSTRUCTION_CLASSES:
            continue
        raw_grid = row.get("grid")
        if not isinstance(raw_grid, dict):
            raw_grid = {}
        cells = raw_grid.get("cells_noted", [])
        if not isinstance(cells, (list, tuple)):
            cells = []  # a non-iterable cells_noted degrades to empty, never raises TypeError
        try:
            grid = GridInfo(
                is_grid=bool(raw_grid.get("is_grid", False)),
                headline_cell=raw_grid.get("headline_cell"),
                cells_noted=tuple(str(c) for c in cells),
            )
            out.append(Construction(name=name, quote=quote, cls=cls, grid=grid, locator=None))
        except LibrarianSchemaError:
            continue
    return tuple(out)


# ---------------------------------------------------------------------------
# The client.
# ---------------------------------------------------------------------------

class RealModelClient:
    """A live ``ModelClient`` (satisfies the Protocol: ``model_id`` + ``answer``).

    ``current_strategy_label`` is set per construction by the caller before its
    fields are filled, so ``method_summary`` renders with the right strategy name
    (the Protocol is field-oriented and carries no construction context)."""

    def __init__(
        self,
        model_id: str,
        vendor: str,
        api_key: str,
        builder: PromptBuilder,
        *,
        temperature: float = 0.0,
        archive_path: str | Path | None = None,
        max_retries: int = 6,
        backoff_base: float = 2.0,
        backoff_cap: float = 30.0,
        min_interval_s: float = 0.0,
    ) -> None:
        self.model_id = model_id
        self.vendor = vendor
        self._builder = builder
        self._backend = make_backend(vendor, model_id, api_key, temperature=temperature)
        self._cache: dict[tuple[str, str, str], ModelAnswer] = {}
        self._archive_path = Path(archive_path) if archive_path is not None else None
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap
        # Proactive pacing: minimum seconds between live calls (free tiers cap
        # requests/sec and tokens/min). 0 = no pacing.
        self._min_interval_s = min_interval_s
        self._last_call_ts: float = 0.0
        self.current_strategy_label: str = "the strategy described in this paper"
        self.format_failures: int = 0

    # -- ModelClient.answer --------------------------------------------------
    def answer(self, query: FieldQuery, canonical_text: CanonicalText) -> ModelAnswer:
        if not isinstance(query, FieldQuery):
            raise LibrarianSchemaError("RealModelClient.answer expects a FieldQuery")
        key = (query.field, query.kind, canonical_text.source_sha256)
        if key in self._cache:
            return self._cache[key]

        instruction = self._builder.render(query, self.current_strategy_label)
        schema = self._builder.schema_for(query.kind)
        paper_text = "\n\n".join(canonical_text.pages)
        prompt = (
            f"{instruction}\n\n"
            "PAPER TEXT (your ONLY source -- quote verbatim, character-for-character):\n"
            f"<<<\n{paper_text}\n>>>\n\n"
            "Return ONLY a single JSON object (no prose, no code fence) matching "
            f"this JSON Schema:\n{json.dumps(schema)}"
        )
        max_tokens = self._builder.max_tokens_for(query.kind)

        raw_text, version = self._generate_with_retry(prompt, max_tokens, query.field)
        parsed = _parse_json(raw_text)
        model_id_stamp = version or self.model_id
        if parsed is None:
            self.format_failures += 1
            answer = ModelAnswer(field=query.field, answered=False, model_id=model_id_stamp)
        else:
            answer = _answer_from_parsed(query.field, query.kind, parsed, model_id_stamp)

        self._archive(query, prompt, raw_text, version, parsed, answer)
        self._cache[key] = answer
        return answer

    # -- enumeration producer (WS-3) ----------------------------------------
    def extract_enumeration(self, canonical_text: CanonicalText) -> tuple[Construction, ...]:
        """Live enumeration (D20): ONE structured whole-paper call returning every
        construction the paper describes, as UNLOCATED ``Construction``s
        (``locator=None``). ``enumerate_constructions`` relocates each quote against
        the canonical text downstream, so this producer never calls ``.locate``.

        A parse failure returns ``()`` and bumps ``format_failures`` -- the same
        run-to-completion degradation the per-field path uses (an empty list routes
        to review via the dual-model agreement gate, never a silent partial spec)."""
        if not isinstance(canonical_text, CanonicalText):
            raise LibrarianSchemaError("RealModelClient.extract_enumeration expects a CanonicalText")
        paper_text = "\n\n".join(canonical_text.pages)
        schema = self._builder.run_schema("enumeration")
        prompt = (
            f"{self._builder.run_prompt('enumeration')}\n\n"
            "PAPER TEXT (your ONLY source -- quote verbatim, character-for-character):\n"
            f"<<<\n{paper_text}\n>>>\n\n"
            "Return ONLY a single JSON object (no prose, no code fence) matching "
            f"this JSON Schema:\n{json.dumps(schema)}"
        )
        max_tokens = self._builder.run_max_tokens("enumeration")
        raw_text, _version = self._generate_with_retry(prompt, max_tokens, "enumeration")
        parsed = _parse_json(raw_text)
        if parsed is None:
            self.format_failures += 1
            return ()
        return _constructions_from_parsed(parsed)

    # -- proactive rate-limit pacing ----------------------------------------
    def _pace(self) -> None:
        if self._min_interval_s <= 0:
            return
        elapsed = time.monotonic() - self._last_call_ts
        if elapsed < self._min_interval_s:
            time.sleep(self._min_interval_s - elapsed)

    # -- vendor call with retry ---------------------------------------------
    def _generate_with_retry(
        self, prompt: str, max_tokens: int, field_label: str
    ) -> tuple[str, str | None]:
        last_exc: Exception | None = None
        attempts = 0
        for attempt in range(self._max_retries):
            attempts = attempt + 1
            self._pace()
            try:
                out = self._backend.generate(prompt, max_tokens)
                self._last_call_ts = time.monotonic()
                return out
            except Exception as exc:  # vendor SDK exception surface is broad
                last_exc = exc
                self._last_call_ts = time.monotonic()
                # Non-retryable client errors (bad key, forbidden, bad request)
                # will never succeed on retry -- fail fast instead of burning the
                # whole backoff budget on every field (a bad key would otherwise
                # cost minutes before the first field completes).
                if _is_non_retryable(exc) or attempt >= self._max_retries - 1:
                    break
                # Prefer the server's Retry-After (rate limits); else exponential
                # backoff, capped, + jitter (desync the two clients' retries).
                wait = _retry_after_seconds(exc)
                if wait is None:
                    wait = min(self._backoff_base**attempt, self._backoff_cap) + random.uniform(0, 0.5)
                time.sleep(wait)
        plural = "attempt" if attempts == 1 else "attempts"
        raise RealClientError(
            f"{self.vendor}:{self.model_id} failed on field {field_label!r} after "
            f"{attempts} {plural}: {type(last_exc).__name__}: {str(last_exc)[:200]}"
        ) from last_exc

    # -- raw-response archive (audit + version log) -------------------------
    def _archive(
        self,
        query: FieldQuery,
        prompt: str,
        raw_text: str,
        version: str | None,
        parsed: dict | None,
        answer: ModelAnswer,
    ) -> None:
        if self._archive_path is None:
            return
        record = {
            "field": query.field,
            "kind": query.kind,
            "configured_model_id": self.model_id,
            "returned_model_version": version,
            "answered": answer.answered,
            "parsed": parsed,
            "raw_text": raw_text,
            "parse_failed": parsed is None,
        }
        self._archive_path.parent.mkdir(parents=True, exist_ok=True)
        with self._archive_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_client_pair(
    stack: dict,
    builder: PromptBuilder,
    api_keys: dict[str, str],
    *,
    archive_dir: str | Path | None = None,
) -> tuple[RealModelClient, RealModelClient]:
    """Build (model_a, model_b) from a ``librarian.model_stack`` phase block
    (``{model_a: {vendor, model_id, api_key_env}, model_b: {...}}``).

    ``api_keys`` maps ``api_key_env`` names to their resolved values (the caller
    reads the environment; this stays testable). ``temperature`` +
    ``structured_decoding`` come from the enclosing stack block."""
    temperature = float(stack.get("temperature", 0))
    archive_dir = Path(archive_dir) if archive_dir is not None else None

    def _one(role: str) -> RealModelClient:
        spec = stack[role]
        env = spec["api_key_env"]
        key = api_keys.get(env)
        if not key:
            raise RealClientError(
                f"missing API key: env var {env!r} is unset (see .env.example). "
                f"Cannot build the {role} client ({spec['vendor']}:{spec['model_id']})."
            )
        archive = (archive_dir / f"raw_{role}.jsonl") if archive_dir is not None else None
        # Rate-limit knobs: per-model override, else stack default, else the
        # RealModelClient default (thresholds.yaml -> librarian.model_stack).
        pace = float(spec.get("min_interval_s", stack.get("min_interval_s", 0)))
        retries = int(spec.get("max_retries", stack.get("max_retries", 6)))
        return RealModelClient(
            model_id=spec["model_id"],
            vendor=spec["vendor"],
            api_key=key,
            builder=builder,
            temperature=temperature,
            archive_path=archive,
            min_interval_s=pace,
            max_retries=retries,
        )

    return _one("model_a"), _one("model_b")
