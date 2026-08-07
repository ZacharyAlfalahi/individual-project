# Architecture
Six-agent LLM pipeline for corporate bond factor replication, bias detection, and repair. Two-layer corpus: anchor (BBW + short-term reversal + six-month momentum, n=3, data-matched, fully manual; KPP methodology-exemplar and Duraj-Giesecke showcase sit outside it) + scale (10–20 papers, pipeline + paper-statistics verification).
Every implementation is validated and verifiable.

## Research Questions
RQ1 — Librarian extraction fidelity on the gold-standard set (BBW, KPP, DG): field-level per-field accuracy, multi-model agreement rate, failure taxonomy (field-level reconstruction, NOT strategy-class routing).
RQ2 — Quant compilation fidelity + coverage: anchor fidelity per the hierarchical gates (evaluation contract §7); layered coverage C_semantic / C_binding / C_execution / C_end-to-end over the frozen candidate set; FIR (headline safety) beside FRR; fraction of the corpus implementable via audited families vs correctly refused (typed refusal taxonomy).
RQ3 — Bias prevalence (Auditor), differential: each strategy run uncorrected (as-published) vs corrected, bias = the gap; effect sizes with CIs (survival counts secondary); scoped, partial estimate over enumerated biases; clean on the anchor set + traded-liquidity negative control, scale layer weaker/confounded.
RQ4 — Repair rate (Scientist): fraction of proposals that repair failing strategies, retain in-sample alpha, survive BH-FDR, show positive OOS Sharpe on the 2022-01..2025-09 holdout (45 months — the complete-months span of the `/data/holdout/` partition; the pull ends 2025-09-30, ruled by post-tag amendment SC-SCI-12 (2026-08-07) correcting SC-SCI-10's false 48/2025-12 and superseding the tag's 2022–2024 window). Conditional on the frontier-completeness citation (`docs/data/registers/citations_verified.md` §3); the tagged 2022–2024 (36-month) sub-window is reported as sensitivity (SC-SCI-13). The holdout claim is descriptive (one OOS observation), not a verdict.
Each RQ has a structurally independent validation path — failure in one component cannot contaminate another.

## Data Rules (inviolable)
- NEVER read or import from /data/holdout/ until walk-forward evaluation (weeks 13–14)
- ALL development, calibration, and evaluation uses /data/development/ (2002–2021)
- BAA-AAA spread median for regime conditioning must be computed on /data/development/ only; commit to /docs/extension_1_config.yaml before holdout opens

## Repository Key Paths
/agents/quant/library/     — ipca.py, characteristic_sort.py, bbw_factors.py (hand-implemented; LLM configures, NEVER modifies; correction-agnostic — run identically on the uncorrected and corrected panels the data layer emits); DNN family deferred (unbuilt)
/agents/auditor/checks/    — deterministic only; zero LLM calls permitted here
/agents/librarian/schema/  — StrategySpec schema (v1.1, built + shipped); JSON field schemas live in /agents/librarian/data/schemas/. Changes to it are deliberate, reviewed migrations
/docs/thresholds.yaml      — ALL numerical thresholds; never hard-code values in agent code
/docs/librarian/specs/inference_rules.md — enumerated rules permitting INFERRED provenance (path reserved; rules not yet authored — see docs/backlog/remaining_work.md)
/data/holdout/             — READ NEVER during development

## Agent Hard Constraints

**Librarian**: dual LLM extraction (stack committed — see D4 / `docs/thresholds.yaml` `librarian.model_stack`). A field is STATED only if both models agree on value AND verbatim quote. No self-reported confidence scores. UNKNOWN is a valid value, not an error.

**Quant**: deterministic compilation — StrategySpec → adapter → QuantConfig → audited runner. Routing is deterministic (closed-enum family table + typed refusal); an LLM appears only as a post-refusal explainer over typed outcomes. The 8-iteration cap applies to the Librarian's retrieval loop if and when the contract's §3.6 gate triggers it. Modifications to /agents/quant/library/ require manual review + all regression tests passing.

**Auditor**: differential comparator — runs each strategy twice (uncorrected/as-published vs corrected) through the same module and measures the gap, NOT single-run inspection (single-run on a pre-cleaned pipeline finds nothing). Checks 1–4 differential; check 5 (multiple-testing) is a non-differential flag. /agents/auditor/checks/ is deterministic — zero LLM calls. LLM appears only in explainer.py after the verdict is produced. Refuses to opine when required fields are UNKNOWN or INFERRED without a matching rule. Silent iteration until pass is forbidden — it is p-hacking.

**Scientist**: input = Auditor-failing strategy. Generates 4–8 repair proposals per the procedure committed to thresholds.yaml before first run. BH-FDR applied jointly across all proposals in-sample. Only FDR-survivors go to holdout. Failure taxonomy: Type 1 (wrong mechanism targeted), Type 2 (repair introduces new bias), Type 3 (repair valid but alpha was never real).

**Reporter**: NEVER regenerates numbers from prose. Every numeric token in the output is asserted against typed pipeline output by verifier.py before commit.

## StrategySpec Schema
Every fact-bearing field is an `Inherited[T]` carrying value, tag (STATED | INFERRED | DESIGN | UNKNOWN), and evidence (STATED requires a verbatim quote + locator; INFERRED requires a rule ID from docs/librarian/specs/inference_rules.md — path reserved, rules not yet authored; DESIGN marks deliberate project substitutions). Schema v1.1 is built and shipped (`agents/librarian/schema/`).

## Replication Success Criterion
**Superseded (2026-07-13): the ±15% primary criterion is retired → evaluation contract §7 hierarchical gates (see RQ2). The paragraph below is retained as history.**
Applies to the data-matched anchors (BBW, str, momentum) on the uncorrected/as-published panel. Primary: Sharpe AND factor loadings AND per-quintile spreads all within 15% tolerance (thresholds.yaml). Coincidental Sharpe match alone is NOT success. KPP is exempt — validated methodologically only (correct IPCA procedure), NO ±15% target, as its ICE-based numbers are unrecoverable on WRDS-MMN. Secondary diagnostic: bootstrap CI overlap reported as an additional column alongside the primary criterion.

## Testing Requirements
- All agents: unit tests in /tests/unit/
- Auditor: synthetic bias injection tests verifying 100% recall for every injected bias type
- Library modules: unit test on synthetic data with known analytical answer + regression test against paper headline metric (data-matched anchors) or published procedure (KPP)

## Open Decisions
D4 — RESOLVED (stack committed to `docs/thresholds.yaml` → `librarian.model_stack`; rationale in `docs/librarian/registers/decision-log_librarian.md` D33). Two-phase dual pair: phase_d (free dev, non-reportable) = Gemini 3.1-flash-lite + Mistral-small; phase_f (reported figures, gated on SKU/cost authorization) = Claude Sonnet 4.6 + Gemini 3.5-flash. The earlier "Sonnet 4.6 + GPT-4o" candidate was NOT adopted.
D5 — OPEN. Dickerson 2026 label availability. If available: Auditor check 5 uses exact name matching against /data/dickerson_zoo/names.csv. If unavailable: check 5 uses t-stat band + free-parameter count only; record outcome in `docs/data/registers/citations_verified.md` (§2 stub — still OPEN).
