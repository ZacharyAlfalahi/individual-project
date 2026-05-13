# Architecture
Six-agent LLM pipeline that ingests academic finance PDFs and produces: (1) provenance-tracked StrategySpec extractions, (2) backtests via audited pre-built library modules, (3) bias audits (5 deterministic checks), (4) two pre-registered extensions, (5) research notes + Signal Registry entries.
Timeline: 10 May – 9 September 2026. Interim: 3 June. Experiment lock: 2 August.


## Research Questions
RQ1 — Extraction fidelity: per-field accuracy of Librarian extraction vs hand-curated gold standard, with provenance; multi-model agreement rate (Claude vs GPT-4o); failure-mode taxonomy.
RQ2 — Bias prevalence: fraction of manually-implemented corpus strategies failing automated bias correction. Evaluated on manual implementations independent of Librarian.
RQ3 — Constrained extension quality: two pre-registered extensions surviving bias correction and showing positive OOS Sharpe on 2022–2024 holdout. Built on manual baselines, independent of Librarian.

Each RQ is answered through a structurally independent evaluation pathway. A failure in any one component cannot contaminate the others.

## Repository Structure
/agents/librarian/         — PDF → StrategySpec with Fact[T] provenance on every field
/agents/quant/             — StrategySpec → BacktestResult (configures library; does NOT author algorithms)
/agents/quant/library/     — ipca.py, four_factor_sort.py, dnn_residual.py (hand-implemented, unit-tested; LLM may call, not modify)
/agents/auditor/           — BacktestResult → AuditReport (deterministic checks; NO LLM calls in checks/)
/agents/auditor/checks/    — Deterministic check logic only
/agents/auditor/explainer.py — LLM plain-English narrative ONLY, after check verdict is produced
/agents/scientist/         — Two pre-registered extensions via FAISS over version-locked mechanism library
/agents/devils_advocate/   — AdversarialReview (deterministic crowding + capacity + regime breakdown)
/agents/reporter/          — Pipeline output → research note + Signal Registry (PostgreSQL)
/orchestration/            — LangGraph state machine
/schema/                   — StrategySpec and Pydantic models (designed end of week 3; NEVER modify without instruction)
/data/development/         — TRACE panel 2002-2021 (safe to use)
/data/holdout/             — TRACE panel 2022-2024 (READ NEVER — holdout only)
/data/mechanism_library/   — mechanisms.json (version-locked; NO LLM writes or modifications)
/data/models/              — duraj_giesecke_2002_2021.pt
/data/dickerson_zoo/       — names.csv (check 5 multiple-testing flag)
/docs/thresholds.yaml      — ALL numerical thresholds with source citations; Auditor reads at runtime; nothing hard-coded
/docs/reporting_delays.yaml — Per-source publication/reporting delays (check 1 look-ahead)
/docs/inference_rules.md   — Enumerated INFERRED provenance rules with rule IDs
/docs/multiple_testing.md  — Precise definition of check 5 flag conditions
/docs/domain_facts.md      — Transaction costs, capacity assumptions with citations
/docs/extension_1_config.yaml — Macro regime boundary committed before holdout opens
/docs/citations_verified.md — Citation verification results (week 3)
/notebooks/                — 01_trace_explore.ipynb (TRACE liquidity distribution; calibrates checks 3)
/tests/                    — pytest suite including synthetic bias injection tests

## StrategySpec Schema (CRITICAL)
Schema is designed at end of week 3, after KPP and BBW manual replications complete. NOT designed in week 1.
Every fact-bearing field is wrapped as Fact[T] with three components:
  value       — the extracted value
  provenance  — STATED | INFERRED | UNKNOWN
  quote       — verbatim PDF span (required when STATED)
INFERRED is permitted only for fields with an explicit rule in /docs/inference_rules.md (rule ID recorded on the field).
UNKNOWN is a first-class value, not an error; the Auditor refuses to opine on checks dependent on UNKNOWN fields.
Librarian does NOT produce self-reported confidence scores (miscalibrated). Only signal recorded is multi-model agreement rate: fraction of fields where Claude Sonnet 4.6 and GPT-4o agree on both value and quote.
DO NOT modify schema files without explicit instruction stating "modify the schema".

## The Six Agents

### Librarian
Input: academic PDF
Output: StrategySpec with Fact[T] provenance on every fact-bearing field
Pipeline: Nougat (equations as LaTeX) → PyMuPDF (text/tables with page offsets) → Nougat/PyMuPDF cross-check
  (equations without matching textual context are flagged) → dual extraction via Instructor (Claude Sonnet 4.6 + GPT-4o)
  → multi-model agreement: field marked STATED only if both models agree on value AND verbatim quote; disagreement → manual review.

### Quant
Input: StrategySpec (all algorithm-determining fields provenance STATED)
Output: BacktestResult (configuration code + metrics + verification artefacts)
Critical: Quant configures pre-built library modules; it does NOT author algorithms.
  /agents/quant/library/ipca.py          — KPP IPCA, hand-implemented in NumPy (ALS); Quant may call, not modify
  /agents/quant/library/four_factor_sort.py — BBW four-factor sort; same constraint
  /agents/quant/library/dnn_residual.py  — Duraj-Giesecke DNN; same constraint
  Modifications require manual code review and all regression tests must pass.
Execution: ReAct loop, hard cap 8 iterations (cost-control; report iteration distribution in project).
Success requires: Sharpe AND factor loadings AND portfolio composition (3 sample dates) AND per-quintile spreads, all within tolerances in /docs/thresholds.yaml. A coincidental Sharpe match alone is NOT success.

### Auditor
Input: BacktestResult + StrategySpec
Output: AuditReport (5 check results + plain-English explanations)
CRITICAL: /agents/auditor/checks/ is DETERMINISTIC — NO LLM calls.
LLMs appear only in /agents/auditor/explainer.py, AFTER the deterministic check has produced its verdict.
Refusal behaviour: Auditor refuses to opine on any check where required fields have provenance ≠ STATED (unless an inference rule applies). Refusal recorded with structured reason code. RQ2 aggregate reported two ways: counting refusals as failures (conservative) and excluding refusals (permissive).
Silent iteration until a check passes is FORBIDDEN — constitutes p-hacking through the Auditor. Failures are documented; any methodology revision must be recorded before rerunning.

The five checks (all thresholds in /docs/thresholds.yaml — never hard-coded in agent code):
  Check 1 — Look-ahead: verify variable lag_specification against reporting delays in /docs/reporting_delays.yaml.
    Refuse to opine if lag_specification provenance is not STATED (unless inference rule applies).
  Check 2 — TRACE measurement error: Sharpe on raw TRACE vs WRDS-MMN corrected; fail if degradation > threshold.
  Check 3 — Stale price: exclude bonds with trading gaps > threshold days; fail/warn on exclusion proportion thresholds.
    Thresholds calibrated against TRACE liquidity distribution in /notebooks/01_trace_explore.ipynb.
  Check 4 — Survivorship: point-in-time universe construction; defaulted/matured bonds retained through actual exit dates.
  Check 5 — Multiple testing flag (WARNING only; does NOT gate Auditor pass for routing):
    Flag if: t-stat in (2.0, 2.5) [Harvey-Liu-Zhu 2016]; OR >2 free parameters per /docs/multiple_testing.md;
    OR matched in /data/dickerson_zoo/names.csv.
    Whether flag counts as RQ2 failure controlled by counted_as_failure in thresholds.yaml; both interpretations reported.

### Scientist
Input: baseline strategy that has passed the Auditor
Output: one of two pre-registered ExtensionProposal objects, each with an explicit mechanism_id
Critical: Economic Mechanism Library (/data/mechanism_library/mechanisms.json) is manually populated from
  the curated reading list and committed to git, version-locked. NO LLM may add or modify mechanisms.
  Scientist retrieves mechanisms via FAISS over sentence-transformers only.
  mechanism_id on each proposal is manually reviewed before any holdout evaluation.

The two pre-registered extensions (binding from interim report 3 June — no further extensions generated):
  extension_1_macro_conditioned_ipca: KPP estimated per binary credit-cycle regime (BAA-AAA spread above/below
    in-sample median). Median committed to /docs/extension_1_config.yaml before holdout opens. Combine via
    regime-weighted average. Mechanism: time-varying risk premia across credit cycle.
  extension_2_dnn_residual_signal: Duraj-Giesecke DNN trained on 2002-2021; OOS residuals used as
    cross-sectional long-short signal. Mechanism: structured features unmodelled by nonlinear baselines.
    Note: technically a separate model, not an IPCA extension.

### Devil's Advocate
Input: ExtensionProposal (passed Auditor)
Output: AdversarialReview with four components:
  Crowding test: alpha + t-stat vs pre-committed factor set (BBW 4 factors + KPP 5 factors); crowded if alpha insignificant at 5%.
  Capacity estimate: turnover-implied AUM ceiling (40 bps IG, 80 bps HY round-trip; citation in /docs/domain_facts.md).
  Regime breakdown: performance split on BAA-AAA spread above vs below in-sample median (defined pre-holdout).
  LLM counter-argument: narrative for Reporter only; NOT used as an evaluation gate.

### Reporter
Input: complete pipeline output for one strategy
Output: Markdown research note + Signal Registry row (PostgreSQL)
Critical: every numerical claim in the research note is a literal copy from upstream typed outputs.
  LLM does NOT regenerate numbers. A validation pass checks every numeric token against structured pipeline output before commit.
  Economic interpretation section must cite mechanism_id explicitly.

## LangGraph Orchestration
Graph: Librarian → Quant → Auditor → [pass] → Scientist → Quant → Auditor → [pass] → Devil's Advocate → Reporter
Failed Auditor check routes back to Quant with specific fix instructions, or pipeline terminates with a structured failure report.
State: typed Pydantic object growing through pipeline. Each agent is a FastAPI service in Docker; orchestrator calls via HTTP.
Redis caches LLM responses keyed by input hash + model version (cost control + reproducibility).

## Data Rules (CRITICAL)
NEVER import from, read, or reference /data/holdout/ for any purpose.
Holdout (2022-2024) is untouched until walk-forward evaluation in weeks 13-14.
Use /data/development/ (2002-2021) for all development, calibration, and evaluation.
extension_1 BAA-AAA spread median must be computed on /data/development/ only; committed to /docs/extension_1_config.yaml before holdout opens.

## Domain Facts Code Cannot Infer
- Return = holding-period return (HPR); excess return = HPR minus duration-matched Treasury yield (not risk-free rate)
- IPCA uses K=5 latent factors for KPP baseline
- PyBondLab's clean_trace() applies WRDS-MMN corrections — always use, never raw TRACE prices
- Compustat quarterly filings ~75 business days delay; annual ~150 business days (verify against WRDS documentation before interim)
- TRACE price errors inflate apparent Sharpe by ~30-50% on average (verify Dickerson et al. page citation before interim)
- Transaction costs: 40 bps round-trip IG, 80 bps HY (citation in /docs/domain_facts.md)
- All numerical thresholds live in /docs/thresholds.yaml — never hard-code in agent code

## Testing Requirements
Every agent:
1. Unit tests in /tests/unit/
2. Auditor: synthetic bias injection tests verifying 100% recall on every injected bias type
3. Library modules (/agents/quant/library/): unit tests on synthetic data with known analytical answer + regression test against paper results within /docs/thresholds.yaml tolerances
4. Integration test: PDF → StrategySpec for KPP, verified field-by-field against gold-standard
