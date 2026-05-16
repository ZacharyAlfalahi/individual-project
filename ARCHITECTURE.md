# Architecture
Six-agent LLM pipeline for corporate bond factor replication, bias detection, and repair. Two-layer corpus: anchor (KPP + BBW, n=2, fully manual) + scale (10–20 papers, pipeline + paper-statistics verification).
Every implementation is validated and verifiable.

## Research Questions
RQ1 — Librarian extraction fidelity on anchor layer (n=2): per-field accuracy, multi-model agreement rate, failure taxonomy.
RQ2 — Bias prevalence on scale corpus with anchor-layer calibration: fraction of strategies failing each of 5 bias checks.
RQ3 — Repair rate: fraction of Scientist proposals that repair failing strategies, retain in-sample alpha, survive BH-FDR, show positive OOS Sharpe on 2022–2024 holdout.
Each RQ has a structurally independent validation path — failure in one component cannot contaminate another.

## Data Rules (inviolable)
- NEVER read or import from /data/holdout/ until walk-forward evaluation (weeks 13–14)
- ALL development, calibration, and evaluation uses /data/development/ (2002–2021)
- BAA-AAA spread median for regime conditioning must be computed on /data/development/ only; commit to /docs/extension_1_config.yaml before holdout opens

## Repository Key Paths
/agents/quant/library/     — ipca.py, four_factor_sort.py, dnn_residual.py (hand-implemented; LLM configures, NEVER modifies)
/agents/auditor/checks/    — deterministic only; zero LLM calls permitted here
/schema/                   — Changes to it are deliberate, reviewed migrations
/docs/thresholds.yaml      — ALL numerical thresholds; never hard-code values in agent code
/docs/inference_rules.md   — enumerated rules permitting INFERRED provenance
/data/holdout/             — READ NEVER during development

## Agent Hard Constraints

**Librarian**: dual LLM extraction (stack TBD — see Open Decisions). A field is STATED only if both models agree on value AND verbatim quote. No self-reported confidence scores. UNKNOWN is a valid value, not an error.

**Quant**: configures library modules; NEVER authors algorithmic code. Modifications to /agents/quant/library/ require manual review + all regression tests passing. ReAct loop hard cap: 8 iterations.

**Auditor**: /agents/auditor/checks/ is deterministic — zero LLM calls. LLM appears only in explainer.py after the verdict is produced. Refuses to opine when required fields are UNKNOWN or INFERRED without a matching rule. Silent iteration until pass is forbidden — it is p-hacking.

**Scientist**: input = Auditor-failing strategy. Generates 4–8 repair proposals per the procedure committed to thresholds.yaml before first run. BH-FDR applied jointly across all proposals in-sample. Only FDR-survivors go to holdout. Failure taxonomy: Type 1 (wrong mechanism targeted), Type 2 (repair introduces new bias), Type 3 (repair valid but alpha was never real).

**Reporter**: NEVER regenerates numbers from prose. Every numeric token in the output is asserted against typed pipeline output by verifier.py before commit.

## StrategySpec Schema
Every fact-bearing field: Fact[T] with value, provenance (STATED|INFERRED|UNKNOWN), and quote (required when STATED). INFERRED requires a rule ID from /docs/inference_rules.md. Schema is designed at end of week 3 after KPP and BBW replications are complete — not before.

## Replication Success Criterion
Primary: Sharpe AND factor loadings AND per-quintile spreads all within 15% tolerance (thresholds.yaml). Coincidental Sharpe match alone is NOT success. Secondary diagnostic: bootstrap CI overlap reported as an additional column alongside the primary criterion.

## Testing Requirements
- All agents: unit tests in /tests/unit/
- Auditor: synthetic bias injection tests verifying 100% recall for every injected bias type
- Library modules: unit test on synthetic data with known analytical answer + regression test against paper headline metric

## Open Decisions (resolve before Librarian v1 runs on any corpus paper)
D4 — LLM stack: commit chosen stack to thresholds.yaml before first Librarian run. Candidate: Claude Sonnet 4.6 + GPT-4o for anchor-layer dual extraction; cost-optimised single-model for scale corpus.
D5 — Dickerson 2026 label availability: verify by end of week 1 (arXiv 2604.07880). If available: Auditor check 5 uses exact name matching against /data/dickerson_zoo/names.csv. If unavailable: check 5 uses t-stat band + free-parameter count only; record outcome in /docs/citations_verified.md.
