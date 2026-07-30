"""The Scientist Researcher — the ONLY generative step (spec §8). This package is import-walled
from magnitude-bearing Auditor schemas (R2 / INVARIANT 1): no module here imports
agents.auditor.schemas.{audit_report,audit_core,decomposition} or
agents.auditor.checks.{economic,inference,bayes}. It consumes the already magnitude-free
ScientistCase (built by shared/handoff/) plus the mechanism library and template registry.

Build order (spec §8.1): the deterministic pieces first — library + eligibility filter +
context-builder wall — then the ProposalSource rungs (random -> retrieval -> llm), rung 3 last.
"""
