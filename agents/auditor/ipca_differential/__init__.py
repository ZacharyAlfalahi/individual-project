"""
The Frozen-Loadings IPCA Differential — an exploratory stretch of the Auditor.

Spec: docs/auditor/evaluation_frozen_ipca_differential_v5_FINAL.md (v5).

Per bias, an exact 2x2 factorial over {panel state, fitted-state provenance}: freeze
the complete IPCA fitted state once per provenance, run both panel arms under it (the
clean data channel), and compare against re-estimation (the total effect). Their
difference-in-differences is the interaction bracket I, identified exactly for the two
realised fitted states.

This package is deterministic. The IPCA module `agents/quant/library/ipca.py` is CALLED,
never modified. The only LLM appears (later) in a downstream explainer over typed output.

This build: the frozen-state contract + the four-cell evaluation engine
+ the build gates (engineering identity, known-truth, entry-boundary). Bootstrap inference,
the randomisation-FPR null, the stability diagnostic, and the 15-pair real-data execution
are implemented separately (see the spec's execution checklist §12).
"""
