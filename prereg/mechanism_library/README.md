# Scientist mechanism library

Manually authored mechanism entries (spec §6), **authored backwards from
`prereg/reachability_census.md`** so every entry is executable by construction. No LLM ever adds
or modifies an entry. Each entry's citation is verified inline (the second source-fidelity
surface, see `docs/data/registers/citations_verified.md §1b`). Sources: KPP, Duraj–Giesecke
("Deep Learning for Corporate Bonds"), Giglio–Kelly–Xiu (a **survey**), BBW, DRR, Jostova, BPW —
PDFs in `papers/pdf/` (untracked).

## Schema (extends the lean spec §6 — the KPP pilot surfaced what "VERIFIED" actually needs)

```yaml
mechanism_id: mech_001
version: 1.0.0
title: <short>
claim: >
  <1-2 sentence economic claim>
sources:
  - source_id: KPP_2023
    kind: primary | survey
    quote: "<verbatim string that locates in the source>"
    verification:                       # makes VERIFIED REPRODUCIBLE (a test re-locates it)
      extractor: frozen_canonical | pymupdf
      text: <path to the frozen canonical text OR the pdf>
      locate_level: L0 | L1 | L2        # ladder level at which the quote locates
      page: <page index into that extraction>
      cite: "<human citation: authors, venue, page, section>"
    chains_to:                          # REQUIRED iff kind == survey
      source_id: <primary the survey cites>
      cite: "<human citation of the primary>"
    status: VERIFIED                    # never set without a re-locating quote
applicability:
  strategy_families: [CHARACTERISTIC_SORT, IPCA]
  required_inputs:
    - {variable_family: <see variable_families.yaml>, timing_rule: observed_before_formation}
allowed_templates: [<template_ids from prereg/templates/>]
forbidden_uses: [<forbidden conditions, enforced at G0/G2>]
```

`variable_families.yaml` maps each `variable_family` to the concrete conditioning variables that
satisfy it (only census-reachable variables appear there).

## What the KPP pilot (n=3) surfaced — schema deltas from the lean §6

1. **`VERIFIED` needs a quote + a `verification` block, not just `{source_id, location, status}`.**
   A bare location cannot be re-checked; the pilot's test re-locates every quote, so each source
   carries the verbatim `quote` plus `extractor`/`text`/`locate_level`/`page`.
2. **`locate_level` is per-source.** KPP quotes locate against the frozen canonical text at **L1**;
   the GKX survey quote locates against a raw PyMuPDF extraction at **L2** (the L1 ladder does not
   fix the "models.It" no-space artifact). One global level would fail.
3. **`extractor` is needed.** Frozen canonical texts exist only for the Librarian's three anchor
   papers (`evaluation/canonical_texts/`); every other source is read straight from the PDF via
   PyMuPDF. The verifier dispatches on `extractor`.
4. **Survey sources need `kind: survey` + `chains_to`.** GKX is a survey; its quote chains to the
   primary it cites (Kelly, Pruitt & Su 2019). The verifier enforces the chain.

An entry may also carry a `caveats:` list — source-fidelity limitations recorded verbatim (e.g.
the BPW 2003–2009 crisis-dominated sample). Consumed by the context builder (surfaced to the
researcher) and the write-up.

## Entries (9 distinct mechanisms)

| id | mechanism | source (chain) | conditioning | templates |
|---|---|---|---|---|
| mech_001 | Characteristic-instrumented conditional betas | KPP 2023 | bond characteristic | T4 |
| mech_002 | Time-varying (conditional) risk exposures | KPP 2023 (+ GKX survey → KPS 2019) | macro regime | T1, T2 |
| mech_003 | Conditional recovery of the credit risk premium | KPP 2023 | credit regime | T1, T3 |
| mech_004 | Macro-conditional time-varying risk premia | GKX survey → Gagliardini–Ossola–Scaillet 2016 | macro (term spread) | T1, T2 |
| mech_005 | Macro state forecasts downturns / low-return regimes | DG 2025 | macro regime | T1, T2 |
| mech_006 | Discrete volatility regimes with sign-flipping returns | DG 2025 | volatility regime | T1, T2 |
| mech_007 | Nonlinear characteristic interactions | DG 2025 | bond characteristic | T4 |
| mech_008 | Within-rating illiquidity premium (robust off-crisis) | BPW 2011 | gamma_illiq | T3, T4 |
| mech_009 | Rating-conditional illiquidity importance (degree crisis-inflated) | BPW 2011 | rating segment | T3, T4 |

Verified by `tests/unit/test_scientist_mechanism_library.py` (quotes re-locate; survey chains;
reachable per census) and exercised by `tests/unit/test_scientist_researcher.py` (eligibility +
wall). The **≥ 8 gate is on conceptually DISTINCT mechanisms, not entry count** — met at **9**.

## Honest yield (spec §6 target 15–25; NOT padded)

The honest yield across the mechanism sources (KPP, GKX, DG, BPW) is **9 distinct mechanisms** —
above the ≥ 8 eligibility gate but below §6's 15–25 target. Per the build directive, entries are
**not manufactured** to reach 15; the number is reported and any amendment to §6's target is
**a project decision**, logged in `docs/data/registers/scope_changes.md` (SC-SCI-6). BBW 2019 is
excluded as a mechanism source (retracted, point 11); DRR/Jostova are bias/anchor sources, not
extension-mechanism sources; the GKX maturity mechanism was dropped as redundant (it chains back
to KPP, so it is not conceptually distinct from the KPP cluster).
