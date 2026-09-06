# Anchor Gold Specs

Hand-authored answer keys for the sort anchor strategies (**four settled RQ1/RQ2 golds: str, drf, mom6, crf**;
a fifth, `lrf`, is a proposed amendment **unresolved**, not settled — see below), per the authoring template
(`fields-spec-anchor.md`, Project root). Forward input for G2 (round trip) and answer key for G3
(calibration). Authored 2026-07-11 from the papers only (rule library off, zero INFERRED); every
STATED value carries a verbatim quote located at L1 by the project's own `locate_quote`
(59/59 — see `locator_backfill_report.md`).

**Correcting a gold?** See the consolidated `docs/evaluation/gold_errata_protocol.md` (append-only + re-stamp, 57/57-locator re-validation, stop-and-report on any diff beyond the errata; extraction errors route to review, never to a gold edit — D27).

| file | anchor | paper | status |
|---|---|---|---|
| `gold_drf_bbw_2019.md` | DRF | BBW (2019), JFE | **INSTANTIATED 2026-07-15 (schema v1.1)** — BBW text frozen; every STATED quote an exact L1 substring with a binding locator; G2 byte-equality green (par-proxy weighting carried as a standing register row) |
| `gold_mom6_jnps_2013.md` | mom6 | Jostova et al. (2013), RFS | **INSTANTIATED 2026-07-15 (schema v1.1)** — JNPS canonical text frozen (`jnps_2013.frozen.yaml`); all STATED locators re-verified binding; `expost_trim=truncate` delegated to the `lab_trim` toggle via `lab_trim_delegation_v1`; G2 byte-equality green |
| `gold_str_drr_2026.md` | str | Dickerson, Robotti & Rossetti (2026) | **INSTANTIATED 2026-07-12 (schema v1.1)** — concept grounded (`prior_1m_excess_return`→`xret`, table v2), DRR text frozen, quotes re-verified against the frozen text |
| `gold_crf_bbw_2019.md` | CRF | BBW (2019), JFE | **INSTANTIATED 2026-08-04 (schema v1.1)** — the first **multi-leg** sort gold (composite of independent rating-signal sorts); G2 composite byte-equality green (`expected_composite_rulebook("crf")`). **RQ1+RQ2 only — RQ3-excluded** (D15). Enters the CRF-inclusive extraction denominator (`ANCHOR_SET_WITH_CRF`). |
| `gold_lrf_bbw_2019.md` | LRF | BBW (2019), JFE | **unresolved — artefact present, role NOT yet ratified as a gold.** Verified: a StrategySpec gold file + a 5th P1 oracle (`bbw_factors.parquet → lrf_corr`), 2026-08-15. The **authoritative Part 4 role is the negative-control construction — NOT a fifth RQ1/RQ2 gold** (crf remains the fourth). Elevating it to a fifth gold is a *proposed amendment*: declare its RQ1/RQ2-only role and **freeze its denominator treatment before scoring it**. It is **not** in the scoring denominator today (absent from `round_trip.py`/`aggregation.PAPER_OF`) — retain the four-gold headline (str/drf/mom6/crf) beside any expanded five-gold denominator. Not external validity (same paper/data/family). |

**Sort** gold family — **four settled RQ1/RQ2 golds: str, drf, mom6, crf** (the authoritative headline;
crf is the fourth, D15 2026-08-04). **str/drf/mom6 are the RQ3 census** (`ANCHOR_SET`); **crf is
RQ1/RQ2-only, RQ3-excluded**, entering the CRF-inclusive extraction denominator (`ANCHOR_SET_WITH_CRF`).
A fifth artefact, **`lrf`** (2026-08-15), exists in the working tree but its role as a fifth RQ1/RQ2 gold
is **unresolved — a proposed amendment, not settled**: the authoritative Part 4 casts `lrf` as the
**negative-control construction**. Ratify only after declaring its RQ1/RQ2-only role and **freezing its
denominator treatment before scoring** — adding it changes the evaluation denominator after earlier
four-gold evidence exists, defensible only if the capability rationale and before/after reporting are
frozen first. The earlier "CRF considered and excluded" ruling (2026-07-11) was superseded by the
corpus BBW extraction (crf gold); it is kept here as history. ◇ control field is named `control_axis` per
the 2026-07-11 brief amendment (shipped code name kept; `control_signal` rejected).

| file | anchor | paper | class | status |
|---|---|---|---|---|
| `gold_kpp_ipca.md` | kpp | Kelly, Palhares & Pruitt (2023), JF | **fitted-factor-model (schema v1.2)** | **INSTANTIATED 2026-08-03** — the first non-sort gold: an `EstimationBlock` (11 fields) + a 29-instrument set (Table A.I), ~90 STATED quotes all L1-located (42 binding rows, zero cross-page/drift). Loaded by `kpp_gold_loader.py` (routed via `load_gold_spec("kpp")`). Graded as a **separate, non-pooled** RQ1 sub-metric — NOT part of the sort G3 set above (different field set; contract v1.3 §3.7 / D42). |

The fitted-model gold is a *separate construction class*, never pooled with the sort family. The RQ1
sort-grading denominator is `ANCHOR_SET` (str/drf/mom6) by default, or `ANCHOR_SET_WITH_CRF` when crf
is in the scored paper set (`evaluation/harness/aggregation.py`; lrf is not yet wired in). See
`docs/librarian/specs/schema_v1_2_estimation_block.md`
and `.../implementation-notes/kpp_rq1_fitted_model_record_2026-08-03.md`.

## Canonical-text status

All three anchor texts are frozen and their locators are binding: BBW (`bbw_2019.frozen.yaml`),
DRR-2026 (`drr_2026.frozen.yaml`, verified byte-identical to the authoring parse, quotes re-verified
2026-07-12), and **JNPS-2013 (`jnps_2013.frozen.yaml`, frozen 2026-07-15; PyMuPDF 1.28.0 / L1 / v2
ladder; source sha256 in the mom6 header)**. The 2026-07-15 JNPS freeze reproduced the earlier
pending parse byte-for-byte, so the mom6 offsets were unchanged and the backfill re-verified all
STATED locators as binding.

## Paper ≠ engine divergences surfaced by the rulebook cross-check (paper wins in the gold)

1. **drf weighting base:** BBW weights by *amount outstanding*; engine uses `offering_amt` par.
2. **str weighting base:** DRR states *bond market capitalization* (market value); engine par.
3. **str leg sign — RESOLVED 2026-07-12.** The flag was correct: `build_str.py`'s old losers−winners
   'Construction A' was the sign-mirror of the paper's stated long-P10 construction, reconciled to
   DRR's −0.99 by coincidence of sign only. The builder is now aligned to this gold (deciles,
   winners−losers, grounded `prior_1m_excess_return`→`xret`). Substantive residual — the gold-aligned
   construction earns ≈ +0.95%/mo momentum on the corr dev panel vs DRR's −0.99 raw reversal — is a
   raw/LIB result owned by the §8 bias-toggle decomposition, not a
   gold or leg issue. `BBW_anchor_implementation_spec.md` §5.1 was corrected to the winners−losers
   construction (2026-07-13); only the §12.1 str −0.696 row stays banner-flagged STALE
   pending its re-run/re-emit.
4. **mom6 trim operation:** JNPS say "eliminated" (truncate) at the full-sample 99.5th percentile;
   engine adjudicated a winsorise/clip repair from DRR's FilterClass. The gold records the paper;
   the clip is a downstream DESIGN call.

## Outstanding (not in scope of this pass)

- D20 enumeration golds (recipe-list per paper) — **authored for the three anchor papers**
  (`enum_bbw_2019.yaml`, `enum_drr_2026.yaml`, `enum_jnps_2013.yaml`; NO-MODEL-CONSULT). Scale-layer
  enumeration golds (RQ2 coverage) remain **outstanding**, gated on the O1 scale-layer scope decision
  (an open decision — see `docs/backlog/remaining_work.md`), and are authored by the human,
  never by an automated agent. The full reconciled inventory is `docs/evaluation/corpus_inventory.md`;
  the scale-layer selection rule (proposing O1's resolution, not yet decided) is
  `docs/evaluation/t2_selection_prereg.md`.

_All three anchor golds are now INSTANTIATED under schema v1.1 (str 2026-07-12; drf + mom6
2026-07-15 on the JNPS freeze). The former outstanding items — the JNPS canonical-text freeze +
mom6 backfill/instantiation, the drf instantiation/re-stamp pass, and the doc-side correction of
`BBW_anchor_implementation_spec.md` §5.1's losers−winners sign note — are all complete._
