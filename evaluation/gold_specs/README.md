# Anchor Gold Specs

Hand-authored answer keys for the three anchor strategies, per the authoring template
(`fields-spec-anchor.md`, Project root). Forward input for G2 (round trip) and answer key for G3
(calibration). Authored 2026-07-11 from the papers only (rule library off, zero INFERRED); every
STATED value carries a verbatim quote located at L1 by the project's own `locate_quote`
(59/59 — see `locator_backfill_report.md`).

| file | anchor | paper | status |
|---|---|---|---|
| `gold_drf_bbw_2019.md` | DRF | BBW (2019), JFE | DRAFT — ◇ pre-filled; awaits its v1.1 instantiation/re-stamp pass (unblocked: BBW frozen; next in line per the template) |
| `gold_mom6_jnps_2013.md` | mom6 | Jostova et al. (2013), RFS | AUTHORED under v1 — ◇ trivial; instantiation blocked only on the JNPS canonical-text freeze |
| `gold_str_drr_2026.md` | str | Dickerson, Robotti & Rossetti (2026) | **INSTANTIATED 2026-07-12 (schema v1.1)** — concept grounded (`prior_1m_excess_return`→`xret`, table v2), DRR text frozen, quotes re-verified against the frozen text |

Gold set = str, drf, mom6 — **final** (project decision 2026-07-11; CRF considered and excluded — combiner
path deferred to the corpus BBW extraction). ◇ control field is named `control_axis` per the
2026-07-11 brief amendment (shipped code name kept; `control_signal` rejected).

## Canonical-text status

BBW and DRR-2026 are frozen (`bbw_2019.frozen.yaml`, `drr_2026.frozen.yaml`) — their locators are
binding (DRR's frozen pages verified byte-identical to the authoring parse, and its quotes
re-verified against the frozen text 2026-07-12). JNPS-2013 was parsed with the frozen recipe
(PyMuPDF 1.28.0 / L1 / v2 ladder; source sha256 in the mom6 header) but remains **PENDING-FREEZE** —
freeze it and re-run the backfill before G2/G3 consume the mom6 gold.

## Paper ≠ engine divergences surfaced by the rulebook cross-check (paper wins in the gold)

1. **drf weighting base:** BBW weights by *amount outstanding*; engine uses `offering_amt` par.
2. **str weighting base:** DRR states *bond market capitalization* (market value); engine par.
3. **str leg sign — RESOLVED 2026-07-12.** The flag was correct: `build_str.py`'s old losers−winners
   'Construction A' was the sign-mirror of the paper's stated long-P10 construction, reconciled to
   DRR's −0.99 by coincidence of sign only. The builder is now aligned to this gold (deciles,
   winners−losers, grounded `prior_1m_excess_return`→`xret`). Substantive residual — the gold-aligned
   construction earns ≈ +0.95%/mo momentum on the corr dev panel vs DRR's −0.99 raw reversal — is a
   raw/LIB result owned by the §8 bias-toggle decomposition, not a
   gold or leg issue. The stale losers−winners note in `BBW_anchor_implementation_spec.md` §5.1 still
   needs its doc-side correction.
4. **mom6 trim operation:** JNPS say "eliminated" (truncate) at the full-sample 99.5th percentile;
   engine adjudicated a winsorise/clip repair from DRR's FilterClass. The gold records the paper;
   the clip is a downstream DESIGN call.

## Outstanding (not in scope of this pass)

- D20 enumeration golds (recipe-list per paper) — separate artifact, still to author.
- Freeze the JNPS-2013 canonical text; re-run the mom6 backfill; then mom6's instantiation pass.
  NB the template note "Jostova PDF still to be sourced" is stale — the PDF is in
  `papers/pdf/Momentum in Corporate Bond Returns.pdf` (source sha256 in the mom6 header) and the
  gold was authored from it; only the freeze is outstanding.
- drf instantiation/re-stamp pass under v1.1 (unblocked; expected diff: ◇ stamps only).
- Doc-side correction of `BBW_anchor_implementation_spec.md` §5.1's stale losers−winners sign note
  (code already fixed; see divergence 3).
