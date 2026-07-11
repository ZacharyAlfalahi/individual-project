# Anchor Gold Specs

Hand-authored answer keys for the three anchor strategies, per the authoring template
(`fields-spec-anchor.md`, Project root). Forward input for G2 (round trip) and answer key for G3
(calibration). Authored 2026-07-11 from the papers only (rule library off, zero INFERRED); every
STATED value carries a verbatim quote located at L1 by the project's own `locate_quote`
(59/59 — see `locator_backfill_report.md`).

| file | anchor | paper | status |
|---|---|---|---|
| `gold_drf_bbw_2019.md` | DRF | BBW (2019), JFE | DRAFT — ◇ fields pre-filled; **re-stamp when schema v1.1 lands** (v1.1 committed: NO as of 2026-07-11) |
| `gold_mom6_jnps_2013.md` | mom6 | Jostova et al. (2013), RFS | AUTHORED under v1 — ◇ trivial; re-stamp on v1.1 is mechanical |
| `gold_str_drr_2026.md` | str | Dickerson, Robotti & Rossetti (2026) | DRAFT — extraction complete; **BLOCKED-ON-RENAME** (reversal column) for instantiation |

## Canonical-text status

Only BBW has a frozen canonical text; its locators are binding. DRR-2026 and JNPS-2013 were parsed
with the frozen recipe (PyMuPDF 1.28.0 / L1 / v2 ladder; source sha256 in each header) but are
**PENDING-FREEZE** — freeze them and re-run the backfill before G2/G3 consume those two golds.

## Paper ≠ engine divergences surfaced by the rulebook cross-check (paper wins in the gold)

1. **drf weighting base:** BBW weights by *amount outstanding*; engine uses `offering_amt` par.
2. **str weighting base:** DRR states *bond market capitalization* (market value); engine par.
3. **str leg sign:** paper prints long-P10 (highest prior return) with premium −0.99; the engine
   spec's §5.1 claims losers−winners ≈ −0.99 and cites the downloadable series. Both cannot hold —
   **re-verify the engine sign note against the stored DRR series before locking targets.**
4. **mom6 trim operation:** JNPS say "eliminated" (truncate) at the full-sample 99.5th percentile;
   engine adjudicated a winsorise/clip repair from DRR's FilterClass. The gold records the paper;
   the clip is a downstream DESIGN call.

## Outstanding (not in scope of this pass)

- D20 enumeration golds (recipe-list per paper) — separate artifact, still to author.
- Freeze DRR-2026 + JNPS-2013 canonical texts; re-run backfill.
- Re-stamp all three on schema v1.1 landing (expected diff: ◇ stamps only).
