# Locator Backfill Report — anchor gold specs

_Generated 2026-07-11 by running every `quote:` field through the project's `locate_quote`
(`agents/librarian/config/locate.py`) at ladder level **L1** (v2 ladder). 59/59 quotes located;
zero cross-page fallbacks. Offsets are into `normalise(page, L1)` (store-L0 / normalise-on-read),
pages 1-based._

**Locator authority:** BBW offsets index the **frozen** canonical text
(`evaluation/canonical_texts/bbw_2019.frozen.yaml`) — these are real D6 locators. DRR-2026 and
JNPS-2013 offsets index a **pending-freeze** parse (PyMuPDF 1.28.0, frozen recipe, source hashes in
each spec header) — re-run this backfill when those texts are frozen; offsets are then binding.

| spec | page | char_start | char_end | quote (first 70 chars) |
|---|---|---|---|---|
| drf_bbw_2019 | 14 | 5459 | 5638 | we form bivariate portfolios by independently sorting bonds into five  |
| drf_bbw_2019 | 14 | 5350 | 5487 | To con- struct the downside risk factor for corporate bonds, for each  |
| drf_bbw_2019 | 6 | 806 | 1038 | Downside risk is the 5% VaR of corporate bond return, defined as the s |
| drf_bbw_2019 | 14 | 5245 | 5349 | We construct the bond factors in a similar vein to Fama and French (20 |
| drf_bbw_2019 | 14 | 5101 | 5243 | it is natural to use credit risk (proxied by credit rating) as the fir |
| drf_bbw_2019 | 14 | 5491 | 5567 | independently sorting bonds into five quintiles based on their credit  |
| drf_bbw_2019 | 14 | 5572 | 5638 | five quin- tiles based on their downside risk (measured by 5% VaR) |
| drf_bbw_2019 | 14 | 5640 | 5813 | The downside risk factor, DRF, is the value-weighted av- erage return  |
| drf_bbw_2019 | 14 | 5640 | 5813 | The downside risk factor, DRF, is the value-weighted av- erage return  |
| drf_bbw_2019 | 9 | 549 | 619 | The portfolios are value weighted using amount outstanding as weights. |
| drf_bbw_2019 | 14 | 5411 | 5518 | for each month from July 2004 to December 2016, we form bivariate port |
| drf_bbw_2019 | 14 | 5674 | 5783 | the value-weighted av- erage return difference between the highest-VaR |
| drf_bbw_2019 | 5 | 4555 | 4701 | We denote R i, t as bond i 's excess return, R i,t = r i,t −r f,t, whe |
| drf_bbw_2019 | 15 | 1801 | 1891 | Model 3 is the ten-factor model that combines the five stock and five  |
| drf_bbw_2019 | 15 | 1892 | 1956 | DRF and CRF cov- ers the period from July 2004 to December 2016. |
| drf_bbw_2019 | 15 | 1892 | 1956 | DRF and CRF cov- ers the period from July 2004 to December 2016. |
| drf_bbw_2019 | 5 | 1566 | 1823 | Remove bonds that are not listed or traded in the US public market, wh |
| drf_bbw_2019 | 15 | 2931 | 3071 | The value-weighted DRF factor has an economically and statistically si |
| mom6_jnps_2013 | 9 | 144 | 287 | each month t, bonds are sorted into decile portfolios, P1 to P10, base |
| mom6_jnps_2013 | 1 | 315 | 413 | This paper documents significant momentum in a comprehensive sample of |
| mom6_jnps_2013 | 9 | 210 | 287 | based on their cumulative returns over months t −6 to t −1 (formation  |
| mom6_jnps_2013 | 9 | 144 | 287 | each month t, bonds are sorted into decile portfolios, P1 to P10, base |
| mom6_jnps_2013 | 9 | 158 | 208 | bonds are sorted into decile portfolios, P1 to P10 |
| mom6_jnps_2013 | 9 | 289 | 380 | The momentum strategy is long the winner portfolio, P10, and short the |
| mom6_jnps_2013 | 9 | 627 | 697 | Portfolio returns are equally weighted across their constituent bonds. |
| mom6_jnps_2013 | 9 | 381 | 447 | The portfolios are held over months t +1 to t +6 (holding period). |
| mom6_jnps_2013 | 9 | 144 | 208 | each month t, bonds are sorted into decile portfolios, P1 to P10 |
| mom6_jnps_2013 | 9 | 289 | 380 | The momentum strategy is long the winner portfolio, P10, and short the |
| mom6_jnps_2013 | 9 | 448 | 626 | Following the equity momentum literature, we skip one month between th |
| mom6_jnps_2013 | 9 | 698 | 882 | The overall momentum strategy month-t return is the equally weighted a |
| mom6_jnps_2013 | 9 | 698 | 882 | The overall momentum strategy month-t return is the equally weighted a |
| mom6_jnps_2013 | 9 | 746 | 881 | the equally weighted average month-t return of strategies implemented  |
| mom6_jnps_2013 | 12 | 1218 | 1332 | the momentum portfolio excess return over the risk-free rate or the mo |
| mom6_jnps_2013 | 12 | 1362 | 1444 | The coefficients are estimated using OLS with Newey-West-adjusted stan |
| mom6_jnps_2013 | 7 | 2474 | 2673 | To ensure that the results are not driven by outliers, we have elimina |
| mom6_jnps_2013 | 2 | 1008 | 1139 | bond-month observations on 81,491 U.S. corporate bonds (8,159 per mont |
| mom6_jnps_2013 | 2 | 1008 | 1139 | bond-month observations on 81,491 U.S. corporate bonds (8,159 per mont |
| mom6_jnps_2013 | 5 | 2365 | 2563 | We eliminate preferred shares, non-U.S. dollar denominated bonds, bond |
| mom6_jnps_2013 | 2 | 1159 | 1290 | past six- month bond winners outperform losers by 37 basis points (bps |
| str_drr_2026 | 17 | 297 | 461 | we sort bonds into deciles each month and form value-weighted portfoli |
| str_drr_2026 | 3 | 94 | 210 | Corporate bond factor research faces a replication crisis. We construc |
| str_drr_2026 | 10 | 1485 | 1595 | For example, cs denotes credit spread, str denotes short-term reversal |
| str_drr_2026 | 17 | 786 | 933 | In Panel A (single-sort), the unad- justed factor earns −0.99% per mon |
| str_drr_2026 | 17 | 297 | 334 | we sort bonds into deciles each month |
| str_drr_2026 | 37 | 295 | 368 | Panel A sorts bonds into deciles each month, going long P10 and short  |
| str_drr_2026 | 17 | 339 | 404 | form value-weighted portfolios (using bond market capitalization) |
| str_drr_2026 | 37 | 178 | 294 | Portfolios are value-weighted (using bond market capitalization) with  |
| str_drr_2026 | 17 | 297 | 334 | we sort bonds into deciles each month |
| str_drr_2026 | 37 | 295 | 368 | Panel A sorts bonds into deciles each month, going long P10 and short  |
| str_drr_2026 | 15 | 420 | 541 | Portfolio weights ωi,t are computed from signals observed at P end i,t |
| str_drr_2026 | 15 | 420 | 541 | Portfolio weights ωi,t are computed from signals observed at P end i,t |
| str_drr_2026 | 9 | 692 | 825 | A valid return requires such a price in both months t and t + 1 (bonds |
| str_drr_2026 | 37 | 830 | 923 | t-statistics (Newey-West, lags = ⌊T 0.25⌋) in parentheses. Sample: 200 |
| str_drr_2026 | 9 | 826 | 888 | Excess returns subtract the Fama-French one-month T-bill rate. |
| str_drr_2026 | 9 | 994 | 1089 | We use a single-factor bond CAPM (CAPMB) comprising MKTB to compute al |
| str_drr_2026 | 37 | 830 | 923 | t-statistics (Newey-West, lags = ⌊T 0.25⌋) in parentheses. Sample: 200 |
| str_drr_2026 | 37 | 830 | 923 | t-statistics (Newey-West, lags = ⌊T 0.25⌋) in parentheses. Sample: 200 |
| str_drr_2026 | 8 | 512 | 774 | We retain USD-denominated, fixed-rate, non-convertible, non-asset-back |
| str_drr_2026 | 17 | 786 | 933 | In Panel A (single-sort), the unad- justed factor earns −0.99% per mon |
