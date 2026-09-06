# Locator Backfill Report — gold specs

_Re-generated 2026-07-12 by running every `quote:` field through the project's `locate_quote`
(`agents/librarian/config/locate.py`) at ladder level **L1** (v2 ladder). 59/59 quotes located
(the original three sort golds — `str`/`drf`/`mom6` — see the 2026-08-19 consolidated totals below
for all six golds); zero cross-page fallbacks. Offsets are into `normalise(page, L1)` (store-L0 /
normalise-on-read), pages 1-based._

**Locator authority:** BBW, DRR-2026, and JNPS-2013 offsets index their **frozen** canonical texts
(`bbw_2019.frozen.yaml`, `drr_2026.frozen.yaml`, `jnps_2013.frozen.yaml`) — real, binding D6 locators.
JNPS-2013 was frozen 2026-07-15 (`scripts/freeze_canonical_text.py`); the freeze reproduced the
pending parse, so its offsets below were unchanged and re-verified binding
(`scripts/regenerate_locator_backfill.py mom6`).

## Consolidated totals — all six golds (2026-08-19)

The table below spans **all six golds** — the original three sort golds plus `crf`, `lrf`, and `kpp`,
folded in as they were built. `crf`/`lrf` offsets index `bbw_2019.frozen.yaml` (shared with `drf`);
`kpp` offsets index `kpp_2023.frozen.yaml` — all real, binding D6 locators.

Re-verified **read-only** on 2026-08-19 by running every `quote:` field through `locate_quote` at L1
across all six golds (`scripts/regenerate_locator_backfill.py` `verify` / `verify_from_markdown`,
tallied — the committed rows below were **not** altered): **every quote located, 0 not-found,
0 cross-page fallbacks, 0 offset-drift.**

| gold | role | rows | distinct spans |
|---|---|---|---|
| str  | sort                         | 20  | 14  |
| drf  | sort                         | 18  | 16  |
| mom6 | sort                         | 21  | 17  |
| crf  | sort (4th)                   | 17  | 17  |
| **sort subtotal** |             | **76**  | **64**  |
| lrf  | negative control / P1 oracle | 15  | 15  |
| kpp  | IPCA methodology exemplar    | 42  | 42  |
| **all golds** |                 | **133** | **121** |

Denominator note: **rows** counts proposed quote instances — the original three sort golds retain
duplicate rows from the pre-dedup 2026-07-12 generation, and their 59 rows are the historical
"59/59"; **distinct spans** dedups by page + char-span and is the figure to cite. `crf`/`lrf`/`kpp`
rows were already deduped at write time, so their two columns coincide. This section is additive —
no data row below was changed.

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
| kpp_2023 | 2 | 142 | 210 | a new conditional factor model for individual corporate bond returns |
| kpp_2023 | 2 | 177 | 210 | individual corporate bond returns |
| kpp_2023 | 2 | 211 | 262 | based on instrumented principal components analysis |
| kpp_2023 | 6 | 2868 | 2921 | a fast-converging alternating least squares algorithm |
| kpp_2023 | 23 | 785 | 809 | models with K = 1, . . . , 5 |
| kpp_2023 | 3 | 1204 | 1271 | Our main five-factor IPCA model explains 51% of the panel variation |
| kpp_2023 | 16 | 186 | 233 | the IPCA model restricted to have no intercepts |
| kpp_2023 | 38 | 1979 | 2025 | returns divided by Duration times Spread (DtS) |
| kpp_2023 | 13 | 1375 | 1419 | we cross-sectionally rank, demean, and scale |
| kpp_2023 | 6 | 2037 | 2070 | characteristic-managed portfolios |
| kpp_2023 | 17 | 299 | 382 | The first out-of-sample test observation is 36 months after the start  |
| kpp_2023 | 38 | 178 | 290 | We use a wild-bootstrap following KPS using a t-distribution with seve |
| kpp_2023 | 36 | 1618 | 1647 | Bond age is measured in years |
| kpp_2023 | 36 | 1649 | 1705 | Coupon, face value, and are attributes of the bond issue |
| kpp_2023 | 36 | 1839 | 1879 | Book-to-price is the sum of shareholders |
| kpp_2023 | 36 | 1972 | 2002 | Debt-to-EBITDA uses total debt |
| kpp_2023 | 36 | 2015 | 2071 | is the derivative of the bond value to the credit spread |
| kpp_2023 | 36 | 2164 | 2206 | Mom. 6m equity is 6-2 momentum of the firm |
| kpp_2023 | 13 | 649 | 666 | Earnings-to-price |
| kpp_2023 | 13 | 701 | 719 | Equity market cap. |
| kpp_2023 | 13 | 748 | 765 | Equity volatility |
| kpp_2023 | 13 | 800 | 815 | Firm total debt |
| kpp_2023 | 36 | 2224 | 2263 | mom. 6m is 6-2 momentum of bond returns |
| kpp_2023 | 36 | 2265 | 2331 | mom. 6m industry is industry-adjusted 6-2 momentum of bond returns |
| kpp_2023 | 36 | 2333 | 2374 | mom. 6m × ratings is bond return momentum |
| kpp_2023 | 36 | 2493 | 2518 | Book leverage shareholder |
| kpp_2023 | 36 | 2659 | 2677 | Market leverage is |
| kpp_2023 | 37 | 1676 | 1758 | Turnover volatility is the quarterly standard deviation of sales divid |
| kpp_2023 | 36 | 1707 | 1755 | Spread is the option-adjusted spread of the bond |
| kpp_2023 | 37 | 1760 | 1819 | Operating leverage is sales minus EBITDA, divided by EBITDA |
| kpp_2023 | 37 | 1821 | 1857 | Profitability is sales minus cost-of |
| kpp_2023 | 37 | 2034 | 2095 | Profitability change is the five-year change in profitability |
| kpp_2023 | 36 | 1757 | 1837 | Rating is a number from 1 to 22, where 1 is S&P AAA and 22 is already  |
| kpp_2023 | 37 | 2118 | 2146 | is defined by Shumway (2001) |
| kpp_2023 | 37 | 2148 | 2176 | Bond skewness is bond return |
| kpp_2023 | 36 | 2403 | 2451 | mom. 6m log(Spread) is the log of the spread six |
| kpp_2023 | 37 | 2213 | 2315 | Spread-to-D2D is the option-adjusted spread, divided by one minus the  |
| kpp_2023 | 37 | 2352 | 2410 | Bond volatility is bond return volatility over the past 24 |
| kpp_2023 | 37 | 2419 | 2488 | Value-at-risk is the second lowest credit excess return over the past |
| kpp_2023 | 37 | 2529 | 2590 | VIX beta is the sum of coefficients on current and lagged VIX |
| kpp_2023 | 16 | 235 | 283 | The sample is January 1999 through December 2020 |
| kpp_2023 | 12 | 2319 | 2379 | We discard bond-month observations with extreme bond spreads |
| kpp_2023 | 12 | 2319 | 2456 | We discard bond-month observations with extreme bond spreads (less tha |
| crf_bbw_2019 | 3 | 1969 | 2137 | These independent sorts also produce three credit risk factors so that |
| crf_bbw_2019 | 15 | 4844 | 4889 | the newly proposed factors of corporate bonds |
| crf_bbw_2019 | 14 | 5245 | 5349 | We construct the bond factors in a similar vein to Fama and French (20 |
| crf_bbw_2019 | 14 | 6685 | 6822 | the average monthly excess returns for the 5 × 5 portfolios independen |
| crf_bbw_2019 | 14 | 5814 | 6049 | The credit risk factor, CRF VaR, is the value-weighted aver- age retur |
| crf_bbw_2019 | 15 | 1346 | 1441 | Credit risk factor ( CRF) is the average of the CRF obtained from form |
| crf_bbw_2019 | 9 | 549 | 619 | The portfolios are value weighted using amount outstanding as weights. |
| crf_bbw_2019 | 14 | 5411 | 5518 | for each month from July 2004 to December 2016, we form bivariate port |
| crf_bbw_2019 | 5 | 4555 | 4701 | We denote R i, t as bond i 's excess return, R i,t = r i,t −r f,t, whe |
| crf_bbw_2019 | 15 | 1801 | 1891 | Model 3 is the ten-factor model that combines the five stock and five  |
| crf_bbw_2019 | 15 | 1892 | 1956 | DRF and CRF cov- ers the period from July 2004 to December 2016. |
| crf_bbw_2019 | 5 | 1566 | 1823 | Remove bonds that are not listed or traded in the US public market, wh |
| crf_bbw_2019 | 15 | 3618 | 3676 | significant premiums of 0.43% per month ( t -stat. = 2.78) |
| crf_bbw_2019 | 14 | 5101 | 5243 | it is natural to use credit risk (proxied by credit rating) as the fir |
| crf_bbw_2019 | 6 | 806 | 1038 | Downside risk is the 5% VaR of corporate bond return, defined as the s |
| crf_bbw_2019 | 8 | 696 | 848 | we follow Bao, Pan, and Wang (2011) to construct bond- level illiquidi |
| crf_bbw_2019 | 15 | 1034 | 1196 | Return re- versal factor ( REV) is constructed by independently sortin |
| lrf_bbw_2019 | 15 | 721 | 1033 | Liquidity risk factor ( LRF ) is con- structed by independently sortin |
| lrf_bbw_2019 | 15 | 4844 | 4889 | the newly proposed factors of corporate bonds |
| lrf_bbw_2019 | 14 | 5245 | 5349 | We construct the bond factors in a similar vein to Fama and French (20 |
| lrf_bbw_2019 | 15 | 813 | 867 | 5 × 5 quintiles based on illiquidity and credit rating |
| lrf_bbw_2019 | 14 | 6152 | 6334 | The liquid- ity risk factor, LRF , is the value-weighted average retur |
| lrf_bbw_2019 | 9 | 549 | 619 | The portfolios are value weighted using amount outstanding as weights. |
| lrf_bbw_2019 | 14 | 5411 | 5518 | for each month from July 2004 to December 2016, we form bivariate port |
| lrf_bbw_2019 | 14 | 6189 | 6304 | the value-weighted average return difference between the highest-illiq |
| lrf_bbw_2019 | 5 | 4555 | 4701 | We denote R i, t as bond i 's excess return, R i,t = r i,t −r f,t, whe |
| lrf_bbw_2019 | 15 | 1801 | 1891 | Model 3 is the ten-factor model that combines the five stock and five  |
| lrf_bbw_2019 | 15 | 1957 | 2020 | LRF and REV cover the period from August 2002 to December 2016. |
| lrf_bbw_2019 | 5 | 1566 | 1823 | Remove bonds that are not listed or traded in the US public market, wh |
| lrf_bbw_2019 | 15 | 2188 | 2225 | Liquidity risk factor (LRF) 0.52 5.02 |
| lrf_bbw_2019 | 8 | 696 | 795 | we follow Bao, Pan, and Wang (2011) to construct bond- level illiquidi |
| lrf_bbw_2019 | 14 | 5101 | 5243 | it is natural to use credit risk (proxied by credit rating) as the ﬁrs |
