# Computational specification

The study evaluates a fixed retrospective fourteen-ETF universe: GLD, IEF, IWM, QQQ, SPY, TLT, XLE, XLF, XLI, XLK, XLP, XLU, XLV and XLY. Primary decisions use previous-close holdings, prices and NAV. Twenty-session mean dollar volume and sample return volatility end at the previous session. Orders execute at the next adjusted open and portfolios are marked at the close.

Weekly equal weight allocates 1/14 to each asset. Daily momentum selects at most four positive prior-close twenty-session scores with equal 1/k weights; ties use ticker order. Capital scenarios start at $10M, $100M and $1B. E1 and E2b contain separately identified diagnostic settings. The registry contains 62 economic scenarios plus one unexecuted-by-default pilot specification. Contrasts comprise 66 comparisons, 12 primary.

Fills are limited by lagged dollar ADV where enabled. Holdings-bounded sales fund buys at the same opening-price event; proportional buy scaling recalculates costs until cash-feasible. Final fill quantities are open-dependent despite prior-close desired gaps. This is a modeled batch convention, not opening-auction attainability. Primary units are fractional adjusted units; daily prices represent price returns excluding ordinary distributions.

For executed notional n, lagged dollar ADV D and volatility sigma, direct cost is
n [0.5 + 15 min(n/D,1)^0.7 (sigma/0.01)] / 10000 dollars.
These are scenario assumptions, not institutional calibration. Uncapped executions may exceed lagged dollar ADV and impact saturates above one ADV. Exogenous prices do not move in response to fills.

Stateful costs alter later cash, NAV, holdings and demand. Post-hoc adjustment retains reference trades and compounds gross returns minus cost/reference-previous-NAV rates. Demand cohorts retain origin prices, FIFO service and newest-first withdrawal. Executed + withdrawn + outstanding equals originated value; completion uses each replay's endogenous originated demand. The 95% completion label is descriptive with matched references.

Annualized arithmetic active return is 252 times the mean aligned daily return difference. Annualized relative wealth catch-up is (reference terminal NAV / arm terminal NAV)^(252 / aligned return intervals) minus one, with matched initial capital and positive wealth. These are distinct estimands with separate 5 bps/year thresholds. H2c uses the four-cell arithmetic-return interaction, not a coefficient or terminal-wealth attribution. WTE materiality is at least 10% relative to the matched reference, or 0.005 absolute if reference WTE is below 0.0001.

Paired stationary resampling uses common indices, 2,000 replicates, geometric mean blocks 20 and 60, PCG64 seed 20261119, and nominal pointwise 95% percentile intervals. Stationarity and sufficiently weak dependence have not been established for evolving portfolios. Future-market, simultaneous and repeated-system coverage are unvalidated. Sharpe uses sqrt(252) times arithmetic mean daily return / population SD, with zero risk-free rate.

Lifecycle implementation shortfall uses policy-specific endpoints and cannot rank common-horizon savings. The E2b factorial investigates accounting assumptions; it does not exactly attribute a historical legacy gap. Three capital levels are not a calibrated continuous capacity frontier. Two strategies and one historical path do not establish universal execution superiority.

Technical specification and scenario values are governed by the exact source and configs/v2 registry; this explanatory document does not change scientific logic.

The preserved registry annotation “quantities fixed pre-open” refers to desired gaps only. Final units depend on opening-price liquidity clipping and cost-inclusive cash allocation, as described above; the frozen record is not silently rewritten.
