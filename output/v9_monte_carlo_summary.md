# Monte Carlo: paper R50d vs paper R50d_asym

Paths: 8,000. Primary method: stationary block bootstrap (block length 21 sessions). Seed 42.

## Historical sample (from_2015)

- R50d: CAGR 21.66%, max DD -22.80%
- R50d_asym: CAGR 22.25%, max DD -21.97%

## 5-year forward block bootstrap (paired)

- P(asym terminal > R50d terminal): **67.4%**
- P(asym CAGR > R50d CAGR): **67.4%**
- P(asym max DD shallower): **70.8%**
- Median terminal $100k — R50d: $267,816; asym: $272,718
- 5th pct terminal $100k — R50d: $143,743; asym: $148,920
- Median max DD — R50d: -21.12%; asym: -20.70%

## Interpretation

Over simulated 5-year block paths from the 2015+ return sample, R50d_asym beat R50d on terminal wealth in 67.4% of paths and had a shallower max drawdown in 70.8%. That supports continuing the paper A/B, not replacing live R50d yet.

## Limitations

- Bootstrap resamples realized strategy returns; it does not re-simulate markets or change regime structure beyond historical blocks.
- from_2015 omits pre-2015 stress that still matters for R50d design.
- Paper trading remains the true out-of-sample test.
