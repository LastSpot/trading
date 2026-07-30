# V10 Orthogonal Premia Research

Data: 1999-03-10 through 2026-07-29.
Primary evaluation window: from_2015. Design: research/v10_orthogonal/DESIGN.md.

## Primary candidate vs live R50d (from_2015)

- Live R50d: 21.06% CAGR, 18.41% vol, Sharpe 1.02, -22.92% max DD, Calmar 0.92
- R50d_plus: 19.94% CAGR, 19.12% vol, Sharpe 0.94, -22.18% max DD, Calmar 0.90
- CAGR delta: -1.12 pp (need >= +1.00)
- Max DD delta: +0.74 pp (positive = shallower; need >= -1.00)
- Sharpe delta: -0.08 (need >= 0)
- 2022: R50d -11.77%, R50d_plus -13.69% (need >= base - 1pp)

## Standalone sleeves (full_common, net of 10 bps)

- VRP_sleeve: 4.02% CAGR, 5.55% vol, Sharpe 0.44, -7.34% max DD, corr to R50d +0.50
- MR_sleeve: 2.39% CAGR, 1.56% vol, Sharpe 0.44, -2.30% max DD, corr to R50d +0.36
- VRP_nogate: 4.62% CAGR, 6.31% vol, Sharpe 0.48, -11.96% max DD, corr to R50d +0.54
- MR_notrend: 2.64% CAGR, 2.31% vol, Sharpe 0.41, -4.40% max DD, corr to R50d +0.29
- VRP_actual_svxy: 5.67% CAGR, 8.06% vol, Sharpe 0.52, -11.40% max DD, corr to R50d +0.48

## MR signal history (QQQ, 1999+)

```
{
  "n_trades": 209,
  "hit_rate": 0.7320574162679426,
  "mean_per_trade": 0.005930916092919076,
  "median_per_trade": 0.010210723853611325,
  "worst_trade": -0.08020101679092406,
  "best_trade": 0.12368378630927501,
  "avg_sessions_in_market_per_year": 28.09348236318769
}
```

## Robustness neighborhood (R50d_plus, from_2015: CAGR / max DD)

- vrp1.03/0.99_mr5: 20.18% / -22.83%
- vrp1.03/0.99_mr10: 19.52% / -23.52%
- vrp1.03/0.99_mr15: 18.98% / -23.78%
- vrp1.05/1.00_mr5: 20.37% / -21.91%
- vrp1.05/1.00_mr10: 19.94% / -22.18%
- vrp1.05/1.00_mr15: 19.14% / -22.27%
- vrp1.07/1.01_mr5: 19.82% / -22.78%
- vrp1.07/1.01_mr10: 19.59% / -23.04%
- vrp1.07/1.01_mr15: 19.33% / -23.13%

Cost stress (2x all legs, from_2015): R50d_plus 18.81% CAGR, -23.14% max DD.

## Decision

**REJECT** — Failed pre-registered check(s): cagr_up_1pp, sharpe_not_lower, y2022_not_worse_1pp.

## Limitations

- SVXY pre-2018 uses the half-leverage proxy of the actual -1x fund; the
  -1x era diagnostic is reported separately.
- Yahoo ^VIX3M can lag a few sessions; live use needs the CBOE CSV fallback.
- VRP history starts 2011-10 (SVXY inception); no 2008-style event in sample.
- No live account was changed.
