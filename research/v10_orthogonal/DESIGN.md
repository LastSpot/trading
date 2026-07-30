# V10 Orthogonal Premia — Pre-registered Design

V7–V9 all tried to improve the live book by reshaping the *same* trend edge and
were rejected. V10 instead adds two return sources that are economically
distinct from trend following, each tradable with only Alpaca + Yahoo data:

1. **VRP** — the volatility risk premium. Hedgers persistently overpay for
   crash insurance, so the VIX futures curve is usually in contango and a
   short-vol position (SVXY) earns the roll-down. The premium disappears (and
   reverses violently) in backwardation, so the sleeve only sells insurance
   when the term structure says the curve is calm.
2. **MR** — short-term mean reversion. Panic sellers pay a liquidity premium;
   buying sharp multi-day pullbacks in QQQ while it remains in a long-term
   uptrend captures it. It fires exactly when trend gates are stepping off,
   which is why it should diversify R50d.

One primary candidate (`R50d_plus` = live R50d + both sleeves), standalone
sleeve evaluations for interpretation, two placebo diagnostics, and a small
pre-declared robustness neighborhood. No promotion after seeing results
outside the pass rule below.

## Baseline (unchanged live R50d)

- Sleeves: TQQQ 40%/3x, TMF 15%/3x, UGL 15%/2x, DBC 10%/1x, DBMF 10%/1x,
  BTC 10%/1x
- Ensemble gate on underlyings: mean of (63/126/252-day return > 0)
- Size: `gate × budget × 0.50 / (leverage × EWMA vol)`, 100% notional cap
- 5-point rebalance band, T+1 execution, 5 bps per unit turnover
- Idle cash at 13-week T-bills

## Sleeve V: VRP (short vol via SVXY)

- Signal at each close: `ratio = ^VIX3M / ^VIX` (both from Yahoo, ffilled).
- Hysteresis state (house asym style): risk-on when ratio ≥ 1.05, risk-off
  when ratio ≤ 1.00, otherwise keep the previous state. Start risk-off.
- Instrument: SVXY. Primary return series is the **modern half-leverage
  series**: actual SVXY daily returns × 0.5 before 2018-02-28, actual
  returns after. A −0.5x daily-rebalanced product returns exactly half of a
  −1x product's daily return gross of fees, so this reproduces what today's
  SVXY (−0.5x since late Feb 2018) would have done. This is instrument
  realism, not parameter hindsight: the −1x product no longer exists.
- Sizing: `state × B_V × 0.50 / EWMA vol(modern series)`, EWMA α = 0.06,
  capped at 1.5 × B_V. **B_V = 0.15.**
- Cost: **10 bps** per unit turnover (SVXY spreads are wider than the ETF
  sleeves).
- Live-signal note: Yahoo's ^VIX3M can lag a few sessions; CBOE publishes a
  free daily CSV (`VIX3M_History.csv`) as the live fallback. Backtest uses
  Yahoo with ffill.

## Sleeve M: short-term mean reversion (QQQ)

- Underlying and traded instrument: QQQ (unlevered — MR entries occur in
  falling markets, no leverage there).
- Qualifier: QQQ close > SMA200.
- Entry: Wilder RSI(2) of QQQ close < 10 while qualified → target weight
  **B_M = 0.20** from the next session (T+1).
- Exit: RSI(2) > 60, or 10 sessions in position, whichever first (T+1).
- Fixed weight while in position; no vol targeting (holds are ~2–6 sessions).
- Cost: **10 bps** per unit turnover.

## Primary candidate: `R50d_plus`

Live R50d weights, unchanged, plus sleeve V and sleeve M as overlay columns.
Joint 100% notional cap: if the combined target sum exceeds 1.0, scale all
eight columns proportionally (cash account, no margin). 5-point band and T+1
apply to all columns. Costs: 5 bps on the six R50d legs, 10 bps on V and M.

## Standalone sleeves and diagnostics (not candidates)

1. `VRP_sleeve` / `MR_sleeve` — each sleeve alone (plus T-bills on idle cash),
   to measure the premium and its correlation to R50d.
2. `VRP_nogate` — placebo: SVXY always on with the same sizing. If this
   matches `VRP_sleeve`, the term-structure gate adds nothing.
3. `MR_notrend` — placebo: MR without the SMA200 qualifier. If this matches
   `MR_sleeve`, the qualifier adds nothing.
4. `MR_sleeve` signal-level history from 1999 (QQQ full history) with trade
   count, hit rate, and mean per-trade return.

## Evaluation

Common data window: SVXY inception (2011-10) + warmup → comparisons start
2012-07-02 ("full_common"). Windows: full_common, **from_2015 (primary)**,
holdout from_2018, last_3y. Stress slices: Aug-2015 flash, Volmageddon
(2018-01-26 → 2018-04-30), COVID crash, 2022 bear, tariff 2025.

Report CAGR, vol, Sharpe, Calmar, exact max-DD episode, worst month,
turnover, yearly returns, and daily-return correlation of each sleeve to
R50d. Cost stress: rerun primary with all leg costs doubled.

## Pass rule for `R50d_plus` (from_2015)

1. CAGR is at least 1.0 percentage point higher than live R50d.
2. Maximum drawdown is not more than 1.0 percentage point deeper.
3. Sharpe is not lower than live R50d.
4. Calendar-year 2022 return is not worse than live R50d by more than 1.0 pp.

All four must hold. Sleeve sanity gates (interpretation only, not promotion):
each standalone sleeve has net Sharpe ≥ 0.50 over its full window and daily
correlation to R50d below 0.40.

## Pre-declared robustness neighborhood (report only)

- VRP hysteresis (enter, exit): {(1.03, 0.99), (1.05, 1.00), (1.07, 1.01)}
- MR entry RSI(2) threshold: {5, 10, 15}

Nine combined-book cells; the primary is the center cell. If the primary
fails but at least 7 of 9 cells beat R50d's from_2015 CAGR with max DD within
1.0 pp, one revision round is permitted, judged primarily on the from_2018
holdout. Otherwise reject without revision.
