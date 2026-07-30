# V9 R50d Improvements — Pre-registered Design

Improve the live R50d book without replacing its architecture. One primary
candidate and three fixed diagnostics. No joint grid. No promotion after
seeing results.

## Baseline (unchanged live R50d)

- Sleeves: TQQQ 40%/3x, TMF 15%/3x, UGL 15%/2x, DBC 10%/1x, DBMF 10%/1x,
  BTC 10%/1x
- Ensemble gate on underlyings: mean of (63/126/252-day return > 0)
- Size: `gate × budget × 0.50 / (leverage × EWMA vol)`, 100% notional cap
- 5-point rebalance band, T+1 execution, 5 bps per unit turnover
- Idle cash at 13-week T-bills

## Primary candidate: `R50d_asym`

Same book as R50d, except the **equity sleeve gate** uses entry/exit asymmetry:

- Risk-on when the QQQ ensemble is at least 2/3
- Risk-off when the QQQ ensemble is at most 1/3
- Between those levels, keep the previous equity risk-on/off state
- When risk-on, size TQQQ with the continuous ensemble gate as in live R50d
- When risk-off, TQQQ weight is zero
- All other sleeves keep the live continuous ensemble gate

No drawdown brake. No whole-book regime switch. No σ* change.

## Fixed diagnostics (not candidates)

1. `R50d_corr` — live R50d plus a portfolio stress scale: if the 63-day average
   pairwise correlation of QQQ, DBC, and BTC exceeds 0.60 and QQQ's 21-day
   return is below -8%, multiply all sleeve targets by 0.50 until the stress
   clears.
2. `R50d_redistrib` — live R50d, but each day redistribute the budgets of
   sleeves with zero raw weight across sleeves that still have positive weight,
   then apply the usual 100% notional cap. No sleeve may receive more than 1.5×
   its base budget via redistribution.
3. `R50d_asym_crash` — `R50d_asym` plus an equity crash cut: while equity is
   risk-on, if QQQ's 21-day return is at or below -2 × (20-day daily std ×
   √21), multiply the TQQQ target by 0.50 for that day.

## Evaluation

Windows: full post-warmup, from_2010, from_2015 (primary), holdout from_2018,
last_3y. Stress: GFC, COVID crash, 2022, tariff 2025.

Report CAGR, vol, Sharpe, Calmar, exact max-DD peak/trough/recovery, turnover,
average invested, and yearly returns. Compare every variant to live R50d.

## Pass rule for `R50d_asym`

On the from_2015 window:

1. CAGR is within 2.0 percentage points of live R50d (not more than 2pp worse).
2. Maximum drawdown improves by at least 1.0 percentage point (less deep).
3. Calendar-year 2022 total return is not worse than live R50d by more than
   1.0 percentage point.

All three must hold. Diagnostics may inform interpretation but cannot replace
the primary candidate.
