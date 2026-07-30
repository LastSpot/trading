# V7 Regime-Adaptive Nasdaq — Pre-registered Design

This document fixes the research rules before the first backtest run. The
strategy is independent of the current live books. Results will not be used to
retune these constants.

## Objective

Pursue high compound return in a $100,000 retail cash account while controlling
risk through regime selection and volatility sizing. No options, short
positions, or broker margin are allowed. Leveraged exposure may only come from
an exchange-traded fund.

## Data and execution

- Daily adjusted closes from Yahoo Finance, starting at QQQ inception on
  1999-03-10 and ending at the latest available session.
- Signals are evaluated after each completed week's final trading session
  (normally Friday). Orders take effect on the next trading session. A
  two-session delay is reported as a stress test.
- Base transaction cost is 10 basis points per unit of turnover; 20 basis
  points is reported as a stress test.
- Idle capital earns the 13-week Treasury-bill yield from `^IRX`. Missing yield
  observations earn zero rather than an invented rate.
- Total portfolio notional is capped at 100% of equity. Fractional shares are
  assumed.
- A five-percentage-point target-weight band suppresses small rebalances.
  Regime changes and assets leaving the defensive selection are closed even
  when the band is not crossed.

## Volatility estimate

Annualized asset volatility is the square root of 252 times the exponentially
weighted mean of squared daily returns. The EWMA span is 63 sessions
(`alpha = 2 / 64`). The estimate uses information available through the signal
close.

## Regime classifier

SPY trend uses a 200-session simple moving average with 2% hysteresis:

- Risk-on begins when SPY closes above 102% of its SMA.
- Risk-on persists until SPY closes below 98% of its SMA.

QQQ momentum is its trailing 63-session and 126-session total return.

- **Bull:** SPY is risk-on and both QQQ momentum measures are positive.
- **Bear:** SPY is risk-off and QQQ 126-session momentum is non-positive.
- **Neutral:** every other combination.

The hysteresis state and regime are updated only on weekly signal dates.

## Position sizing

### Bull

Hold TQQQ at:

`weight = min(1, 0.30 / TQQQ_EWMA_vol)`

Before TQQQ inception, its return is modeled as daily-reset 3× QQQ less
financing and a 0.95% annual expense ratio. Actual adjusted TQQQ returns replace
the synthetic series after inception.

### Neutral

Hold QQQ at:

`weight = min(1, 0.15 / QQQ_EWMA_vol)`

### Bear

The deployable defensive universe is IEF, TLT, GLD, DBC, and DBMF. An asset is
eligible only after 126 valid daily observations and when its 126-session
momentum is positive.

Select the top two eligible assets by momentum. Normalize inverse-volatility
weights, estimate their annualized covariance over the latest 63 valid common
sessions, and scale the combined allocation toward 12% annualized volatility:

`weights = inverse_vol_weights * min(1, 0.12 / portfolio_vol)`

If fewer than two assets qualify, use the qualifying asset alone. If none
qualify, hold Treasury bills. Each ETF is unavailable before its real
inception; no synthetic defensive history is created.

## Benchmarks

- Buy-and-hold SPY
- Buy-and-hold QQQ
- 60% SPY / 40% IEF, rebalanced monthly; the unavailable IEF allocation remains
  in Treasury bills before inception
- Buy-and-hold synthetic/actual TQQQ

The current live strategies are not inputs, candidates, or optimization
benchmarks.

## Evaluation

Report the full post-warmup sample, actual-TQQQ era, 2015-present, and trailing
three years. Fixed stress periods are dot-com, GFC, COVID crash, 2022 inflation
bear, and the 2025 tariff period.

For every main window report CAGR, annualized volatility, Sharpe, Sortino,
Calmar, turnover, terminal value of $100,000, and the exact maximum-drawdown
peak, trough, recovery date, depth, dollar loss, and durations. Also report the
longest underwater period, yearly returns, rolling 1/3/5-year returns, and
regime occupancy/contribution.

The strategy passes only if, over the full sample:

1. Calmar exceeds QQQ buy-and-hold.
2. CAGR exceeds QQQ by at least three percentage points.
3. Its maximum drawdown recovers within 756 trading sessions.

An unrecovered maximum drawdown fails the third condition.

## Fixed robustness checks

- Next-session execution with 20 bps costs
- Two-session execution with 10 bps costs
- SPY trend SMA lengths of 150 and 250 sessions
- Constant defensive universe restricted to IEF, TLT, and GLD

These are diagnostics only. No robustness variant replaces the primary strategy
after results are observed.
