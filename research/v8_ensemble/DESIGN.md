# V8 Independent Alpha Ensemble — Pre-registered Design

This specification is fixed before the first backtest. The strategy combines
three simultaneous return sources instead of classifying the whole market into
one regime. No result from V7 or the live strategies is used to choose the
parameters below.

## Objective and account constraints

- $100,000 retail cash account.
- Long-only liquid ETFs; no options, short positions, or broker margin.
- TQQQ may provide embedded leverage but is capped at 25% of account equity.
- Total cash invested in ETFs may not exceed 100%.
- Target high compound return while requiring drawdown recovery within roughly
  three trading years.

## Data and execution

- Yahoo Finance adjusted daily closes from QQQ inception on 1999-03-10 through
  the latest complete session.
- Signals formed after the close and positions applied to the next session.
- Base trading cost: 10 basis points per unit of turnover.
- Stress tests: 20 basis points and a two-session execution delay.
- Idle capital earns the available 13-week T-bill yield from `^IRX`; missing
  observations earn zero.
- Small target changes are ignored until the difference from the held weight
  exceeds five percentage points. Required exits are not suppressed.

## Sleeve 1: always-on equity premium

Raw economic exposure is:

- 25% SPY
- 25% QQQ

This sleeve does not use a market regime or trend filter. Its purpose is to
capture the persistent equity risk premium and avoid missing recoveries.

## Sleeve 2: cross-asset relative momentum

At each completed month end, rank QQQ, IWM, EFA, EEM, IEF, TLT, GLD, DBC, and
DBMF.

- An ETF requires 252 valid observations.
- Both 126-session and 252-session total returns must be positive.
- Score is the mean of those two returns.
- Select at most the top three.
- Divide a 30% raw economic allocation using inverse 63-session EWMA
  volatility.
- If no ETF qualifies, this sleeve remains in T-bills.
- An ETF cannot be selected before its actual inception.

## Sleeve 3: panic reversal

This sleeve targets short-lived forced selling rather than persistent trends.

- Use QQQ.
- Entry requires its five-session return to be at or below negative two times
  its trailing 20-session daily standard deviation times the square root of
  five.
- QQQ must remain above its 200-session moving average at entry. This limits
  the sleeve to sharp selloffs inside a longer-term uptrend.
- Raw economic allocation is 20%.
- Exit after five trading sessions or when QQQ closes at or above its
  20-session moving average, whichever occurs first.
- No overlapping entries and no TQQQ is assigned specifically to this sleeve.

## Portfolio construction

Add the three sleeves' raw one-times-equivalent exposures. Estimate annualized
portfolio volatility from the latest 63 common daily returns and scale all
economic exposures toward an 18% target. The scale multiplier is capped at 2.

Then enforce retail constraints:

1. Each non-TQQQ ETF is capped at 35% of account equity.
2. Combined non-QQQ positions are capped at 75%.
3. QQQ exposure that fits within remaining cash is held as QQQ.
4. Additional desired QQQ-equivalent exposure is represented with TQQQ, using
   three units of QQQ-equivalent exposure per unit of TQQQ capital.
5. TQQQ is capped at 25% and total ETF capital at 100%.

Before TQQQ inception, its returns are modeled as daily-reset 3x QQQ less
financing and a 0.95% annual expense ratio. Actual adjusted TQQQ returns replace
the synthetic series after inception.

There is no drawdown brake and no whole-portfolio regime switch.

## Evaluation

Report:

- Full post-warmup history
- Development label: start through 2009
- Validation label: 2010 through 2017
- Untouched holdout label: 2018 through the latest session
- Actual-TQQQ era and 2015-present
- Dot-com, GFC, COVID crash, 2022 inflation bear, and 2025 tariff stress

Metrics include CAGR, volatility, Sharpe, Sortino, Calmar, turnover, terminal
value, exact maximum-drawdown peak/trough/recovery, longest underwater period,
rolling returns, yearly returns, average invested capital, and TQQQ usage.

Fixed diagnostics:

- Core sleeve alone
- Momentum sleeve alone
- Reversal sleeve alone
- Ensemble without reversal
- 20-basis-point costs
- Two-session execution delay

Diagnostics explain the result; they do not replace the primary specification.

## Pass criteria

The primary ensemble passes only if:

1. Full-history CAGR exceeds QQQ by at least two percentage points.
2. Holdout Calmar exceeds holdout QQQ Calmar.
3. Full-history maximum drawdown recovers within 756 trading sessions.
4. Both cost and execution stress tests retain positive holdout CAGR.
