# V8 Independent Alpha Ensemble

Data begins 1999-03-10; evaluation begins 2000-03-20 after warmup and ends 2026-07-22.

## Full-history result

- CAGR: **11.27%**
- Annualized volatility: **17.15%**
- Maximum drawdown: **-41.34%**
- Drawdown peak / trough / recovery: **2000-03-24 / 2002-10-09 / 2005-08-02**
- Peak-to-recovery: **1345 sessions**
- Terminal value from $100,000: **$1,663,682**
- Calmar: **0.27**

QQQ returned 7.98% with a -82.96% maximum drawdown over the same period.

## Chronological result

- Development label (2000–2009): 3.82% CAGR, -41.34% max drawdown
- Validation label (2010–2017): 16.82% CAGR, -16.49% max drawdown
- Holdout label (2018–latest): 15.12% CAGR, -22.62% max drawdown

## Holdout (2018–latest)

- Ensemble: 15.12% CAGR, -22.62% max drawdown, 0.67 Calmar
- QQQ: 20.09% CAGR, -35.12% max drawdown, 0.57 Calmar

## Fixed diagnostics

- core_only: 8.06% CAGR, -43.24% max drawdown, 0.19 Calmar
- momentum_only: 6.65% CAGR, -11.41% max drawdown, 0.58 Calmar
- reversal_only: 2.47% CAGR, -4.46% max drawdown, 0.55 Calmar
- without_reversal: 11.13% CAGR, -41.17% max drawdown, 0.27 Calmar
- cost_20bps: 10.86% CAGR, -41.53% max drawdown, 0.26 Calmar
- execution_t2: 11.05% CAGR, -41.94% max drawdown, 0.26 Calmar

The momentum sleeve supplied the strongest standalone drawdown efficiency. The reversal sleeve was low-risk but sparse: adding it changed ensemble CAGR from 11.13% to 11.27% and max drawdown from -41.17% to -41.34%. It did not solve the recovery problem.

## Decision

**REJECT** — Failed pre-registered check(s): max_dd_recovery_within_756_sessions.

No diagnostic replaces the pre-registered primary ensemble.

## Limitations

- TQQQ returns before 2010 are synthetic.
- The fixed ETF universe is selected with hindsight, although every fund enters only after actual inception.
- Yahoo adjusted closes do not model intraday gaps, taxes, or variable spreads.
- The reversal sleeve is a sparse historical pattern and requires live paper evidence.
