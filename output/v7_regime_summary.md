# V7 Regime-Adaptive Nasdaq Research

Data: 1999-03-10 through 2026-07-22. Evaluation begins 2000-03-20 after signal warmup. Initial capital: $100,000.

## Result

- Annualized return: **11.56%**
- Annualized volatility: **24.25%**
- Maximum drawdown: **-36.75%**
- Drawdown peak: **2004-01-26**
- Drawdown trough: **2006-07-21**
- Recovery date: **2009-12-21**
- Peak-to-recovery: **1488 trading sessions**
- Maximum dollar loss on the simulated $100k path: **$45,573**
- Terminal value: **$1,785,353**
- Calmar ratio: **0.31**

## Independent benchmark

QQQ returned 7.98% annualized with a -82.96% maximum drawdown and a 0.10 Calmar ratio.

## Actual-TQQQ era

From 2010-02-11, the strategy returned 18.23% annualized versus 19.58% for QQQ. Maximum drawdowns were -35.45% and -35.12%, respectively. The strategy therefore delivered lower return without a meaningful drawdown improvement during the non-synthetic period.

## Decision

**REJECT** — Failed pre-registered check(s): max_dd_recovery_within_756_sessions.

The result was evaluated against the rules fixed in `DESIGN.md`; no post-result parameter was promoted.

## Robustness

- cost_20bps: CAGR 10.38%, max drawdown -39.26%, Calmar 0.26
- execution_t2: CAGR 10.37%, max drawdown -38.74%, Calmar 0.27
- sma_150: CAGR 11.71%, max drawdown -35.45%, Calmar 0.33
- sma_250: CAGR 11.12%, max drawdown -35.97%, Calmar 0.31
- defense_IEF_TLT_GLD: CAGR 11.37%, max drawdown -37.34%, Calmar 0.30

## Important limitations

- TQQQ history before its 2010 inception is synthetic and depends on the stated financing and expense assumptions.
- Yahoo adjusted closes are a research data source, not guaranteed execution prices.
- The ETF universe is selected with hindsight. DBC and DBMF enter only after actual inception; the constant-universe diagnostic shows their effect.
- Taxes, bid/ask variation, market impact, and intraday gaps are not fully modeled.
