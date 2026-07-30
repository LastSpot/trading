# V10b MR-on-Idle-Cash — Pre-registered Revision

V10's primary (`R50d_plus`) was rejected: the overlay displaced core R50d legs
through the joint 100% notional cap and band interaction (-1.1 pp CAGR, worse
2022) even though **both sleeves made money in 2022**. Sleeve diagnostics:

- MR passed its sanity gates in spirit (corr to R50d 0.36, 73% hit rate over
  209 trades since 1999, sleeve max DD -2.3%) and the trend qualifier beat the
  no-trend placebo on DD.
- VRP failed both sanity gates (corr to R50d 0.50, net Sharpe 0.43). Short vol
  is largely equity beta on an already equity-heavy book. **VRP is dropped
  from the candidate** and kept only as documented sleeve research.

V10b changes the *architecture*, not the signal: the MR sleeve may only use
**idle cash** — capital the trend core has already vacated — so it can never
displace or perturb the R50d legs.

## Candidate: `R50d_mr`

- Core: live R50d, byte-for-byte unchanged (own gates, sizing, 100% cap,
  5-point band, 5 bps, T+1). Its positions are identical to the baseline by
  construction.
- MR signal: unchanged from V10 — Wilder RSI(2) of QQQ < 10 while
  QQQ > SMA200 to enter; exit at RSI(2) > 60 or 10 sessions. T+1.
- Sizing: `w_MR(t) = signal(t) × min(0.50, headroom(t))` where
  `headroom(t) = max(0, 1 − Σ core positions after band at close t)`.
  **B_M_MAX = 0.50.**
- MR column trades through the 5-point band while a position is open, but is
  always hard-capped at current headroom (cash account, never above 100%
  notional).
- Cost: 10 bps on the MR leg. Idle cash still earns T-bills.

## Diagnostics (not candidates)

1. Headroom/fill stats: average headroom while the MR signal is on, fraction
   of signals that could deploy at least 10% weight.
2. `R50d_vrpcash` — VRP sleeve (V10 rules) sized the same idle-cash way, to
   confirm the expectation that it rarely deploys (contango = risk-on = core
   fully invested).
3. MR entry neighborhood {5, 10, 15} on the candidate (report only).
4. Cost stress: MR leg at 20 bps.

## Evaluation

Windows: full post-warmup, from_2010, **from_2015 (primary)**, holdout
from_2018, last_3y. Crises: GFC, COVID crash, 2022 bear, tariff 2025.
Report the usual metrics plus MR contribution per year.

## Pass rule for `R50d_mr` (from_2015)

1. CAGR at least 0.5 pp higher than live R50d.
2. Max drawdown not more than 0.5 pp deeper.
3. Sharpe not lower.
4. 2022 return not worse by more than 0.5 pp.

All four must hold. Bars are tighter than V10 because the overlay is strictly
additive cash deployment; if it cannot clear +0.5 pp without hurting the risk
profile, it is not worth the operational complexity. No further revision
rounds: fail → reject V10b entirely.
