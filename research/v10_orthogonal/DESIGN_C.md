# V10c Idle-Cash Premia Overlay — Pre-registered Final Iteration

V10b's candidate (MR alone on idle cash) failed on CAGR (+0.26 pp vs +0.50
bar) and max DD (-1.27 pp, 2015-08 → 2016-01 episode). Two diagnostics
motivate one final composition:

- The DD deepening came from MR buying dips during vol-regime breaks
  (Aug-2015, Jan-2016); in COVID the overlay actually reduced the book's
  crisis loss. Liquidity provision loses when volatility is exploding — the
  VIX term structure already measures that.
- `R50d_vrpcash` (VRP sized from idle cash) added +2.7 pp CAGR from_2015.
  Headroom exists exactly in choppy, non-trending markets where contango
  carry pays, so idle cash — not an overlay on top of the core — is the
  right home for VRP.

Because this is the third pass over the same history, a pass earns at most a
**paper A/B recommendation**, never direct promotion. This is declared before
running.

## Candidate: `R50d_cash`

Core: live R50d, unchanged. Idle cash headroom
`H(t) = max(0, 1 − Σ core positions)` funds two slots, MR first:

- **MR slot** (priority): V10 MR signal (RSI(2) < 10, QQQ > SMA200; exit
  RSI(2) > 60 or 10 sessions) with a **backwardation veto**: no new entries
  while VIX3M/VIX ≤ 1.00 (open positions are not force-closed).
  Weight = `min(0.30, H)`. **CAP_M = 0.30.**
- **VRP slot**: V10 VRP hysteresis (on ≥ 1.05, off ≤ 1.00) on SVXY (modern
  half-leverage series), vol-targeted exactly as V10
  (`0.15 × 0.50 / EWMA vol`, cap 0.225), further capped by remaining
  headroom `H − w_MR`.
- Both slots: 5-point band while open, hard-capped at available headroom
  every session, 10 bps per unit turnover, T+1. Idle cash at T-bills.

## Ablations (structural, report only)

1. `R50d_mrveto` — MR slot only (with veto), CAP_M = 0.30.
2. `R50d_vrponly` — VRP slot only.
3. Cap neighborhood on the candidate: CAP_M ∈ {0.20, 0.30, 0.40}.
4. Cost stress: overlay legs at 20 bps.

## Evaluation

Windows: full post-warmup, from_2010, **from_2015 (primary)**, holdout
from_2018, last_3y. Crises: GFC, Aug-2015, Volmageddon, COVID, 2022, tariff
2025. VRP contributes nothing before SVXY inception (2011-10) by
construction.

## Pass rule for `R50d_cash` (from_2015)

1. CAGR at least 1.0 pp higher than live R50d (raised from V10b's 0.5 —
   two sleeves must clear a higher bar).
2. Max drawdown not more than 1.0 pp deeper.
3. Sharpe not lower.
4. 2022 return not worse by more than 0.5 pp.
5. Holdout from_2018: CAGR higher than live R50d and max DD not more than
   1.0 pp deeper (added because from_2015 has now been examined three times).

All five must hold → recommend **paper A/B account only**. Any failure →
V10 research line closes with a reject; no v10d.
