# V9 R50d Improvement Research

Data: 1999-03-10 through 2026-07-22.
Primary evaluation window: from_2015.

## Primary candidate vs live R50d (from_2015)

- Live R50d: 21.52% CAGR, -22.86% max DD, 0.94 Calmar
- R50d_asym: 22.10% CAGR, -22.03% max DD, 1.00 Calmar
- CAGR delta: +0.58 pp
- Max DD delta: +0.83 pp (positive = shallower / better)
- 2022 return: R50d -11.77%, R50d_asym -6.78%

## Full-history snapshot

- R50d: 15.84% CAGR, -22.86% max DD
- R50d_asym: 16.26% CAGR, -22.03% max DD

## Diagnostics (from_2015)

- R50d_corr: 21.47% CAGR, -22.86% max DD, 0.94 Calmar
- R50d_redistrib: 22.91% CAGR, -22.63% max DD, 1.01 Calmar
- R50d_asym_crash: 22.10% CAGR, -22.03% max DD, 1.00 Calmar

## Decision

**REJECT** — Failed pre-registered check(s): max_dd_improves_1pp.

Diagnostics do not replace the primary candidate.

## Limitations

- Pre-2010 TQQQ/TMF/UGL history is synthetic where needed.
- DBMF uses AQMIX splice before ETF inception.
- No live account was changed.
