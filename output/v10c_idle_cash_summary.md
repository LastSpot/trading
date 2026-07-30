# V10c Idle-Cash Premia Overlay Research

Data: 1999-03-10 through 2026-07-29.
Primary window: from_2015 with a from_2018 holdout gate.
Design: research/v10_orthogonal/DESIGN_C.md.

## Candidate vs live R50d

- from_2015 R50d:      20.87% CAGR, Sharpe 1.01, -22.96% max DD, Calmar 0.91
- from_2015 R50d_cash: 22.10% CAGR, Sharpe 1.03, -23.57% max DD, Calmar 0.94
- from_2018 R50d:      20.68% CAGR, -20.92% max DD
- from_2018 R50d_cash: 21.63% CAGR, -20.92% max DD
- 2022: R50d -11.77%, R50d_cash -11.14%

## Ablations and stress (from_2015, CAGR / max DD / Sharpe)

- R50d_mrveto: 21.10% / -23.48% / 1.02
- R50d_vrponly: 22.36% / -22.44% / 1.04
- R50d_cash_cap20: 22.17% / -23.25% / 1.03
- R50d_cash_cap40: 22.06% / -23.57% / 1.03
- R50d_cash_cost2x: 21.48% / -24.22% / 1.00

## Sleeve contribution

```
{
  "mr_by_year": {
    "2000": 0.031705268180265936,
    "2003": 0.053916525120942055,
    "2004": 0.026735003888902285,
    "2005": -0.003257124407979003,
    "2006": 0.00517336657865277,
    "2007": 0.005861757280077774,
    "2008": -0.004676499111361933,
    "2009": 0.021542561061430675,
    "2010": -0.007745459726384663,
    "2011": 0.0019124266061945187,
    "2012": 0.005717394977872284,
    "2013": 0.005130725034845162,
    "2014": 0.002033545662694495,
    "2015": 0.007454843801011706,
    "2016": -0.009836687123296758,
    "2017": 0.0011908374337868764,
    "2018": 0.021387406467243443,
    "2019": 0.0008299324545234101,
    "2020": -0.006401685293225901,
    "2021": 0.0030425662125651443,
    "2022": -0.004279361070680613,
    "2023": 0.009268032098311235,
    "2024": 0.009227700999228846,
    "2025": 0.00891212510824916,
    "2026": 0.002486730096608904
  },
  "vrp_by_year": {
    "2011": -0.017768912492858824,
    "2012": 0.044384941273159104,
    "2013": 0.014435146639878067,
    "2014": -0.0014126098186906075,
    "2015": 0.019654686067544523,
    "2016": 0.03423928608989477,
    "2017": 0.02316341214001074,
    "2018": -0.03177864326964426,
    "2019": 0.010138637484188029,
    "2020": 0.024013256215684974,
    "2021": 0.004607701516717962,
    "2022": 0.021193456608992824,
    "2023": 0.06386875936913683,
    "2024": -0.00999755737314788,
    "2025": 0.007579284333962394,
    "2026": -0.00034475029127788517
  },
  "avg_mr_pos_when_on": 0.16762761873546864,
  "avg_vrp_pos_when_on": 0.09613878951711935,
  "pct_days_vrp_on": 0.2692698504862825,
  "pct_days_mr_on": 0.07141820293221077
}
```

## Decision

**PASS_PAPER_AB_ONLY** — R50d_cash passed all five pre-registered checks; eligible for a paper A/B account only (third pass over the same history).

## Limitations

- Third pass over the same history: from_2015 numbers are partially
  burned; only paper trading provides true out-of-sample evidence.
- SVXY pre-2018 is the half-leverage proxy of the -1x fund.
- Yahoo ^VIX3M lags; live requires the CBOE daily CSV fallback.
- Headroom must be computed from broker positions live.
- No live account was changed.
