# V11 VOO/QQQ allocation research (with retail tax)

Universe: SPY/VOO + QQQ only. Monthly T+1, 5 bps, cash at T-bills.
**Tax (~$150k income):** ST 24% / LT 15% (FIFO lots, LT after 365d). Gain-only.
Data from 1999-03-10; **primary metrics from 2000-03-08** (252d warmup).
STATIC_BEST: **100% QQQ** (max Calmar on 2003-01-01 -> 2012-12-31, selected pre-tax).

## Pre-tax vs after-tax (full sample)

Path CAGR uses taxes paid on rebalance sells. **Liq. terminal** = spendable cash if all holdings are sold on the last day (BH pays ~LT 15% then; active books already paid some tax along the way).

| Strategy | CAGR pre | CAGR after | Delta | Calmar after | Tax drag/yr | Term pre | Term path | Term after liq |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| COMBO | 8.12% | 6.06% | -2.06pp | 0.17 | 1.91% | $785,230 | $472,846 | $468,366 |
| INV_VOL | 9.21% | 8.21% | -1.00pp | 0.13 | 0.91% | $1,023,922 | $803,486 | $750,920 |
| REL_MOM | 8.61% | 6.47% | -2.14pp | 0.10 | 1.97% | $885,113 | $523,851 | $512,167 |
| MV_MOM | 8.93% | 5.33% | -3.60pp | 0.16 | 3.33% | $957,122 | $394,315 | $392,929 |
| VOL_TARGET_5050 | 8.46% | 7.13% | -1.34pp | 0.17 | 1.24% | $854,558 | $615,555 | $603,231 |
| REGIME_TILT | 8.56% | 6.95% | -1.61pp | 0.15 | 1.47% | $874,163 | $589,778 | $577,583 |
| BH_5050 | 8.05% | 8.05% | +0.00pp | 0.11 | 0.00% | $772,595 | $772,595 | $665,989 |
| STATIC_5050 | 8.32% | 7.93% | -0.38pp | 0.11 | 0.35% | $824,028 | $750,127 | $670,656 |
| STATIC_BEST | 7.95% | 7.95% | +0.00pp | 0.10 | 0.00% | $753,570 | $753,570 | $647,481 |
| STATIC_070 | 8.22% | 7.93% | -0.29pp | 0.10 | 0.27% | $805,431 | $750,035 | $664,506 |
| STATIC_100 | 7.95% | 7.95% | +0.00pp | 0.10 | 0.00% | $753,570 | $753,570 | $647,481 |
| BH_QQQ | 7.95% | 7.95% | +0.00pp | 0.10 | 0.00% | $753,570 | $753,570 | $647,481 |
| BH_SPY_VOO | 8.25% | 8.25% | +0.00pp | 0.15 | 0.00% | $810,945 | $810,945 | $703,292 |
| PLACEBO_REL_MOM | 9.10% | 5.57% | -3.53pp | 0.08 | 3.27% | $996,112 | $417,902 | $409,508 |
| LAG_REL_MOM_63 | 8.17% | 6.11% | -2.07pp | 0.09 | 1.92% | $795,585 | $478,186 | $464,286 |

## After-tax full sample (primary ranking)

| Strategy | CAGR | Vol | Sharpe | MaxDD | Calmar | U0.5 | Avg QQQ | Invested | Tax/yr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| COMBO | 6.06% | 14.70% | 0.47 | -34.93% | 0.17 | -11.40% | 43.9% | 80.6% | 1.91% |
| INV_VOL | 8.21% | 20.93% | 0.48 | -61.26% | 0.13 | -22.42% | 42.8% | 100.0% | 0.91% |
| REL_MOM | 6.47% | 21.70% | 0.40 | -67.87% | 0.10 | -27.46% | 52.8% | 99.9% | 1.97% |
| MV_MOM | 5.33% | 16.36% | 0.40 | -34.32% | 0.16 | -11.83% | 40.6% | 77.5% | 3.33% |
| VOL_TARGET_5050 | 7.13% | 14.74% | 0.54 | -41.27% | 0.17 | -13.51% | 40.8% | 81.6% | 1.24% |
| REGIME_TILT | 6.95% | 16.91% | 0.48 | -46.39% | 0.15 | -16.24% | 50.6% | 83.4% | 1.47% |
| BH_5050 | 8.05% | 22.28% | 0.46 | -70.78% | 0.11 | -27.34% | 52.6% | 100.0% | 0.00% |
| STATIC_5050 | 7.93% | 22.02% | 0.46 | -69.30% | 0.11 | -26.72% | 50.0% | 100.0% | 0.35% |
| STATIC_BEST | 7.95% | 26.44% | 0.42 | -82.96% | 0.10 | -33.53% | 100.0% | 100.0% | 0.00% |
| STATIC_070 | 7.93% | 23.62% | 0.44 | -75.57% | 0.10 | -29.86% | 70.0% | 100.0% | 0.27% |
| STATIC_100 | 7.95% | 26.44% | 0.42 | -82.96% | 0.10 | -33.53% | 100.0% | 100.0% | 0.00% |
| BH_QQQ | 7.95% | 26.44% | 0.42 | -82.96% | 0.10 | -33.53% | 100.0% | 100.0% | 0.00% |
| BH_SPY_VOO | 8.25% | 19.32% | 0.51 | -55.19% | 0.15 | -19.35% | 0.0% | 100.0% | 0.00% |
| PLACEBO_REL_MOM | 5.57% | 22.53% | 0.35 | -66.74% | 0.08 | -27.80% | 49.1% | 99.9% | 3.27% |
| LAG_REL_MOM_63 | 6.11% | 22.16% | 0.38 | -69.90% | 0.09 | -28.84% | 53.3% | 99.9% | 1.92% |

## After-tax ranking (Calmar -> CAGR -> Sharpe)

1. **COMBO** — Calmar 0.17, CAGR 6.06%, Sharpe 0.47, maxDD -34.93%, U0.5 -11.40%
2. **VOL_TARGET_5050** — Calmar 0.17, CAGR 7.13%, Sharpe 0.54, maxDD -41.27%, U0.5 -13.51%
3. **MV_MOM** — Calmar 0.16, CAGR 5.33%, Sharpe 0.40, maxDD -34.32%, U0.5 -11.83%
4. **REGIME_TILT** — Calmar 0.15, CAGR 6.95%, Sharpe 0.48, maxDD -46.39%, U0.5 -16.24%
5. **BH_SPY_VOO** — Calmar 0.15, CAGR 8.25%, Sharpe 0.51, maxDD -55.19%, U0.5 -19.35%
6. **INV_VOL** — Calmar 0.13, CAGR 8.21%, Sharpe 0.48, maxDD -61.26%, U0.5 -22.42%
7. **STATIC_5050** — Calmar 0.11, CAGR 7.93%, Sharpe 0.46, maxDD -69.30%, U0.5 -26.72%
8. **BH_5050** — Calmar 0.11, CAGR 8.05%, Sharpe 0.46, maxDD -70.78%, U0.5 -27.34%
9. **STATIC_070** — Calmar 0.10, CAGR 7.93%, Sharpe 0.44, maxDD -75.57%, U0.5 -29.86%
10. **STATIC_BEST** — Calmar 0.10, CAGR 7.95%, Sharpe 0.42, maxDD -82.96%, U0.5 -33.53%
11. **STATIC_100** — Calmar 0.10, CAGR 7.95%, Sharpe 0.42, maxDD -82.96%, U0.5 -33.53%
12. **BH_QQQ** — Calmar 0.10, CAGR 7.95%, Sharpe 0.42, maxDD -82.96%, U0.5 -33.53%
13. **REL_MOM** — Calmar 0.10, CAGR 6.47%, Sharpe 0.40, maxDD -67.87%, U0.5 -27.46%
14. **LAG_REL_MOM_63** — Calmar 0.09, CAGR 6.11%, Sharpe 0.38, maxDD -69.90%, U0.5 -28.84%

## Useful rule (after-tax full sample)

Useful = Calmar > STATIC_5050, and (Calmar or CAGR) > STATIC_070; momentum books must also beat PLACEBO_REL_MOM Calmar.

- `INV_VOL`: **YES** (CAGR 8.21%, Calmar 0.13, maxDD -61.26%)
- `REL_MOM`: **no** (CAGR 6.47%, Calmar 0.10, maxDD -67.87%)
- `MV_MOM`: **YES** (CAGR 5.33%, Calmar 0.16, maxDD -34.32%)
- `VOL_TARGET_5050`: **YES** (CAGR 7.13%, Calmar 0.17, maxDD -41.27%)
- `REGIME_TILT`: **YES** (CAGR 6.95%, Calmar 0.15, maxDD -46.39%)
- `COMBO`: **YES** (CAGR 6.06%, Calmar 0.17, maxDD -34.93%)

## After-tax major crashes (total return / max DD)

| Strategy | dotcom | gfc | euro_2011 | china_2015_16 | volmageddon_2018 | covid_crash | bear_2022 | tariff_2025 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| COMBO | -33.3%/-34.1% | -33.3%/-34.8% | -13.1%/-16.4% | -13.9%/-15.6% | -15.2%/-20.0% | -26.6%/-27.2% | -21.0%/-21.6% | -7.6%/-17.1% |
| REL_MOM | -67.0%/-67.5% | -55.3%/-56.3% | -16.2%/-16.4% | -12.5%/-14.9% | -16.5%/-22.0% | -29.4%/-30.0% | -29.1%/-29.6% | 1.3%/-21.0% |
| REGIME_TILT | -40.1%/-40.7% | -45.1%/-46.4% | -13.0%/-16.2% | -16.5%/-18.5% | -8.5%/-16.1% | -29.2%/-29.7% | -18.8%/-19.5% | -8.6%/-18.9% |
| INV_VOL | -60.9%/-61.3% | -53.3%/-53.9% | -16.1%/-16.3% | -11.9%/-14.4% | -16.6%/-20.8% | -30.7%/-31.2% | -28.8%/-29.3% | 2.0%/-20.5% |
| MV_MOM | -31.7%/-33.5% | -27.3%/-31.7% | -10.2%/-17.8% | -13.7%/-17.6% | -16.2%/-22.3% | -27.2%/-28.5% | -17.4%/-18.0% | -4.6%/-10.9% |
| VOL_TARGET_5050 | -41.0%/-41.3% | -33.2%/-34.2% | -12.9%/-15.9% | -14.5%/-15.5% | -14.1%/-18.2% | -25.6%/-26.1% | -21.0%/-21.6% | -6.5%/-18.3% |
| BH_5050 | -70.5%/-70.8% | -53.3%/-53.9% | -15.8%/-16.2% | -11.6%/-14.5% | -14.8%/-21.3% | -29.8%/-30.3% | -30.2%/-30.8% | 2.5%/-21.3% |
| STATIC_5050 | -69.0%/-69.3% | -52.9%/-53.7% | -15.9%/-16.3% | -11.6%/-14.3% | -15.4%/-21.1% | -30.5%/-31.0% | -29.4%/-30.0% | 2.0%/-20.8% |
| STATIC_070 | -75.3%/-75.6% | -52.3%/-53.4% | -14.8%/-16.2% | -11.8%/-15.0% | -14.8%/-21.8% | -29.2%/-29.8% | -31.2%/-31.8% | 2.2%/-21.6% |
| STATIC_BEST | -82.7%/-83.0% | -51.3%/-53.4% | -13.2%/-16.1% | -12.0%/-16.1% | -14.0%/-22.8% | -27.2%/-28.6% | -33.7%/-34.3% | 2.8%/-22.8% |
| BH_QQQ | -82.7%/-83.0% | -51.3%/-53.4% | -13.2%/-16.1% | -12.0%/-16.1% | -14.0%/-22.8% | -27.2%/-28.6% | -33.7%/-34.3% | 2.8%/-22.8% |
| BH_SPY_VOO | -47.2%/-47.5% | -54.8%/-55.2% | -18.5%/-18.8% | -11.1%/-12.8% | -15.8%/-19.5% | -33.7%/-34.0% | -24.1%/-24.5% | 2.0%/-18.7% |

## After-tax holdout from 2018

| Strategy | CAGR | Sharpe | MaxDD | Calmar | Tax/yr |
|---|---:|---:|---:|---:|---:|
| COMBO | 9.43% | 0.62 | -27.19% | 0.35 | 2.55% |
| INV_VOL | 15.60% | 0.80 | -31.21% | 0.50 | 1.39% |
| REL_MOM | 15.05% | 0.76 | -30.82% | 0.49 | 2.40% |
| MV_MOM | 8.45% | 0.54 | -29.98% | 0.28 | 3.99% |
| VOL_TARGET_5050 | 10.55% | 0.71 | -26.15% | 0.40 | 1.91% |
| REGIME_TILT | 13.83% | 0.80 | -29.74% | 0.47 | 2.24% |
| BH_5050 | 17.84% | 0.86 | -31.15% | 0.57 | 0.00% |
| STATIC_5050 | 16.52% | 0.83 | -30.99% | 0.53 | 0.73% |
| STATIC_BEST | 19.90% | 0.88 | -35.12% | 0.57 | 0.00% |
| STATIC_070 | 17.73% | 0.85 | -32.31% | 0.55 | 0.56% |
| STATIC_100 | 19.90% | 0.88 | -35.12% | 0.57 | 0.00% |
| BH_QQQ | 19.90% | 0.88 | -35.12% | 0.57 | 0.00% |
| BH_SPY_VOO | 14.72% | 0.81 | -33.99% | 0.43 | 0.00% |
| PLACEBO_REL_MOM | 12.90% | 0.68 | -30.15% | 0.43 | 3.49% |
| LAG_REL_MOM_63 | 14.88% | 0.75 | -30.68% | 0.48 | 2.27% |

## Pre-tax full sample (reference)

| Strategy | CAGR | MaxDD | Calmar |
|---|---:|---:|---:|
| MV_MOM | 8.93% | -32.41% | 0.28 |
| COMBO | 8.12% | -34.20% | 0.24 |
| REGIME_TILT | 8.56% | -45.48% | 0.19 |
| REL_MOM | 8.61% | -65.28% | 0.13 |
| BH_5050 | 8.05% | -70.78% | 0.11 |
| STATIC_5050 | 8.32% | -69.13% | 0.12 |
| STATIC_070 | 8.22% | -75.46% | 0.11 |
| BH_QQQ | 7.95% | -82.96% | 0.10 |

## Verdict

Tax model ST 24%/LT 15% at ~$150k income. After-tax path Calmar leader: **COMBO**. MV_MOM pre 8.93% -> after 5.33% (-3.60pp), Calmar 0.16. COMBO pre 8.12% -> after 6.06% (-2.06pp), Calmar 0.17. REGIME_TILT pre 8.56% -> after 6.95% (-1.61pp), Calmar 0.15. BH_5050 pre 8.05% -> after 8.05% (+0.00pp), Calmar 0.11. STATIC_5050 pre 8.32% -> after 7.93% (-0.38pp), Calmar 0.11. BH_QQQ pre 7.95% -> after 7.95% (+0.00pp), Calmar 0.10. Largest path-CAGR tax hit: **MV_MOM** -3.60pp (drag ~3.33%/yr). Least path hit: **BH_5050** +0.00pp. Best end liquidation (spendable) terminal: **INV_VOL** $750,920 (BH pays ~15% LT on embedded gains at exit). Useful after tax: INV_VOL, MV_MOM, VOL_TARGET_5050, REGIME_TILT, COMBO.

## Limitations

- Tax model: federal ST 24% / LT 15% only; no state tax, NIIT, or wash-sale rules.
- Assumes ~$150k taxable income stays in the 15% LTCG band.
- Losses do not create a tax credit (tax = 0 on losing sells).
- T-bill interest untaxed; path CAGR for BH defers tax until liquidation column.
- Two-asset universe only; crash protection is cash/tilt only.
