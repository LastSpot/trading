# V11 VOO/QQQ Allocation — Design (refined)

Optimize a two-asset book (VOO + QQQ; SPY spliced before VOO inception).
No other assets. Goal: maximize compound return while controlling risk
(drawdown / vol). No hard pass thresholds.

**Primary evaluation window:** full sample from QQQ inception after signal
warmup (`1999-03-10` data start → metrics from first valid signal date,
~2000-03). This window includes every major crash below.

## Universe and execution

- Assets: VOO (SPY before VOO inception), QQQ, idle cash at `^IRX` / 252
- Signal frequency: monthly (month-end), T+1 execution
- Cost: 5 bps per unit turnover
- Notional cap: 100% of equity (cash account)
- Rebalance: month-end target reset

## Retail tax (after-tax layer)

Model a taxable retail account at roughly **$150k** ordinary income
(US federal only; no state, no NIIT). Report **pre-tax** and **after-tax**.

Assumed rates (2025-style):

| Holding period | Rate | Why |
|---|---:|---|
| ≤ 1 year (short-term) | **24%** | Ordinary-income bracket near $150k |
| > 1 year (long-term) | **15%** | LTCG 15% band for ~$150k taxable income |

Mechanics:

- FIFO **tax lots** per sleeve (VOO/SPY, QQQ). Each buy opens a lot
  `{basis, market_value, buy_date}`.
- On a sell, consume oldest lots first. For each lot:
  `tax += rate(lot_age) × max(proceeds − basis, 0)`.
- Losing lots: tax = 0 (no loss-harvest credit in this layer).
- Weights drift between month-end rebalances; tax only on trade days.
- Buy-and-hold deploys once: **no in-path tax**; unrealized gains are
  mostly long-term. Also report **end liquidation** after-tax terminal
  (sell everything on the last day at lot-appropriate ST/LT rates) so BH
  is comparable on spendable wealth, not just deferred tax.
- No tax on T-bill interest in this layer.

## Benchmarks (fixed)

1. `BH_5050` — **true** 50/50 VOO/QQQ buy-and-hold: deploy once, never
   rebalance (weights drift with relative performance)
2. `BH_QQQ` — QQQ buy-and-hold
3. `BH_SPY_VOO` — SPY/VOO buy-and-hold
4. `STATIC_5050` — 50/50 **monthly rebalance** (not BH)
5. `STATIC_070` — 70/30 QQQ/VOO monthly rebalance
6. `STATIC_100` — 100% QQQ monthly
7. `STATIC_BEST` — best static `w_QQQ ∈ {0,0.1,…,1}` by Calmar on
   **select window** `2003-01-01` → `2012-12-31` (post-dotcom trough through
   GFC + recovery). Locked thereafter.

## Candidates (params unchanged from first pass)

### 1. `INV_VOL`
`w_i ∝ 1 / σ_i` (EWMA span 63), fully invested.

### 2. `REL_MOM`
`m = R_QQQ(126) − R_VOO(126)`; `w_QQQ = clip(0.5 + 2·m, 0, 1)`.

### 3. `MV_MOM`
`μ` = annualized 126d return (cap ±40%); `Σ` = EWMA cov + 0.3 diagonal
shrink; `w ∝ Σ⁻¹μ` long-only; all `μ≤0` → cash.

### 4. `VOL_TARGET_5050`
50/50 scaled to 15% port vol, `scale ≤ 1`, residual T-bills.

### 5. `REGIME_TILT`
QQQ SMA200 ±2% hysteresis: risk-on 70/30 QQQ/VOO; risk-off 0/40 + 60% cash.

### 6. `COMBO`
REL_MOM base; ×0.5 equity if QQQ < SMA200×0.98; scale to 18% vol (`≤1`).

## Diagnostics

- `PLACEBO_REL_MOM` — random ±1 monthly signs on relative momentum
- `LAG_REL_MOM_63` — lag relative momentum 63 sessions

## Evaluation windows

| Window | Role |
|---|---|
| `full` | **Primary** — after 252d warmup through latest |
| `from_2010` | post-GFC era |
| `from_2015` | recent |
| `holdout_from_2018` | locked OOS (secondary) |
| `last_24m` | recent stress |
| `select_2003_2012` | STATIC_BEST selection only |

## Major crashes (peak→trough style windows)

| Id | Window | Event |
|---|---|---|
| `dotcom` | 2000-03-24 → 2002-10-09 | Nasdaq / tech bust |
| `gfc` | 2007-10-09 → 2009-03-09 | Global financial crisis |
| `euro_2011` | 2011-04-29 → 2011-10-03 | Eurozone / US downgrade |
| `china_2015_16` | 2015-08-10 → 2016-02-11 | China deval / oil smash |
| `volmageddon_2018` | 2018-01-26 → 2018-12-24 | Vol spike + Q4 bear |
| `covid_crash` | 2020-02-19 → 2020-03-23 | COVID liquidation |
| `bear_2022` | 2022-01-03 → 2022-10-12 | Inflation / rates bear |
| `tariff_2025` | 2025-02-15 → 2025-06-30 | Tariff drawdown |

For each crash report total return and max DD for all books.

## Ranking (on `full`)

1. Calmar, then CAGR, then Sharpe
2. Also report utility `U_λ = CAGR − λ·|MaxDD|` for `λ ∈ {0.5, 1.0}`
3. **Useful** on `full` only if:
   - Calmar > `STATIC_5050`, **and**
   - Calmar > `STATIC_070` **or** CAGR > `STATIC_070`
   - and Calmar > `PLACEBO_REL_MOM` when the candidate is momentum-based
     (`REL_MOM`, `COMBO`)
