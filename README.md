# Trading

Systematic backtesting research for trend-following strategies on leveraged ETFs.

## Strategy

The flagship design is **Trend Trio**: a three-sleeve portfolio sized by volatility targeting and gated by 200-day SMA regimes (with 2% hysteresis). Signals are computed on underlying ETFs, not the leveraged products.

| Sleeve | Signal source | Risk budget | Leverage |
|--------|---------------|-------------|----------|
| TQQQ   | QQQ           | 60%         | 3×       |
| TMF    | TLT           | 20%         | 3×       |
| UGL    | GLD           | 20%         | 2×       |

Sizing: `weight = gate × budget × 50% / (leverage × EWMA vol)`, total capped at 100%. Rebalance only when a sleeve moves more than 5 percentage points. T+1 execution, ~20 trades per year.

Earlier research rounds explored TQQQ-only variants (OLS trend scoring, vol kill-switches, parameter grids) before converging on Trend Trio. Round 4 (`research/v3_extended/`) extended the book to six sleeves with ensemble momentum gates; that variant is what trades live. See `docs/INFRASTRUCTURE.md` for live-deployment planning.

## Live paper trading

`live/trade.py` trades the six-sleeve cash-account strategy (variant R50d in `research/v3_extended/v3_revised.py` — the cash book of `v3_cash.py` with EFA dropped, a managed-futures trend sleeve added, and the vol target stepped down to 50%) on an Alpaca paper account:

| Sleeve  | Signal source | Risk budget | Leverage |
|---------|---------------|-------------|----------|
| TQQQ    | QQQ           | 40%         | 3× (in fund) |
| TMF     | TLT           | 15%         | 3× (in fund) |
| UGL     | GLD           | 15%         | 2× (in fund) |
| DBC     | DBC           | 10%         | 1× |
| DBMF    | DBMF          | 10%         | 1× |
| BTC/USD | BTC/USD       | 10%         | 1× |

Rules: each sleeve is gated by ensemble momentum votes (63/126/252-day trailing return > 0), sized by `weight = gate × budget × 50% / (leverage × EWMA vol)`, total capped at 100% of equity (no margin borrowing — unlevered sleeves buy the same risk contribution with more capital). Rebalance only when a sleeve drifts more than 5 points from target.

Each run recomputes targets from adjusted Alpaca daily closes and reads held weights from broker positions — the ensemble gate is stateless, so there is no state file. Equity orders are notional market DAY orders submitted after the close, which Alpaca queues for the next open (T+1, matching the backtest). Safety rails: stale/absurd-data checks, an unexpected-position reconciliation halt, a 70%-of-equity order ceiling, an open-order duplicate-run guard, and a `HALT` file kill switch at the repo root.

`.github/workflows/daily-trade.yml` runs it at 6 PM Eastern every weekday. GitHub cron is UTC-only, so two crons fire (22:00 and 23:00 UTC) and a guard step proceeds only when it is actually 6 PM in New York — DST-proof. Setup: add `ALPACA_KEY` and `ALPACA_SECRET` (paper keys) as repository secrets.

Every run emails a summary (signals table, orders, equity) via Gmail SMTP, and a crash sends a failure alert — silence means the schedule itself is broken. To enable, add `GMAIL_USER` and `GMAIL_APP_PASSWORD` secrets (create an app password at Google Account → Security → 2-Step Verification → App passwords; recipients are configured in `live/trade.py`). Without these secrets the run still trades and just skips the email.

Test locally with:

```bash
poetry run python live/trade.py --dry-run
```

## Repository layout

```
trading/
├── research/
│   ├── v1_tqqq/          # Round 1–2: TQQQ-only strategies and variant grids
│   ├── v2_trend_trio/    # Round 3: Trend Trio (source of truth for live rules)
│   └── v3_extended/      # Round 4: extended sleeves (EFA, DBC, BTC)
├── live/                 # Daily Alpaca paper-trading engine
├── output/               # Backtest JSON artifacts
├── tools/                # Utilities (report generator)
├── docs/                 # Deployment and infrastructure notes
├── setup.sh              # Bootstrap (macOS / Linux)
├── setup.cmd             # Bootstrap (Windows — preferred)
└── setup.ps1             # Bootstrap (Windows — called by setup.cmd)
```

| Folder / file | Contents |
|---------------|----------|
| `research/v1_tqqq/` | `backtest.py` (original RFT-VT), `variants.py`, `variants2.py`, `final.py` (simplified TQQQ winner) |
| `research/v2_trend_trio/` | `v2_research.py` (Trend Trio backtest), `v2_tune.py` (parameter sensitivity) |
| `research/v3_extended/` | `v3_research.py` (quint/six-sleeve extensions), `v3_cash.py` (cash-account revision — live rules), `v3_revised.py` (σ\*=50% + managed-futures sleeve candidates vs live config) |
| `live/` | `trade.py` — daily signal + order engine (see “Live paper trading”) |
| `output/` | `results.json`, `final_results.json`, `v2_results.json`, `v3_results.json`, `canvas_data.json` |
| `tools/` | `generate_report.py` — interactive Plotly HTML from result JSON |
| `docs/` | `INFRASTRUCTURE.md` — broker options, signal engine design, scheduling |

## Setup

| OS | Command |
|----|---------|
| Windows | `setup.cmd` |
| macOS / Linux | `./setup.sh` |

Then run a backtest:

```bash
poetry run python research/v2_trend_trio/v2_research.py
```

Generate an interactive HTML report from a result file:

```bash
poetry run python tools/generate_report.py output/v3_results.json -o report.html
```

Omit `-o` to write `<input_stem>_report.html` next to the JSON.

Research artifacts only — not investment advice.
