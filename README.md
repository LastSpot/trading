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

Earlier research rounds explored TQQQ-only variants (OLS trend scoring, vol kill-switches, parameter grids) before converging on Trend Trio. See `docs/INFRASTRUCTURE.md` for live-deployment planning.

## Repository layout

```
trading/
├── research/
│   ├── v1_tqqq/          # Round 1–2: TQQQ-only strategies and variant grids
│   ├── v2_trend_trio/    # Round 3: Trend Trio (source of truth for live rules)
│   └── v3_extended/      # Round 4: extended sleeves (EFA, DBC, BTC)
├── output/               # Backtest JSON artifacts
├── docs/                 # Deployment and infrastructure notes
├── setup.sh              # Bootstrap (macOS / Linux)
├── setup.cmd             # Bootstrap (Windows — preferred)
└── setup.ps1             # Bootstrap (Windows — called by setup.cmd)
```

| Folder / file | Contents |
|---------------|----------|
| `research/v1_tqqq/` | `backtest.py` (original RFT-VT), `variants.py`, `variants2.py`, `final.py` (simplified TQQQ winner) |
| `research/v2_trend_trio/` | `v2_research.py` (Trend Trio backtest), `v2_tune.py` (parameter sensitivity) |
| `research/v3_extended/` | `v3_research.py` (quint/six-sleeve extensions) |
| `output/` | `results.json`, `final_results.json`, `v2_results.json`, `v3_results.json`, `canvas_data.json` |
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

Research artifacts only — not investment advice.
