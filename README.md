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

Earlier research rounds explored TQQQ-only variants (OLS trend scoring, vol kill-switches, parameter grids) before converging on Trend Trio. Round 4 (`research/v3_extended/`) extended the book to six sleeves with ensemble momentum gates; that variant (R50d) is what trades live. Later rounds (`research/v4_improve/`, `research/v5_retail/`) stress-test risk overlays and return-first tweaks against that live baseline — none of those have replaced live yet. See `docs/INFRASTRUCTURE.md` for live-deployment planning.

## Live paper trading

Named Alpaca accounts run through a shared engine. Strategy logic lives in
`live/strategies/`; account identity (paper vs live, credential env names) in
`live/accounts.py`. Entry point:

```bash
poetry run python live/run.py --account paper_r50d
poetry run python live/run.py --account paper_r50d --dry-run
# compat shim (same as paper_r50d):
poetry run python live/trade.py --dry-run
```

**Current paper book** (`paper_r50d` → strategy `r50d`): six-sleeve cash-account
variant R50d from `research/v3_extended/v3_revised.py` (v3_cash variant F with
EFA dropped, managed-futures sleeve added, σ\*=50%):

| Sleeve  | Signal source | Risk budget | Leverage |
|---------|---------------|-------------|----------|
| TQQQ    | QQQ           | 40%         | 3× (in fund) |
| TMF     | TLT           | 15%         | 3× (in fund) |
| UGL     | GLD           | 15%         | 2× (in fund) |
| DBC     | DBC           | 10%         | 1× |
| DBMF    | DBMF          | 10%         | 1× |
| BTC/USD | BTC/USD       | 10%         | 1× |

Rules: each sleeve is gated by ensemble momentum votes (63/126/252-day trailing return > 0), sized by `weight = gate × budget × 50% / (leverage × EWMA vol)`, total capped at 100% of equity (no margin borrowing — unlevered sleeves buy the same risk contribution with more capital). Rebalance only when a sleeve drifts more than 5 points from target.

Each run recomputes targets from adjusted Alpaca daily closes and reads held weights from broker positions — the ensemble gate is stateless, so there is no state file. Equity orders are notional market DAY orders submitted after the close, which Alpaca queues for the next open (T+1, matching the backtest). Safety rails: stale/absurd-data checks, an unexpected-position reconciliation halt, a 70%-of-equity order ceiling, an open-order duplicate-run guard, a global `HALT` file at the repo root, and per-account halt files under `live/halts/<account_id>`.

### Credentials

| Env var | Purpose |
|---------|---------|
| `ALPACA_PAPER_R50D_KEY` / `ALPACA_PAPER_R50D_SECRET` | Paper account for `paper_r50d` |
| `ALPACA_PAPER_R50D_ASYM_KEY` / `ALPACA_PAPER_R50D_ASYM_SECRET` | Paper A/B for `paper_r50d_asym` (asymmetric TQQQ gate) |
| `ALPACA_KEY` / `ALPACA_SECRET` | Legacy fallback for `paper_r50d` only (soft cutover) |
| `GMAIL_USER` / `GMAIL_APP_PASSWORD` | Optional daily email summary |

Add repository secrets with the same names for CI. Prefer the `ALPACA_PAPER_R50D_*` names going forward.

### Adding another paper account

1. Implement (or reuse) a strategy module under `live/strategies/` and register it in `live/strategies/__init__.py`.
2. Add an `Account(...)` entry in `live/accounts.py` with its own `ALPACA_PAPER_*` env names.
3. Append the account id to the matrix in `.github/workflows/daily-trade.yml` and wire the new secrets into the job `env`.

### Promoting paper → live money

1. Add a `paper=False` account in `live/accounts.py` (e.g. `live_r50d`) pointing at the same strategy and `ALPACA_LIVE_*` secrets.
2. Run with `poetry run python live/run.py --account live_r50d --confirm-live` (required for real-money submits; `--dry-run` still works without it).
3. Use a **separate** GitHub workflow for live — do not add live accounts to the paper matrix.

`.github/workflows/daily-trade.yml` runs the paper matrix after 6 PM Eastern every weekday. One cron fires at 23:00 UTC (6 PM EST / 7 PM EDT — always at or after 6 PM ET). A guard requires New York hour ≥ 18 so GitHub’s common schedule delay still trades; manual dispatch always runs. Each matrix account has its own concurrency group.

Every run emails a summary (signals table, orders, equity) via Gmail SMTP, and a crash sends a failure alert — silence means the schedule itself is broken. Recipients are configured in `live/engine.py`. Without Gmail secrets the run still trades and just skips the email. Sends try SSL:465 then STARTTLS:587, with up to 3 attempts and backoff, so a single Gmail timeout (common on GitHub Actions) does not drop the summary.
## Repository layout

```
trading/
├── research/
│   ├── v1_tqqq/          # Round 1–2: TQQQ-only strategies and variant grids
│   ├── v2_trend_trio/    # Round 3: Trend Trio
│   ├── v3_extended/      # Round 4: extended sleeves → live R50d
│   ├── v4_improve/       # Continuous TSMOM / cov-aware sizing experiments
│   └── v5_retail/        # Retail risk overlays + return-first DD-brake sweeps
├── live/                 # Multi-account Alpaca engine (paper + promote-to-live)
│   ├── run.py            # CLI: --account <id> [--dry-run] [--confirm-live]
│   ├── engine.py         # Shared broker I/O, safety rails, email
│   ├── accounts.py       # Named account registry
│   ├── strategies/       # Strategy modules (r50d, …)
│   ├── halts/            # Per-account kill switches
│   └── trade.py          # Shim → paper_r50d
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
| `research/v3_extended/` | `v3_research.py` (quint/six-sleeve extensions), `v3_cash.py` (cash-account revision), `v3_revised.py` (σ\*=50% + MF sleeve → live R50d) |
| `research/v4_improve/` | Continuous TSMOM, covariance scaling, growth-tilt experiments on R50d |
| `research/v5_retail/` | `v5_retail_risk.py` (vol/chop/corr/DD overlays), `v5_r50dd_sweep.py` (return-first DD-brake grid), `v5_soft_dd.py` (soft DD brake on live R50d), `v5_long_horizon.py` (long-window / 5-sleeve twins) |
| `live/` | Multi-account engine: `run.py`, `engine.py`, `accounts.py`, `strategies/r50d.py` (see “Live paper trading”) |
| `output/` | Backtest JSON artifacts (`v3_revised_results.json`, `v5_*_results.json`, …) |
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
