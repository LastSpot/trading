# Trend Trio — Live Deployment Infrastructure

Deployment guide for the systematic strategy backtested in this folder
(`v2_research.py` = the Trend Trio backtest, `final.py` = the TQQQ-only variant).

**Strategy recap:** three sleeves — TQQQ (signals on QQQ, 60% risk budget),
TMF (signals on TLT, 20%), UGL (signals on GLD, 20%) — each gated by its
underlying's 200-day SMA with a 2% hysteresis band, sized by
`weight = budget x 50% / (leverage x EWMA vol)`, total capped at 100%,
rebalanced only when a sleeve's target moves more than 5 points. Idle cash
in SGOV or BIL. ~20–25 order events per year.

---

## 1. The broker question: should you leave Fidelity?

**Short answer: don't move your portfolio. Open one additional account for
the strategy capital, OR keep everything at Fidelity and run semi-automated.**

Fidelity has **no public trading API**. There is no supported way for a script
to place orders in a Fidelity account. (Third-party screen-scraping libraries
exist; do not use them — they break silently, violate the ToS, and a silent
failure in an automated leveraged-ETF system is exactly the catastrophic risk
this strategy exists to avoid.)

Because the strategy only trades ~2x/month, you have three legitimate options:

### Option A — Stay at Fidelity, semi-automated (lowest effort, zero migration)

The script runs daily, computes the target weights, and **notifies you only
when an order is needed** (email/text/push). You place the order yourself in
the Fidelity app in under two minutes, ~23 times per year.

- Pros: no new account, no API keys to secure, you keep Fidelity's platform,
  works with your existing IRA if you have one there.
- Cons: you are the execution layer. The failure mode is *you* — skipping the
  exit order during a crash because it "feels wrong" is precisely the
  discipline failure that destroys trend followers. Be honest with yourself
  about whether you'll click sell in October-2008 conditions.

### Option B — Alpaca for the strategy sleeve (simplest full automation)

[Alpaca](https://alpaca.markets) is an API-first, commission-free broker.
Clean REST API, official Python SDK (`alpaca-py`), first-class **paper-trading
environment** with identical API semantics.

- Pros: easiest API in the industry; paper account for free rehearsal;
  fractional shares; commission-free.
- Cons: taxable brokerage accounts only for most retail users (no IRA) —
  which matters, see §7. A smaller broker than Fidelity/IBKR.

### Option C — Interactive Brokers (most robust, supports IRA automation)

IBKR has the most battle-tested retail API (Web API or TWS API via
`ib_async`). Crucially, **IBKR offers IRAs, and the API works in IRA
accounts** — the only mainstream way to fully automate this strategy in a
tax-sheltered wrapper.

- Pros: IRA + API in one place; institutional-grade execution; IBKR Lite is
  commission-free for US ETFs.
- Cons: the API is more complex (gateway process, sessions, pacing rules);
  steeper learning curve than Alpaca.

### Recommendation

| Your situation | Choice |
|---|---|
| Want minimum change, trust your own discipline | **A** — Fidelity + signal-only automation |
| Taxable account, want true hands-off | **B** — Alpaca with the strategy capital only |
| Can run this inside an IRA (best after-tax result) | **C** — IBKR IRA |

A sane path: start with **Option A for 1–3 months** (the signal engine is
identical in all three options), then move the strategy sleeve to Alpaca/IBKR
once you trust the pipeline.

---

## 2. Architecture

Deliberately boring. One machine, one scheduled job, one state file.

```
                    (16:05 ET, after market close, Mon-Fri)
                                   |
                             signal_engine.py
                                   |
    1. Fetch EOD adjusted closes: QQQ, TLT, GLD, ^IRX   (yfinance, Stooq fallback)
    2. Data sanity checks (dates fresh, no NaN, |move| < 25%)
    3. Compute per sleeve: SMA200 gate (2% hysteresis), EWMA vol (lambda=0.94)
    4. Target weights: w_i = gate_i * b_i * 0.50 / (L_i * sigma_i); cap sum at 1.0
    5. Compare vs held positions (state.json): trade only if |delta| > 5 pts
                                   |
              +--------------------+--------------------+
              |                                         |
        [Option A]                              [Options B/C]
        Email/SMS/push:                         Submit orders via broker API
        "SELL 120 TQQQ, BUY 85 TMF"             for next session (or MOC),
        You execute at Fidelity                 then reconcile fills next run
              |                                         |
              +--------------------+--------------------+
                                   |
    6. Write state.json (positions, date, signal values), append to trades.log
    7. Always send a daily heartbeat ("ran OK, no trades") -- silence = broken
```

**Key design rules**

- **State lives in a file** (`state.json`), not in your head or the broker UI.
  The hysteresis gate is stateful — the script must remember yesterday's gate.
- **Heartbeat every day, not just on trades.** A cron job that dies stays dead
  silently. If you don't get the "ran OK" message by 16:30 ET, investigate.
- **Reconcile before trading** (B/C): fetch actual broker positions at the
  start of each run and compare to `state.json`. Mismatch → alert and halt,
  never trade on stale state.
- **T+1 execution matches the backtest.** Signals from tonight's close are
  traded tomorrow (market-on-close preferred, market-on-open acceptable).
  The backtest assumed exactly this, so don't "improve" it by rushing.

---

## 3. Machine: where the script runs

| Host | Cost | Notes |
|---|---|---|
| **GitHub Actions** (recommended) | Free | Scheduled workflow, no server to maintain, secrets manager built in, runs even when your Mac is asleep. Commit `state.json` back to a **private** repo after each run. |
| Your Mac + `launchd` | Free | Fine for Option A. Fragile if the laptop is closed at 16:05. Use `launchd` (not cron) so a missed run fires on wake. |
| Cheap VPS (Hetzner/DO, ~$5/mo) | ~$60/yr | Always-on cron; only worth it if you dislike GitHub Actions. |

Compute is trivial — three SMAs and three EWMAs on daily data is under one
second of CPU. **Do not** build anything real-time; there is nothing to react
to intraday, and per the original design, intraday is a game reserved for HFT
capital you don't want to compete with.

---

## 4. Setup, step by step

### 4.1 Project skeleton

```
~/Documents/trades/
├── INFRASTRUCTURE.md        # this file
├── v2_research.py           # backtest (source of truth for the rules)
├── live/
│   ├── signal_engine.py     # daily job (skeleton in §5)
│   ├── state.json           # held weights + gate states  (gitignore if public!)
│   ├── trades.log           # append-only audit trail
│   └── requirements.txt     # yfinance, pandas, numpy (+ alpaca-py or ib_async)
└── .env                     # API keys -- NEVER commit this
```

```bash
python3 -m venv ~/Documents/trades/live/.venv
source ~/Documents/trades/live/.venv/bin/activate
pip install yfinance pandas numpy
# Option B:  pip install alpaca-py
# Option C:  pip install ib_async
```

### 4.2 Notifications (all options)

Simplest reliable channels, pick one:

- **Email via SMTP** (Gmail app-password) — 10 lines of `smtplib`.
- **Pushover** (~$5 one-time) — real push notifications, dead simple API.
- **Telegram bot** — free, instant, 5 lines with `requests`.

### 4.3 Broker setup (Option B — Alpaca)

1. Open account at alpaca.markets → get **paper** API key + secret first.
2. Put keys in `.env` (`ALPACA_KEY`, `ALPACA_SECRET`), load with `os.environ`.
3. Run against the paper endpoint (`paper-api.alpaca.markets`) for 1–3 months.
4. Go live by swapping the base URL and keys. Nothing else changes.

### 4.4 Broker setup (Option C — IBKR)

1. Open IBKR account (IRA if applicable), subscribe to IBKR Lite.
2. Easiest API path: **Client Portal Web API** (REST, OAuth) or run the
   **IB Gateway** headless with `ib_async`.
3. IBKR has a paper-trading mode too — same 1–3 month rehearsal rule.

### 4.5 Scheduling (GitHub Actions example)

```yaml
# .github/workflows/daily.yml  (private repo!)
name: trend-trio-daily
on:
  schedule:
    - cron: "10 21 * * 1-5"   # 21:10 UTC ~ 16:10 ET (adjust for DST or run twice)
  workflow_dispatch: {}        # manual trigger button
jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -r live/requirements.txt
      - run: python live/signal_engine.py
        env:
          ALPACA_KEY: ${{ secrets.ALPACA_KEY }}
          ALPACA_SECRET: ${{ secrets.ALPACA_SECRET }}
          PUSHOVER_TOKEN: ${{ secrets.PUSHOVER_TOKEN }}
      - run: |                 # persist state
          git config user.name bot && git config user.email bot@local
          git add live/state.json live/trades.log
          git commit -m "state $(date -u +%F)" || true
          git push
```

Note: GitHub cron fires in UTC and doesn't shift for US daylight saving —
either schedule two crons (21:10 and 22:10 UTC) with an in-script market-hours
check, or accept a 1-hour drift twice a year (harmless for T+1 execution).

---

## 5. Signal engine skeleton

The exact math from the backtest, restructured for daily incremental runs.
Port the constants from `v2_research.py` — they must match.

```python
#!/usr/bin/env python3
"""Trend Trio daily signal engine. Rules identical to v2_research.py."""
import json, os, sys, datetime as dt
import numpy as np, pandas as pd, yfinance as yf

SLEEVES = {          # ticker_traded: (ticker_signal, leverage, risk_budget)
    "TQQQ": ("QQQ", 3.0, 0.60),
    "TMF":  ("TLT", 3.0, 0.20),
    "UGL":  ("GLD", 2.0, 0.20),
}
SIGMA_TARGET = 0.50      # portfolio vol target
HYSTERESIS   = 0.02      # SMA200 exit band
EWMA_LAMBDA  = 0.94
TRADE_BAND   = 0.05      # min weight change to trade
STATE_FILE   = os.path.join(os.path.dirname(__file__), "state.json")

def load_state():
    if os.path.exists(STATE_FILE):
        return json.load(open(STATE_FILE))
    return {"held": {t: 0.0 for t in SLEEVES}, "gate": {t: 0 for t in SLEEVES},
            "last_run": None}

def fetch(signal_tickers):
    px = yf.download(signal_tickers, period="2y", auto_adjust=True,
                     progress=False)["Close"].dropna()
    # --- sanity gates: refuse to signal on bad data ---
    assert px.index[-1].date() >= dt.date.today() - dt.timedelta(days=4), "stale data"
    last_moves = px.pct_change().iloc[-1].abs()
    assert (last_moves < 0.25).all(), f"absurd move, check data: {dict(last_moves)}"
    return px

def main():
    state = load_state()
    px = fetch([s[0] for s in SLEEVES.values()])
    targets, diags = {}, {}
    for traded, (sig, lev, budget) in SLEEVES.items():
        p = px[sig]
        sma = p.rolling(200).mean().iloc[-1]
        gate = state["gate"][traded]
        if p.iloc[-1] > sma:                      gate = 1
        elif p.iloc[-1] < (1 - HYSTERESIS) * sma: gate = 0
        var = (p.pct_change().dropna() ** 2).ewm(alpha=1 - EWMA_LAMBDA,
                                                 adjust=False).mean().iloc[-1]
        sigma = float(np.sqrt(252 * var))
        w = gate * budget * SIGMA_TARGET / (lev * sigma)
        targets[traded] = w
        state["gate"][traded] = gate
        diags[traded] = {"gate": gate, "vol": round(sigma, 3), "w_raw": round(w, 3)}
    tot = sum(targets.values())                   # cap at 100% invested
    if tot > 1.0:
        targets = {k: v / tot for k, v in targets.items()}
    orders = {}
    for t, w in targets.items():
        held = state["held"][t]
        if abs(w - held) > TRADE_BAND or (w == 0.0 and held != 0.0):
            orders[t] = {"from": round(held, 3), "to": round(w, 3)}
            state["held"][t] = w
    state["last_run"] = str(dt.date.today())
    json.dump(state, open(STATE_FILE, "w"), indent=1)
    notify(orders, targets, diags)   # implement: Pushover/Telegram/SMTP
    # Option B/C: also place orders via broker API here (or notify-only for A)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        notify_failure(str(e))       # ALWAYS alert on failure -- silence is deadly
        sys.exit(1)
```

For Options B/C, convert weight deltas to share quantities with
`shares = round(delta_w * account_equity / last_price)` using the broker's
reported equity, and submit market-on-close (Alpaca: `order_class` MOC;
IBKR: `MOC` order type) or next-open market orders.

---

## 6. Safety rails (non-negotiable)

1. **Paper-trade 1–3 months minimum.** Compare every paper signal to a manual
   run of the backtest code on the same dates. They must agree exactly.
2. **Daily heartbeat** even when no trade. No message = assume broken.
3. **Position reconciliation** (B/C): broker positions must match
   `state.json` before any order is placed. Mismatch → halt + alert.
4. **Data sanity gates** (in skeleton): stale date or a >25% one-day "move"
   (usually a split/API glitch) → no signal, alert instead. yfinance breaks a
   few times a year; a Stooq or broker-data fallback is worth adding.
5. **Order-size ceiling:** refuse any single order > 70% of equity. The
   largest legitimate single move is a full gate-close of the TQQQ sleeve.
6. **Kill switch:** a `HALT` file in the repo/folder that the script checks
   first — lets you stop trading from your phone via a one-line commit.
7. **Audit log:** append every run's signals, targets, and orders to
   `trades.log`. When something looks wrong, the log answers "what did it
   know and when."
8. **No manual overrides.** If you must intervene (fund emergency), halt the
   system, flatten manually, restart from a clean reconciled state. Never
   "adjust" a live position the script doesn't know about.

---

## 7. Taxes and account placement (bigger than all infra costs combined)

- ~23 order events/yr → mostly **short-term capital gains** in a taxable
  account. At a 30%+ marginal rate this eats roughly a third of the edge over
  buy-and-hold QQQ.
- **Best structure: run the strategy inside an IRA.** Fidelity IRA works for
  Option A (manual execution). **IBKR IRA (Option C) is the only mainstream
  full-automation + IRA combination.**
- If taxable is the only option, the strategy still dominates holding TQQQ
  outright on after-tax survival terms — but expect the realized edge vs
  plain QQQ to compress.
- Keep the strategy in its **own account** regardless of broker. Mixing
  discretionary holdings with a systematic book breaks reconciliation and
  invites tinkering.

---

## 8. Running costs

| Item | Cost |
|---|---|
| Market data (yfinance / Stooq / broker EOD) | $0 |
| Compute (GitHub Actions or launchd) | $0 |
| Commissions (Fidelity / Alpaca / IBKR Lite) | $0 |
| Spread + slippage (~4.5x turnover on penny-wide ETFs) | ~$30–50/yr per $100k |
| Notifications (Pushover one-time / Telegram free) | ~$0–5 |
| Fund expense ratios (~0.9–1.0%, embedded in ETF prices) | already in backtest |
| **Total explicit infra** | **≈ $50/yr** |

---

## 9. Go-live checklist

- [ ] Port signal logic; verify outputs match `v2_research.py` targets on the
      last 60 trading days, bar for bar
- [ ] Notifications firing (test both the trade alert and the failure alert)
- [ ] Scheduler runs at 16:05–16:15 ET and survives a laptop-asleep /
      runner-outage day (check the missed-run behavior)
- [ ] `state.json` persistence + reconciliation tested (B/C)
- [ ] Paper traded ≥ 1 month with zero unexplained divergences
- [ ] `HALT` kill switch tested
- [ ] Strategy capital in its own account (IRA if possible)
- [ ] Position sizes double-checked against the order-size ceiling
- [ ] You have pre-committed, in writing, to take every exit signal

---

*Backtests in this folder: `final.py` (TQQQ-only RG-VT, 2010–2026),
`v2_research.py` (Trend Trio with synthetic 3x history, 1999–2026),
`v2_tune.py` (Kelly/vol-target sensitivity grid). Research artifacts,
not investment advice.*
