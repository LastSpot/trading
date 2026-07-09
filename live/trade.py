#!/usr/bin/env python3
"""Daily Alpaca paper-trading engine for the v3 cash-account strategy.

Implements variant R50d from research/v3_extended/v3_revised.py -- the revised
cash-account book (v3_cash.py variant F with EFA dropped, a managed-futures
trend sleeve added, and the vol target stepped down to 50%):

  - Six sleeves gated by ensemble trend votes (63/126/252-day momentum on the
    underlying), sized by EWMA vol targeting at a 50% portfolio vol target.
  - DBC and DBMF are held UNLEVERED (L=1). The research's synthetic 2x assumed
    margin borrowing; in a cash account the same risk contribution is bought
    with more capital instead of borrowed money.
  - Total notional is capped at 100% of equity, so the account never borrows.

Designed to run once per day after the US close. Signals use tonight's close;
equity orders are notional market DAY orders, which Alpaca queues for the next
open (T+1 execution, matching the backtest). BTC/USD fills immediately (24/7).

The ensemble gate is stateless and held weights are read from the broker, so
there is no state file: every run recomputes targets from scratch and trades
only when a sleeve drifts more than TRADE_BAND from target.

Usage:  python live/trade.py [--dry-run]
Env:    ALPACA_KEY, ALPACA_SECRET   (paper keys; also read from ./.env locally)
        GMAIL_USER, GMAIL_APP_PASSWORD  (optional; emails the run summary)
"""
from __future__ import annotations

import argparse
import os
import smtplib
import sys
import traceback
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetCalendarRequest,
    GetOrdersRequest,
    MarketOrderRequest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Sleeve:
    signal: str      # ticker whose closes drive the gate and vol estimate
    budget: float    # share of the portfolio risk budget
    leverage: float  # leverage embedded in the traded product
    crypto: bool = False


SLEEVES: dict[str, Sleeve] = {
    "TQQQ": Sleeve("QQQ", 0.40, 3.0),
    "TMF": Sleeve("TLT", 0.15, 3.0),
    "UGL": Sleeve("GLD", 0.15, 2.0),
    "DBC": Sleeve("DBC", 0.10, 1.0),
    "DBMF": Sleeve("DBMF", 0.10, 1.0),
    "BTC/USD": Sleeve("BTC/USD", 0.10, 1.0, crypto=True),
}
SIGMA_TARGET = 0.50          # portfolio vol target (variant R50d)
LOOKBACKS = (63, 126, 252)   # momentum votes, in trading days
EWMA_ALPHA = 0.06            # daily EWMA vol estimator
TRADE_BAND = 0.05            # rebalance only if |target - held| > 5 pts
HISTORY_DAYS = 600           # calendar days fetched (>= 252 bars + vol warm-up)
MAX_MOVE = 0.25              # reject data with an absurd one-day move
MAX_ORDER_FRACTION = 0.70    # refuse any single order above 70% of equity
MIN_NOTIONAL = 1.0           # Alpaca's fractional order minimum

EMAIL_TO = ["anhminhle402@gmail.com", "artificial.voidstorage@gmail.com"]


def load_env() -> None:
    """Read ./.env into the environment for local runs (no-op in CI)."""
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def last_completed_session(trading: TradingClient) -> date:
    """Most recent trading session whose close has already passed."""
    now = datetime.now(ET)
    calendar = trading.get_calendar(
        GetCalendarRequest(start=now.date() - timedelta(days=14), end=now.date())
    )
    done = [c.date for c in calendar if c.close.replace(tzinfo=ET) <= now]
    if not done:
        raise RuntimeError("no completed trading session in the last 14 days?")
    return done[-1]


def fetch_closes(key: str, secret: str, last_session: date) -> pd.DataFrame:
    """Split/dividend-adjusted daily closes for all signals through last_session."""
    start = datetime.now(timezone.utc) - timedelta(days=HISTORY_DAYS)
    # Free data plan: SIP history is available except the most recent 15 minutes.
    end = datetime.now(timezone.utc) - timedelta(minutes=16)

    stock_symbols = [s.signal for s in SLEEVES.values() if not s.crypto]
    bars = StockHistoricalDataClient(key, secret).get_stock_bars(
        StockBarsRequest(
            symbol_or_symbols=stock_symbols,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            adjustment=Adjustment.ALL,
            feed=DataFeed.SIP,
        )
    ).df
    px = bars.reset_index().pivot(index="timestamp", columns="symbol", values="close")
    px.index = [ts.date() for ts in px.index]

    crypto_bars = CryptoHistoricalDataClient().get_crypto_bars(
        CryptoBarsRequest(
            symbol_or_symbols=["BTC/USD"], timeframe=TimeFrame.Day, start=start
        )
    ).df
    btc = crypto_bars.reset_index().pivot(
        index="timestamp", columns="symbol", values="close"
    )["BTC/USD"]
    btc.index = [ts.date() for ts in btc.index]
    # Sample crypto on equity trading days, as the research did.
    px["BTC/USD"] = btc.reindex(px.index)

    px = px[px.index <= last_session]

    if px.index[-1] != last_session:
        raise RuntimeError(f"stale data: last bar {px.index[-1]}, expected {last_session}")
    if len(px) < max(LOOKBACKS) + 60:
        raise RuntimeError(f"insufficient history: {len(px)} bars")
    if px.iloc[-max(LOOKBACKS) - 1:].isna().any().any():
        raise RuntimeError("missing closes inside the signal window")
    moves = px.pct_change().iloc[-1].abs()
    if (moves > MAX_MOVE).any():
        raise RuntimeError(f"absurd one-day move, check data: {moves.round(3).to_dict()}")
    return px


def ensemble_gate(px: pd.Series) -> float:
    return float(np.mean([px.iloc[-1] > px.iloc[-1 - L] for L in LOOKBACKS]))


def ewma_vol(px: pd.Series) -> float:
    r = px.pct_change().dropna()
    var = (r**2).ewm(alpha=EWMA_ALPHA, adjust=False).mean().iloc[-1]
    return float(np.sqrt(252.0 * var))


def target_weights(closes: pd.DataFrame) -> tuple[dict[str, float], dict[str, dict]]:
    raw, diag = {}, {}
    for sym, s in SLEEVES.items():
        px = closes[s.signal].dropna()
        gate, vol = ensemble_gate(px), ewma_vol(px)
        raw[sym] = gate * s.budget * SIGMA_TARGET / (s.leverage * vol)
        diag[sym] = {"gate": gate, "vol": vol}
    total = sum(raw.values())
    scale = 1.0 / total if total > 1.0 else 1.0  # never exceed 100% invested
    return {k: v * scale for k, v in raw.items()}, diag


def broker_state(trading: TradingClient) -> tuple[float, dict[str, dict]]:
    """Account equity and held weight/qty per sleeve, from broker positions."""
    equity = float(trading.get_account().equity)
    api_to_sleeve = {sym.replace("/", ""): sym for sym in SLEEVES}
    held: dict[str, dict] = {}
    for pos in trading.get_all_positions():
        sym = api_to_sleeve.get(pos.symbol)
        if sym is None:
            raise RuntimeError(
                f"unexpected position {pos.symbol} in the account -- "
                "this account must hold strategy sleeves only; reconcile manually"
            )
        held[sym] = {"weight": float(pos.market_value) / equity, "qty": pos.qty}
    return equity, held


def plan_orders(
    targets: dict[str, float], held: dict[str, dict], equity: float
) -> list[dict]:
    orders = []
    for sym in SLEEVES:
        target_w = targets[sym]
        pos = held.get(sym)
        held_w = pos["weight"] if pos else 0.0
        if pos and target_w == 0.0:
            orders.append({"symbol": sym, "action": "exit", "qty": pos["qty"]})
        elif abs(target_w - held_w) > TRADE_BAND:
            notional = round(abs(target_w - held_w) * equity, 2)
            if notional < MIN_NOTIONAL:
                continue
            if notional > MAX_ORDER_FRACTION * equity:
                raise RuntimeError(
                    f"order for {sym} (${notional:,.2f}) exceeds "
                    f"{MAX_ORDER_FRACTION:.0%} of equity -- refusing to trade"
                )
            action = "buy" if target_w > held_w else "sell"
            orders.append({"symbol": sym, "action": action, "notional": notional})
    orders.sort(key=lambda o: o["action"] == "buy")  # free cash first: sells before buys
    return orders


def submit_orders(trading: TradingClient, orders: list[dict]) -> None:
    for o in orders:
        sym = o["symbol"]
        tif = TimeInForce.GTC if SLEEVES[sym].crypto else TimeInForce.DAY
        if o["action"] == "exit":
            request = MarketOrderRequest(
                symbol=sym, qty=o["qty"], side=OrderSide.SELL, time_in_force=tif
            )
        else:
            side = OrderSide.BUY if o["action"] == "buy" else OrderSide.SELL
            request = MarketOrderRequest(
                symbol=sym, notional=o["notional"], side=side, time_in_force=tif
            )
        order = trading.submit_order(request)
        detail = f"qty {o['qty']}" if o["action"] == "exit" else f"${o['notional']:,.2f}"
        print(f"submitted {o['action']:>4} {sym:<8} {detail}  (id {order.id})")


def send_email(subject: str, text: str, html: str | None = None) -> None:
    """Email the run summary; never fails the run if the email itself fails."""
    user = os.environ.get("GMAIL_USER")
    password = os.environ.get("GMAIL_APP_PASSWORD")
    if not user or not password:
        print("GMAIL_USER / GMAIL_APP_PASSWORD not set -- skipping email")
        return
    msg = EmailMessage()
    msg["From"] = user
    msg["To"] = ", ".join(EMAIL_TO)
    msg["Subject"] = subject
    msg.set_content(text)
    if html:
        msg.add_alternative(html, subtype="html")
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as smtp:
            smtp.login(user, password)
            smtp.send_message(msg)
        print(f"emailed summary to {', '.join(EMAIL_TO)}")
    except Exception as exc:  # noqa: BLE001 -- notification must not fail the run
        print(f"WARNING: email failed: {exc}")


def build_rows(targets, diag, held, orders) -> list[dict]:
    planned = {o["symbol"]: o for o in orders}
    rows = []
    for sym in SLEEVES:
        o = planned.get(sym)
        action = "hold" if o is None else (
            "exit" if o["action"] == "exit" else f"{o['action']} ${o['notional']:,.2f}"
        )
        rows.append({
            "sym": sym, "gate": diag[sym]["gate"], "vol": diag[sym]["vol"],
            "target": targets[sym],
            "held": held[sym]["weight"] if sym in held else 0.0,
            "action": action,
        })
    return rows


def render_text(rows: list[dict], footer: str, status: str) -> str:
    lines = [f"{'sleeve':<8} {'gate':>5} {'vol':>6} {'target':>7} {'held':>7}  action"]
    for r in rows:
        lines.append(
            f"{r['sym']:<8} {r['gate']:>5.2f} {r['vol']:>6.3f} "
            f"{r['target']:>7.1%} {r['held']:>7.1%}  {r['action']}"
        )
    return "\n".join(lines) + f"\n\n{footer}\n{status}"


def render_html(rows: list[dict], footer: str, status: str) -> str:
    cell = 'style="padding:4px 10px;border-bottom:1px solid #e0e0e0;text-align:right"'
    left = 'style="padding:4px 10px;border-bottom:1px solid #e0e0e0;text-align:left"'
    body = ""
    for r in rows:
        color = ("#1a7f37" if r["action"].startswith("buy")
                 else "#c62828" if r["action"] in ("exit",) or r["action"].startswith("sell")
                 else "#757575")
        body += (
            f"<tr><td {left}><b>{r['sym']}</b></td>"
            f"<td {cell}>{r['gate']:.2f}</td><td {cell}>{r['vol']:.1%}</td>"
            f"<td {cell}>{r['target']:.1%}</td><td {cell}>{r['held']:.1%}</td>"
            f"<td {left}><span style='color:{color}'>{r['action']}</span></td></tr>"
        )
    head = "".join(
        f"<th {left if h in ('sleeve', 'action') else cell}>{h}</th>"
        for h in ("sleeve", "gate", "vol", "target", "held", "action")
    )
    return (
        "<div style='font-family:Segoe UI,Arial,sans-serif;font-size:14px;color:#222'>"
        f"<p><b>{status}</b></p>"
        "<table style='border-collapse:collapse'>"
        f"<tr>{head}</tr>{body}</table>"
        f"<p style='color:#555'>{footer}</p>"
        "<p style='color:#999;font-size:12px'>Automated run of live/trade.py "
        "(six-sleeve trend strategy, variant R50d, Alpaca paper account).</p></div>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="compute and print orders without submitting"
    )
    args = parser.parse_args()

    key, secret = os.environ.get("ALPACA_KEY"), os.environ.get("ALPACA_SECRET")
    if not key or not secret:
        sys.exit("ALPACA_KEY / ALPACA_SECRET are not set")

    if (REPO_ROOT / "HALT").exists():
        print("HALT file present -- skipping run")
        return

    trading = TradingClient(key, secret, paper=True)
    last_session = last_completed_session(trading)
    closes = fetch_closes(key, secret, last_session)
    targets, diag = target_weights(closes)
    equity, held = broker_state(trading)
    orders = plan_orders(targets, held, equity)

    if orders and not args.dry_run:
        open_orders = trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
        if open_orders:
            raise RuntimeError(
                f"{len(open_orders)} open order(s) already queued -- duplicate run? aborting"
            )
        submit_orders(trading, orders)

    if not orders:
        status = "No trades -- all sleeves within band."
        subject_status = "no trades"
    elif args.dry_run:
        status = f"DRY RUN -- {len(orders)} order(s) computed but NOT submitted."
        subject_status = f"dry run: {len(orders)} order(s)"
    else:
        status = f"{len(orders)} order(s) submitted, queued for the next market open."
        subject_status = f"{len(orders)} order(s) queued"

    rows = build_rows(targets, diag, held, orders)
    footer = (
        f"equity ${equity:,.2f} &middot; target invested {sum(targets.values()):.1%} "
        f"&middot; data through {closes.index[-1]}"
    )
    footer_text = footer.replace("&middot;", "|")
    text = render_text(rows, footer_text, status)
    print(text)

    send_email(
        subject=f"[trading] {subject_status} | equity ${equity:,.0f} | {closes.index[-1]}",
        text=text,
        html=render_html(rows, footer, status),
    )


if __name__ == "__main__":
    load_env()
    try:
        main()
    except Exception:
        send_email(
            subject=f"[trading] RUN FAILED | {date.today()}",
            text="The daily trading run crashed before completing. No orders were "
                 "verified as placed -- check the GitHub Actions log.\n\n"
                 + traceback.format_exc(),
        )
        raise
