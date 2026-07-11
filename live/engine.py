"""Shared Alpaca trading runtime used by all named accounts.

Strategy logic lives in live/strategies/; account identity in live/accounts.py.
This module owns broker I/O, safety rails, email, and the daily run loop.
"""
from __future__ import annotations

import os
import smtplib
import sys
import time
import traceback
from datetime import date, datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from types import ModuleType
from zoneinfo import ZoneInfo

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

from accounts import Account
from strategies import get_strategy

REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = Path(__file__).resolve().parent
HALTS_DIR = LIVE_DIR / "halts"
ET = ZoneInfo("America/New_York")

MAX_ORDER_FRACTION = 0.70
MIN_NOTIONAL = 1.0
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


def resolve_credentials(account: Account) -> tuple[str, str]:
    """Return (key, secret) from account env names, with optional legacy fallback."""
    key = os.environ.get(account.key_env)
    secret = os.environ.get(account.secret_env)
    if key and secret:
        return key, secret
    if account.legacy_key_env and account.legacy_secret_env:
        key = os.environ.get(account.legacy_key_env)
        secret = os.environ.get(account.legacy_secret_env)
        if key and secret:
            print(
                f"NOTE: using legacy {account.legacy_key_env}/"
                f"{account.legacy_secret_env}; prefer {account.key_env}/"
                f"{account.secret_env}"
            )
            return key, secret
    names = f"{account.key_env} / {account.secret_env}"
    if account.legacy_key_env:
        names += f" (or legacy {account.legacy_key_env} / {account.legacy_secret_env})"
    sys.exit(f"{names} are not set for account {account.id}")


def halt_paths(account: Account) -> list[Path]:
    """Global repo HALT plus per-account live/halts/<id>."""
    return [REPO_ROOT / "HALT", HALTS_DIR / account.halt_path_name]


def check_halts(account: Account) -> Path | None:
    for path in halt_paths(account):
        if path.exists():
            return path
    return None


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


def fetch_closes(
    key: str, secret: str, last_session: date, strategy: ModuleType
) -> pd.DataFrame:
    """Split/dividend-adjusted daily closes for all strategy signals."""
    start = datetime.now(timezone.utc) - timedelta(days=strategy.HISTORY_DAYS)
    end = datetime.now(timezone.utc) - timedelta(minutes=16)

    stock_symbols = [s.signal for s in strategy.SLEEVES.values() if not s.crypto]
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

    crypto_syms = [sym for sym, s in strategy.SLEEVES.items() if s.crypto]
    if crypto_syms:
        crypto_bars = CryptoHistoricalDataClient().get_crypto_bars(
            CryptoBarsRequest(
                symbol_or_symbols=crypto_syms, timeframe=TimeFrame.Day, start=start
            )
        ).df
        crypto_px = crypto_bars.reset_index().pivot(
            index="timestamp", columns="symbol", values="close"
        )
        crypto_px.index = [ts.date() for ts in crypto_px.index]
        for sym in crypto_syms:
            px[sym] = crypto_px[sym].reindex(px.index)

    px = px[px.index <= last_session]

    lookbacks = strategy.LOOKBACKS
    if px.index[-1] != last_session:
        raise RuntimeError(f"stale data: last bar {px.index[-1]}, expected {last_session}")
    if len(px) < max(lookbacks) + 60:
        raise RuntimeError(f"insufficient history: {len(px)} bars")
    if px.iloc[-max(lookbacks) - 1:].isna().any().any():
        raise RuntimeError("missing closes inside the signal window")
    moves = px.pct_change().iloc[-1].abs()
    if (moves > strategy.MAX_MOVE).any():
        raise RuntimeError(f"absurd one-day move, check data: {moves.round(3).to_dict()}")
    return px


def broker_state(
    trading: TradingClient, strategy: ModuleType
) -> tuple[float, dict[str, dict]]:
    """Account equity and held weight/qty per sleeve, from broker positions."""
    acct = trading.get_account()
    equity = float(acct.equity)
    cash = float(acct.cash)
    if equity <= 0:
        raise RuntimeError(
            f"broker equity is ${equity:,.2f} -- cannot size orders; "
            "reset/liquidate the Alpaca paper account before trading"
        )
    api_to_sleeve = {sym.replace("/", ""): sym for sym in strategy.SLEEVES}
    held: dict[str, dict] = {}
    positions_mv = 0.0
    for pos in trading.get_all_positions():
        sym = api_to_sleeve.get(pos.symbol)
        if sym is None:
            raise RuntimeError(
                f"unexpected position {pos.symbol} in the account -- "
                "this account must hold strategy sleeves only; reconcile manually"
            )
        mv = float(pos.market_value)
        positions_mv += abs(mv)
        held[sym] = {"weight": mv / equity, "qty": pos.qty}
    # Cash deeply negative with large positions (paper double-buy / bad reset)
    # makes weights >> 1 and trips order rails with a confusing error.
    if positions_mv > 2.0 * equity:
        raise RuntimeError(
            f"broker state incoherent: positions ${positions_mv:,.2f} vs equity "
            f"${equity:,.2f} (cash ${cash:,.2f}). Likely a paper-account "
            "double-buy or reset glitch -- liquidate positions and reset the "
            "Alpaca paper account, then clear live/state/<account>.json"
        )
    return equity, held


def plan_orders(
    targets: dict[str, float],
    held: dict[str, dict],
    equity: float,
    strategy: ModuleType,
) -> list[dict]:
    orders = []
    for sym in strategy.SLEEVES:
        target_w = targets[sym]
        pos = held.get(sym)
        held_w = pos["weight"] if pos else 0.0
        if pos and target_w == 0.0:
            orders.append({"symbol": sym, "action": "exit", "qty": pos["qty"]})
        elif abs(target_w - held_w) > strategy.TRADE_BAND:
            notional = round(abs(target_w - held_w) * equity, 2)
            if notional < MIN_NOTIONAL:
                continue
            action = "buy" if target_w > held_w else "sell"
            # Ceiling is a runaway-buy rail; sells that reduce exposure are fine.
            if action == "buy" and notional > MAX_ORDER_FRACTION * equity:
                raise RuntimeError(
                    f"buy for {sym} (${notional:,.2f}) exceeds "
                    f"{MAX_ORDER_FRACTION:.0%} of equity -- refusing to trade"
                )
            orders.append({"symbol": sym, "action": action, "notional": notional})
    orders.sort(key=lambda o: o["action"] == "buy")
    return orders


def submit_orders(
    trading: TradingClient, orders: list[dict], strategy: ModuleType
) -> None:
    for o in orders:
        sym = o["symbol"]
        tif = TimeInForce.GTC if strategy.SLEEVES[sym].crypto else TimeInForce.DAY
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


def _smtp_send(msg: EmailMessage, user: str, password: str) -> None:
    """Try SSL:465 then STARTTLS:587. Gmail from CI is flaky on either alone."""
    last_exc: Exception | None = None
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=45) as smtp:
            smtp.login(user, password)
            smtp.send_message(msg)
        return
    except Exception as exc:  # noqa: BLE001 -- fall through to STARTTLS
        last_exc = exc
    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=45) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(user, password)
            smtp.send_message(msg)
        return
    except Exception as exc:  # noqa: BLE001 -- surface both failures
        raise RuntimeError(
            f"SMTP_SSL(465) failed: {last_exc}; STARTTLS(587) failed: {exc}"
        ) from exc


def send_email(subject: str, text: str, html: str | None = None) -> None:
    """Email the run summary; never fails the run if the email itself fails.

    Retries a few times with backoff and alternates SSL/STARTTLS so a single
    Gmail timeout (common on GitHub Actions) does not drop the daily summary.
    """
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

    attempts = 3
    for attempt in range(1, attempts + 1):
        try:
            _smtp_send(msg, user, password)
            print(f"emailed summary to {', '.join(EMAIL_TO)}")
            return
        except Exception as exc:  # noqa: BLE001 -- notification must not fail the run
            if attempt == attempts:
                print(f"WARNING: email failed after {attempts} attempts: {exc}")
                return
            wait = 2 ** attempt  # 2s, 4s
            print(f"WARNING: email attempt {attempt}/{attempts} failed ({exc}); retry in {wait}s")
            time.sleep(wait)


def build_rows(targets, diag, held, orders, strategy: ModuleType) -> list[dict]:
    planned = {o["symbol"]: o for o in orders}
    rows = []
    for sym in strategy.SLEEVES:
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


def render_html(
    rows: list[dict], footer: str, status: str, account: Account, strategy: ModuleType
) -> str:
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
    mode = "paper" if account.paper else "LIVE"
    return (
        "<div style='font-family:Segoe UI,Arial,sans-serif;font-size:14px;color:#222'>"
        f"<p><b>{status}</b></p>"
        "<table style='border-collapse:collapse'>"
        f"<tr>{head}</tr>{body}</table>"
        f"<p style='color:#555'>{footer}</p>"
        f"<p style='color:#999;font-size:12px'>Automated run of live/run.py "
        f"(account {account.id}, strategy {strategy.NAME}, Alpaca {mode}).</p></div>"
    )


def run_account(account: Account, *, dry_run: bool = False, confirm_live: bool = False) -> None:
    """Execute one daily cycle for a named account."""
    if not account.paper and not dry_run and not confirm_live:
        sys.exit(
            f"account {account.id} is LIVE money -- pass --confirm-live to submit orders "
            "(or --dry-run to compute without trading)"
        )

    strategy = get_strategy(account.strategy)
    key, secret = resolve_credentials(account)

    halted = check_halts(account)
    if halted is not None:
        print(f"HALT present at {halted} -- skipping run for {account.id}")
        return

    trading = TradingClient(key, secret, paper=account.paper)
    last_session = last_completed_session(trading)
    closes = fetch_closes(key, secret, last_session, strategy)
    targets, diag = strategy.target_weights(closes)
    equity, held = broker_state(trading, strategy)
    if hasattr(strategy, "scale_for_risk"):
        targets, risk_diag = strategy.scale_for_risk(
            targets, equity, state_key=account.id
        )
        diag = {**diag, "_risk": risk_diag}
        print(
            f"risk overlay: scale={risk_diag.get('scale', 1.0):.2f} "
            f"dd={risk_diag.get('dd', 0.0):.1%} "
            f"braked={risk_diag.get('braked', False)} "
            f"peak=${risk_diag.get('peak_equity', equity):,.2f}"
        )
    orders = plan_orders(targets, held, equity, strategy)

    if orders and not dry_run:
        open_orders = trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
        if open_orders:
            raise RuntimeError(
                f"{len(open_orders)} open order(s) already queued -- duplicate run? aborting"
            )
        submit_orders(trading, orders, strategy)

    if not orders:
        status = "No trades -- all sleeves within band."
        subject_status = "no trades"
    elif dry_run:
        status = f"DRY RUN -- {len(orders)} order(s) computed but NOT submitted."
        subject_status = f"dry run: {len(orders)} order(s)"
    else:
        status = f"{len(orders)} order(s) submitted, queued for the next market open."
        subject_status = f"{len(orders)} order(s) queued"

    rows = build_rows(targets, diag, held, orders, strategy)
    mode = "paper" if account.paper else "LIVE"
    footer = (
        f"{account.email_tag} ({mode}) &middot; equity ${equity:,.2f} "
        f"&middot; target invested {sum(targets.values()):.1%} "
        f"&middot; data through {closes.index[-1]}"
    )
    footer_text = footer.replace("&middot;", "|")
    text = render_text(rows, footer_text, status)
    print(text)

    send_email(
        subject=(
            f"[trading:{account.email_tag}] {subject_status} | "
            f"equity ${equity:,.0f} | {closes.index[-1]}"
        ),
        text=text,
        html=render_html(rows, footer, status, account, strategy),
    )


def notify_run_failure(account_id: str | None = None) -> None:
    tag = account_id or "unknown"
    send_email(
        subject=f"[trading:{tag}] RUN FAILED | {date.today()}",
        text=(
            f"The daily trading run for account {tag} crashed before completing. "
            "No orders were verified as placed -- check the GitHub Actions log.\n\n"
            + traceback.format_exc()
        ),
    )
