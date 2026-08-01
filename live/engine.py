"""Shared Alpaca trading runtime used by all named accounts.

Strategy logic lives in live/strategies/; account identity in live/accounts.py.
This module owns broker I/O, safety rails, email, and the run loop (including
cash-rotate reduce/increase phases for weekly rotators like DM42).
"""
from __future__ import annotations

import json
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
from alpaca.common.exceptions import APIError
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
STATE_DIR = LIVE_DIR / "state"
ET = ZoneInfo("America/New_York")

MAX_ORDER_FRACTION = 0.70
MIN_NOTIONAL = 1.0
# Sells at >= this fraction of the position go by qty instead of notional:
# Alpaca converts notional sells to shares at its own reference price, and a
# small price drift can push the share count above what the account holds.
NEAR_FULL_SELL_FRACTION = 0.97
EMAIL_TO = ["anhminhle402@gmail.com", "artificial.voidstorage@gmail.com"]

# cash_rotate phases: reduce after signal close; increase next session after cash frees.
PHASES = frozenset({"auto", "reduce", "increase"})
BUY_POWER_BUFFER = 0.99  # leave a haircut so Alpaca BP rounding does not 403
ALPACA_INSUFFICIENT_BP = 40310000  # Alpaca error code on the 403 order reject
FILL_WAIT_SEC = 600
FILL_POLL_SEC = 15


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

    # dict.fromkeys dedupes while keeping order (a traded overlay sleeve may
    # share its signal ticker with a core sleeve, e.g. QQQ in r50d_cash).
    stock_symbols = list(
        dict.fromkeys(s.signal for s in strategy.SLEEVES.values() if not s.crypto)
    )
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
            if action == "sell" and notional >= NEAR_FULL_SELL_FRACTION * held_w * equity:
                orders.append({"symbol": sym, "action": "exit", "qty": pos["qty"]})
            else:
                orders.append({"symbol": sym, "action": action, "notional": notional})
    orders.sort(key=lambda o: o["action"] == "buy")
    return orders


def split_orders(orders: list[dict]) -> tuple[list[dict], list[dict]]:
    """Partition into reduces (exit/sell) then increases (buy)."""
    reduces = [o for o in orders if o["action"] != "buy"]
    buys = [o for o in orders if o["action"] == "buy"]
    return reduces, buys


def rebalance_path(account_id: str) -> Path:
    return STATE_DIR / f"{account_id}.rebalance.json"


def load_pending_rebalance(account_id: str) -> dict | None:
    path = rebalance_path(account_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def save_pending_rebalance(account_id: str, payload: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = rebalance_path(account_id)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def clear_pending_rebalance(account_id: str) -> None:
    path = rebalance_path(account_id)
    if path.exists():
        path.unlink()


def open_orders(trading: TradingClient) -> list:
    return list(trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN)))


def market_is_open(trading: TradingClient) -> bool:
    return bool(trading.get_clock().is_open)


def wait_for_no_open_orders(
    trading: TradingClient, *, timeout_sec: float = FILL_WAIT_SEC
) -> bool:
    """Poll until no open orders. Return True if clear, False on timeout."""
    deadline = time.time() + timeout_sec
    while True:
        open_list = open_orders(trading)
        if not open_list:
            return True
        if time.time() >= deadline:
            syms = ", ".join(sorted({o.symbol for o in open_list}))
            print(
                f"WARNING: still {len(open_list)} open order(s) after "
                f"{timeout_sec:.0f}s ({syms})"
            )
            return False
        print(f"waiting for {len(open_list)} open order(s) to fill...")
        time.sleep(FILL_POLL_SEC)


def clip_buys_to_buying_power(
    trading: TradingClient, buys: list[dict], *, strict: bool = True
) -> list[dict]:
    """Scale buy notionals down if broker buying power is short.

    strict=True (increase phase, reduces already filled) raises when buying
    power cannot fund even MIN_NOTIONAL. strict=False (after-close single-phase
    runs, where queued sells have not yet freed cash) drops the buys instead so
    the next scheduled run retries once proceeds settle.
    """
    if not buys:
        return buys
    bp = float(trading.get_account().buying_power) * BUY_POWER_BUFFER
    need = sum(float(o["notional"]) for o in buys)
    if need <= bp or need <= 0:
        return buys
    if bp < MIN_NOTIONAL:
        if strict:
            raise RuntimeError(
                f"insufficient buying power ${bp:,.2f} for ${need:,.2f} of buys -- "
                "reduces may not have filled yet"
            )
        print(
            f"WARNING: buying power ${bp:,.2f} cannot fund ${need:,.2f} of "
            "buys -- deferring buys to the next run"
        )
        return []
    scale = bp / need
    print(
        f"NOTE: scaling buys by {scale:.3f} to fit buying power "
        f"${bp:,.2f} (planned ${need:,.2f})"
    )
    out = []
    for o in buys:
        notional = round(float(o["notional"]) * scale, 2)
        if notional < MIN_NOTIONAL:
            continue
        out.append({**o, "notional": notional})
    return out


def _api_error_code(exc: APIError) -> int | None:
    """APIError.code re-parses the response body and throws on non-JSON errors."""
    try:
        return int(exc.code)
    except Exception:  # noqa: BLE001
        return None


def submit_orders(
    trading: TradingClient, orders: list[dict], strategy: ModuleType
) -> list[dict]:
    """Submit orders in sequence; return the ones actually accepted.

    A buy the broker rejects for insufficient buying power is skipped (crypto
    buys draw on settled cash, which the buying-power clip cannot see); the
    next scheduled run re-plans and retries it. Any other reject aborts with
    context on what was already submitted.
    """
    submitted: list[dict] = []
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
        try:
            order = trading.submit_order(request)
        except APIError as exc:
            if o["action"] == "buy" and _api_error_code(exc) == ALPACA_INSUFFICIENT_BP:
                print(
                    f"WARNING: broker rejected buy {sym} "
                    f"${o['notional']:,.2f} (insufficient buying power) -- "
                    "skipping; next run will retry"
                )
                continue
            done = ", ".join(f"{s['action']} {s['symbol']}" for s in submitted)
            raise RuntimeError(
                f"order submit failed for {o['action']} {sym} -- already "
                f"submitted: {done or 'none'}"
            ) from exc
        submitted.append(o)
        detail = f"qty {o['qty']}" if o["action"] == "exit" else f"${o['notional']:,.2f}"
        print(f"submitted {o['action']:>4} {sym:<8} {detail}  (id {order.id})")
    return submitted


def _ensure_no_surprise_open_orders(
    trading: TradingClient, *, allow_symbols: set[str] | None = None
) -> None:
    """Abort on unexpected open orders; allow listed sleeve symbols (in-flight reduces)."""
    open_list = open_orders(trading)
    if not open_list:
        return
    allow_symbols = allow_symbols or set()
    # Alpaca crypto symbols may omit the slash.
    allow_api = {s.replace("/", "") for s in allow_symbols} | set(allow_symbols)
    unexpected = [o for o in open_list if o.symbol not in allow_api]
    if unexpected:
        raise RuntimeError(
            f"{len(unexpected)} unexpected open order(s) already queued -- "
            "duplicate run? aborting"
        )
    print(
        f"NOTE: {len(open_list)} open order(s) on "
        f"{', '.join(sorted({o.symbol for o in open_list}))} "
        "(treating as in-flight reduces)"
    )


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


def _resolve_phase(account: Account, phase: str) -> str:
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}; expected one of {sorted(PHASES)}")
    if not account.cash_rotate:
        return "all"
    if phase == "auto":
        pending = load_pending_rebalance(account.id)
        return "increase" if pending else "reduce"
    return phase


def _compute_targets(
    strategy: ModuleType,
    closes: pd.DataFrame,
    equity: float,
    account: Account,
    *,
    apply_risk: bool,
) -> tuple[dict[str, float], dict]:
    targets, diag = strategy.target_weights(closes)
    if apply_risk and hasattr(strategy, "scale_for_risk"):
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
    return targets, diag


def run_account(
    account: Account,
    *,
    dry_run: bool = False,
    confirm_live: bool = False,
    phase: str = "auto",
) -> None:
    """Execute one cycle for a named account.

    cash_rotate accounts split rebalance into reduce (sells/exits after signal
    close) and increase (buys next session once cash is free), matching the
    research Friday-close -> next-session T+1 path.
    """
    if not account.paper and not dry_run and not confirm_live:
        sys.exit(
            f"account {account.id} is LIVE money -- pass --confirm-live to submit orders "
            "(or --dry-run to compute without trading)"
        )

    strategy = get_strategy(account.strategy)
    key, secret = resolve_credentials(account)
    resolved = _resolve_phase(account, phase)

    halted = check_halts(account)
    if halted is not None:
        print(f"HALT present at {halted} -- skipping run for {account.id}")
        return

    trading = TradingClient(key, secret, paper=account.paper)
    last_session = last_completed_session(trading)
    closes = fetch_closes(key, secret, last_session, strategy)
    equity, held = broker_state(trading, strategy)
    pending = load_pending_rebalance(account.id)

    if resolved == "increase" and pending and isinstance(pending.get("targets"), dict):
        # Freeze Friday targets (weekly_hold); do not re-signal or re-brake.
        targets = {sym: float(pending["targets"].get(sym, 0.0)) for sym in strategy.SLEEVES}
        _, diag = strategy.target_weights(closes)  # email diagnostics only
        print(
            f"phase=increase using frozen targets from signal_date="
            f"{pending.get('signal_date', '?')}"
        )
    else:
        targets, diag = _compute_targets(
            strategy, closes, equity, account, apply_risk=True
        )

    orders = plan_orders(targets, held, equity, strategy)
    reduces, buys = split_orders(orders)

    if resolved == "all":
        to_submit = orders
    elif resolved == "reduce":
        if pending and not dry_run:
            raise RuntimeError(
                f"pending rebalance from {pending.get('signal_date')} still awaits "
                f"increase -- run with --phase increase before a new reduce"
            )
        to_submit = reduces + buys  # dry-run shows full plan; submit path splits
    else:  # increase
        to_submit = reduces + buys  # may still need leftover reduces (crash recovery)

    submitted: list[dict] = []
    deferred = False

    if not dry_run and (to_submit or (resolved == "increase" and pending)):
        if resolved == "all":
            if not to_submit:
                pass
            elif open_orders(trading):
                raise RuntimeError(
                    f"{len(open_orders(trading))} open order(s) already queued -- "
                    "duplicate run? aborting"
                )
            else:
                # Runs happen after the close (daily-trade.yml, 6 PM ET), so the
                # reduces queue for the next open and their proceeds are NOT in
                # buying power yet. Clip buys to what the broker can fund now;
                # the next daily run tops up once the sells fill.
                submitted = submit_orders(trading, reduces, strategy)
                still_buy = clip_buys_to_buying_power(trading, buys, strict=False)
                if still_buy:
                    submitted += submit_orders(trading, still_buy, strategy)

        elif resolved == "reduce":
            if open_orders(trading):
                raise RuntimeError(
                    f"{len(open_orders(trading))} open order(s) already queued -- "
                    "duplicate run? aborting"
                )
            if reduces and buys:
                # Persist before reduce submit so a mid-crash can finish on increase.
                save_pending_rebalance(
                    account.id,
                    {
                        "signal_date": str(closes.index[-1]),
                        "targets": targets,
                        "status": "awaiting_increase",
                        "reduces": [
                            {"symbol": o["symbol"], "action": o["action"]}
                            for o in reduces
                        ],
                        "buy_symbols": [o["symbol"] for o in buys],
                    },
                )
                submitted = submit_orders(trading, reduces, strategy)
            elif reduces:
                submitted = submit_orders(trading, reduces, strategy)
            elif buys:
                # Already have cash; no sell leg -- queue buys same session.
                submitted = submit_orders(trading, buys, strategy)

        else:  # increase
            allow = {o["symbol"] for o in reduces}
            if pending:
                allow |= {
                    r["symbol"]
                    for r in pending.get("reduces", [])
                    if isinstance(r, dict) and "symbol" in r
                }
            _ensure_no_surprise_open_orders(trading, allow_symbols=allow)

            opened = open_orders(trading)
            open_syms = {o.symbol for o in opened} | {
                o.symbol.replace("/", "") for o in opened
            }
            need_reduce = [
                o for o in reduces
                if o["symbol"] not in open_syms
                and o["symbol"].replace("/", "") not in open_syms
            ]
            if need_reduce:
                print(
                    f"phase=increase: submitting {len(need_reduce)} missing reduce(s) "
                    "before buys"
                )
                submitted.extend(submit_orders(trading, need_reduce, strategy))

            if open_orders(trading):
                if market_is_open(trading):
                    if not wait_for_no_open_orders(trading):
                        raise RuntimeError(
                            "open reduce orders did not fill in time -- "
                            "rerun --phase increase after fills"
                        )
                else:
                    deferred = True
                    if pending is None:
                        # Crash recovery with no pending file: freeze targets now.
                        save_pending_rebalance(
                            account.id,
                            {
                                "signal_date": str(closes.index[-1]),
                                "targets": targets,
                                "status": "awaiting_increase",
                                "reduces": [
                                    {"symbol": o["symbol"], "action": o["action"]}
                                    for o in reduces
                                ],
                                "buy_symbols": [o["symbol"] for o in buys],
                            },
                        )
                    print(
                        "market closed with open reduce order(s) -- "
                        "keeping pending rebalance; buys deferred to next session"
                    )

            if not deferred:
                equity, held = broker_state(trading, strategy)
                _, replanned_buys = split_orders(
                    plan_orders(targets, held, equity, strategy)
                )
                still_buy = clip_buys_to_buying_power(trading, replanned_buys)
                if still_buy:
                    if open_orders(trading):
                        raise RuntimeError(
                            "open orders remain before buy submit -- aborting"
                        )
                    submitted.extend(submit_orders(trading, still_buy, strategy))
                clear_pending_rebalance(account.id)
                equity, held = broker_state(trading, strategy)
                orders = plan_orders(targets, held, equity, strategy)

    # Display / email: show the full planned set for context, highlight phase.
    display_orders = orders
    if resolved == "reduce":
        display_orders = reduces + buys
    elif resolved == "increase":
        display_orders = submitted if submitted else (reduces + buys)

    if deferred:
        status = (
            f"phase=increase deferred -- reduce order(s) still open until next "
            "session; pending rebalance kept."
        )
        subject_status = "increase deferred"
    elif dry_run:
        n = len(orders) if resolved == "all" else len(reduces) + len(buys)
        status = (
            f"DRY RUN phase={resolved} -- {n} order(s) computed but NOT submitted."
        )
        subject_status = f"dry run {resolved}: {n} order(s)"
    elif resolved == "reduce":
        if not reduces and not buys:
            status = "No trades -- all sleeves within band."
            subject_status = "no trades"
        elif submitted and buys and reduces:
            status = (
                f"phase=reduce: {len(submitted)} reduce order(s) submitted "
                f"(queued next open); {len(buys)} buy(s) pending for increase phase."
            )
            subject_status = (
                f"reduce: {len(submitted)} queued, {len(buys)} buy(s) pending"
            )
        elif submitted:
            status = (
                f"phase=reduce: {len(submitted)} order(s) submitted "
                "(queued for the next market open)."
            )
            subject_status = f"reduce: {len(submitted)} order(s)"
        else:
            status = "phase=reduce: nothing to submit."
            subject_status = "reduce: idle"
    elif resolved == "increase":
        if submitted:
            status = (
                f"phase=increase: {len(submitted)} order(s) submitted "
                "(queued / working)."
            )
            subject_status = f"increase: {len(submitted)} order(s)"
        else:
            status = "phase=increase -- book already at targets (no buys needed)."
            subject_status = "increase: flat"
            clear_pending_rebalance(account.id)
    elif not orders:
        status = "No trades -- all sleeves within band."
        subject_status = "no trades"
    else:
        status = f"{len(submitted)} order(s) submitted, queued for the next market open."
        subject_status = f"{len(submitted)} order(s) queued"

    rows = build_rows(targets, diag, held, display_orders, strategy)
    mode = "paper" if account.paper else "LIVE"
    footer = (
        f"{account.email_tag} ({mode}) &middot; equity ${equity:,.2f} "
        f"&middot; target invested {sum(targets.values()):.1%} "
        f"&middot; data through {closes.index[-1]}"
        f" &middot; phase {resolved}"
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
            "Orders submitted before the crash (if any) are listed in the "
            "traceback below and the GitHub Actions log -- reconcile manually.\n\n"
            + traceback.format_exc()
        ),
    )
