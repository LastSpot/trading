"""V11: optimize VOO/QQQ allocation with quantitative methods.

Rules in DESIGN.md. Universe is SPY/VOO + QQQ only. Monthly T+1, 5 bps,
cash earns ^IRX. Primary ranking on full sample from ~2000 (after warmup)
covering major crashes since QQQ inception.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = REPO_ROOT / "output" / "v11_voo_qqq_results.json"
OUT_SUMMARY = REPO_ROOT / "output" / "v11_voo_qqq_summary.md"

START = "1999-03-10"
SELECT_START = "2003-01-01"
SELECT_END = "2012-12-31"
WARMUP_DAYS = 252
COST = 0.0005
# Retail federal CG rates at ~$150k taxable income (2025-style)
TAX_ST = 0.24  # short-term (<= 1y): ordinary income
TAX_LT = 0.15  # long-term (> 1y): LTCG 15% band
LT_CALENDAR_DAYS = 365
VOL_SPAN = 63
EWMA_ALPHA = 2.0 / (VOL_SPAN + 1.0)
INITIAL = 100_000.0
VOL_TARGET_5050 = 0.15
VOL_TARGET_COMBO = 0.18
REL_MOM_LOOKBACK = 126
REL_MOM_GAIN = 2.0
MV_MOM_LOOKBACK = 126
MV_MU_CAP = 0.40
MV_SHRINK = 0.30
SMA = 200
HYST = 0.02

CRISES = {
    "dotcom": ("2000-03-24", "2002-10-09"),
    "gfc": ("2007-10-09", "2009-03-09"),
    "euro_2011": ("2011-04-29", "2011-10-03"),
    "china_2015_16": ("2015-08-10", "2016-02-11"),
    "volmageddon_2018": ("2018-01-26", "2018-12-24"),
    "covid_crash": ("2020-02-19", "2020-03-23"),
    "bear_2022": ("2022-01-03", "2022-10-12"),
    "tariff_2025": ("2025-02-15", "2025-06-30"),
}

CRASH_REPORT_BOOKS = [
    "COMBO",
    "REL_MOM",
    "REGIME_TILT",
    "INV_VOL",
    "MV_MOM",
    "VOL_TARGET_5050",
    "BH_5050",
    "STATIC_5050",
    "STATIC_070",
    "STATIC_BEST",
    "BH_QQQ",
    "BH_SPY_VOO",
]

CANDIDATES = [
    "INV_VOL",
    "REL_MOM",
    "MV_MOM",
    "VOL_TARGET_5050",
    "REGIME_TILT",
    "COMBO",
]
MOMENTUM_BASED = {"REL_MOM", "COMBO"}
BENCHMARKS = [
    "BH_5050",
    "STATIC_5050",
    "STATIC_BEST",
    "STATIC_070",
    "STATIC_100",
    "BH_QQQ",
    "BH_SPY_VOO",
]
DIAGNOSTICS = ["PLACEBO_REL_MOM", "LAG_REL_MOM_63"]
DISPLAY_ORDER = (
    ["COMBO"]
    + [c for c in CANDIDATES if c != "COMBO"]
    + BENCHMARKS
    + DIAGNOSTICS
)


def native(value):
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value).date())
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    return value


def month_end_flags(index: pd.DatetimeIndex) -> np.ndarray:
    month = index.to_period("M")
    flags = np.zeros(len(index), dtype=bool)
    flags[:-1] = month[:-1] != month[1:]
    flags[-1] = True
    return flags


def ewma_vol_series(returns: pd.Series) -> pd.Series:
    return np.sqrt(
        252.0
        * returns.pow(2)
        .ewm(alpha=EWMA_ALPHA, adjust=False, min_periods=20)
        .mean()
    )


def ewma_cov_pair(ra: pd.Series, rb: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Annualized EWMA var_a, var_b, cov_ab."""
    a2 = (ra**2).ewm(alpha=EWMA_ALPHA, adjust=False, min_periods=20).mean() * 252.0
    b2 = (rb**2).ewm(alpha=EWMA_ALPHA, adjust=False, min_periods=20).mean() * 252.0
    ab = (ra * rb).ewm(alpha=EWMA_ALPHA, adjust=False, min_periods=20).mean() * 252.0
    return a2, b2, ab


def drawdown_episode(equity: pd.Series) -> dict:
    dd = equity / equity.cummax() - 1.0
    trough = dd.idxmin()
    max_dd = float(dd.loc[trough])
    peak_val = float(equity.loc[:trough].max())
    peak_date = equity.loc[:trough][equity.loc[:trough] >= peak_val - 1e-12].index[-1]
    recovered = equity.loc[trough:][equity.loc[trough:] >= peak_val]
    recovery = recovered.index[0] if not recovered.empty else None
    peak_loc = equity.index.get_loc(peak_date)
    trough_loc = equity.index.get_loc(trough)
    recovery_loc = equity.index.get_loc(recovery) if recovery is not None else None
    return native(
        {
            "max_dd": max_dd,
            "peak_date": peak_date,
            "trough_date": trough,
            "recovery_date": recovery,
            "peak_to_trough_trading_days": trough_loc - peak_loc,
            "trough_to_recovery_trading_days": (
                recovery_loc - trough_loc if recovery_loc is not None else None
            ),
            "peak_to_recovery_trading_days": (
                recovery_loc - peak_loc if recovery_loc is not None else None
            ),
            "time_underwater_pct": float((dd < 0).mean()),
        }
    )


def metrics(
    ret: pd.Series,
    rf: pd.Series,
    start,
    end=None,
    turnover: pd.Series | None = None,
) -> dict:
    selected = ret.loc[start:end].dropna()
    if selected.empty or len(selected) < 5:
        return {}
    equity = (1.0 + selected).cumprod()
    years = max(
        (selected.index[-1] - selected.index[0]).days / 365.25,
        len(selected) / 252.0,
    )
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0)
    vol = float(selected.std() * np.sqrt(252.0))
    excess = selected - rf.reindex(selected.index).fillna(0.0)
    sharpe = (
        float(excess.mean() / selected.std() * np.sqrt(252.0))
        if selected.std() > 0
        else None
    )
    episode = drawdown_episode(equity)
    turn = (
        float(turnover.reindex(selected.index).sum() / years)
        if turnover is not None
        else 0.0
    )
    return native(
        {
            "start": selected.index[0],
            "end": selected.index[-1],
            "cagr": cagr,
            "annualized_vol": vol,
            "sharpe": sharpe,
            "calmar": cagr / abs(episode["max_dd"]) if episode["max_dd"] < 0 else None,
            "total_return": float(equity.iloc[-1] - 1.0),
            "terminal_100k": float(INITIAL * equity.iloc[-1]),
            "worst_month": float(
                ((1.0 + selected).resample("ME").prod().sub(1.0)).min()
            ),
            "annual_turnover": turn,
            "avg_qqq_weight": None,
            "avg_invested": None,
            "tax_paid_frac_equity_years": None,
            "drawdown": episode,
        }
    )


def slim(m: dict) -> dict:
    if not m:
        return {}
    return {
        "cagr": m.get("cagr"),
        "vol": m.get("annualized_vol"),
        "sharpe": m.get("sharpe"),
        "calmar": m.get("calmar"),
        "max_dd": m.get("drawdown", {}).get("max_dd"),
        "total_return": m.get("total_return"),
        "terminal_100k": m.get("terminal_100k"),
        "worst_month": m.get("worst_month"),
        "annual_turnover": m.get("annual_turnover"),
        "avg_qqq_weight": m.get("avg_qqq_weight"),
        "avg_invested": m.get("avg_invested"),
        "annual_tax_drag": m.get("annual_tax_drag"),
        "terminal_after_liquidation": m.get("terminal_after_liquidation"),
        "liquidation_tax": m.get("liquidation_tax"),
        "start": m.get("start"),
        "end": m.get("end"),
    }


def load_data() -> dict:
    raw = yf.download(
        ["SPY", "VOO", "QQQ", "^IRX"],
        start=START,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no data")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    qqq = close["QQQ"].dropna()
    idx = qqq.index
    spy = close["SPY"].reindex(idx).ffill()
    voo = close["VOO"].reindex(idx)
    sap = voo.combine_first(spy)  # VOO when live else SPY
    irx = close["^IRX"].reindex(idx).ffill().fillna(0.0) / 100.0
    rf = irx / 252.0
    r_sap = sap.pct_change().fillna(0.0)
    r_qqq = qqq.pct_change().fillna(0.0)
    return {
        "index": idx,
        "sap": sap,
        "qqq": qqq,
        "r_sap": r_sap,
        "r_qqq": r_qqq,
        "rf": rf,
        "voo_inception": voo.dropna().index.min() if not voo.dropna().empty else None,
    }


def _cap_weights(w_sap: np.ndarray, w_qqq: np.ndarray) -> np.ndarray:
    W = np.column_stack([w_sap, w_qqq])
    W = np.clip(W, 0.0, None)
    row_sum = W.sum(axis=1)
    over = row_sum > 1.0 + 1e-12
    if over.any():
        W[over] = W[over] / row_sum[over, None]
    return W


def _lot_rate(buy_date: pd.Timestamp, sell_date: pd.Timestamp) -> float:
    age = (sell_date - buy_date).days
    return TAX_LT if age > LT_CALENDAR_DAYS else TAX_ST


def _sleeve_mv(lots_j: list[dict]) -> float:
    return float(sum(lot["mv"] for lot in lots_j))


def _sleeve_basis(lots_j: list[dict]) -> float:
    return float(sum(lot["basis"] for lot in lots_j))


def _sell_fifo(
    lots_j: list[dict],
    sell_mv: float,
    sell_date: pd.Timestamp,
    tax_enabled: bool,
) -> tuple[float, float]:
    """Sell sell_mv dollars FIFO. Returns (tax_paid, sold_proceeds)."""
    remaining = sell_mv
    tax_paid = 0.0
    sold = 0.0
    while remaining > 1e-9 and lots_j:
        lot = lots_j[0]
        take = min(remaining, lot["mv"])
        frac = take / lot["mv"]
        basis_sold = frac * lot["basis"]
        gain = take - basis_sold
        if tax_enabled and gain > 0:
            tax_paid += _lot_rate(lot["buy_date"], sell_date) * gain
        lot["mv"] -= take
        lot["basis"] -= basis_sold
        remaining -= take
        sold += take
        if lot["mv"] <= 1e-9:
            lots_j.pop(0)
    return tax_paid, sold


def _rebalance_to_targets(
    lots: list[list[dict]],
    cash: float,
    target_w: np.ndarray,
    trade_date: pd.Timestamp,
    tax_enabled: bool,
) -> tuple[float, float, float]:
    """Trade once toward target weights. Returns cash, traded, tax_paid."""
    mv = np.array([_sleeve_mv(lots[j]) for j in range(2)])
    equity = cash + float(mv.sum())
    if equity < 1e-8:
        return cash, 0.0, 0.0
    target = target_w * equity
    traded = 0.0
    tax_paid = 0.0

    for j in range(2):
        if mv[j] > 1e-12 and target[j] < mv[j] - 1e-12:
            sell = mv[j] - target[j]
            t, sold = _sell_fifo(lots[j], sell, trade_date, tax_enabled)
            tax_paid += t
            cash += sold
            traded += sold
            mv[j] = _sleeve_mv(lots[j])

    cash -= tax_paid
    if cash < 0:
        cash = 0.0

    mv = np.array([_sleeve_mv(lots[j]) for j in range(2)])
    equity = cash + float(mv.sum())
    target = target_w * equity
    for j in range(2):
        if target[j] > mv[j] + 1e-12:
            buy = min(target[j] - mv[j], cash)
            if buy > 0:
                lots[j].append(
                    {"basis": buy, "mv": buy, "buy_date": trade_date}
                )
                cash -= buy
                traded += buy
                mv[j] += buy

    mv = np.array([_sleeve_mv(lots[j]) for j in range(2)])
    equity = cash + float(mv.sum())
    target = target_w * equity
    for j in range(2):
        if mv[j] > 1e-12 and target[j] < mv[j] - 1e-12:
            sell = mv[j] - target[j]
            t, sold = _sell_fifo(lots[j], sell, trade_date, tax_enabled)
            tax_paid += t
            cash += sold - t
            traded += sold
            mv[j] = _sleeve_mv(lots[j])

    cost = COST * traded
    cash -= cost
    if cash < 0:
        # Pay residual cost by trimming largest sleeve (basis pro-rata, no tax event
        # beyond the sale itself)
        jmax = int(np.argmax(mv))
        take = min(-cash, mv[jmax])
        if take > 0:
            t, sold = _sell_fifo(lots[jmax], take, trade_date, tax_enabled)
            tax_paid += t
            cash += sold - t
        cash = max(cash, 0.0)

    return cash, traded, tax_paid


def _liquidate_all(
    lots: list[list[dict]],
    cash: float,
    sell_date: pd.Timestamp,
    tax_enabled: bool,
) -> float:
    """Sell all lots; return after-tax cash (spendable wealth)."""
    for j in range(2):
        mv_j = _sleeve_mv(lots[j])
        if mv_j > 0:
            t, sold = _sell_fifo(lots[j], mv_j, sell_date, tax_enabled)
            cash += sold - t
    return cash


def simulate(
    w_sap: np.ndarray,
    w_qqq: np.ndarray,
    data: dict,
    tax_enabled: bool = False,
    rebalance_flags: np.ndarray | None = None,
) -> dict:
    """Dollar path: drift between rebalance days; FIFO ST/LT tax on sells.

    ST = 24%, LT = 15% after 365 calendar days. Tax only on gains.
    """
    idx = data["index"]
    n = len(idx)
    R = np.column_stack([data["r_sap"].to_numpy(), data["r_qqq"].to_numpy()])
    rf = data["rf"].to_numpy()
    W = _cap_weights(w_sap, w_qqq)
    if rebalance_flags is None:
        flags = np.zeros(n, dtype=bool)
        flags[0] = True
        for i in range(1, n):
            flags[i] = not np.allclose(W[i], W[i - 1], atol=1e-6, rtol=0.0)
    else:
        flags = np.asarray(rebalance_flags, dtype=bool).copy()
        flags[0] = True

    cash = float(INITIAL)
    lots: list[list[dict]] = [[], []]
    equity_prev = float(INITIAL)

    rets = np.zeros(n)
    turns = np.zeros(n)
    taxes = np.zeros(n)
    weights = np.zeros((n, 2))

    for i in range(n):
        for j in range(2):
            for lot in lots[j]:
                lot["mv"] *= 1.0 + R[i, j]
        cash = cash * (1.0 + rf[i])
        mv = np.array([_sleeve_mv(lots[j]) for j in range(2)])
        equity_mtm = max(cash + float(mv.sum()), 1e-8)

        traded = 0.0
        tax_paid = 0.0
        if flags[i]:
            cash, traded, tax_paid = _rebalance_to_targets(
                lots, cash, W[i], idx[i], tax_enabled
            )

        mv = np.array([_sleeve_mv(lots[j]) for j in range(2)])
        equity = max(cash + float(mv.sum()), 1e-8)
        rets[i] = equity / equity_prev - 1.0
        turns[i] = traded / equity_mtm
        taxes[i] = tax_paid / equity_mtm
        weights[i] = mv / equity
        equity_prev = equity

    # End-of-sample liquidation (spendable wealth if everything is sold)
    terminal_pre_liq = cash + sum(_sleeve_mv(lots[j]) for j in range(2))
    if tax_enabled:
        after = _liquidate_all(copy.deepcopy(lots), cash, idx[-1], True)
        liq_tax = max(terminal_pre_liq - after, 0.0)
        terminal_after_liq = after
    else:
        liq_tax = 0.0
        terminal_after_liq = terminal_pre_liq

    return {
        "returns": pd.Series(rets, index=idx),
        "turnover": pd.Series(turns, index=idx),
        "tax_drag": pd.Series(taxes, index=idx),
        "weights": pd.DataFrame(weights, index=idx, columns=["SAP", "QQQ"]),
        "targets": pd.DataFrame(W, index=idx, columns=["SAP", "QQQ"]),
        "tax_enabled": tax_enabled,
        "tax_st_rate": TAX_ST,
        "tax_lt_rate": TAX_LT,
        "terminal_pre_liquidation": float(terminal_pre_liq),
        "terminal_after_liquidation": float(terminal_after_liq),
        "liquidation_tax": float(liq_tax),
    }


def hold_constant(w_qqq: float, data: dict, tax_enabled: bool = False) -> dict:
    n = len(data["index"])
    flags = np.zeros(n, dtype=bool)
    return simulate(
        np.full(n, 1.0 - w_qqq),
        np.full(n, w_qqq),
        data,
        tax_enabled=tax_enabled,
        rebalance_flags=flags,
    )


def buy_and_hold(which: str, data: dict, tax_enabled: bool = False) -> dict:
    n = len(data["index"])
    flags = np.zeros(n, dtype=bool)
    if which == "QQQ":
        return simulate(
            np.zeros(n),
            np.ones(n),
            data,
            tax_enabled=tax_enabled,
            rebalance_flags=flags,
        )
    return simulate(
        np.ones(n),
        np.zeros(n),
        data,
        tax_enabled=tax_enabled,
        rebalance_flags=flags,
    )


def monthly_static(w_qqq: float, data: dict, tax_enabled: bool = False) -> dict:
    """Rebalance to fixed weights at month-end."""
    flags = month_end_flags(data["index"])
    n = len(data["index"])
    w_s = np.full(n, 1.0 - w_qqq)
    w_q = np.full(n, w_qqq)
    return simulate(
        w_s, w_q, data, tax_enabled=tax_enabled, rebalance_flags=flags
    )


def build_signals(data: dict) -> dict:
    idx = data["index"]
    sap, qqq = data["sap"], data["qqq"]
    r_sap, r_qqq = data["r_sap"], data["r_qqq"]

    vol_s = ewma_vol_series(r_sap)
    vol_q = ewma_vol_series(r_qqq)
    var_s, var_q, cov_sq = ewma_cov_pair(r_sap, r_qqq)

    mom_s = sap / sap.shift(REL_MOM_LOOKBACK) - 1.0
    mom_q = qqq / qqq.shift(REL_MOM_LOOKBACK) - 1.0
    rel = (mom_q - mom_s).fillna(0.0)

    sma = qqq.rolling(SMA).mean()
    # hysteresis state on daily closes; sampled at month-end
    risk_on = np.zeros(len(idx), dtype=bool)
    on = False
    q = qqq.to_numpy()
    s = sma.to_numpy()
    for i in range(len(idx)):
        if np.isnan(s[i]):
            risk_on[i] = on
            continue
        if on:
            if q[i] < s[i] * (1.0 - HYST):
                on = False
        else:
            if q[i] > s[i] * (1.0 + HYST):
                on = True
        risk_on[i] = on

    return {
        "vol_s": vol_s,
        "vol_q": vol_q,
        "var_s": var_s,
        "var_q": var_q,
        "cov_sq": cov_sq,
        "mom_s": mom_s,
        "mom_q": mom_q,
        "rel": rel,
        "risk_on": pd.Series(risk_on, index=idx),
        "sma": sma,
    }


def monthly_weights_from_fn(data: dict, weight_fn, tax_enabled: bool = False) -> dict:
    """Call weight_fn(i, signals) on month-end days; trade only on those days."""
    flags = month_end_flags(data["index"])
    n = len(data["index"])
    sig = data["signals"]
    w_s = np.zeros(n)
    w_q = np.zeros(n)
    cur_s, cur_q = 0.5, 0.5
    for i in range(n):
        if flags[i] or i == 0:
            cur_s, cur_q = weight_fn(i, sig, data)
        w_s[i], w_q[i] = cur_s, cur_q
    return simulate(
        w_s, w_q, data, tax_enabled=tax_enabled, rebalance_flags=flags
    )


def w_inv_vol(i: int, sig: dict, data: dict) -> tuple[float, float]:
    vs = float(sig["vol_s"].iloc[i])
    vq = float(sig["vol_q"].iloc[i])
    if not np.isfinite(vs) or not np.isfinite(vq) or vs <= 0 or vq <= 0:
        return 0.5, 0.5
    inv_s, inv_q = 1.0 / vs, 1.0 / vq
    tot = inv_s + inv_q
    return inv_s / tot, inv_q / tot


def w_rel_mom(i: int, sig: dict, data: dict) -> tuple[float, float]:
    m = float(sig["rel"].iloc[i])
    if not np.isfinite(m):
        m = 0.0
    wq = float(np.clip(0.5 + REL_MOM_GAIN * m, 0.0, 1.0))
    return 1.0 - wq, wq


def w_mv_mom(i: int, sig: dict, data: dict) -> tuple[float, float]:
    ms = float(sig["mom_s"].iloc[i])
    mq = float(sig["mom_q"].iloc[i])
    if not np.isfinite(ms) or not np.isfinite(mq):
        return 0.5, 0.5
    # annualize ~126d return
    mu_s = float(np.clip(ms * (252.0 / MV_MOM_LOOKBACK), -MV_MU_CAP, MV_MU_CAP))
    mu_q = float(np.clip(mq * (252.0 / MV_MOM_LOOKBACK), -MV_MU_CAP, MV_MU_CAP))
    if mu_s <= 0 and mu_q <= 0:
        return 0.0, 0.0  # cash

    vs = float(sig["var_s"].iloc[i])
    vq = float(sig["var_q"].iloc[i])
    cv = float(sig["cov_sq"].iloc[i])
    if not all(np.isfinite(x) for x in (vs, vq, cv)) or vs <= 0 or vq <= 0:
        return 0.5, 0.5

    avg_v = 0.5 * (vs + vq)
    # shrink toward diagonal average variance
    s = MV_SHRINK
    cov = np.array(
        [
            [(1 - s) * vs + s * avg_v, (1 - s) * cv],
            [(1 - s) * cv, (1 - s) * vq + s * avg_v],
        ]
    )
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return 0.5, 0.5
    raw = inv @ np.array([mu_s, mu_q])
    raw = np.maximum(raw, 0.0)
    if raw.sum() <= 1e-12:
        return 0.0, 0.0
    w = raw / raw.sum()
    return float(w[0]), float(w[1])


def w_vol_target_5050(i: int, sig: dict, data: dict) -> tuple[float, float]:
    vs = float(sig["var_s"].iloc[i])
    vq = float(sig["var_q"].iloc[i])
    cv = float(sig["cov_sq"].iloc[i])
    if not all(np.isfinite(x) for x in (vs, vq, cv)):
        return 0.5, 0.5
    w = np.array([0.5, 0.5])
    port_var = float(w @ np.array([[vs, cv], [cv, vq]]) @ w)
    if port_var <= 0:
        return 0.5, 0.5
    scale = min(1.0, VOL_TARGET_5050 / np.sqrt(port_var))
    return 0.5 * scale, 0.5 * scale


def w_regime_tilt(i: int, sig: dict, data: dict) -> tuple[float, float]:
    if bool(sig["risk_on"].iloc[i]):
        return 0.30, 0.70
    return 0.40, 0.0  # cash 0.60


def w_combo(i: int, sig: dict, data: dict) -> tuple[float, float]:
    ws, wq = w_rel_mom(i, sig, data)
    # soft risk-off: cut equity book in half when below SMA exit
    q = float(data["qqq"].iloc[i])
    sma = float(sig["sma"].iloc[i])
    if np.isfinite(sma) and q < sma * (1.0 - HYST):
        ws *= 0.5
        wq *= 0.5
    vs = float(sig["var_s"].iloc[i])
    vq = float(sig["var_q"].iloc[i])
    cv = float(sig["cov_sq"].iloc[i])
    invested = ws + wq
    if invested <= 1e-12 or not all(np.isfinite(x) for x in (vs, vq, cv)):
        return ws, wq
    w = np.array([ws, wq]) / invested
    port_var = float(w @ np.array([[vs, cv], [cv, vq]]) @ w)
    if port_var <= 0:
        return ws, wq
    scale = min(1.0, VOL_TARGET_COMBO / np.sqrt(port_var))
    return ws * scale, wq * scale


def w_placebo_rel_mom(i: int, sig: dict, data: dict, signs: np.ndarray) -> tuple[float, float]:
    m = float(sig["rel"].iloc[i]) * float(signs[i])
    if not np.isfinite(m):
        m = 0.0
    wq = float(np.clip(0.5 + REL_MOM_GAIN * m, 0.0, 1.0))
    return 1.0 - wq, wq


def w_lag_rel_mom(i: int, sig: dict, data: dict, lag: int = 63) -> tuple[float, float]:
    j = max(0, i - lag)
    m = float(sig["rel"].iloc[j])
    if not np.isfinite(m):
        m = 0.0
    wq = float(np.clip(0.5 + REL_MOM_GAIN * m, 0.0, 1.0))
    return 1.0 - wq, wq


def annotate(run: dict, start, end=None) -> dict:
    m = metrics(run["returns"], run.get("_rf"), start, end, run["turnover"])
    if not m:
        return {}
    w = run["weights"].loc[start:end]
    m["avg_qqq_weight"] = float(w["QQQ"].mean())
    m["avg_invested"] = float(w.sum(axis=1).mean())
    td = run.get("tax_drag")
    if td is not None:
        sel = td.loc[start:end]
        years = max(
            (sel.index[-1] - sel.index[0]).days / 365.25,
            len(sel) / 252.0,
        )
        # sum of (tax/equity) over days ≈ cumulative tax drag; report annualized
        m["annual_tax_drag"] = float(sel.sum() / years) if years > 0 else 0.0
    else:
        m["annual_tax_drag"] = 0.0
    # Scale end-liquidation haircut onto this window's path terminal
    pre_liq = run.get("terminal_pre_liquidation")
    after_liq = run.get("terminal_after_liquidation")
    path_term = m.get("terminal_100k")
    if (
        pre_liq
        and after_liq is not None
        and path_term is not None
        and pre_liq > 0
    ):
        factor = after_liq / pre_liq
        m["terminal_after_liquidation"] = float(path_term * factor)
        m["liquidation_tax"] = float(path_term - m["terminal_after_liquidation"])
    else:
        m["terminal_after_liquidation"] = path_term
        m["liquidation_tax"] = 0.0
    return m


def pick_static_best(data: dict, tax_enabled: bool = False) -> float:
    """Max Calmar on 2003–2012 (post-dotcom trough through GFC recovery)."""
    best_w, best_calmar = 0.5, -np.inf
    for k in range(0, 11):
        wq = k / 10.0
        run = monthly_static(wq, data, tax_enabled=tax_enabled)
        run["_rf"] = data["rf"]
        m = annotate(run, SELECT_START, SELECT_END)
        cal = m.get("calmar")
        if cal is not None and cal > best_calmar:
            best_calmar = cal
            best_w = wq
    return best_w


def score_window(books: dict[str, dict], data: dict, start, end=None) -> dict:
    out = {}
    for name, run in books.items():
        run["_rf"] = data["rf"]
        out[name] = slim(annotate(run, start, end))
    return out


def build_books(
    data: dict, static_best_w: float, signs: np.ndarray, tax_enabled: bool
) -> dict:
    return {
        # True BH 50/50: buy once, never rebalance (weights drift)
        "BH_5050": hold_constant(0.5, data, tax_enabled=tax_enabled),
        "STATIC_5050": monthly_static(0.5, data, tax_enabled=tax_enabled),
        "STATIC_BEST": monthly_static(static_best_w, data, tax_enabled=tax_enabled),
        "STATIC_070": monthly_static(0.7, data, tax_enabled=tax_enabled),
        "STATIC_100": monthly_static(1.0, data, tax_enabled=tax_enabled),
        "BH_QQQ": buy_and_hold("QQQ", data, tax_enabled=tax_enabled),
        "BH_SPY_VOO": buy_and_hold("SAP", data, tax_enabled=tax_enabled),
        "INV_VOL": monthly_weights_from_fn(data, w_inv_vol, tax_enabled=tax_enabled),
        "REL_MOM": monthly_weights_from_fn(data, w_rel_mom, tax_enabled=tax_enabled),
        "MV_MOM": monthly_weights_from_fn(data, w_mv_mom, tax_enabled=tax_enabled),
        "VOL_TARGET_5050": monthly_weights_from_fn(
            data, w_vol_target_5050, tax_enabled=tax_enabled
        ),
        "REGIME_TILT": monthly_weights_from_fn(
            data, w_regime_tilt, tax_enabled=tax_enabled
        ),
        "COMBO": monthly_weights_from_fn(data, w_combo, tax_enabled=tax_enabled),
        "PLACEBO_REL_MOM": monthly_weights_from_fn(
            data,
            lambda i, s, d: w_placebo_rel_mom(i, s, d, signs),
            tax_enabled=tax_enabled,
        ),
        "LAG_REL_MOM_63": monthly_weights_from_fn(
            data,
            lambda i, s, d: w_lag_rel_mom(i, s, d, 63),
            tax_enabled=tax_enabled,
        ),
    }


def yearly(ret: pd.Series, start) -> dict:
    y = (1.0 + ret.loc[start:]).resample("YE").prod() - 1.0
    return {str(d.year): float(v) for d, v in y.items()}


def utility(m: dict, lam: float) -> float | None:
    if not m or m.get("cagr") is None or m.get("max_dd") is None:
        return None
    return float(m["cagr"] - lam * abs(m["max_dd"]))


def rank_scores(scores: dict) -> list[dict]:
    rows = []
    for name, m in scores.items():
        if not m or m.get("calmar") is None:
            continue
        rows.append(
            {
                "name": name,
                "calmar": m["calmar"],
                "cagr": m["cagr"],
                "sharpe": m["sharpe"],
                "max_dd": m["max_dd"],
                "vol": m["vol"],
                "U_0.5": utility(m, 0.5),
                "U_1.0": utility(m, 1.0),
            }
        )
    rows.sort(key=lambda r: (-r["calmar"], -r["cagr"], -(r["sharpe"] or -99)))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def useful_check(scores: dict) -> dict:
    """Beat 50/50 Calmar and beat 70/30 on Calmar or CAGR; mom books beat placebo."""
    out = {}
    base = scores.get("STATIC_5050") or {}
    agg = scores.get("STATIC_070") or {}
    placebo = scores.get("PLACEBO_REL_MOM") or {}
    for name in CANDIDATES:
        m = scores.get(name) or {}
        if not m or not base or not agg:
            out[name] = {"useful": False, "reason": "missing metrics"}
            continue
        beat_5050 = (m.get("calmar") or -99) > (base.get("calmar") or -99)
        beat_070 = (m.get("calmar") or -99) > (agg.get("calmar") or -99) or (
            m.get("cagr") or -99
        ) > (agg.get("cagr") or -99)
        beat_placebo = True
        if name in MOMENTUM_BASED and placebo:
            beat_placebo = (m.get("calmar") or -99) > (placebo.get("calmar") or -99)
        out[name] = {
            "useful": bool(beat_5050 and beat_070 and beat_placebo),
            "beat_5050_calmar": bool(beat_5050),
            "beat_070_calmar_or_cagr": bool(beat_070),
            "beat_placebo_calmar": bool(beat_placebo),
            "cagr": m.get("cagr"),
            "calmar": m.get("calmar"),
            "max_dd": m.get("max_dd"),
            "U_0.5": utility(m, 0.5),
            "U_1.0": utility(m, 1.0),
        }
    return out


def fmt_row(name: str, m: dict, with_tax: bool = False) -> str:
    tax_col = (
        f" | {(m.get('annual_tax_drag') or 0):.2%}" if with_tax else ""
    )
    return (
        f"| {name} | {m['cagr']:.2%} | {m['vol']:.2%} | "
        f"{m['sharpe']:.2f} | {m['max_dd']:.2%} | {m['calmar']:.2f} | "
        f"{utility(m, 0.5):.2%} | "
        f"{(m.get('avg_qqq_weight') or 0):.1%} | "
        f"{(m.get('avg_invested') or 0):.1%}{tax_col} |"
    )


def compare_pre_post(pre: dict, post: dict) -> list[dict]:
    rows = []
    for name in DISPLAY_ORDER:
        a = pre.get(name) or {}
        b = post.get(name) or {}
        if not a or not b:
            continue
        rows.append(
            {
                "name": name,
                "cagr_pre": a.get("cagr"),
                "cagr_post": b.get("cagr"),
                "cagr_delta_pp": (
                    None
                    if a.get("cagr") is None or b.get("cagr") is None
                    else b["cagr"] - a["cagr"]
                ),
                "calmar_pre": a.get("calmar"),
                "calmar_post": b.get("calmar"),
                "max_dd_pre": a.get("max_dd"),
                "max_dd_post": b.get("max_dd"),
                "terminal_pre": a.get("terminal_100k"),
                "terminal_post": b.get("terminal_100k"),
                "terminal_after_liq": b.get("terminal_after_liquidation"),
                "liquidation_tax": b.get("liquidation_tax"),
                "annual_tax_drag": b.get("annual_tax_drag"),
            }
        )
    rows.sort(key=lambda r: (r["cagr_delta_pp"] is None, r["cagr_delta_pp"] or 0))
    return rows


def run_layer(
    data: dict,
    books: dict,
    eval_start,
    windows: dict,
) -> dict:
    window_scores = {}
    for wname, (a, b) in windows.items():
        window_scores[wname] = score_window(books, data, a, b)
    crisis_scores = {}
    for cname, (a, b) in CRISES.items():
        crisis_scores[cname] = score_window(books, data, a, b)
    return {
        "windows": window_scores,
        "crises": crisis_scores,
        "full_ranks": rank_scores(window_scores["full"]),
        "holdout_ranks": rank_scores(window_scores["holdout_from_2018"]),
        "useful": useful_check(window_scores["full"]),
        "yearly": {
            name: yearly(run["returns"], eval_start) for name, run in books.items()
        },
    }


def write_summary(results: dict) -> None:
    pre = results["pre_tax"]
    post = results["after_tax"]
    full_pre = pre["windows"]["full"]
    full_post = post["windows"]["full"]
    hold_post = post["windows"]["holdout_from_2018"]
    ranks = post["full_ranks"]
    useful = post["useful"]
    static_best_w = results["static_best_w_qqq"]
    eval_start = results["eval_start"]
    cmp_rows = results["tax_comparison_full"]

    lines = [
        "# V11 VOO/QQQ allocation research (with retail tax)",
        "",
        "Universe: SPY/VOO + QQQ only. Monthly T+1, 5 bps, cash at T-bills.",
        f"**Tax (~$150k income):** ST {TAX_ST:.0%} / LT {TAX_LT:.0%} "
        f"(FIFO lots, LT after {LT_CALENDAR_DAYS}d). Gain-only.",
        f"Data from {START}; **primary metrics from {eval_start}** (252d warmup).",
        f"STATIC_BEST: **{static_best_w:.0%} QQQ** "
        f"(max Calmar on {SELECT_START} -> {SELECT_END}, selected pre-tax).",
        "",
        "## Pre-tax vs after-tax (full sample)",
        "",
        "Path CAGR uses taxes paid on rebalance sells. "
        "**Liq. terminal** = spendable cash if all holdings are sold on the last day "
        "(BH pays ~LT 15% then; active books already paid some tax along the way).",
        "",
        "| Strategy | CAGR pre | CAGR after | Delta | Calmar after | "
        "Tax drag/yr | Term pre | Term path | Term after liq |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by_name = {r["name"]: r for r in cmp_rows}
    for name in DISPLAY_ORDER:
        r = by_name.get(name)
        if not r:
            continue
        liq = r.get("terminal_after_liq") or r["terminal_post"]
        lines.append(
            f"| {name} | {r['cagr_pre']:.2%} | {r['cagr_post']:.2%} | "
            f"{r['cagr_delta_pp']*100:+.2f}pp | "
            f"{r['calmar_post']:.2f} | {r['annual_tax_drag']:.2%} | "
            f"${r['terminal_pre']:,.0f} | ${r['terminal_post']:,.0f} | "
            f"${liq:,.0f} |"
        )

    lines += [
        "",
        "## After-tax full sample (primary ranking)",
        "",
        "| Strategy | CAGR | Vol | Sharpe | MaxDD | Calmar | U0.5 | Avg QQQ | Invested | Tax/yr |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in DISPLAY_ORDER:
        m = full_post.get(name) or {}
        if m:
            lines.append(fmt_row(name, m, with_tax=True))

    lines += ["", "## After-tax ranking (Calmar -> CAGR -> Sharpe)", ""]
    for r in ranks[:14]:
        lines.append(
            f"{r['rank']}. **{r['name']}** — Calmar {r['calmar']:.2f}, "
            f"CAGR {r['cagr']:.2%}, Sharpe {r['sharpe']:.2f}, "
            f"maxDD {r['max_dd']:.2%}, U0.5 {r['U_0.5']:.2%}"
        )

    lines += ["", "## Useful rule (after-tax full sample)", ""]
    lines.append(
        "Useful = Calmar > STATIC_5050, and (Calmar or CAGR) > STATIC_070; "
        "momentum books must also beat PLACEBO_REL_MOM Calmar."
    )
    lines.append("")
    for name in CANDIDATES:
        u = useful[name]
        flag = "YES" if u["useful"] else "no"
        lines.append(
            f"- `{name}`: **{flag}** "
            f"(CAGR {u.get('cagr', float('nan')):.2%}, "
            f"Calmar {u.get('calmar', float('nan')):.2f}, "
            f"maxDD {u.get('max_dd', float('nan')):.2%})"
        )

    lines += [
        "",
        "## After-tax major crashes (total return / max DD)",
        "",
    ]
    crash_names = list(CRISES.keys())
    lines.append("| Strategy | " + " | ".join(crash_names) + " |")
    lines.append("|---|" + "|".join(["---:" for _ in crash_names]) + "|")
    for name in CRASH_REPORT_BOOKS:
        cells = []
        for cname in crash_names:
            m = (post["crises"].get(cname) or {}).get(name) or {}
            if not m:
                cells.append("--")
            else:
                cells.append(
                    f"{m.get('total_return', float('nan')):.1%}/"
                    f"{m.get('max_dd', float('nan')):.1%}"
                )
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    lines += ["", "## After-tax holdout from 2018", ""]
    lines.append(
        "| Strategy | CAGR | Sharpe | MaxDD | Calmar | Tax/yr |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name in DISPLAY_ORDER:
        m = hold_post.get(name) or {}
        if not m:
            continue
        lines.append(
            f"| {name} | {m['cagr']:.2%} | {m['sharpe']:.2f} | "
            f"{m['max_dd']:.2%} | {m['calmar']:.2f} | "
            f"{(m.get('annual_tax_drag') or 0):.2%} |"
        )

    lines += ["", "## Pre-tax full sample (reference)", ""]
    lines.append(
        "| Strategy | CAGR | MaxDD | Calmar |"
    )
    lines.append("|---|---:|---:|---:|")
    for name in [
        "MV_MOM",
        "COMBO",
        "REGIME_TILT",
        "REL_MOM",
        "BH_5050",
        "STATIC_5050",
        "STATIC_070",
        "BH_QQQ",
    ]:
        m = full_pre.get(name) or {}
        if m:
            lines.append(
                f"| {name} | {m['cagr']:.2%} | {m['max_dd']:.2%} | {m['calmar']:.2f} |"
            )

    lines += ["", "## Verdict", "", results["verdict"], ""]
    lines += [
        "## Limitations",
        "",
        "- Tax model: federal ST 24% / LT 15% only; no state tax, NIIT, or wash-sale rules.",
        "- Assumes ~$150k taxable income stays in the 15% LTCG band.",
        "- Losses do not create a tax credit (tax = 0 on losing sells).",
        "- T-bill interest untaxed; path CAGR for BH defers tax until liquidation column.",
        "- Two-asset universe only; crash protection is cash/tilt only.",
        "",
    ]
    OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")


def build_verdict(results: dict) -> str:
    pre_full = results["pre_tax"]["windows"]["full"]
    post_full = results["after_tax"]["windows"]["full"]
    ranks = results["after_tax"]["full_ranks"]
    useful = results["after_tax"]["useful"]
    best_name = ranks[0]["name"] if ranks else "n/a"
    useful_names = [n for n, u in useful.items() if u["useful"]]

    def pack(name: str) -> str:
        a = pre_full.get(name) or {}
        b = post_full.get(name) or {}
        d = (b.get("cagr") or 0) - (a.get("cagr") or 0)
        return (
            f"{name} pre {a.get('cagr', float('nan')):.2%} -> "
            f"after {b.get('cagr', float('nan')):.2%} "
            f"({d*100:+.2f}pp), Calmar {b.get('calmar', float('nan')):.2f}"
        )

    cmp_rows = results["tax_comparison_full"]
    worst = min(
        cmp_rows,
        key=lambda r: r["cagr_delta_pp"] if r["cagr_delta_pp"] is not None else 0,
    )
    least = max(
        cmp_rows,
        key=lambda r: r["cagr_delta_pp"] if r["cagr_delta_pp"] is not None else -99,
    )
    # Best spendable wealth after full liquidation
    liq_best = max(
        cmp_rows,
        key=lambda r: r.get("terminal_after_liq") or r.get("terminal_post") or 0,
    )

    parts = []
    parts.append(
        f"Tax model ST {TAX_ST:.0%}/LT {TAX_LT:.0%} at ~$150k income. "
        f"After-tax path Calmar leader: **{best_name}**. "
        f"{pack('MV_MOM')}. {pack('COMBO')}. {pack('REGIME_TILT')}. "
        f"{pack('BH_5050')}. {pack('STATIC_5050')}. {pack('BH_QQQ')}."
    )
    parts.append(
        f"Largest path-CAGR tax hit: **{worst['name']}** "
        f"{worst['cagr_delta_pp']*100:+.2f}pp "
        f"(drag ~{worst['annual_tax_drag']:.2%}/yr). "
        f"Least path hit: **{least['name']}** "
        f"{least['cagr_delta_pp']*100:+.2f}pp."
    )
    parts.append(
        f"Best end liquidation (spendable) terminal: **{liq_best['name']}** "
        f"${(liq_best.get('terminal_after_liq') or 0):,.0f} "
        f"(BH pays ~{TAX_LT:.0%} LT on embedded gains at exit)."
    )
    if useful_names:
        parts.append(f"Useful after tax: {', '.join(useful_names)}.")
    else:
        parts.append("No candidate cleared the after-tax useful rule.")
    return " ".join(parts)


def static_ladder(
    data: dict, start, end=None, tax_enabled: bool = False
) -> list[dict]:
    rows = []
    for k in range(0, 11):
        wq = k / 10.0
        run = monthly_static(wq, data, tax_enabled=tax_enabled)
        run["_rf"] = data["rf"]
        m = slim(annotate(run, start, end))
        rows.append(
            {
                "w_qqq": wq,
                "cagr": m.get("cagr"),
                "calmar": m.get("calmar"),
                "max_dd": m.get("max_dd"),
                "sharpe": m.get("sharpe"),
                "U_0.5": utility(m, 0.5),
                "U_1.0": utility(m, 1.0),
                "annual_tax_drag": m.get("annual_tax_drag"),
            }
        )
    return rows


def main() -> None:
    data = load_data()
    data["signals"] = build_signals(data)
    idx = data["index"]
    eval_start = idx[WARMUP_DAYS]

    static_best_w = pick_static_best(data, tax_enabled=False)
    print(f"STATIC_BEST select w_QQQ = {static_best_w:.1%}")
    print(f"Primary eval start = {eval_start.date()}")
    print(f"Tax ST/LT = {TAX_ST:.0%}/{TAX_LT:.0%} (income ~$150k)")

    rng = np.random.default_rng(11)
    month_ids = idx.to_period("M")
    month_sign = {m: rng.choice([-1.0, 1.0]) for m in month_ids.unique()}
    signs = np.array([month_sign[m] for m in month_ids])

    last = idx[-1]
    last_24 = last - pd.DateOffset(months=24)
    windows = {
        "full": (eval_start, None),
        "from_2010": ("2010-01-01", None),
        "from_2015": ("2015-01-01", None),
        "holdout_from_2018": ("2018-01-01", None),
        "last_24m": (last_24, None),
        "select_2003_2012": (SELECT_START, SELECT_END),
    }

    layers = {}
    for label, enabled in (("pre_tax", False), ("after_tax", True)):
        print(f"building books ({label})...")
        books = build_books(data, static_best_w, signs, tax_enabled=enabled)
        layers[label] = run_layer(data, books, eval_start, windows)
        print(f"scored {label}")

    tax_comparison_full = compare_pre_post(
        layers["pre_tax"]["windows"]["full"],
        layers["after_tax"]["windows"]["full"],
    )
    tax_comparison_holdout = compare_pre_post(
        layers["pre_tax"]["windows"]["holdout_from_2018"],
        layers["after_tax"]["windows"]["holdout_from_2018"],
    )

    results = {
        "design": "research/v11_voo_qqq/DESIGN.md",
        "tax_st": TAX_ST,
        "tax_lt": TAX_LT,
        "income_assumption": 150_000,
        "voo_inception": native(data["voo_inception"]),
        "eval_start": native(eval_start),
        "static_best_w_qqq": static_best_w,
        "static_ladder_select_pretax": native(
            static_ladder(data, SELECT_START, SELECT_END, tax_enabled=False)
        ),
        "static_ladder_full_aftertax": native(
            static_ladder(data, eval_start, None, tax_enabled=True)
        ),
        "pre_tax": layers["pre_tax"],
        "after_tax": layers["after_tax"],
        "tax_comparison_full": native(tax_comparison_full),
        "tax_comparison_holdout": native(tax_comparison_holdout),
    }
    results["verdict"] = build_verdict(results)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(native(results), indent=2), encoding="utf-8")
    write_summary(results)
    print(results["verdict"].encode("ascii", "replace").decode("ascii"))
    print(f"Wrote {OUT_SUMMARY}")



if __name__ == "__main__":
    main()


