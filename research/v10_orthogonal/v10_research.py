"""V10: orthogonal premia — VRP (short vol) + short-term mean reversion.

Primary candidate: R50d_plus (live R50d + VRP sleeve + MR sleeve)
Standalones: VRP_sleeve, MR_sleeve
Placebos: VRP_nogate, MR_notrend
Baseline: live R50d

Pre-registered in research/v10_orthogonal/DESIGN.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = REPO_ROOT / "output" / "v10_orthogonal_results.json"
OUT_SUMMARY = REPO_ROOT / "output" / "v10_orthogonal_summary.md"

START = "1999-03-10"
SIGP = 0.50
BAND = 0.05
COST_CORE = 0.0005   # R50d legs
COST_NEW = 0.0010    # SVXY and MR legs
EWMA_ALPHA = 0.06
SPREAD = 0.005 / 252.0
EXP = 0.0095 / 252.0
INITIAL = 100_000.0

SLEEVE_ORDER = ["TQQQ", "TMF", "UGL", "DBC", "DBMF", "BTC"]
BASE_BUDGETS = {
    "TQQQ": 0.40,
    "TMF": 0.15,
    "UGL": 0.15,
    "DBC": 0.10,
    "DBMF": 0.10,
    "BTC": 0.10,
}
LEVERAGE = {
    "TQQQ": 3.0,
    "TMF": 3.0,
    "UGL": 2.0,
    "DBC": 1.0,
    "DBMF": 1.0,
    "BTC": 1.0,
}

# V10 sleeve parameters (pre-registered)
B_V = 0.15
VRP_ENTER = 1.05
VRP_EXIT = 1.00
SVXY_HALF_DATE = pd.Timestamp("2018-02-28")

B_M = 0.20
MR_ENTRY = 10.0
MR_EXIT = 60.0
MR_MAX_HOLD = 10

FULL_COMMON = pd.Timestamp("2012-07-02")


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


def gate_ens(px: pd.Series) -> np.ndarray:
    votes = [
        (px / px.shift(L) - 1 > 0).astype(float).where(~px.isna(), 0.0)
        for L in (63, 126, 252)
    ]
    return pd.concat(votes, axis=1).mean(axis=1).fillna(0.0).to_numpy()


def ewma_vol(px: pd.Series) -> np.ndarray:
    r = px.pct_change().fillna(0.0)
    return np.sqrt(252.0 * (r**2).ewm(alpha=EWMA_ALPHA, adjust=False).mean()).to_numpy()


def synth(u: pd.Series, L: float, rf: pd.Series) -> tuple[pd.Series, pd.Series]:
    r = u.pct_change()
    return (L * r - (L - 1) * (rf + SPREAD) - EXP), ~r.isna()


def band_positions(W: np.ndarray) -> np.ndarray:
    P = np.zeros_like(W)
    held = np.zeros(W.shape[1])
    for i in range(len(W)):
        for j in range(W.shape[1]):
            if abs(W[i, j] - held[j]) > BAND or (W[i, j] == 0.0 and held[j] != 0.0):
                held[j] = W[i, j]
            P[i, j] = held[j]
    return P


def rsi2(px: pd.Series) -> pd.Series:
    """Wilder RSI, period 2."""
    d = px.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    au = up.ewm(alpha=0.5, adjust=False).mean()
    ad = dn.ewm(alpha=0.5, adjust=False).mean()
    rsi = 100.0 - 100.0 / (1.0 + au / ad.where(ad > 0))
    rsi = rsi.where(ad > 0, np.where(au > 0, 100.0, 50.0))
    return rsi


def vrp_state(ratio: pd.Series, enter: float, exit_: float) -> np.ndarray:
    out = np.zeros(len(ratio))
    on = False
    for i, x in enumerate(ratio.to_numpy()):
        if np.isfinite(x):
            if on and x <= exit_ + 1e-12:
                on = False
            elif not on and x >= enter - 1e-12:
                on = True
        out[i] = 1.0 if on else 0.0
    return out


def mr_positions(
    px: pd.Series,
    entry_th: float = MR_ENTRY,
    exit_th: float = MR_EXIT,
    max_hold: int = MR_MAX_HOLD,
    trend: bool = True,
) -> np.ndarray:
    """1/0 target as of each close (T+1 shift happens in PnL)."""
    r = rsi2(px).to_numpy()
    sma = px.rolling(200).mean().to_numpy()
    p = px.to_numpy()
    pos = np.zeros(len(px))
    in_pos = False
    held = 0
    for i in range(len(px)):
        if in_pos:
            held += 1
            if (np.isfinite(r[i]) and r[i] > exit_th) or held >= max_hold:
                in_pos = False
                held = 0
        else:
            ok_trend = (not trend) or (np.isfinite(sma[i]) and p[i] > sma[i])
            if np.isfinite(r[i]) and r[i] < entry_th and ok_trend and np.isfinite(sma[i]):
                in_pos = True
                held = 0
        pos[i] = 1.0 if in_pos else 0.0
    return pos


def load_data() -> dict:
    tick = [
        "QQQ", "SPY", "TLT", "GLD", "DBC", "DBMF", "TQQQ", "TMF", "UGL",
        "BTC-USD", "AQMIX", "^IRX", "^VIX", "^VIX3M", "SVXY",
    ]
    raw = yf.download(tick, start=START, auto_adjust=True, progress=False, threads=True)
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no data")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    qqq = close["QQQ"].dropna()
    idx = qqq.index
    n = len(idx)

    def al(c: str) -> pd.Series:
        return close[c].reindex(idx)

    spy = al("SPY").ffill()
    tlt, gld, dbc = al("TLT"), al("GLD"), al("DBC")
    tqqq, tmf, ugl = al("TQQQ"), al("TMF"), al("UGL")
    btc = al("BTC-USD")
    irx = al("^IRX").ffill().fillna(0.0) / 100.0
    rf = irx / 252.0

    r_ndx3, _ = synth(qqq, 3, rf)
    r_ndx3 = r_ndx3.fillna(0.0)
    ov = tqqq.pct_change().dropna().index
    r_ndx3.loc[ov] = tqqq.pct_change().loc[ov]

    r_tlt3, v_tlt = synth(tlt, 3, rf)
    r_tlt3 = r_tlt3.fillna(0.0)
    ov_tmf = tmf.pct_change().dropna().index
    r_tlt3.loc[ov_tmf] = tmf.pct_change().loc[ov_tmf]

    r_gld2, v_gld = synth(gld, 2, rf)
    r_gld2 = r_gld2.fillna(0.0)
    ov_ugl = ugl.pct_change().dropna().index
    r_gld2.loc[ov_ugl] = ugl.pct_change().loc[ov_ugl]

    r_dbc1 = dbc.pct_change().fillna(0.0)
    v_dbc = ~dbc.pct_change().isna()

    r_mf = al("AQMIX").pct_change()
    r_db = al("DBMF").pct_change()
    ovm = r_db.dropna().index
    r_mf.loc[ovm] = r_db.loc[ovm]
    v_mf = ~r_mf.isna()
    px_mf = (1 + r_mf.fillna(0.0)).cumprod().where(v_mf.cummax())
    r_mf = r_mf.fillna(0.0)

    r_btc1 = (btc.pct_change() - 0.01 / 252.0).fillna(0.0)
    v_btc = ~btc.pct_change().isna()

    gates = {
        "TQQQ": gate_ens(qqq),
        "TMF": gate_ens(tlt),
        "UGL": gate_ens(gld),
        "DBC": gate_ens(dbc),
        "DBMF": gate_ens(px_mf),
        "BTC": gate_ens(btc),
    }
    vols = {
        "TQQQ": ewma_vol(qqq),
        "TMF": ewma_vol(tlt),
        "UGL": ewma_vol(gld),
        "DBC": ewma_vol(dbc),
        "DBMF": ewma_vol(px_mf),
        "BTC": ewma_vol(btc),
    }
    valid = {
        "TQQQ": np.ones(n, dtype=bool),
        "TMF": v_tlt.to_numpy(),
        "UGL": v_gld.to_numpy(),
        "DBC": v_dbc.to_numpy(),
        "DBMF": v_mf.to_numpy(),
        "BTC": v_btc.to_numpy(),
    }
    rets = {
        "TQQQ": r_ndx3,
        "TMF": r_tlt3,
        "UGL": r_gld2,
        "DBC": r_dbc1,
        "DBMF": r_mf,
        "BTC": r_btc1,
    }

    # V10 inputs
    vix = al("^VIX").ffill()
    vix3m = al("^VIX3M").ffill()
    ratio = vix3m / vix

    svxy = al("SVXY")
    r_svxy = svxy.pct_change()
    v_svxy = ~r_svxy.isna()
    r_svxy_mod = r_svxy.copy()
    pre = r_svxy_mod.index < SVXY_HALF_DATE
    r_svxy_mod[pre] = r_svxy_mod[pre] * 0.5
    px_svxy_mod = (1 + r_svxy_mod.fillna(0.0)).cumprod().where(v_svxy.cummax())
    r_svxy_mod = r_svxy_mod.fillna(0.0)
    r_svxy_act = r_svxy.fillna(0.0)

    return {
        "index": idx,
        "n": n,
        "rf": rf,
        "gates": gates,
        "vols": vols,
        "valid": valid,
        "rets": rets,
        "spy": spy.pct_change().fillna(0.0),
        "qqq_px": qqq,
        "qqq": qqq.pct_change().fillna(0.0),
        "ts_ratio": ratio,
        "svxy_valid": v_svxy.to_numpy(),
        "r_svxy_mod": r_svxy_mod,
        "r_svxy_act": r_svxy_act,
        "svxy_vol_mod": ewma_vol(px_svxy_mod),
        "inception": {
            "SVXY": str(svxy.dropna().index.min().date()) if not svxy.dropna().empty else None,
            "VIX3M": str(vix3m.dropna().index.min().date()) if not vix3m.dropna().empty else None,
        },
    }


def r50d_weights(data: dict) -> np.ndarray:
    n = data["n"]
    W = np.zeros((n, len(SLEEVE_ORDER)))
    for j, sym in enumerate(SLEEVE_ORDER):
        gate = data["gates"][sym]
        vol = data["vols"][sym]
        valid = data["valid"][sym]
        w = gate * BASE_BUDGETS[sym] * SIGP / (LEVERAGE[sym] * np.where(vol > 1e-8, vol, np.nan))
        W[:, j] = np.nan_to_num(np.where(valid, w, 0.0), nan=0.0)
    tot = W.sum(axis=1)
    W *= np.where(tot > 1.0, 1.0 / np.maximum(tot, 1e-12), 1.0)[:, None]
    return W


def vrp_weights(data: dict, enter: float = VRP_ENTER, exit_: float = VRP_EXIT, gated: bool = True) -> np.ndarray:
    state = vrp_state(data["ts_ratio"], enter, exit_) if gated else np.ones(data["n"])
    vol = data["svxy_vol_mod"]
    w = state * B_V * SIGP / np.where(vol > 1e-8, vol, np.nan)
    w = np.minimum(np.nan_to_num(w, nan=0.0), 1.5 * B_V)
    return np.where(data["svxy_valid"], w, 0.0)


def mr_weights(data: dict, entry_th: float = MR_ENTRY, trend: bool = True) -> np.ndarray:
    return mr_positions(data["qqq_px"], entry_th=entry_th, trend=trend) * B_M


def book_returns(data: dict, W: np.ndarray, R: np.ndarray, costs: np.ndarray) -> dict:
    P = band_positions(W)
    Pl = np.vstack([np.zeros((1, P.shape[1])), P[:-1]])
    dP = np.abs(np.diff(np.vstack([np.zeros((1, Pl.shape[1])), Pl]), axis=0))
    turn = dP.sum(axis=1)
    ret = (Pl * R).sum(axis=1) + (1.0 - Pl.sum(axis=1)) * data["rf"].to_numpy() - dP @ costs
    return {
        "returns": pd.Series(ret, index=data["index"]),
        "turnover": pd.Series(turn, index=data["index"]),
        "positions": pd.DataFrame(P, index=data["index"]),
    }


def build_all(data: dict, vrp_th=(VRP_ENTER, VRP_EXIT), mr_entry=MR_ENTRY, cost_mult=1.0) -> dict:
    R6 = np.column_stack([data["rets"][s].to_numpy() for s in SLEEVE_ORDER])
    r_v = data["r_svxy_mod"].to_numpy()
    r_m = data["qqq"].to_numpy()

    W6 = r50d_weights(data)
    Wv = vrp_weights(data, *vrp_th)
    Wm = mr_weights(data, entry_th=mr_entry)

    c6 = np.full(6, COST_CORE * cost_mult)
    c1 = np.array([COST_NEW * cost_mult])
    c8 = np.concatenate([c6, c1, c1])

    books = {
        "R50d": book_returns(data, W6, R6, c6),
        "VRP_sleeve": book_returns(data, Wv[:, None], r_v[:, None], c1),
        "MR_sleeve": book_returns(data, Wm[:, None], r_m[:, None], c1),
    }

    W8 = np.column_stack([W6, Wv, Wm])
    tot = W8.sum(axis=1)
    W8 = W8 * np.where(tot > 1.0, 1.0 / np.maximum(tot, 1e-12), 1.0)[:, None]
    R8 = np.column_stack([R6, r_v, r_m])
    books["R50d_plus"] = book_returns(data, W8, R8, c8)

    if cost_mult == 1.0 and vrp_th == (VRP_ENTER, VRP_EXIT) and mr_entry == MR_ENTRY:
        Wvn = vrp_weights(data, gated=False)
        Wmn = mr_weights(data, trend=False)
        books["VRP_nogate"] = book_returns(data, Wvn[:, None], r_v[:, None], c1)
        books["MR_notrend"] = book_returns(data, Wmn[:, None], r_m[:, None], c1)
        # actual −1x-era SVXY prices as diagnostic
        Wva = vrp_weights(data)
        books["VRP_actual_svxy"] = book_returns(
            data, Wva[:, None], data["r_svxy_act"].to_numpy()[:, None], c1
        )
    return books


def drawdown_episode(equity: pd.Series) -> dict:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    trough = dd.idxmin()
    peak_val = float(peak.loc[trough])
    peak_date = equity.loc[:trough][equity.loc[:trough] >= peak_val - 1e-12].index[-1]
    recovered = equity.loc[trough:][equity.loc[trough:] >= peak_val]
    recovery = recovered.index[0] if not recovered.empty else None
    peak_loc = equity.index.get_loc(peak_date)
    trough_loc = equity.index.get_loc(trough)
    recovery_loc = equity.index.get_loc(recovery) if recovery is not None else None
    return native(
        {
            "max_dd": float(dd.loc[trough]),
            "peak_date": peak_date,
            "trough_date": trough,
            "recovery_date": recovery,
            "peak_to_trough_trading_days": trough_loc - peak_loc,
            "trough_to_recovery_trading_days": (
                recovery_loc - trough_loc if recovery_loc is not None else None
            ),
            "time_underwater_pct": float((dd < 0).mean()),
        }
    )


def metrics(ret: pd.Series, rf: pd.Series, start, end=None, turnover: pd.Series | None = None) -> dict:
    selected = ret.loc[start:end].dropna()
    if selected.empty:
        return {}
    equity = (1.0 + selected).cumprod()
    years = max((selected.index[-1] - selected.index[0]).days / 365.25, len(selected) / 252.0)
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0)
    vol = float(selected.std() * np.sqrt(252.0))
    excess = selected - rf.reindex(selected.index).fillna(0.0)
    sharpe = float(excess.mean() / selected.std() * np.sqrt(252.0)) if selected.std() > 0 else None
    episode = drawdown_episode(equity)
    turn = float(turnover.reindex(selected.index).sum() / years) if turnover is not None else 0.0
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
            "worst_month": float(((1.0 + selected).resample("ME").prod().sub(1.0)).min()),
            "annual_turnover": turn,
            "drawdown": episode,
        }
    )


def yearly(ret: pd.Series, start) -> dict:
    y = (1.0 + ret.loc[start:]).resample("YE").prod() - 1.0
    return {str(d.year): float(v) for d, v in y.items()}


def mr_trade_stats(data: dict, entry_th: float = MR_ENTRY, trend: bool = True) -> dict:
    """Signal-level per-trade stats from full QQQ history (T+1 holding)."""
    pos = mr_positions(data["qqq_px"], entry_th=entry_th, trend=trend)
    held = np.concatenate([[0.0], pos[:-1]])  # T+1
    r = data["qqq"].to_numpy()
    trades = []
    cur = None
    for i in range(len(held)):
        if held[i] > 0:
            cur = (1.0 + r[i]) if cur is None else cur * (1.0 + r[i])
        elif cur is not None:
            trades.append(cur - 1.0)
            cur = None
    if cur is not None:
        trades.append(cur - 1.0)
    t = np.array(trades)
    if len(t) == 0:
        return {"n_trades": 0}
    return native(
        {
            "n_trades": len(t),
            "hit_rate": float((t > 0).mean()),
            "mean_per_trade": float(t.mean()),
            "median_per_trade": float(np.median(t)),
            "worst_trade": float(t.min()),
            "best_trade": float(t.max()),
            "avg_sessions_in_market_per_year": float(held.sum() / (len(held) / 252.0)),
        }
    )


def write_summary(results: dict) -> None:
    p = results["windows"]["from_2015"]
    base, cand = p["R50d"], p["R50d_plus"]
    fc = results["windows"]["full_common"]
    y = results["yearly"]
    c = results["conclusion"]
    lines = [
        "# V10 Orthogonal Premia Research",
        "",
        f"Data: {results['meta']['data_start']} through {results['meta']['data_end']}.",
        "Primary evaluation window: from_2015. Design: research/v10_orthogonal/DESIGN.md.",
        "",
        "## Primary candidate vs live R50d (from_2015)",
        "",
        f"- Live R50d: {base['cagr']:.2%} CAGR, {base['annualized_vol']:.2%} vol, "
        f"Sharpe {base['sharpe']:.2f}, {base['drawdown']['max_dd']:.2%} max DD, Calmar {base['calmar']:.2f}",
        f"- R50d_plus: {cand['cagr']:.2%} CAGR, {cand['annualized_vol']:.2%} vol, "
        f"Sharpe {cand['sharpe']:.2f}, {cand['drawdown']['max_dd']:.2%} max DD, Calmar {cand['calmar']:.2f}",
        f"- CAGR delta: {(cand['cagr'] - base['cagr']) * 100:+.2f} pp (need >= +1.00)",
        f"- Max DD delta: {(cand['drawdown']['max_dd'] - base['drawdown']['max_dd']) * 100:+.2f} pp "
        f"(positive = shallower; need >= -1.00)",
        f"- Sharpe delta: {cand['sharpe'] - base['sharpe']:+.2f} (need >= 0)",
        f"- 2022: R50d {y['R50d'].get('2022', float('nan')):.2%}, "
        f"R50d_plus {y['R50d_plus'].get('2022', float('nan')):.2%} (need >= base - 1pp)",
        "",
        "## Standalone sleeves (full_common, net of 10 bps)",
        "",
    ]
    for name in ("VRP_sleeve", "MR_sleeve", "VRP_nogate", "MR_notrend", "VRP_actual_svxy"):
        m = fc.get(name)
        if m:
            corr = results["sleeve_correlation_to_R50d"].get(name)
            corr_s = f", corr to R50d {corr:+.2f}" if corr is not None else ""
            lines.append(
                f"- {name}: {m['cagr']:.2%} CAGR, {m['annualized_vol']:.2%} vol, "
                f"Sharpe {m['sharpe']:.2f}, {m['drawdown']['max_dd']:.2%} max DD{corr_s}"
            )
    lines += [
        "",
        "## MR signal history (QQQ, 1999+)",
        "",
        f"```\n{json.dumps(results['mr_trade_stats'], indent=2)}\n```",
        "",
        "## Robustness neighborhood (R50d_plus, from_2015: CAGR / max DD)",
        "",
    ]
    for k, v in results["neighborhood"].items():
        lines.append(f"- {k}: {v['cagr']:.2%} / {v['max_dd']:.2%}")
    stress = results["cost_stress_from_2015"]
    lines += [
        "",
        f"Cost stress (2x all legs, from_2015): R50d_plus {stress['cagr']:.2%} CAGR, "
        f"{stress['max_dd']:.2%} max DD.",
        "",
        "## Decision",
        "",
        f"**{c['verdict'].upper()}** — {c['reason']}",
        "",
        "## Limitations",
        "",
        "- SVXY pre-2018 uses the half-leverage proxy of the actual -1x fund; the",
        "  -1x era diagnostic is reported separately.",
        "- Yahoo ^VIX3M can lag a few sessions; live use needs the CBOE CSV fallback.",
        "- VRP history starts 2011-10 (SVXY inception); no 2008-style event in sample.",
        "- No live account was changed.",
    ]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    data = load_data()
    built = build_all(data)

    last = data["index"][-1]
    windows = {
        "full_common": (FULL_COMMON, None),
        "from_2015": (pd.Timestamp("2015-01-01"), None),
        "from_2018": (pd.Timestamp("2018-01-01"), None),
        "last_3y": (last - pd.DateOffset(years=3), None),
    }
    crises = {
        "aug_2015_flash": ("2015-08-01", "2015-10-01"),
        "volmageddon_2018": ("2018-01-26", "2018-04-30"),
        "covid_crash": ("2020-02-19", "2020-03-23"),
        "inflation_bear_2022": ("2022-01-03", "2022-10-12"),
        "tariff_2025": ("2025-02-15", "2025-06-30"),
    }

    window_results = {}
    for label, (a, b) in windows.items():
        window_results[label] = {
            name: metrics(run["returns"], data["rf"], a, b, run["turnover"])
            for name, run in built.items()
        }
        window_results[label]["bench_SPY"] = metrics(data["spy"], data["rf"], a, b)
        window_results[label]["bench_QQQ"] = metrics(data["qqq"], data["rf"], a, b)

    crisis_results = {
        label: {
            name: metrics(run["returns"], data["rf"], a, b, run["turnover"])
            for name, run in built.items()
        }
        for label, (a, b) in crises.items()
    }

    yearly_map = {name: yearly(run["returns"], FULL_COMMON) for name, run in built.items()}

    # sleeve correlations to R50d on common window
    base_ret = built["R50d"]["returns"].loc[FULL_COMMON:]
    corr = {}
    for name in ("VRP_sleeve", "MR_sleeve", "VRP_nogate", "MR_notrend", "VRP_actual_svxy"):
        if name in built:
            corr[name] = float(built[name]["returns"].loc[FULL_COMMON:].corr(base_ret))

    # pre-declared neighborhood
    neighborhood = {}
    for vth in ((1.03, 0.99), (1.05, 1.00), (1.07, 1.01)):
        for me in (5.0, 10.0, 15.0):
            b = build_all(data, vrp_th=vth, mr_entry=me)["R50d_plus"]
            m = metrics(b["returns"], data["rf"], pd.Timestamp("2015-01-01"), None, b["turnover"])
            neighborhood[f"vrp{vth[0]:.2f}/{vth[1]:.2f}_mr{int(me)}"] = {
                "cagr": m["cagr"],
                "max_dd": m["drawdown"]["max_dd"],
                "sharpe": m["sharpe"],
            }

    # cost stress
    b2 = build_all(data, cost_mult=2.0)["R50d_plus"]
    m2 = metrics(b2["returns"], data["rf"], pd.Timestamp("2015-01-01"), None, b2["turnover"])
    cost_stress = {"cagr": m2["cagr"], "max_dd": m2["drawdown"]["max_dd"], "sharpe": m2["sharpe"]}

    base = window_results["from_2015"]["R50d"]
    cand = window_results["from_2015"]["R50d_plus"]
    y_base = yearly_map["R50d"].get("2022")
    y_cand = yearly_map["R50d_plus"].get("2022")
    checks = {
        "cagr_up_1pp": cand["cagr"] >= base["cagr"] + 0.01,
        "max_dd_within_1pp": cand["drawdown"]["max_dd"] >= base["drawdown"]["max_dd"] - 0.01,
        "sharpe_not_lower": cand["sharpe"] >= base["sharpe"],
        "y2022_not_worse_1pp": (
            y_cand is not None and y_base is not None and y_cand >= y_base - 0.01
        ),
    }
    passed = all(checks.values())
    failed = [k for k, ok in checks.items() if not ok]
    conclusion = {
        "verdict": "pass" if passed else "reject",
        "checks": checks,
        "deltas_from_2015": {
            "cagr_pp": (cand["cagr"] - base["cagr"]) * 100,
            "max_dd_pp": (cand["drawdown"]["max_dd"] - base["drawdown"]["max_dd"]) * 100,
            "sharpe": cand["sharpe"] - base["sharpe"],
            "y2022_pp": None if y_cand is None or y_base is None else (y_cand - y_base) * 100,
        },
        "reason": (
            "R50d_plus passed all pre-registered from_2015 checks versus live R50d."
            if passed
            else "Failed pre-registered check(s): " + ", ".join(failed) + "."
        ),
    }

    results = native(
        {
            "meta": {
                "strategy": "V10 orthogonal premia (VRP + MR)",
                "primary": "R50d_plus",
                "baseline": "R50d",
                "data_start": data["index"][0],
                "data_end": data["index"][-1],
                "full_common_start": FULL_COMMON,
                "design": "research/v10_orthogonal/DESIGN.md",
                "inception": data["inception"],
                "params": {
                    "B_V": B_V,
                    "vrp_enter": VRP_ENTER,
                    "vrp_exit": VRP_EXIT,
                    "B_M": B_M,
                    "mr_entry": MR_ENTRY,
                    "mr_exit": MR_EXIT,
                    "mr_max_hold": MR_MAX_HOLD,
                    "cost_core_bps": COST_CORE * 1e4,
                    "cost_new_bps": COST_NEW * 1e4,
                },
            },
            "windows": window_results,
            "crises": crisis_results,
            "yearly": yearly_map,
            "sleeve_correlation_to_R50d": corr,
            "mr_trade_stats": mr_trade_stats(data),
            "mr_trade_stats_notrend": mr_trade_stats(data, trend=False),
            "neighborhood": neighborhood,
            "cost_stress_from_2015": cost_stress,
            "turnover": {
                name: float(run["turnover"].loc[FULL_COMMON:].sum() / (len(run["turnover"].loc[FULL_COMMON:]) / 252.0))
                for name, run in built.items()
            },
            "conclusion": conclusion,
        }
    )
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_summary(results)
    print(
        json.dumps(
            {
                "verdict": conclusion["verdict"],
                "checks": checks,
                "from_2015": {
                    "R50d": {"cagr": base["cagr"], "mdd": base["drawdown"]["max_dd"], "sharpe": base["sharpe"]},
                    "R50d_plus": {"cagr": cand["cagr"], "mdd": cand["drawdown"]["max_dd"], "sharpe": cand["sharpe"]},
                },
                "y2022": {"R50d": y_base, "R50d_plus": y_cand},
                "sleeves_full_common": {
                    k: {
                        "cagr": window_results["full_common"][k]["cagr"],
                        "sharpe": window_results["full_common"][k]["sharpe"],
                        "mdd": window_results["full_common"][k]["drawdown"]["max_dd"],
                        "corr_R50d": corr.get(k),
                    }
                    for k in ("VRP_sleeve", "MR_sleeve", "VRP_nogate", "MR_notrend", "VRP_actual_svxy")
                },
                "mr_trades": results["mr_trade_stats"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
