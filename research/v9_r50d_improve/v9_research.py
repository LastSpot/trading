"""V9: improve live R50d with asymmetric equity gate (pre-registered).

Primary candidate: R50d_asym
Diagnostics: R50d_corr, R50d_redistrib, R50d_asym_crash
Baseline: live R50d
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = REPO_ROOT / "output" / "v9_r50d_improve_results.json"
OUT_SUMMARY = REPO_ROOT / "output" / "v9_r50d_improve_summary.md"

START = "1999-03-10"
SIGP = 0.50
BAND = 0.05
COST = 0.0005
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


def load_data() -> dict:
    tick = [
        "QQQ", "SPY", "TLT", "GLD", "DBC", "DBMF", "TQQQ", "TMF", "UGL",
        "BTC-USD", "AQMIX", "^IRX",
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

    # correlation stress inputs
    rets_on = pd.DataFrame(
        {
            "q": qqq.pct_change(),
            "d": dbc.pct_change(),
            "b": btc.pct_change(),
        },
        index=idx,
    )
    roll_corr = rets_on.rolling(63).corr()
    avg_corr = pd.Series(0.0, index=idx)
    for a, b in (("q", "d"), ("q", "b"), ("d", "b")):
        try:
            c = roll_corr.xs(a, level=1)[b]
            avg_corr = avg_corr.add(c.reindex(idx).fillna(0.0), fill_value=0.0)
        except Exception:
            pass
    avg_corr = (avg_corr / 3.0).fillna(0.0)
    qqq_21 = (qqq / qqq.shift(21) - 1.0).fillna(0.0)
    corr_scale = np.where((avg_corr > 0.60) & (qqq_21 < -0.08), 0.5, 1.0)

    daily_std_20 = qqq.pct_change().rolling(20).std()
    crash = (qqq_21 <= (-2.0 * daily_std_20 * np.sqrt(21.0))).fillna(False).to_numpy()

    return {
        "index": idx,
        "n": n,
        "rf": rf,
        "gates": gates,
        "vols": vols,
        "valid": valid,
        "rets": rets,
        "corr_scale": corr_scale,
        "crash": crash,
        "spy": spy.pct_change().fillna(0.0),
        "qqq": qqq.pct_change().fillna(0.0),
        "inception": {
            "TLT": str(tlt.dropna().index.min().date()) if not tlt.dropna().empty else None,
            "GLD": str(gld.dropna().index.min().date()) if not gld.dropna().empty else None,
            "DBC": str(dbc.dropna().index.min().date()) if not dbc.dropna().empty else None,
            "DBMF": str(al("DBMF").dropna().index.min().date())
            if not al("DBMF").dropna().empty
            else None,
            "TQQQ": str(tqqq.dropna().index.min().date()) if not tqqq.dropna().empty else None,
            "BTC": str(btc.dropna().index.min().date()) if not btc.dropna().empty else None,
        },
    }


def asymmetric_equity_gate(gate: np.ndarray) -> np.ndarray:
    """Hysteresis on QQQ ensemble for TQQQ only."""
    out = np.zeros_like(gate)
    on = False
    for i, g in enumerate(gate):
        if on:
            if g <= 1.0 / 3.0 + 1e-12:
                on = False
        else:
            if g >= 2.0 / 3.0 - 1e-12:
                on = True
        out[i] = g if on else 0.0
    return out


def raw_weights(
    data: dict,
    equity_gate: np.ndarray,
    redistrib: bool = False,
    crash_cut: bool = False,
) -> np.ndarray:
    n = data["n"]
    W = np.zeros((n, len(SLEEVE_ORDER)))
    for j, sym in enumerate(SLEEVE_ORDER):
        gate = equity_gate if sym == "TQQQ" else data["gates"][sym]
        if crash_cut and sym == "TQQQ":
            gate = np.where((gate > 0) & data["crash"], gate * 0.5, gate)
        vol = data["vols"][sym]
        valid = data["valid"][sym]
        budget = BASE_BUDGETS[sym]
        lev = LEVERAGE[sym]
        w = gate * budget * SIGP / (lev * np.where(vol > 1e-8, vol, np.nan))
        W[:, j] = np.nan_to_num(np.where(valid, w, 0.0), nan=0.0)

    if redistrib:
        budgets = np.array([BASE_BUDGETS[s] for s in SLEEVE_ORDER])
        for i in range(n):
            active = W[i] > 0
            if not active.any() or active.all():
                continue
            dead_budget = float(budgets[~active].sum())
            if dead_budget <= 0:
                continue
            active_budget = float(budgets[active].sum())
            if active_budget <= 0:
                continue
            # scale active raw weights so they absorb dead budget, with 1.5x cap
            scale = 1.0 + dead_budget / active_budget
            capped = np.minimum(scale, 1.5)
            # if scale would exceed 1.5 for some, use uniform capped scale
            W[i, active] *= capped

    tot = W.sum(axis=1)
    W *= np.where(tot > 1.0, 1.0 / np.maximum(tot, 1e-12), 1.0)[:, None]
    return W


def build_variant(
    data: dict,
    mode: str,
) -> dict:
    equity_gate = data["gates"]["TQQQ"]
    redistrib = False
    crash_cut = False
    book_scale = np.ones(data["n"])

    if mode in ("R50d_asym", "R50d_asym_crash"):
        equity_gate = asymmetric_equity_gate(data["gates"]["TQQQ"])
    if mode == "R50d_asym_crash":
        crash_cut = True
    if mode == "R50d_corr":
        book_scale = data["corr_scale"]
    if mode == "R50d_redistrib":
        redistrib = True

    W = raw_weights(data, equity_gate, redistrib=redistrib, crash_cut=crash_cut)
    W = W * book_scale[:, None]
    tot = W.sum(axis=1)
    W *= np.where(tot > 1.0, 1.0 / np.maximum(tot, 1e-12), 1.0)[:, None]
    P = band_positions(W)
    R = np.column_stack([data["rets"][s].to_numpy() for s in SLEEVE_ORDER])
    Pl = np.vstack([np.zeros((1, P.shape[1])), P[:-1]])
    turn = np.abs(np.diff(np.vstack([np.zeros((1, P.shape[1])), Pl]), axis=0)).sum(axis=1)
    ret = (
        (Pl * R).sum(axis=1)
        + (1.0 - Pl.sum(axis=1)) * data["rf"].to_numpy()
        - COST * turn
    )
    return {
        "returns": pd.Series(ret, index=data["index"]),
        "turnover": pd.Series(turn, index=data["index"]),
        "positions": pd.DataFrame(P, index=data["index"], columns=SLEEVE_ORDER),
        "targets": pd.DataFrame(W, index=data["index"], columns=SLEEVE_ORDER),
    }


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
            "peak_to_recovery_trading_days": (
                recovery_loc - peak_loc if recovery_loc is not None else None
            ),
            "max_dd_dollars_on_100k_path": abs(float(dd.loc[trough])) * peak_val * INITIAL,
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


def write_summary(results: dict) -> None:
    primary = results["windows"]["from_2015"]
    base = primary["R50d"]
    cand = primary["R50d_asym"]
    full = results["windows"]["full"]
    y2022 = results["yearly"]
    conclusion = results["conclusion"]
    lines = [
        "# V9 R50d Improvement Research",
        "",
        f"Data: {results['meta']['data_start']} through {results['meta']['data_end']}.",
        "Primary evaluation window: from_2015.",
        "",
        "## Primary candidate vs live R50d (from_2015)",
        "",
        f"- Live R50d: {base['cagr']:.2%} CAGR, {base['drawdown']['max_dd']:.2%} max DD, "
        f"{base['calmar']:.2f} Calmar",
        f"- R50d_asym: {cand['cagr']:.2%} CAGR, {cand['drawdown']['max_dd']:.2%} max DD, "
        f"{cand['calmar']:.2f} Calmar",
        f"- CAGR delta: {(cand['cagr'] - base['cagr']) * 100:+.2f} pp",
        f"- Max DD delta: {(cand['drawdown']['max_dd'] - base['drawdown']['max_dd']) * 100:+.2f} pp "
        f"(positive = shallower / better)",
        f"- 2022 return: R50d {y2022['R50d'].get('2022', float('nan')):.2%}, "
        f"R50d_asym {y2022['R50d_asym'].get('2022', float('nan')):.2%}",
        "",
        "## Full-history snapshot",
        "",
        f"- R50d: {full['R50d']['cagr']:.2%} CAGR, {full['R50d']['drawdown']['max_dd']:.2%} max DD",
        f"- R50d_asym: {full['R50d_asym']['cagr']:.2%} CAGR, "
        f"{full['R50d_asym']['drawdown']['max_dd']:.2%} max DD",
        "",
        "## Diagnostics (from_2015)",
        "",
    ]
    for name in ("R50d_corr", "R50d_redistrib", "R50d_asym_crash"):
        m = primary[name]
        lines.append(
            f"- {name}: {m['cagr']:.2%} CAGR, {m['drawdown']['max_dd']:.2%} max DD, "
            f"{m['calmar']:.2f} Calmar"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"**{conclusion['verdict'].upper()}** — {conclusion['reason']}",
            "",
            "Diagnostics do not replace the primary candidate.",
            "",
            "## Limitations",
            "",
            "- Pre-2010 TQQQ/TMF/UGL history is synthetic where needed.",
            "- DBMF uses AQMIX splice before ETF inception.",
            "- No live account was changed.",
        ]
    )
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_baseline(data: dict) -> dict:
    W = raw_weights(data, data["gates"]["TQQQ"])
    P = band_positions(W)
    R = np.column_stack([data["rets"][s].to_numpy() for s in SLEEVE_ORDER])
    Pl = np.vstack([np.zeros((1, P.shape[1])), P[:-1]])
    turn = np.abs(np.diff(np.vstack([np.zeros((1, P.shape[1])), Pl]), axis=0)).sum(axis=1)
    ret = (
        (Pl * R).sum(axis=1)
        + (1.0 - Pl.sum(axis=1)) * data["rf"].to_numpy()
        - COST * turn
    )
    return {
        "returns": pd.Series(ret, index=data["index"]),
        "turnover": pd.Series(turn, index=data["index"]),
        "positions": pd.DataFrame(P, index=data["index"], columns=SLEEVE_ORDER),
    }


def main() -> None:
    data = load_data()
    built = {"R50d": build_baseline(data)}
    for mode in ("R50d_asym", "R50d_corr", "R50d_redistrib", "R50d_asym_crash"):
        built[mode] = build_variant(data, mode)

    eval_start = data["index"][260]
    last = data["index"][-1]
    windows = {
        "full": (eval_start, None),
        "from_2010": (pd.Timestamp("2010-01-01"), None),
        "from_2015": (pd.Timestamp("2015-01-01"), None),
        "from_2018": (pd.Timestamp("2018-01-01"), None),
        "last_3y": (last - pd.DateOffset(years=3), None),
    }
    crises = {
        "gfc": ("2007-10-09", "2009-03-09"),
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

    yearly_map = {name: yearly(run["returns"], eval_start) for name, run in built.items()}

    base = window_results["from_2015"]["R50d"]
    cand = window_results["from_2015"]["R50d_asym"]
    y2022_base = yearly_map["R50d"].get("2022")
    y2022_cand = yearly_map["R50d_asym"].get("2022")
    checks = {
        "cagr_within_2pp": cand["cagr"] >= base["cagr"] - 0.02,
        "max_dd_improves_1pp": cand["drawdown"]["max_dd"] >= base["drawdown"]["max_dd"] + 0.01,
        "y2022_not_worse_1pp": (
            y2022_cand is not None
            and y2022_base is not None
            and y2022_cand >= y2022_base - 0.01
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
            "y2022_pp": (
                None
                if y2022_cand is None or y2022_base is None
                else (y2022_cand - y2022_base) * 100
            ),
        },
        "reason": (
            "R50d_asym passed all pre-registered from_2015 checks versus live R50d."
            if passed
            else "Failed pre-registered check(s): " + ", ".join(failed) + "."
        ),
    }

    results = native(
        {
            "meta": {
                "strategy": "V9 R50d improvements",
                "primary": "R50d_asym",
                "baseline": "R50d",
                "data_start": data["index"][0],
                "data_end": data["index"][-1],
                "evaluation_start": eval_start,
                "design": "research/v9_r50d_improve/DESIGN.md",
                "inception": data["inception"],
            },
            "windows": window_results,
            "crises": crisis_results,
            "yearly": yearly_map,
            "turnover": {
                name: float(run["turnover"].sum() / (len(data["index"]) / 252.0))
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
                "from_2015_R50d_cagr": base["cagr"],
                "from_2015_asym_cagr": cand["cagr"],
                "from_2015_R50d_mdd": base["drawdown"]["max_dd"],
                "from_2015_asym_mdd": cand["drawdown"]["max_dd"],
                "y2022_R50d": y2022_base,
                "y2022_asym": y2022_cand,
                "checks": checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
