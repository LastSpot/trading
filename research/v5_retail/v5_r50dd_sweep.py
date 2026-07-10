"""
Round 5b: return-first DD-brake sweep on the R50d cash book.

User preference: maximize CAGR; accept higher DD. Compare soft brakes
(so we don't donate ~3 CAGR pts) vs raw R50d / R35 baselines.

Brake (position-level, not return-scale approx):
  - Track strategy equity from T+1 P&L of the unbraked book.
  - When peak-to-trough DD <= trigger, multiply ALL sleeve targets by floor
    until equity makes a new high, then restore full size.
  - Rebalance band still applies on the scaled targets.

Grid:
  triggers: -15%, -20%, -25%
  floors:   0.50, 0.65, 0.75
  + baselines: R50d (no brake), R35d, R35@-15%/0.50

Also tests a mild equity tilt on the best soft brake:
  TQQQ budget 0.45 / diversifiers trimmed (DBC+DBMF+BTC = 0.25 total).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "output" / "v5_r50dd_sweep_results.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

tick = [
    "QQQ", "SPY", "TLT", "GLD", "DBC", "DBMF", "TQQQ", "TMF", "UGL",
    "BTC-USD", "^IRX",
]
raw = yf.download(tick, start="1999-03-10", auto_adjust=True, progress=False)["Close"]
qqq = raw["QQQ"].dropna()
idx = qqq.index
n = len(idx)

def al(c):
    return raw[c].reindex(idx)

spy = al("SPY").ffill()
tlt, gld, dbc, dbmf = al("TLT"), al("GLD"), al("DBC"), al("DBMF")
tqqq, tmf, ugl = al("TQQQ"), al("TMF"), al("UGL")
btc = al("BTC-USD")
irx = al("^IRX").ffill().fillna(4.5) / 100.0
rf = irx / 252.0
rf_np = rf.to_numpy()

SPREAD = 0.005 / 252.0
EXP = 0.0095 / 252.0
COST = 0.0005
TRUE = np.ones(n, dtype=bool)

def synth(u, L):
    r = u.pct_change()
    return (L * r - (L - 1) * (rf + SPREAD) - EXP), ~r.isna()

r_ndx3, _ = synth(qqq, 3)
ov = tqqq.pct_change().dropna().index
r_ndx3 = r_ndx3.fillna(0.0)
r_ndx3.loc[ov] = tqqq.pct_change().loc[ov]

r_tlt3, v_tlt = synth(tlt, 3)
r_tlt3 = r_tlt3.fillna(0.0)
ov_tmf = tmf.pct_change().dropna().index
r_tlt3.loc[ov_tmf] = tmf.pct_change().loc[ov_tmf]

r_gld2, v_gld = synth(gld, 2)
r_gld2 = r_gld2.fillna(0.0)
ov_ugl = ugl.pct_change().dropna().index
r_gld2.loc[ov_ugl] = ugl.pct_change().loc[ov_ugl]

r_dbc1 = dbc.pct_change().fillna(0.0)
v_dbc = ~dbc.pct_change().isna()
r_dbmf1 = dbmf.pct_change().fillna(0.0)
v_dbmf = ~dbmf.pct_change().isna()
r_btc1 = (btc.pct_change() - 0.01 / 252.0).fillna(0.0)
v_btc = ~btc.pct_change().isna()
r_spy = spy.pct_change().fillna(0.0)
r_qqq = qqq.pct_change().fillna(0.0)

def gate_ens(px):
    votes = [
        (px / px.shift(L) - 1 > 0).astype(float).where(~px.isna(), 0.0)
        for L in (63, 126, 252)
    ]
    return pd.concat(votes, axis=1).mean(axis=1).fillna(0.0).to_numpy()

def ewma_vol(px):
    r = px.pct_change().fillna(0.0)
    return np.sqrt(252.0 * (r ** 2).ewm(alpha=0.06, adjust=False).mean()).to_numpy()

g = {
    "q": gate_ens(qqq), "t": gate_ens(tlt), "gl": gate_ens(gld),
    "d": gate_ens(dbc), "m": gate_ens(dbmf), "b": gate_ens(btc),
}
s = {
    "q": ewma_vol(qqq), "t": ewma_vol(tlt), "gl": ewma_vol(gld),
    "d": ewma_vol(dbc), "m": ewma_vol(dbmf), "b": ewma_vol(btc),
}

def sleeve_w(gate, budget, sigp, L, sigma, valid):
    w = gate * budget * sigp / (L * np.where(sigma > 1e-8, sigma, np.nan))
    return np.nan_to_num(np.where(valid, w, 0.0), nan=0.0)

def pack(budgets=None):
    """Default R50d budgets; optional override dict for q/t/gl/d/m/b."""
    b = {"q": 0.40, "t": 0.15, "gl": 0.15, "d": 0.10, "m": 0.10, "b": 0.10}
    if budgets:
        b.update(budgets)
    return [
        ("q", b["q"], 3, "q", TRUE, r_ndx3),
        ("t", b["t"], 3, "t", v_tlt.to_numpy(), r_tlt3),
        ("gl", b["gl"], 2, "gl", v_gld.to_numpy(), r_gld2),
        ("d", b["d"], 1, "d", v_dbc.to_numpy(), r_dbc1),
        ("m", b["m"], 1, "m", v_dbmf.to_numpy(), r_dbmf1),
        ("b", b["b"], 1, "b", v_btc.to_numpy(), r_btc1),
    ]

def raw_targets(sleeves, sigp):
    W = np.column_stack([
        sleeve_w(g[gk], bud, sigp, L, s[sk], valid)
        for gk, bud, L, sk, valid, _ in sleeves
    ])
    tot = W.sum(axis=1)
    W *= np.where(tot > 1.0, 1.0 / tot, 1.0)[:, None]
    R = np.column_stack([r.to_numpy() for *_, r in sleeves])
    return W, R

def band_apply(WT, band=0.05):
    P = np.zeros_like(WT)
    held = np.zeros(WT.shape[1])
    for i in range(len(WT)):
        for j in range(WT.shape[1]):
            if abs(WT[i, j] - held[j]) > band or (WT[i, j] == 0.0 and held[j] != 0.0):
                held[j] = WT[i, j]
            P[i, j] = held[j]
    return P

def simulate(W, R, dd_trig=None, scale_floor=0.5, band=0.05):
    """T+1 execution with optional path-dependent DD brake on targets.

    Brake uses realized strategy equity (after costs) to decide today's scale.
    Scale decided at close t applies to positions that earn return at t+1
    (same causality as the rest of the book).
    """
    k = W.shape[1]
    P = np.zeros_like(W)
    held = np.zeros(k)
    rets = np.zeros(n)
    turns = np.zeros(n)
    scales = np.ones(n)
    braked = False
    equity = 1.0
    peak = 1.0
    prev_pos = np.zeros(k)

    for i in range(n):
        # decide scale from equity path *before* today's target (known at t)
        dd = equity / peak - 1.0
        if dd_trig is not None:
            if dd <= dd_trig:
                braked = True
            if braked and abs(dd) < 1e-12:
                braked = False
            scales[i] = scale_floor if braked else 1.0
        else:
            scales[i] = 1.0

        target = W[i] * scales[i]
        for j in range(k):
            if abs(target[j] - held[j]) > band or (target[j] == 0.0 and held[j] != 0.0):
                held[j] = target[j]
            P[i, j] = held[j]

        # P&L: yesterday's position earns today's return
        if i == 0:
            rets[i] = rf_np[i]
            turns[i] = 0.0
        else:
            turn = np.abs(prev_pos - P[i - 1]).sum()  # turnover when we set P[i-1]
            # actually turnover should be |P[i]-P[i-1]| paid when position changes
            # Standard v3: Pl = P shifted; turn on Pl changes. Match that:
            pass
        prev_pos = P[i].copy()

    # Vectorized P&L matching v3/v4 (shift-1 positions)
    Pl = np.vstack([np.zeros((1, k)), P[:-1]])
    turn = np.abs(np.diff(np.vstack([np.zeros((1, k)), Pl]), axis=0)).sum(axis=1)
    ret = (Pl * R).sum(axis=1) + (1.0 - Pl.sum(axis=1)) * rf_np - COST * turn

    # The loop above set scales using a pilot equity that wasn't cost-correct.
    # Rebuild scales causally with the same brake rule on the *unbraked* book
    # first, then apply — cleaner two-pass used below instead.
    return None  # placeholder — real impl is simulate_brake

def simulate_brake(W, R, dd_trig=None, scale_floor=0.5, band=0.05):
    """Two-pass causal brake:
    Pass 1: unbraked banded positions -> pilot equity.
    Pass 2: derive daily scale from pilot equity path; apply to targets; re-band; P&L.

    This matches the v4 research approximation's intent but sizes positions
    correctly (half weight, not half return).
    """
    k = W.shape[1]
    # Pass 1: unbraked
    P0 = band_apply(W, band=band)
    Pl0 = np.vstack([np.zeros((1, k)), P0[:-1]])
    turn0 = np.abs(np.diff(np.vstack([np.zeros((1, k)), Pl0]), axis=0)).sum(axis=1)
    ret0 = (Pl0 * R).sum(axis=1) + (1.0 - Pl0.sum(axis=1)) * rf_np - COST * turn0
    eq = np.cumprod(1.0 + ret0)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0

    scales = np.ones(n)
    if dd_trig is not None:
        braked = False
        for i in range(n):
            if dd[i] <= dd_trig:
                braked = True
            if braked and abs(dd[i]) < 1e-12:
                braked = False
            scales[i] = scale_floor if braked else 1.0

    # Pass 2: scale targets, re-band, P&L
    Ws = W * scales[:, None]
    P = band_apply(Ws, band=band)
    Pl = np.vstack([np.zeros((1, k)), P[:-1]])
    turn = np.abs(np.diff(np.vstack([np.zeros((1, k)), Pl]), axis=0)).sum(axis=1)
    ret = (Pl * R).sum(axis=1) + (1.0 - Pl.sum(axis=1)) * rf_np - COST * turn
    return (
        pd.Series(ret, index=idx),
        pd.Series(turn, index=idx),
        P,
        pd.Series(scales, index=idx),
        float((scales < 1.0 - 1e-12).mean()),
    )

def metrics(ret, start=None, end=None):
    r = ret.loc[start:end].dropna()
    if len(r) < 40:
        return {k: None for k in (
            "cagr", "vol", "sharpe", "max_dd", "calmar", "total",
            "worst_month", "p5_roll12", "ulcer", "final_100k")}
    eq = (1 + r).cumprod()
    yrs = len(r) / 252.0
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    rfl = rf.loc[r.index]
    vol = float(r.std() * np.sqrt(252))
    sharpe = float(((r - rfl).mean() / r.std()) * np.sqrt(252)) if r.std() > 0 else 0.0
    dd = eq / eq.cummax() - 1
    mdd = float(dd.min())
    monthly = (1 + r).resample("ME").prod() - 1
    roll12 = (1 + r).rolling(252).apply(lambda x: np.prod(1 + x) - 1, raw=True)
    p5 = float(roll12.quantile(0.05)) if roll12.notna().sum() > 20 else None
    ulcer = float(np.sqrt((dd.clip(upper=0) ** 2).mean()))  # ulcer index
    return dict(
        cagr=round(cagr, 4),
        vol=round(vol, 3),
        sharpe=round(sharpe, 2),
        max_dd=round(mdd, 3),
        calmar=round(cagr / abs(mdd), 2) if mdd < 0 else None,
        total=round(float(eq.iloc[-1] - 1), 2),
        worst_month=round(float(monthly.min()), 3),
        p5_roll12=round(p5, 3) if p5 is not None else None,
        ulcer=round(ulcer, 3),
        final_100k=round(float(100_000 * eq.iloc[-1]), 0),
    )

# ---- build base target matrices ----
W50, R50 = raw_targets(pack(), 0.50)
W35, R35 = raw_targets(pack(), 0.35)
W50_tilt, R50_tilt = raw_targets(
    pack({"q": 0.45, "t": 0.15, "gl": 0.15, "d": 0.08, "m": 0.08, "b": 0.09}),
    0.50,
)

s_dbmf = dbmf.dropna().index.min()
s_live = max(pd.Timestamp("2019-06-01"), s_dbmf)
start3 = idx[-1] - pd.DateOffset(years=3)
WINDOWS = [
    ("live_dbmf", s_live, None),
    ("from_2015", "2015-01-01", None),
    ("last_3y", start3, None),
]
CRISES = [
    ("covid", "2020-02-15", "2020-04-30"),
    ("y2022", "2022-01-01", "2022-12-31"),
    ("tariff_2025", "2025-02-15", "2025-06-30"),
]

# Spec list: (name, W, R, trig, floor)
specs = [
    ("R50d", W50, R50, None, 1.0),
    ("R35d", W35, R35, None, 1.0),
    ("R35_dd15_50", W35, R35, -0.15, 0.50),
]
for trig in (-0.15, -0.20, -0.25):
    for floor in (0.50, 0.65, 0.75):
        tlab = abs(int(trig * 100))
        flab = int(floor * 100)
        specs.append((f"R50_dd{tlab}_{flab}", W50, R50, trig, floor))

# Soft-brake + equity tilt
specs.append(("R50tilt_dd20_65", W50_tilt, R50_tilt, -0.20, 0.65))
specs.append(("R50tilt", W50_tilt, R50_tilt, None, 1.0))

built = {}
for name, W, R, trig, floor in specs:
    ret, turn, P, scales, pct_braked = simulate_brake(
        W, R, dd_trig=trig, scale_floor=floor
    )
    built[name] = dict(ret=ret, turn=turn, P=P, scales=scales, pct_braked=pct_braked)

res = {
    "meta": {
        "live_start": str(pd.Timestamp(s_live).date()),
        "data_end": str(idx[-1].date()),
        "brake": "two-pass position scale from unbraked pilot equity; release at new high",
        "note": "return-first sweep; prefer high CAGR with soft DD brake",
    },
    "windows": {},
    "crises": {},
    "brake_stats": {},
    "yearly_live": {},
}

for name, b in built.items():
    res["windows"][name] = {w: metrics(b["ret"], a, c) for w, a, c in WINDOWS}
    yrs = n / 252.0
    res["brake_stats"][name] = {
        "pct_days_braked": round(b["pct_braked"], 3),
        "turnover_yr": round(float(b["turn"].sum() / yrs), 2),
        "avg_gross": round(float(b["P"].sum(axis=1).mean()), 3),
    }

for w, a, c in WINDOWS:
    res["windows"].setdefault("bench_SPY", {})[w] = metrics(r_spy, a, c)
    res["windows"].setdefault("bench_QQQ", {})[w] = metrics(r_qqq, a, c)

for lab, a, c in CRISES:
    res["crises"][lab] = {
        name: {
            "max_dd": metrics(b["ret"], a, c)["max_dd"],
            "cagr": metrics(b["ret"], a, c)["cagr"],
            "total": metrics(b["ret"], a, c)["total"],
        }
        for name, b in built.items()
    }
    res["crises"][lab]["bench_SPY"] = {
        "max_dd": metrics(r_spy, a, c)["max_dd"],
        "cagr": metrics(r_spy, a, c)["cagr"],
        "total": metrics(r_spy, a, c)["total"],
    }

def yearly(ret, start):
    y = (1 + ret.loc[start:].dropna()).resample("YE").prod() - 1
    return {str(k.year): round(float(v), 3) for k, v in y.items()}

focus = [
    "R50d", "R35d", "R35_dd15_50",
    "R50_dd15_50", "R50_dd20_50", "R50_dd20_65", "R50_dd20_75",
    "R50_dd25_65", "R50_dd25_75",
    "R50tilt", "R50tilt_dd20_65",
]
for name in focus:
    if name in built:
        res["yearly_live"][name] = yearly(built[name]["ret"], s_live)

# Scoreboards
def board(window, sort_key="cagr"):
    rows = []
    for name, wins in res["windows"].items():
        if name.startswith("bench_"):
            continue
        m = wins[window]
        if m["cagr"] is None:
            continue
        rows.append({
            "name": name,
            "cagr": m["cagr"],
            "max_dd": m["max_dd"],
            "calmar": m["calmar"],
            "sharpe": m["sharpe"],
            "vol": m["vol"],
            "ulcer": m["ulcer"],
            "final_100k": m["final_100k"],
            "worst_month": m["worst_month"],
            "pct_braked": res["brake_stats"].get(name, {}).get("pct_days_braked"),
        })
    rows.sort(key=lambda x: (x[sort_key] if x[sort_key] is not None else -999), reverse=True)
    return rows

res["scoreboard_cagr"] = board("live_dbmf", "cagr")
res["scoreboard_calmar"] = board("live_dbmf", "calmar")

base = res["windows"]["R50d"]["live_dbmf"]
res["vs_R50d"] = {}
for name in built:
    m = res["windows"][name]["live_dbmf"]
    res["vs_R50d"][name] = {
        "cagr_delta_pp": round((m["cagr"] - base["cagr"]) * 100, 2),
        "mdd_delta_pp": round((m["max_dd"] - base["max_dd"]) * 100, 2),
        "calmar_delta": round(m["calmar"] - base["calmar"], 2),
        "final_delta_$": round(m["final_100k"] - base["final_100k"], 0),
        "ulcer_delta": round(m["ulcer"] - base["ulcer"], 3),
    }

# Pareto-ish: among variants within 2 CAGR pts of R50d, best max_dd / calmar
near = [
    r for r in res["scoreboard_cagr"]
    if r["cagr"] is not None and r["cagr"] >= base["cagr"] - 0.02
]
near_by_dd = sorted(near, key=lambda x: x["max_dd"], reverse=True)  # least negative first
near_by_cal = sorted(near, key=lambda x: x["calmar"], reverse=True)
res["return_first_picks"] = {
    "rule": "among variants within 2 CAGR pts of R50d",
    "best_max_dd": near_by_dd[0] if near_by_dd else None,
    "best_calmar": near_by_cal[0] if near_by_cal else None,
    "candidates": near,
}

with open(OUT, "w") as f:
    json.dump(res, f, indent=1)

print("=== SCOREBOARD by CAGR (live DBMF window) ===")
for r in res["scoreboard_cagr"]:
    print(
        f"{r['name']:<22} CAGR {r['cagr']*100:5.1f}%  DD {r['max_dd']*100:6.1f}%  "
        f"Calmar {r['calmar']:4.2f}  Sharpe {r['sharpe']:4.2f}  "
        f"$100k→${r['final_100k']:,.0f}  braked {100*(r['pct_braked'] or 0):4.1f}%"
    )

print("\n=== vs R50d (CAGR / DD / $ final) ===")
for name, d in sorted(res["vs_R50d"].items(), key=lambda kv: -kv[1]["cagr_delta_pp"]):
    print(
        f"{name:<22} CAGR {d['cagr_delta_pp']:+5.2f}pp  "
        f"DD {d['mdd_delta_pp']:+5.2f}pp  "
        f"final {d['final_delta_$']:+,.0f}  Calmar {d['calmar_delta']:+.2f}"
    )

print("\n=== Return-first picks (within 2 CAGR pts of R50d) ===")
print(json.dumps(res["return_first_picks"], indent=1))

print("\n=== Crisis max DD ===")
for lab, d in res["crises"].items():
    print(lab, {k: v["max_dd"] for k, v in d.items()})

print("\nwrote", OUT)
