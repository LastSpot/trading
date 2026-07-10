"""
Round 5c: longer-horizon comparison for return-first R50d variants.

Honest constraint: DBMF only exists from 2019-05. Pre-2019 the DBMF sleeve
is flat (valid mask) — same convention as v3. So "long R50d" is really
TQQQ/TMF/UGL/DBC/BTC(+DBMF when live).

Windows:
  full_2007  — post-GFC-prep; TLT/GLD/DBC live; no BTC/DBMF yet
  from_2015  — BTC live; still no DBMF
  live_dbmf  — full 6-sleeve book (fair live comparison)
  last_3y

Also reports a 5-sleeve twin (no DBMF budget redistributed to others) so
pre-2019 isn't silently carrying a dead 10% budget that just becomes cash
via the vol formula when invalid... actually invalid -> weight 0, and other
sleeves are NOT renormalized on budget — capital just sits in cash. We also
test R50_5sl where DBMF's 10% is folded into TQQQ (0.50) for a cleaner
long-only-available book.

Crises: GFC, COVID, 2022, tariff 2025.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "output" / "v5_long_horizon_results.json"
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
svol = {
    "q": ewma_vol(qqq), "t": ewma_vol(tlt), "gl": ewma_vol(gld),
    "d": ewma_vol(dbc), "m": ewma_vol(dbmf), "b": ewma_vol(btc),
}

def sleeve_w(gate, budget, sigp, L, sigma, valid):
    w = gate * budget * sigp / (L * np.where(sigma > 1e-8, sigma, np.nan))
    return np.nan_to_num(np.where(valid, w, 0.0), nan=0.0)

def pack(budgets, include_dbmf=True):
    b = {"q": 0.40, "t": 0.15, "gl": 0.15, "d": 0.10, "m": 0.10, "b": 0.10}
    b.update(budgets or {})
    sleeves = [
        ("q", b["q"], 3, "q", TRUE, r_ndx3),
        ("t", b["t"], 3, "t", v_tlt.to_numpy(), r_tlt3),
        ("gl", b["gl"], 2, "gl", v_gld.to_numpy(), r_gld2),
        ("d", b["d"], 1, "d", v_dbc.to_numpy(), r_dbc1),
    ]
    if include_dbmf:
        sleeves.append(("m", b["m"], 1, "m", v_dbmf.to_numpy(), r_dbmf1))
    sleeves.append(("b", b["b"], 1, "b", v_btc.to_numpy(), r_btc1))
    return sleeves

def raw_targets(sleeves, sigp):
    W = np.column_stack([
        sleeve_w(g[gk], bud, sigp, L, svol[sk], valid)
        for gk, bud, L, sk, valid, _ in sleeves
    ])
    tot = W.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
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

def simulate_brake(W, R, dd_trig=None, scale_floor=0.75, band=0.05):
    k = W.shape[1]
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
    Ws = W * scales[:, None]
    P = band_apply(Ws, band=band)
    Pl = np.vstack([np.zeros((1, k)), P[:-1]])
    turn = np.abs(np.diff(np.vstack([np.zeros((1, k)), Pl]), axis=0)).sum(axis=1)
    ret = (Pl * R).sum(axis=1) + (1.0 - Pl.sum(axis=1)) * rf_np - COST * turn
    return pd.Series(ret, index=idx), pd.Series(turn, index=idx), P, float((scales < 0.999).mean())

def metrics(ret, start=None, end=None):
    r = ret.loc[start:end].dropna()
    if len(r) < 60:
        return {k: None for k in (
            "cagr", "vol", "sharpe", "max_dd", "calmar", "total",
            "worst_month", "final_100k", "yrs")}
    eq = (1 + r).cumprod()
    yrs = len(r) / 252.0
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    rfl = rf.loc[r.index]
    vol = float(r.std() * np.sqrt(252))
    sharpe = float(((r - rfl).mean() / r.std()) * np.sqrt(252)) if r.std() > 0 else 0.0
    dd = eq / eq.cummax() - 1
    mdd = float(dd.min())
    monthly = (1 + r).resample("ME").prod() - 1
    return dict(
        cagr=round(cagr, 4),
        vol=round(vol, 3),
        sharpe=round(sharpe, 2),
        max_dd=round(mdd, 3),
        calmar=round(cagr / abs(mdd), 2) if mdd < 0 else None,
        total=round(float(eq.iloc[-1] - 1), 2),
        worst_month=round(float(monthly.min()), 3),
        final_100k=round(float(100_000 * eq.iloc[-1]), 0),
        yrs=round(yrs, 1),
    )

# Target books
books = {
    "R50d": raw_targets(pack({}), 0.50),
    "R50tilt": raw_targets(pack({"q": 0.45, "d": 0.08, "m": 0.08, "b": 0.09}), 0.50),
    "R50_dd25_75": None,  # filled below from R50d W
    "R50_dd20_75": None,
    "R35d": raw_targets(pack({}), 0.35),
    # 5-sleeve: no DBMF, its 10% given to TQQQ — cleaner pre-2019 book
    "R50_5sl": raw_targets(
        pack({"q": 0.50, "t": 0.15, "gl": 0.15, "d": 0.10, "b": 0.10}, include_dbmf=False),
        0.50,
    ),
    "R50tilt_5sl": raw_targets(
        pack({"q": 0.55, "t": 0.15, "gl": 0.15, "d": 0.08, "b": 0.07}, include_dbmf=False),
        0.50,
    ),
    "Trio50": raw_targets(
        [
            ("q", 0.60, 3, "q", TRUE, r_ndx3),
            ("t", 0.20, 3, "t", v_tlt.to_numpy(), r_tlt3),
            ("gl", 0.20, 2, "gl", v_gld.to_numpy(), r_gld2),
        ],
        0.50,
    ),
}

W50, R50 = books["R50d"]
built = {}
for name, wr in books.items():
    if name.startswith("R50_dd"):
        continue
    W, R = wr
    ret, turn, P, pct = simulate_brake(W, R, dd_trig=None)
    built[name] = (ret, turn, P, pct)

ret, turn, P, pct = simulate_brake(W50, R50, dd_trig=-0.25, scale_floor=0.75)
built["R50_dd25_75"] = (ret, turn, P, pct)
ret, turn, P, pct = simulate_brake(W50, R50, dd_trig=-0.20, scale_floor=0.75)
built["R50_dd20_75"] = (ret, turn, P, pct)

# Windows — DBC exists from 2006, so 2007 is the honest "multi-sleeve long" start
s_dbc = dbc.dropna().index.min()
s2007 = max(pd.Timestamp("2007-01-01"), s_dbc + pd.Timedelta(days=300))
s_btc = btc.dropna().index.min()
s2015 = "2015-01-01"
s_dbmf = dbmf.dropna().index.min()
s_live = max(pd.Timestamp("2019-06-01"), s_dbmf)
start3 = idx[-1] - pd.DateOffset(years=3)

WINDOWS = [
    ("from_2007", s2007, None),
    ("from_2015", s2015, None),
    ("live_dbmf", s_live, None),
    ("last_3y", start3, None),
]
CRISES = [
    ("gfc", "2007-10-01", "2009-06-30"),
    ("covid", "2020-02-15", "2020-04-30"),
    ("y2022", "2022-01-01", "2022-12-31"),
    ("tariff_2025", "2025-02-15", "2025-06-30"),
]

res = {
    "meta": {
        "note": (
            "Long BT possible with sleeves off until inception. "
            "DBMF only from 2019-05 — pre-2019 R50d has that sleeve at 0. "
            "R50_5sl redistributes DBMF budget into TQQQ for a cleaner long book."
        ),
        "inception": {
            "TLT": str(tlt.dropna().index.min().date()),
            "GLD": str(gld.dropna().index.min().date()),
            "DBC": str(s_dbc.date()),
            "BTC": str(s_btc.date()) if pd.notna(s_btc) else None,
            "DBMF": str(s_dbmf.date()) if pd.notna(s_dbmf) else None,
        },
        "from_2007_start": str(pd.Timestamp(s2007).date()),
        "data_end": str(idx[-1].date()),
    },
    "windows": {},
    "crises": {},
    "yearly": {},
}

for name, (ret, turn, P, pct) in built.items():
    res["windows"][name] = {w: metrics(ret, a, b) for w, a, b in WINDOWS}
for w, a, b in WINDOWS:
    res["windows"].setdefault("bench_SPY", {})[w] = metrics(r_spy, a, b)
    res["windows"].setdefault("bench_QQQ", {})[w] = metrics(r_qqq, a, b)

for lab, a, b in CRISES:
    res["crises"][lab] = {
        name: {
            "max_dd": metrics(ret, a, b)["max_dd"],
            "cagr": metrics(ret, a, b)["cagr"],
            "total": metrics(ret, a, b)["total"],
        }
        for name, (ret, *_) in built.items()
    }
    res["crises"][lab]["bench_SPY"] = {
        "max_dd": metrics(r_spy, a, b)["max_dd"],
        "cagr": metrics(r_spy, a, b)["cagr"],
        "total": metrics(r_spy, a, b)["total"],
    }

def yearly(ret, start):
    y = (1 + ret.loc[start:].dropna()).resample("YE").prod() - 1
    return {str(k.year): round(float(v), 3) for k, v in y.items()}

for name in ("R50d", "R50tilt", "R50_5sl", "R50tilt_5sl", "R35d", "Trio50", "R50_dd25_75"):
    res["yearly"][name] = yearly(built[name][0], s2007)

# Scoreboards per window
res["scoreboards"] = {}
for w, *_ in WINDOWS:
    rows = []
    for name in built:
        m = res["windows"][name][w]
        if m["cagr"] is None:
            continue
        rows.append({
            "name": name,
            "cagr": m["cagr"],
            "max_dd": m["max_dd"],
            "calmar": m["calmar"],
            "sharpe": m["sharpe"],
            "final_100k": m["final_100k"],
            "yrs": m["yrs"],
        })
    rows.sort(key=lambda x: x["cagr"], reverse=True)
    res["scoreboards"][w] = rows

with open(OUT, "w") as f:
    json.dump(res, f, indent=1)

print("META:", json.dumps(res["meta"], indent=1))
for w, rows in res["scoreboards"].items():
    print(f"\n=== {w} (by CAGR) ===")
    for r in rows:
        print(
            f"{r['name']:<14} CAGR {r['cagr']*100:5.1f}%  DD {r['max_dd']*100:6.1f}%  "
            f"Calmar {r['calmar']:4.2f}  Sharpe {r['sharpe']:4.2f}  "
            f"$100k→${r['final_100k']:,.0f}  ({r['yrs']}y)"
        )
    spy = res["windows"]["bench_SPY"][w]
    print(
        f"{'SPY':<14} CAGR {spy['cagr']*100:5.1f}%  DD {spy['max_dd']*100:6.1f}%  "
        f"Calmar {spy['calmar']:4.2f}  $100k→${spy['final_100k']:,.0f}"
    )

print("\n=== Crises max DD ===")
for lab, d in res["crises"].items():
    print(lab, {k: v["max_dd"] for k, v in d.items()})

print("\nwrote", OUT)
