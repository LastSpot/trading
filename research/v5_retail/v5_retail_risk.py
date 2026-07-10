"""
v5 retail risk: $100k book — grow fast with survivable risk (Round 5).

Tests the claim that R50d (50% vol, levered ETFs, independent sleeve sizing)
is too aggressive for limited capital, and that a few dumb risk upgrades
improve the growth/drawdown tradeoff.

Variants
--------
R50d     : live cash-account book (TQQQ/TMF/UGL/DBC/DBMF/BTC), ensemble gate,
           sigma*=50%, DBC/DBMF unlevered, 100% notional cap. Baseline.
R35d     : same book, sigma*=35%
R25d     : same book, sigma*=25%
R50_1x   : same budgets/gates, but trade QQQ/TLT/GLD (1x) instead of 3x/2x ETFs
           (same risk contribution via larger capital weight; less ETF decay)
R35_dd   : R35d + peak-to-trough drawdown brake (halve risk after -15% DD;
           restore after equity makes a new high)
R35_chop : R35d + chop filter (gate*=0 when |63d return|/vol_63d < 0.5)
R35_corr : R35d + correlation stress (if 63d avg pairwise corr of risk-on
           sleeves > 0.6 AND QQQ 21d return < -8%, scale book 0.5x)
R35_all  : R35d + dd brake + chop + corr stress (the proposed retail stack)
Trio50   : classic 3-sleeve (TQQQ/TMF/UGL) sigma*=50% from v2 (reference)

Objective for ranking: maximize Calmar, then CAGR, subject to maxDD not
much worse than ~-25% on the full sample where all sleeves are live.

Execution: T+1, 5bps/turnover, idle cash at T-bills. Same machinery as v3.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "output" / "v5_retail_risk_results.json"
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

SPREAD = 0.005 / 252.0
EXP = 0.0095 / 252.0

def synth(u, L):
    r = u.pct_change()
    return (L * r - (L - 1) * (rf + SPREAD) - EXP), ~r.isna()

# Levered sleeve returns: synth pre-inception, real after
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

# 1x underlyings (for R50_1x)
r_qqq1 = qqq.pct_change().fillna(0.0)
r_tlt1 = tlt.pct_change().fillna(0.0)
r_gld1 = gld.pct_change().fillna(0.0)
r_spy = spy.pct_change().fillna(0.0)

TRUE = np.ones(n, dtype=bool)

def gate_ens(px: pd.Series) -> np.ndarray:
    votes = [
        (px / px.shift(L) - 1 > 0).astype(float).where(~px.isna(), 0.0)
        for L in (63, 126, 252)
    ]
    return pd.concat(votes, axis=1).mean(axis=1).fillna(0.0).to_numpy()

def ewma_vol(px: pd.Series) -> np.ndarray:
    r = px.pct_change().fillna(0.0)
    return np.sqrt(252.0 * (r ** 2).ewm(alpha=0.06, adjust=False).mean()).to_numpy()

def chop_mult(px: pd.Series, thresh: float = 0.5) -> np.ndarray:
    """0 when 63d trend is weak vs its own vol (chop); else 1."""
    r63 = px / px.shift(63) - 1
    vol63 = px.pct_change().rolling(63).std() * np.sqrt(252)
    strength = (r63.abs() / vol63.replace(0, np.nan)).fillna(0.0)
    return (strength >= thresh).astype(float).to_numpy()

g = {
    "q": gate_ens(qqq), "t": gate_ens(tlt), "gl": gate_ens(gld),
    "d": gate_ens(dbc), "m": gate_ens(dbmf), "b": gate_ens(btc),
}
s = {
    "q": ewma_vol(qqq), "t": ewma_vol(tlt), "gl": ewma_vol(gld),
    "d": ewma_vol(dbc), "m": ewma_vol(dbmf), "b": ewma_vol(btc),
}
ch = {
    "q": chop_mult(qqq), "t": chop_mult(tlt), "gl": chop_mult(gld),
    "d": chop_mult(dbc), "m": chop_mult(dbmf), "b": chop_mult(btc),
}

# Correlation stress on risk-on underlyings (QQQ, DBC, BTC) when available
rets_on = pd.DataFrame({
    "q": qqq.pct_change(), "d": dbc.pct_change(), "b": btc.pct_change(),
}).reindex(idx)
roll_corr = rets_on.rolling(63).corr()
# average pairwise corr of the three
pairs = [("q", "d"), ("q", "b"), ("d", "b")]
avg_corr = pd.Series(0.0, index=idx)
for a, b in pairs:
    # rolling corr MultiIndex: (date, col)
    try:
        c = roll_corr.xs(a, level=1)[b]
        avg_corr = avg_corr.add(c.reindex(idx).fillna(0.0), fill_value=0.0)
    except Exception:
        pass
avg_corr = (avg_corr / len(pairs)).fillna(0.0)
qqq_21 = (qqq / qqq.shift(21) - 1).fillna(0.0)
corr_stress = ((avg_corr > 0.60) & (qqq_21 < -0.08)).astype(float).to_numpy()
# scale = 0.5 when stressed, else 1.0
corr_scale = np.where(corr_stress > 0, 0.5, 1.0)

def sleeve_w(gate, budget, sigp, L, sigma, valid):
    w = gate * budget * sigp / (L * np.where(sigma > 1e-8, sigma, np.nan))
    return np.nan_to_num(np.where(valid, w, 0.0), nan=0.0)

def band_positions(WT, band=0.05):
    P = np.zeros_like(WT)
    held = np.zeros(WT.shape[1])
    for i in range(n):
        for j in range(WT.shape[1]):
            if abs(WT[i, j] - held[j]) > band or (WT[i, j] == 0.0 and held[j] != 0.0):
                held[j] = WT[i, j]
            P[i, j] = held[j]
    return P

COST = 0.0005

def pnl(P, R):
    Pl = np.vstack([np.zeros((1, P.shape[1])), P[:-1]])
    turn = np.abs(np.diff(np.vstack([np.zeros((1, P.shape[1])), Pl]), axis=0)).sum(axis=1)
    ret = (Pl * R).sum(axis=1) + (1.0 - Pl.sum(axis=1)) * rf.to_numpy() - COST * turn
    return pd.Series(ret, index=idx), pd.Series(turn, index=idx)

def apply_dd_brake(ret: pd.Series, scale_floor: float = 0.5, dd_trig: float = -0.15):
    """Path-dependent: after DD hits dd_trig, multiply subsequent *target* exposure
    by scale_floor until a new equity high. Applied by scaling the return stream's
    risky part — implemented as post-hoc position scale on a second pass via
    equity path. Here we scale daily returns' active component approximately by
    rebuilding with a scale series derived from a pilot equity.
    """
    # Pilot equity assuming full risk, then derive scale, then caller rebuilds.
    # For simplicity in this research pass: scale the strategy return relative
    # to rf when in brake mode (approximate half-risk).
    eq = (1 + ret).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    scale = pd.Series(1.0, index=ret.index)
    braked = False
    for i, dt in enumerate(ret.index):
        if dd.iloc[i] <= dd_trig:
            braked = True
        if braked and eq.iloc[i] >= peak.iloc[i] * 0.999:  # new high releases
            # peak already updated; release when at high
            if abs(dd.iloc[i]) < 1e-12:
                braked = False
        scale.iloc[i] = scale_floor if braked else 1.0
    # Blend: r' = rf + scale*(r - rf)  ≈ half the active risk when braked
    return rf + scale * (ret - rf), scale

def metrics(ret, start=None, end=None):
    r = ret.loc[start:end].dropna()
    if len(r) < 40:
        return dict(cagr=None, vol=None, sharpe=None, max_dd=None, calmar=None,
                    total=None, worst_month=None, p5_roll12=None)
    eq = (1 + r).cumprod()
    yrs = len(r) / 252.0
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    rfl = rf.loc[r.index]
    sharpe = ((r - rfl).mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0
    dd = eq / eq.cummax() - 1
    monthly = (1 + r).resample("ME").prod() - 1
    roll12 = (1 + r).rolling(252).apply(lambda x: np.prod(1 + x) - 1, raw=False)
    p5 = float(roll12.quantile(0.05)) if roll12.notna().sum() > 20 else None
    return dict(
        cagr=round(float(cagr), 4),
        vol=round(float(r.std() * np.sqrt(252)), 3),
        sharpe=round(float(sharpe), 2),
        max_dd=round(float(dd.min()), 3),
        calmar=round(float(cagr / abs(dd.min())), 2) if dd.min() < 0 else None,
        total=round(float(eq.iloc[-1]) - 1, 2),
        worst_month=round(float(monthly.min()), 3),
        p5_roll12=round(p5, 3) if p5 is not None else None,
    )

def build(sleeves, sigp, use_chop=False, use_corr=False, vol_band_widen=False):
    """sleeves: list of (gate_key, budget, L, sigma_key, valid, returns, chop_key)"""
    cols = []
    for gate_key, budget, L, sig_key, valid, _r, chop_key in sleeves:
        gate = g[gate_key].copy()
        if use_chop:
            gate = gate * ch[chop_key]
        cols.append(sleeve_w(gate, budget, sigp, L, s[sig_key], valid))
    W = np.column_stack(cols)
    if use_corr:
        W = W * corr_scale[:, None]
    if vol_band_widen:
        # widen effective band in high-vol regimes by pre-smoothing targets
        # (skip — band handled below with dynamic band)
        pass
    tot = W.sum(axis=1)
    W *= np.where(tot > 1.0, 1.0 / tot, 1.0)[:, None]
    # dynamic band: 5% normally, 8% when QQQ vol > 25%
    band = np.where(s["q"] > 0.25, 0.08, 0.05)
    P = np.zeros_like(W)
    held = np.zeros(W.shape[1])
    for i in range(n):
        bd = float(band[i])
        for j in range(W.shape[1]):
            if abs(W[i, j] - held[j]) > bd or (W[i, j] == 0.0 and held[j] != 0.0):
                held[j] = W[i, j]
            P[i, j] = held[j]
    R = np.column_stack([r.to_numpy() for *_, r, __ in sleeves])
    return pnl(P, R) + (P,)

# Sleeve packs
# Live R50d cash book
def pack_r50d(L_eq=3, L_bd=3, L_gl=2, L_dbc=1, use_1x=False):
    if use_1x:
        # unlevered: L=1 on underlyings; same risk budgets → larger capital weights
        return [
            ("q", 0.40, 1, "q", TRUE, r_qqq1, "q"),
            ("t", 0.15, 1, "t", v_tlt.to_numpy(), r_tlt1, "t"),
            ("gl", 0.15, 1, "gl", v_gld.to_numpy(), r_gld1, "gl"),
            ("d", 0.10, 1, "d", v_dbc.to_numpy(), r_dbc1, "d"),
            ("m", 0.10, 1, "m", v_dbmf.to_numpy(), r_dbmf1, "m"),
            ("b", 0.10, 1, "b", v_btc.to_numpy(), r_btc1, "b"),
        ]
    return [
        ("q", 0.40, L_eq, "q", TRUE, r_ndx3, "q"),
        ("t", 0.15, L_bd, "t", v_tlt.to_numpy(), r_tlt3, "t"),
        ("gl", 0.15, L_gl, "gl", v_gld.to_numpy(), r_gld2, "gl"),
        ("d", 0.10, L_dbc, "d", v_dbc.to_numpy(), r_dbc1, "d"),
        ("m", 0.10, 1, "m", v_dbmf.to_numpy(), r_dbmf1, "m"),
        ("b", 0.10, 1, "b", v_btc.to_numpy(), r_btc1, "b"),
    ]

trio = [
    ("q", 0.60, 3, "q", TRUE, r_ndx3, "q"),
    ("t", 0.20, 3, "t", v_tlt.to_numpy(), r_tlt3, "t"),
    ("gl", 0.20, 2, "gl", v_gld.to_numpy(), r_gld2, "gl"),
]

variants_spec = {
    "R50d": dict(sleeves=pack_r50d(), sigp=0.50),
    "R35d": dict(sleeves=pack_r50d(), sigp=0.35),
    "R25d": dict(sleeves=pack_r50d(), sigp=0.25),
    "R50_1x": dict(sleeves=pack_r50d(use_1x=True), sigp=0.50),
    "R35_chop": dict(sleeves=pack_r50d(), sigp=0.35, use_chop=True),
    "R35_corr": dict(sleeves=pack_r50d(), sigp=0.35, use_corr=True),
    "R35_all": dict(sleeves=pack_r50d(), sigp=0.35, use_chop=True, use_corr=True),
    "Trio50": dict(sleeves=trio, sigp=0.50),
}

built = {}
for name, kw in variants_spec.items():
    ret, turn, P = build(**kw)
    built[name] = (ret, turn, P)

# DD brake variants: apply to R35d and R35_all
for src, dst in (("R35d", "R35_dd"), ("R35_all", "R35_all_dd")):
    ret0, turn0, P0 = built[src]
    ret_b, sc = apply_dd_brake(ret0)
    built[dst] = (ret_b, turn0, P0)  # turnover approx unchanged

# Also test R50d with dd brake (does brake alone fix 50% vol?)
ret_b, sc = apply_dd_brake(built["R50d"][0])
built["R50_dd"] = (ret_b, built["R50d"][1], built["R50d"][2])

last = idx[-1]
start3 = last - pd.DateOffset(years=3)
# DBMF live from ~2019-05; require all sleeves for "live-like" window
s_dbmf = dbmf.dropna().index.min()
s_live = max(pd.Timestamp("2019-06-01"), s_dbmf) if pd.notna(s_dbmf) else pd.Timestamp("2019-06-01")
s2015 = "2015-01-01"
s2007 = "2007-01-01"
s1999 = idx[260]

WINDOWS = [
    ("full_live_dbmf", s_live, None),   # fair comparison for 6-sleeve book
    ("from_2015", s2015, None),         # BTC+DBC live; DBMF mostly cash early
    ("from_2007", s2007, None),
    ("last_3y", start3, None),
]

CRISES = [
    ("gfc", "2007-10-01", "2009-06-30"),
    ("covid", "2020-02-15", "2020-04-30"),
    ("y2022", "2022-01-01", "2022-12-31"),
    ("tariff_2025", "2025-02-15", "2025-06-30"),
]

res = {"windows": {}, "crises": {}, "turnover": {}, "expo": {}, "meta": {}}
res["meta"] = {
    "dbmf_start": str(s_dbmf.date()) if pd.notna(s_dbmf) else None,
    "live_window_start": str(pd.Timestamp(s_live).date()),
    "data_end": str(last.date()),
    "note": "R50d matches live cash book; R35_* test retail risk upgrades",
}

for name, (ret, turn, P) in built.items():
    res["windows"][name] = {w: metrics(ret, a, b) for w, a, b in WINDOWS}
    yrs = n / 252.0
    res["turnover"][name] = round(float(turn.sum() / yrs), 2)
    res["expo"][name] = {
        "avg_gross": round(float(P.sum(axis=1).mean()), 3),
        "avg_cash": round(float(1 - P.sum(axis=1).mean()), 3),
    }

for w, a, b in WINDOWS:
    res["windows"].setdefault("bench_SPY", {})[w] = metrics(r_spy, a, b)
    res["windows"].setdefault("bench_QQQ", {})[w] = metrics(r_qqq1, a, b)

for lab, a, b in CRISES:
    res["crises"][lab] = {name: metrics(v[0], a, b) for name, v in built.items()}
    res["crises"][lab]["bench_SPY"] = metrics(r_spy, a, b)

# $100k terminal wealth on live window
res["wealth_100k"] = {}
for name, (ret, _, _) in built.items():
    r = ret.loc[s_live:].dropna()
    eq = (1 + r).cumprod()
    res["wealth_100k"][name] = {
        "start": str(pd.Timestamp(s_live).date()),
        "end": str(r.index[-1].date()) if len(r) else None,
        "final": round(float(100_000 * eq.iloc[-1]), 0) if len(eq) else None,
        "max_dd": metrics(ret, s_live)["max_dd"],
        "cagr": metrics(ret, s_live)["cagr"],
        "calmar": metrics(ret, s_live)["calmar"],
    }
res["wealth_100k"]["bench_SPY"] = {
    "final": round(float(100_000 * (1 + r_spy.loc[s_live:]).cumprod().iloc[-1]), 0),
    **{k: metrics(r_spy, s_live)[k] for k in ("max_dd", "cagr", "calmar")},
}

# Yearly returns for key variants
def yearly(ret, start):
    y = (1 + ret.loc[start:].dropna()).resample("YE").prod() - 1
    return {str(k.year): round(float(v), 3) for k, v in y.items()}

res["yearly"] = {
    k: yearly(built[k][0], s_live)
    for k in ("R50d", "R35d", "R35_all_dd", "R25d", "R50_1x", "Trio50")
}

# Scoreboard: live window rank by calmar then cagr
board = []
for name in built:
    m = res["windows"][name]["full_live_dbmf"]
    if m["calmar"] is None:
        continue
    board.append((name, m["calmar"], m["cagr"], m["max_dd"], m["sharpe"], m["vol"]))
board.sort(key=lambda x: (x[1], x[2]), reverse=True)
res["scoreboard_live"] = [
    {"name": n, "calmar": c, "cagr": g, "max_dd": d, "sharpe": sh, "vol": v}
    for n, c, g, d, sh, v in board
]

# Verdict helper numbers: growth sacrificed vs DD saved vs R50d
base = res["windows"]["R50d"]["full_live_dbmf"]
res["vs_R50d_live"] = {}
for name in built:
    m = res["windows"][name]["full_live_dbmf"]
    if m["cagr"] is None or base["cagr"] is None:
        continue
    res["vs_R50d_live"][name] = {
        "cagr_delta_pp": round((m["cagr"] - base["cagr"]) * 100, 1),
        "mdd_delta_pp": round((m["max_dd"] - base["max_dd"]) * 100, 1),  # less neg = better
        "calmar_delta": round(m["calmar"] - base["calmar"], 2) if m["calmar"] and base["calmar"] else None,
    }

with open(OUT, "w") as f:
    json.dump(res, f, indent=1)

print("=== SCOREBOARD (DBMF-live window) ===")
print(json.dumps(res["scoreboard_live"], indent=1))
print("\n=== $100k terminal wealth ===")
print(json.dumps(res["wealth_100k"], indent=1))
print("\n=== vs R50d ===")
print(json.dumps(res["vs_R50d_live"], indent=1))
print("\n=== crises (max_dd) ===")
for lab in res["crises"]:
    print(lab, {k: v.get("max_dd") for k, v in res["crises"][lab].items()})
print("\nwrote", OUT)
