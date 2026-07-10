"""v4: improve R50d with continuous TSMOM, covariance-aware sizing, and growth tilt.

Hypothesis (math / game theory):
  1. Discrete ensemble votes {0,1/3,2/3,1} discard signal magnitude. Continuous
     TSMOM (Moskowitz-Ooi-Pedersen) keeps signed strength and should raise
     Sharpe by sitting out weak trends and sizing into strong ones.
  2. Independent inverse-vol budgets ignore covariance. When sleeves co-move,
     true portfolio risk > sum of sleeve risks. Scale by predicted portfolio
     vol from a rolling cov so the 50% vol target is actually hit.
  3. For a $100k retail cash book maximizing growth under drawdown, a mild
     relative-momentum tilt (reallocate risk toward the strongest absolute
     trends among active sleeves) is closer to fractional-Kelly concentration
     than fixed budgets, without abandoning diversification.

Variants (all cash-account, 100% notional cap, T+1, 5bps, 5pt band):
  R50d       -- live baseline
  R50        -- drop DBC (prior research favorite on Calmar)
  CONT       -- R50d book + continuous long-only TSMOM gate
  COV        -- R50d book + rolling-cov portfolio vol scale
  CONT_COV   -- continuous gate + cov scale
  GROWTH     -- CONT_COV + relative-momentum risk tilt (top-heavy)
  GROWTH60   -- GROWTH at sigma*=0.60 (spend diversification if cov works)

Same data construction as v3_revised.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "output" / "v4_improve_results.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

tick = ["QQQ", "SPY", "TLT", "GLD", "DBC", "TQQQ", "BTC-USD", "AQMIX", "DBMF", "^IRX"]
raw = yf.download(tick, start="1999-03-10", auto_adjust=True, progress=False)["Close"]
qqq = raw["QQQ"].dropna()
idx = qqq.index
n = len(idx)
spy = raw["SPY"].reindex(idx).ffill()
irx = raw["^IRX"].reindex(idx).ffill().fillna(4.5) / 100.0
rf = irx / 252.0


def al(c):
    return raw[c].reindex(idx)


tlt, gld, dbc, btc, tqqq = al("TLT"), al("GLD"), al("DBC"), al("BTC-USD"), al("TQQQ")

SPREAD = 0.005 / 252
EXP = 0.0095 / 252


def synth(u, L):
    r = u.pct_change()
    return (L * r - (L - 1) * (rf + SPREAD) - EXP), ~r.isna()


r_ndx3, _ = synth(qqq, 3)
rtr = tqqq.pct_change()
ov = rtr.dropna().index
r_ndx3 = r_ndx3.fillna(0.0)
r_ndx3.loc[ov] = rtr.loc[ov]
r_tlt3, v_tlt = synth(tlt, 3)
r_tlt3 = r_tlt3.fillna(0.0)
r_gld2, v_gld = synth(gld, 2)
r_gld2 = r_gld2.fillna(0.0)
r_dbc1 = dbc.pct_change().fillna(0.0)
v_dbc = ~dbc.pct_change().isna()
r_btc1 = (btc.pct_change() - 0.01 / 252).fillna(0.0)
v_btc = ~btc.pct_change().isna()
r_spy = spy.pct_change().fillna(0.0)

r_mf = al("AQMIX").pct_change()
r_db = al("DBMF").pct_change()
ovm = r_db.dropna().index
r_mf.loc[ovm] = r_db.loc[ovm]
v_mf = ~r_mf.isna()
px_mf = (1 + r_mf.fillna(0.0)).cumprod().where(v_mf.cummax())
r_mf = r_mf.fillna(0.0)

LOOKBACKS = (63, 126, 252)
TRUE = np.ones(n, dtype=bool)


def gate_ens(px):
    votes = [
        (px / px.shift(L) - 1 > 0).astype(float).where(~px.isna(), 0.0)
        for L in LOOKBACKS
    ]
    return pd.concat(votes, axis=1).mean(axis=1).fillna(0.0).to_numpy()


def gate_cont(px):
    """Long-only continuous TSMOM: avg of soft scores mom/(vol*sqrt(L)), clipped."""
    r = px.pct_change()
    vol = np.sqrt((r**2).ewm(alpha=0.06, adjust=False).mean())
    scores = []
    for L in LOOKBACKS:
        mom = px / px.shift(L) - 1
        # annualize-ish: mom over L days / (daily vol * sqrt(L))
        z = mom / (vol * np.sqrt(L) + 1e-12)
        # soft long-only: map z>0 into (0,1] via tanh; weak trends ~0
        scores.append(np.tanh(np.maximum(z, 0.0) / 1.5).where(~px.isna(), 0.0))
    return pd.concat(scores, axis=1).mean(axis=1).fillna(0.0).to_numpy()


def evol(px):
    r = px.pct_change().fillna(0.0)
    return np.sqrt(252 * (r**2).ewm(alpha=0.06, adjust=False).mean()).to_numpy()


# underlying returns for cov (use signal assets, not leveraged products)
u_rets = {
    "q": qqq.pct_change().fillna(0.0),
    "t": tlt.pct_change().fillna(0.0),
    "gl": gld.pct_change().fillna(0.0),
    "d": dbc.pct_change().fillna(0.0),
    "m": r_mf,  # already filled
    "b": btc.pct_change().fillna(0.0),
}

g_ens = {k: gate_ens(v) for k, v in dict(q=qqq, t=tlt, gl=gld, d=dbc, b=btc, m=px_mf).items()}
g_cont = {k: gate_cont(v) for k, v in dict(q=qqq, t=tlt, gl=gld, d=dbc, b=btc, m=px_mf).items()}
s = {k: evol(v) for k, v in dict(q=qqq, t=tlt, gl=gld, d=dbc, b=btc, m=px_mf).items()}


def sleeve_w(gate, b, sigp, L, sig, valid):
    w = gate * b * sigp / (L * np.where(sig > 1e-8, sig, np.nan))
    return np.nan_to_num(np.where(valid, w, 0.0), nan=0.0)


def band(WT, bd=0.05):
    P = np.zeros_like(WT)
    held = np.zeros(WT.shape[1])
    for i in range(n):
        for j in range(WT.shape[1]):
            if abs(WT[i, j] - held[j]) > bd or (WT[i, j] == 0.0 and held[j] != 0.0):
                held[j] = WT[i, j]
            P[i, j] = held[j]
    return P


def pnl(P, R):
    Pl = np.vstack([np.zeros((1, P.shape[1])), P[:-1]])
    turn = np.abs(np.diff(np.vstack([np.zeros((1, P.shape[1])), Pl]), axis=0)).sum(axis=1)
    ret = pd.Series(
        (Pl * R).sum(axis=1)
        + (1 - Pl.sum(axis=1)) * rf.to_numpy()
        - 0.0005 * turn,
        index=idx,
    )
    return ret, float(turn.sum() / (n / 252))


def met(ret, a=None, b=None):
    r = ret.loc[a:b].dropna()
    if len(r) < 20:
        return dict(cagr=None, vol=None, sharpe=None, max_dd=None, calmar=None, worst_month=None)
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    rfl = rf.loc[r.index]
    sh = ((r - rfl).mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0.0
    dd = (eq / eq.cummax() - 1).min()
    monthly = (1 + r).resample("ME").prod() - 1
    return dict(
        cagr=round(float(cagr), 4),
        vol=round(float(r.std() * np.sqrt(252)), 3),
        sharpe=round(float(sh), 2),
        max_dd=round(float(dd), 3),
        calmar=round(float(cagr / abs(dd)), 2) if dd != 0 else None,
        worst_month=round(float(monthly.min()), 3),
    )


def rolling_cov_scale(W, keys, leverages, sigp, window=63):
    """Scale notional weights so predicted portfolio vol ~= sigp.

    Predicted vol uses underlying returns * leverage * weight, because sleeve
    returns are approximately L * underlying for the leveraged products.
    """
    # build aligned return matrix of underlyings
    U = np.column_stack([u_rets[k].to_numpy() for k in keys])
    # effective exposure to underlying = weight * L
    E = W * np.array(leverages)[None, :]
    out = np.zeros_like(W)
    # warm-up: no scale until window
    for i in range(n):
        if i < window:
            out[i] = W[i]
            continue
        block = U[i - window : i]
        # sample cov of daily underlying returns
        C = np.cov(block, rowvar=False)
        e = E[i]
        var = float(e @ C @ e)
        pred = np.sqrt(max(var, 0.0) * 252)
        if pred < 1e-8:
            out[i] = W[i]
        else:
            out[i] = W[i] * (sigp / pred)
    return out


def relative_tilt(W, gates, strength=0.5):
    """Tilt risk budgets toward sleeves with stronger gates (relative momentum).

    strength in [0,1]: 0 = no tilt, 1 = fully proportional to gate among actives.
    Preserves total raw weight sum before cash cap.
    """
    G = np.column_stack(gates)
    out = np.zeros_like(W)
    for i in range(n):
        w = W[i].copy()
        g = G[i]
        active = (w > 1e-8) & (g > 1e-8)
        if active.sum() < 2:
            out[i] = w
            continue
        total = w[active].sum()
        # blend equal-ish current weights with gate-proportional
        gate_w = g[active] / g[active].sum() * total
        blended = (1 - strength) * w[active] + strength * gate_w
        w[active] = blended
        out[i] = w
    return out


def cap100(W):
    tot = W.sum(axis=1)
    return W * np.where(tot > 1.0, 1.0 / np.maximum(tot, 1e-12), 1.0)[:, None]


def build_book(gates, budgets, leverages, sigs, valids, rets, keys, sigp,
               use_cov=False, tilt=0.0):
    W = np.column_stack([
        sleeve_w(gg, bb, sigp, L, ss, vv)
        for gg, bb, L, ss, vv in zip(gates, budgets, leverages, sigs, valids)
    ])
    if tilt > 0:
        W = relative_tilt(W, gates, strength=tilt)
    if use_cov:
        # cov scale uses target sigp; then cash-cap
        W = rolling_cov_scale(W, keys, leverages, sigp)
    W = cap100(W)
    P = band(W)
    R = np.column_stack([r.to_numpy() for r in rets])
    ret, turn = pnl(P, R)
    return ret, turn, P


# sleeve specs for R50d book
def r50d_spec(gate_map):
    keys = ["q", "t", "gl", "d", "m", "b"]
    gates = [gate_map[k] for k in keys]
    budgets = [0.40, 0.15, 0.15, 0.10, 0.10, 0.10]
    leverages = [3, 3, 2, 1, 1, 1]
    sigs = [s[k] for k in keys]
    valids = [TRUE, v_tlt.to_numpy(), v_gld.to_numpy(), v_dbc.to_numpy(),
              v_mf.to_numpy(), v_btc.to_numpy()]
    rets = [r_ndx3, r_tlt3, r_gld2, r_dbc1, r_mf, r_btc1]
    return keys, gates, budgets, leverages, sigs, valids, rets


def r50_spec(gate_map):
    keys = ["q", "t", "gl", "m", "b"]
    gates = [gate_map[k] for k in keys]
    budgets = [0.40, 0.20, 0.20, 0.10, 0.10]
    leverages = [3, 3, 2, 1, 1]
    sigs = [s[k] for k in keys]
    valids = [TRUE, v_tlt.to_numpy(), v_gld.to_numpy(), v_mf.to_numpy(), v_btc.to_numpy()]
    rets = [r_ndx3, r_tlt3, r_gld2, r_mf, r_btc1]
    return keys, gates, budgets, leverages, sigs, valids, rets


variants = {}

# R50d baseline
k, g, b, L, sg, v, r = r50d_spec(g_ens)
variants["R50d"] = build_book(g, b, L, sg, v, r, k, 0.50)

# R50 no DBC
k, g, b, L, sg, v, r = r50_spec(g_ens)
variants["R50"] = build_book(g, b, L, sg, v, r, k, 0.50)

# Continuous gate on R50d book
k, g, b, L, sg, v, r = r50d_spec(g_cont)
variants["CONT"] = build_book(g, b, L, sg, v, r, k, 0.50)

# Cov scale on R50d + ensemble
k, g, b, L, sg, v, r = r50d_spec(g_ens)
variants["COV"] = build_book(g, b, L, sg, v, r, k, 0.50, use_cov=True)

# Continuous + cov
k, g, b, L, sg, v, r = r50d_spec(g_cont)
variants["CONT_COV"] = build_book(g, b, L, sg, v, r, k, 0.50, use_cov=True)

# Growth: continuous + cov + relative tilt on R50 (drop DBC, more bond/gold budget)
k, g, b, L, sg, v, r = r50_spec(g_cont)
variants["GROWTH"] = build_book(g, b, L, sg, v, r, k, 0.50, use_cov=True, tilt=0.45)

# Growth at 60% vol target
k, g, b, L, sg, v, r = r50_spec(g_cont)
variants["GROWTH60"] = build_book(g, b, L, sg, v, r, k, 0.60, use_cov=True, tilt=0.45)

# Also: continuous + cov on R50 book at 50 (cleaner than R50d)
k, g, b, L, sg, v, r = r50_spec(g_cont)
variants["R50_CONT_COV"] = build_book(g, b, L, sg, v, r, k, 0.50, use_cov=True)

last = idx[-1]
start3 = last - pd.DateOffset(years=3)
s1999 = idx[260]
WINDOWS = [
    ("full_1999", s1999, None),
    ("from_2010", "2010-01-01", None),
    ("from_2015", "2015-01-01", None),
    ("last_3y", start3, None),
]
CRISES = [
    ("y2022", "2022-01-01", "2022-12-31"),
    ("tariff_2025", "2025-02-15", "2025-06-30"),
    ("covid_crash", "2020-02-15", "2020-03-31"),
    ("gfc", "2007-10-01", "2009-03-31"),
]

res = {
    "windows": {},
    "crises": {},
    "turnover": {},
    "capital": {},
    "yearly": {},
    "equity_curves": {},
    "notes": {
        "objective": "Maximize CAGR for $100k cash retail with disciplined drawdowns",
        "baseline": "R50d live config",
        "improvements": [
            "continuous long-only TSMOM gate (tanh of mom/vol)",
            "rolling 63d cov portfolio-vol scaling",
            "relative momentum tilt among active sleeves",
            "optional drop DBC (R50 book)",
        ],
    },
}

for name, (ret, turn, P) in variants.items():
    res["windows"][name] = {w: met(ret, a, b) for w, a, b in WINDOWS}
    res["turnover"][name] = round(turn, 2)
    tot = P.sum(axis=1)
    res["capital"][name] = {
        "avg_total_weight": round(float(tot.mean()), 3),
        "pct_days_cap_binds": round(float((tot > 0.99).mean()), 3),
        "avg_n_sleeves": round(float((P > 1e-4).sum(axis=1).mean()), 2),
    }
    y = (1 + ret.loc["2010":].dropna()).resample("YE").prod() - 1
    res["yearly"][name] = {str(k.year): round(float(v), 3) for k, v in y.items()}
    # downsample equity for canvas (monthly)
    eq = (1 + ret.loc[s1999:].dropna()).cumprod()
    monthly_eq = eq.resample("ME").last()
    res["equity_curves"][name] = {
        str(d.date()): round(float(v), 4) for d, v in monthly_eq.items()
    }

for w, a, b in WINDOWS:
    res["windows"].setdefault("bench_SPY", {})[w] = met(r_spy, a, b)
    res["windows"].setdefault("bench_QQQ", {})[w] = met(
        qqq.pct_change().fillna(0.0), a, b
    )

for lab, a, b in CRISES:
    res["crises"][lab] = {name: met(v[0], a, b) for name, v in variants.items()}
    res["crises"][lab]["bench_SPY"] = met(r_spy, a, b)

# $100k terminal wealth from 2010 and from 2015
res["terminal_100k"] = {}
for name, (ret, _, _) in variants.items():
    for label, start in (("from_2010", "2010-01-01"), ("from_2015", "2015-01-01")):
        r = ret.loc[start:].dropna()
        wealth = float((1 + r).prod() * 100_000)
        res["terminal_100k"].setdefault(name, {})[label] = round(wealth, 0)

res["terminal_100k"]["bench_SPY"] = {}
for label, start in (("from_2010", "2010-01-01"), ("from_2015", "2015-01-01")):
    r = r_spy.loc[start:].dropna()
    res["terminal_100k"]["bench_SPY"][label] = round(float((1 + r).prod() * 100_000), 0)

with open(OUT, "w") as f:
    json.dump(res, f)
print(json.dumps({k: v for k, v in res.items() if k != "equity_curves"}, indent=1))
print(f"\nWrote {OUT}")
