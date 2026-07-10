"""v4c: spend the diversification dividend (capped), vs R50d / R50.

Independent inverse-vol sizing assumes zero correlation, so realized portfolio
vol (~18%) sits far below the 50% sleeve-budget target. The cash-account fix:

  scale = clip(sigma_target / pred_port_vol, min_scale, max_scale)
  then cash-cap at 100%, and optionally cap any single sleeve.

This is the Kelly/vol-target consistent approach under a no-borrow constraint.
We also re-test R50 (drop DBC) and a concentrated growth book.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "output" / "v4c_improve_results.json"

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
u_rets = {
    "q": qqq.pct_change().fillna(0.0),
    "t": tlt.pct_change().fillna(0.0),
    "gl": gld.pct_change().fillna(0.0),
    "d": dbc.pct_change().fillna(0.0),
    "m": r_mf,
    "b": btc.pct_change().fillna(0.0),
}


def gate_ens(px):
    votes = [
        (px / px.shift(L) - 1 > 0).astype(float).where(~px.isna(), 0.0)
        for L in LOOKBACKS
    ]
    return pd.concat(votes, axis=1).mean(axis=1).fillna(0.0).to_numpy()


def evol(px):
    r = px.pct_change().fillna(0.0)
    return np.sqrt(252 * (r**2).ewm(alpha=0.06, adjust=False).mean()).to_numpy()


px_map = dict(q=qqq, t=tlt, gl=gld, d=dbc, b=btc, m=px_mf)
g_ens = {k: gate_ens(v) for k, v in px_map.items()}
s = {k: evol(v) for k, v in px_map.items()}


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


def cov_scale(W, keys, leverages, sigp, window=63, min_s=0.5, max_s=2.5):
    U = np.column_stack([u_rets[k].to_numpy() for k in keys])
    E0 = W * np.array(leverages)[None, :]
    out = np.zeros_like(W)
    scales = np.ones(n)
    for i in range(n):
        if i < window:
            out[i] = W[i]
            continue
        C = np.cov(U[i - window : i], rowvar=False)
        # ridge for stability
        C = C + np.eye(C.shape[0]) * 1e-8
        e = E0[i]
        pred = np.sqrt(max(float(e @ C @ e), 0.0) * 252)
        if pred < 1e-8:
            sc = 1.0
        else:
            sc = float(np.clip(sigp / pred, min_s, max_s))
        scales[i] = sc
        out[i] = W[i] * sc
    return out, scales


def cap100(W):
    tot = W.sum(axis=1)
    return W * np.where(tot > 1.0, 1.0 / np.maximum(tot, 1e-12), 1.0)[:, None]


def cap_sleeve(W, max_w=0.55):
    """Cap any single sleeve, redistribute leftover pro-rata to others under cap."""
    out = W.copy()
    for i in range(n):
        w = out[i].copy()
        for _ in range(6):
            over = w > max_w
            if not over.any():
                break
            excess = (w[over] - max_w).sum()
            w[over] = max_w
            under = ~over
            if under.sum() == 0 or w[under].sum() <= 0:
                break
            w[under] += excess * (w[under] / w[under].sum())
        out[i] = w
    return out


def build(gates, budgets, leverages, sigs, valids, rets, keys, sigp,
          spend_div=False, max_sleeve=None, max_s=2.0):
    W = np.column_stack([
        sleeve_w(gg, bb, sigp, L, ss, vv)
        for gg, bb, L, ss, vv in zip(gates, budgets, leverages, sigs, valids)
    ])
    scale_mean = 1.0
    if spend_div:
        W, scales = cov_scale(W, keys, leverages, sigp, max_s=max_s)
        scale_mean = float(np.mean(scales[63:]))
    if max_sleeve is not None:
        W = cap_sleeve(W, max_sleeve)
    W = cap100(W)
    P = band(W)
    R = np.column_stack([r.to_numpy() for r in rets])
    ret, turn = pnl(P, R)
    return ret, turn, P, scale_mean


def r50d():
    keys = ["q", "t", "gl", "d", "m", "b"]
    return (
        keys,
        [g_ens[k] for k in keys],
        [0.40, 0.15, 0.15, 0.10, 0.10, 0.10],
        [3, 3, 2, 1, 1, 1],
        [s[k] for k in keys],
        [TRUE, v_tlt.to_numpy(), v_gld.to_numpy(), v_dbc.to_numpy(), v_mf.to_numpy(), v_btc.to_numpy()],
        [r_ndx3, r_tlt3, r_gld2, r_dbc1, r_mf, r_btc1],
    )


def r50():
    keys = ["q", "t", "gl", "m", "b"]
    return (
        keys,
        [g_ens[k] for k in keys],
        [0.40, 0.20, 0.20, 0.10, 0.10],
        [3, 3, 2, 1, 1],
        [s[k] for k in keys],
        [TRUE, v_tlt.to_numpy(), v_gld.to_numpy(), v_mf.to_numpy(), v_btc.to_numpy()],
        [r_ndx3, r_tlt3, r_gld2, r_mf, r_btc1],
    )


variants = {}
meta = {}

for name, spec, kwargs in [
    ("R50d", r50d(), dict(sigp=0.50)),
    ("R50", r50(), dict(sigp=0.50)),
    ("DIV_R50d", r50d(), dict(sigp=0.50, spend_div=True, max_s=2.0)),
    ("DIV_R50d_cap", r50d(), dict(sigp=0.50, spend_div=True, max_s=2.0, max_sleeve=0.50)),
    ("DIV_R50", r50(), dict(sigp=0.50, spend_div=True, max_s=2.0, max_sleeve=0.50)),
    ("DIV_R50_55", r50(), dict(sigp=0.55, spend_div=True, max_s=2.0, max_sleeve=0.50)),
    ("DIV_R50d_mild", r50d(), dict(sigp=0.50, spend_div=True, max_s=1.4, max_sleeve=0.50)),
]:
    keys, g, b, L, sg, v, r = spec
    sigp = kwargs.pop("sigp")
    ret, turn, P, sc = build(g, b, L, sg, v, r, keys, sigp, **kwargs)
    variants[name] = (ret, turn, P)
    meta[name] = {"avg_cov_scale": round(sc, 3)}

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
    "terminal_100k": {},
    "meta": meta,
}

for name, (ret, turn, P) in variants.items():
    res["windows"][name] = {w: met(ret, a, b) for w, a, b in WINDOWS}
    res["turnover"][name] = round(turn, 2)
    tot = P.sum(axis=1)
    res["capital"][name] = {
        "avg_total_weight": round(float(tot.mean()), 3),
        "pct_days_cap_binds": round(float((tot > 0.99).mean()), 3),
        "max_sleeve_avg": round(float(P.max(axis=1).mean()), 3),
    }
    y = (1 + ret.loc["2010":].dropna()).resample("YE").prod() - 1
    res["yearly"][name] = {str(yy.year): round(float(vv), 3) for yy, vv in y.items()}
    eq = (1 + ret.loc["2010":].dropna()).cumprod()
    monthly_eq = eq.resample("ME").last()
    # normalize to 1.0 at start for canvas
    monthly_eq = monthly_eq / monthly_eq.iloc[0]
    res["equity_curves"][name] = {
        str(d.date()): round(float(vv), 4) for d, vv in monthly_eq.items()
    }
    for label, start in (("from_2010", "2010-01-01"), ("from_2015", "2015-01-01")):
        rr = ret.loc[start:].dropna()
        res["terminal_100k"].setdefault(name, {})[label] = round(
            float((1 + rr).prod() * 100_000), 0
        )

for w, a, b in WINDOWS:
    res["windows"].setdefault("bench_SPY", {})[w] = met(r_spy, a, b)

res["terminal_100k"]["bench_SPY"] = {}
for label, start in (("from_2010", "2010-01-01"), ("from_2015", "2015-01-01")):
    rr = r_spy.loc[start:].dropna()
    res["terminal_100k"]["bench_SPY"][label] = round(float((1 + rr).prod() * 100_000), 0)

# spy equity curve from 2010
eqs = (1 + r_spy.loc["2010":].dropna()).cumprod()
eqs = eqs.resample("ME").last()
eqs = eqs / eqs.iloc[0]
res["equity_curves"]["bench_SPY"] = {str(d.date()): round(float(v), 4) for d, v in eqs.items()}

for lab, a, b in CRISES:
    res["crises"][lab] = {name: met(v[0], a, b) for name, v in variants.items()}
    res["crises"][lab]["bench_SPY"] = met(r_spy, a, b)

with open(OUT, "w") as f:
    json.dump(res, f)

print("variant          CAGR10  Sh   MDD    Vol   CAGR15  term2010   avgW  scale")
for name in variants:
    w = res["windows"][name]
    t = res["terminal_100k"][name]["from_2010"]
    c = res["capital"][name]
    print(
        f"{name:16} {w['from_2010']['cagr']:.1%} {w['from_2010']['sharpe']:.2f} "
        f"{w['from_2010']['max_dd']:.1%} {w['from_2010']['vol']:.1%}  "
        f"{w['from_2015']['cagr']:.1%}  ${t:,.0f}  {c['avg_total_weight']:.2f}  "
        f"{meta[name]['avg_cov_scale']:.2f}"
    )
print("\nCrises:")
for name in variants:
    c = res["crises"]
    print(
        f"{name:16} 2022 ret={c['y2022'][name]['cagr']:+.1%} mdd={c['y2022'][name]['max_dd']:.1%}  "
        f"covid_mdd={c['covid_crash'][name]['max_dd']:.1%}  gfc_mdd={c['gfc'][name]['max_dd']:.1%}"
    )
print(f"\nWrote {OUT}")
