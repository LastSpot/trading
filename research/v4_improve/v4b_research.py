"""Refine v4: fix cov-scale blowups and continuous-gate under-sizing.

Problems from first pass:
  - CONT alone: tanh gate averages ~0.4-0.6 so portfolio sits under-invested
    (avg weight 44%) -> lower CAGR despite nicer risk.
  - COV/CONT_COV/GROWTH: scaling by predicted vol WITHOUT a leverage ceiling
    lets notional explode when estimated cov is low, then cash-cap binds at
    100% with concentrated risk -> GFC/full-sample MDD ~70%.

Fixes (still cash-account, 100% notional cap):
  1. Renormalize continuous gate so its long-run mean matches ensemble mean
     (preserve shape, restore average risk budget).
  2. Cov scale with MAX_GROSS_LEVERAGE = 1.0 already via cash cap, but also
     clamp the scale factor to [0.5, 1.5] so we never 3x into a quiet cov
     estimate (the failure mode). Prefer DOWN-scaling when cov is high.
  3. Add a "risk parity budgets" variant: budgets proportional to 1/vol of
     sleeve contribution, refreshed slowly — closer to equal risk.
  4. Add dual-momentum style: only allocate to sleeves with positive absolute
     AND positive relative (vs SPY) momentum — classic retail-friendly filter.

Also test a pragmatic "best of both": R50d book + soft continuous gate
(renormalized) + downside-only cov scale (scale down when pred_vol > target,
never scale up past 1.0).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "output" / "v4b_improve_results.json"

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


def gate_cont_raw(px):
    r = px.pct_change()
    vol = np.sqrt((r**2).ewm(alpha=0.06, adjust=False).mean())
    scores = []
    for L in LOOKBACKS:
        mom = px / px.shift(L) - 1
        z = mom / (vol * np.sqrt(L) + 1e-12)
        scores.append(np.tanh(np.maximum(z, 0.0) / 1.5).where(~px.isna(), 0.0))
    return pd.concat(scores, axis=1).mean(axis=1).fillna(0.0)


def gate_cont_calibrated(px, ref_ens):
    """Match long-run mean of continuous gate to ensemble (when both > 0)."""
    raw = gate_cont_raw(px)
    ens = pd.Series(ref_ens, index=px.index)
    # scale so E[cont | cont>0] ~= E[ens | ens>0]
    m_c = raw[raw > 0].mean()
    m_e = ens[ens > 0].mean()
    scale = float(m_e / m_c) if m_c and m_c > 0 else 1.0
    return np.clip((raw * scale).to_numpy(), 0.0, 1.0)


def evol(px):
    r = px.pct_change().fillna(0.0)
    return np.sqrt(252 * (r**2).ewm(alpha=0.06, adjust=False).mean()).to_numpy()


u_rets = {
    "q": qqq.pct_change().fillna(0.0),
    "t": tlt.pct_change().fillna(0.0),
    "gl": gld.pct_change().fillna(0.0),
    "d": dbc.pct_change().fillna(0.0),
    "m": r_mf,
    "b": btc.pct_change().fillna(0.0),
}

px_map = dict(q=qqq, t=tlt, gl=gld, d=dbc, b=btc, m=px_mf)
g_ens = {k: gate_ens(v) for k, v in px_map.items()}
g_cont = {k: gate_cont_calibrated(v, g_ens[k]) for k, v in px_map.items()}
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


def cov_scale_down(W, keys, leverages, sigp, window=63, max_up=1.0, min_scale=0.35):
    """Scale weights so predicted vol <= sigp; never lever up past max_up.

    This is the risk-management version: diversification dividend is spent by
    keeping more capital deployed when cov is low ONLY up to the cash-cap path
    already in sleeve_w; we refuse to amplify beyond the independent-vol size.
    """
    U = np.column_stack([u_rets[k].to_numpy() for k in keys])
    E = W * np.array(leverages)[None, :]
    out = np.zeros_like(W)
    for i in range(n):
        if i < window:
            out[i] = W[i]
            continue
        C = np.cov(U[i - window : i], rowvar=False)
        e = E[i]
        pred = np.sqrt(max(float(e @ C @ e), 0.0) * 252)
        if pred < 1e-8:
            scale = 1.0
        else:
            scale = sigp / pred
        scale = float(np.clip(scale, min_scale, max_up))
        out[i] = W[i] * scale
    return out


def relative_tilt(W, gates, strength=0.4):
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
        gate_w = g[active] / g[active].sum() * total
        w[active] = (1 - strength) * w[active] + strength * gate_w
        out[i] = w
    return out


def dual_mom_mask(keys, abs_gates, horizon=126):
    """Relative momentum vs SPY: sleeve trailing return > SPY trailing return."""
    spy_mom = spy / spy.shift(horizon) - 1
    masks = []
    px = {"q": qqq, "t": tlt, "gl": gld, "d": dbc, "m": px_mf, "b": btc}
    for k, g in zip(keys, abs_gates):
        sm = px[k] / px[k].shift(horizon) - 1
        rel = (sm > spy_mom).astype(float).fillna(0.0).to_numpy()
        # keep absolute gate, multiply by relative filter (0 or 1)
        # soft: if relative fails, cut gate by half rather than zero (less binary)
        masks.append(g * (0.5 + 0.5 * rel))
    return masks


def cap100(W):
    tot = W.sum(axis=1)
    return W * np.where(tot > 1.0, 1.0 / np.maximum(tot, 1e-12), 1.0)[:, None]


def build_book(gates, budgets, leverages, sigs, valids, rets, keys, sigp,
               use_cov_down=False, tilt=0.0):
    W = np.column_stack([
        sleeve_w(gg, bb, sigp, L, ss, vv)
        for gg, bb, L, ss, vv in zip(gates, budgets, leverages, sigs, valids)
    ])
    if tilt > 0:
        W = relative_tilt(W, gates, strength=tilt)
    if use_cov_down:
        W = cov_scale_down(W, keys, leverages, sigp)
    W = cap100(W)
    P = band(W)
    R = np.column_stack([r.to_numpy() for r in rets])
    ret, turn = pnl(P, R)
    return ret, turn, P


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


# Equal-risk budgets: inverse of typical sleeve vol contribution (L * sigma)
# refreshed annually would be ideal; use static long-run for simplicity
def equal_risk_budgets(keys, leverages):
    # use median vol
    vols = np.array([np.nanmedian(s[k]) for k in keys])
    risk = vols * np.array(leverages)
    inv = 1.0 / np.maximum(risk, 1e-6)
    return (inv / inv.sum()).tolist()


variants = {}

k, g, b, L, sg, v, r = r50d_spec(g_ens)
variants["R50d"] = build_book(g, b, L, sg, v, r, k, 0.50)

k, g, b, L, sg, v, r = r50_spec(g_ens)
variants["R50"] = build_book(g, b, L, sg, v, r, k, 0.50)

# calibrated continuous on R50d
k, g, b, L, sg, v, r = r50d_spec(g_cont)
variants["CONT_CAL"] = build_book(g, b, L, sg, v, r, k, 0.50)

# downside-only cov on R50d
k, g, b, L, sg, v, r = r50d_spec(g_ens)
variants["COV_DOWN"] = build_book(g, b, L, sg, v, r, k, 0.50, use_cov_down=True)

# continuous calibrated + cov down
k, g, b, L, sg, v, r = r50d_spec(g_cont)
variants["CONT_COV_DOWN"] = build_book(g, b, L, sg, v, r, k, 0.50, use_cov_down=True)

# R50 + cont cal + cov down + mild tilt  (recommended candidate)
k, g, b, L, sg, v, r = r50_spec(g_cont)
variants["IMPROVED"] = build_book(g, b, L, sg, v, r, k, 0.50, use_cov_down=True, tilt=0.35)

# R50d + dual momentum soft filter
k, g, b, L, sg, v, r = r50d_spec(g_ens)
g_dual = dual_mom_mask(k, g)
variants["DUAL"] = build_book(g_dual, b, L, sg, v, r, k, 0.50)

# equal-risk budget R50d
k, g, b0, L, sg, v, r = r50d_spec(g_ens)
b_er = equal_risk_budgets(k, L)
variants["EQRISK"] = build_book(g, b_er, L, sg, v, r, k, 0.50)

# IMPROVED at 55% vol (mild growth)
k, g, b, L, sg, v, r = r50_spec(g_cont)
variants["IMPROVED55"] = build_book(g, b, L, sg, v, r, k, 0.55, use_cov_down=True, tilt=0.35)

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
    "gate_means": {
        "ens": {k: round(float(np.mean(g_ens[k])), 3) for k in g_ens},
        "cont_cal": {k: round(float(np.mean(g_cont[k])), 3) for k in g_cont},
    },
}

for name, (ret, turn, P) in variants.items():
    res["windows"][name] = {w: met(ret, a, b) for w, a, b in WINDOWS}
    res["turnover"][name] = round(turn, 2)
    tot = P.sum(axis=1)
    res["capital"][name] = {
        "avg_total_weight": round(float(tot.mean()), 3),
        "pct_days_cap_binds": round(float((tot > 0.99).mean()), 3),
    }
    y = (1 + ret.loc["2010":].dropna()).resample("YE").prod() - 1
    res["yearly"][name] = {str(yy.year): round(float(vv), 3) for yy, vv in y.items()}
    eq = (1 + ret.loc[s1999:].dropna()).cumprod()
    monthly_eq = eq.resample("ME").last()
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
    res["windows"].setdefault("bench_QQQ", {})[w] = met(qqq.pct_change().fillna(0.0), a, b)

res["terminal_100k"]["bench_SPY"] = {}
for label, start in (("from_2010", "2010-01-01"), ("from_2015", "2015-01-01")):
    rr = r_spy.loc[start:].dropna()
    res["terminal_100k"]["bench_SPY"][label] = round(float((1 + rr).prod() * 100_000), 0)

for lab, a, b in CRISES:
    res["crises"][lab] = {name: met(v[0], a, b) for name, v in variants.items()}
    res["crises"][lab]["bench_SPY"] = met(r_spy, a, b)

with open(OUT, "w") as f:
    json.dump(res, f)

# compact print
print("variant        CAGR10  Sh   MDD    CAGR15  Sh   MDD    CAGR3y  term2010")
for name in variants:
    w = res["windows"][name]
    t = res["terminal_100k"][name]["from_2010"]
    print(
        f"{name:14} {w['from_2010']['cagr']:.1%} {w['from_2010']['sharpe']:.2f} "
        f"{w['from_2010']['max_dd']:.1%}  {w['from_2015']['cagr']:.1%} "
        f"{w['from_2015']['sharpe']:.2f} {w['from_2015']['max_dd']:.1%}  "
        f"{w['last_3y']['cagr']:.1%}  ${t:,.0f}"
    )
print("\nCrises MDD:")
for name in variants:
    c = res["crises"]
    print(
        f"{name:14} 2022={c['y2022'][name]['max_dd']:.1%} "
        f"covid={c['covid_crash'][name]['max_dd']:.1%} "
        f"gfc={c['gfc'][name]['max_dd']:.1%} "
        f"tariff={c['tariff_2025'][name]['max_dd']:.1%}"
    )
print(f"\nWrote {OUT}")
print("gate means", res["gate_means"])
