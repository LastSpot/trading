"""
Round 4: unconstrained redesign ladder.

A) Trio baseline (binary SMA200+hysteresis gate)      -- current recommendation
B) Trio with ensemble-speed gates (63/126/252d TSMOM votes)
C) Quint: + international equity (EFA 2x) + commodities (DBC 2x)
D) Six: + BTC spot trend sleeve (10% risk budget)
E) Six at sigma*_p = 65% (spend the diversification dividend)

All synthetic leverage: L*r - (L-1)*(rf+50bps) - 0.95%/yr expense.
Real TQQQ spliced in from 2010-02. BTC held unlevered (cost 1%/yr).
Gates/vols always on underlying (QQQ, TLT, GLD, EFA, DBC, BTC).
T+1 execution, 5 bps per unit turnover, idle cash at T-bills.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "v3_results.json"

tick = ["QQQ", "SPY", "TLT", "GLD", "EFA", "DBC", "TQQQ", "BTC-USD", "^IRX"]
raw = yf.download(tick, start="1999-03-10", auto_adjust=True, progress=False)["Close"]
qqq = raw["QQQ"].dropna()
idx = qqq.index
spy = raw["SPY"].reindex(idx).ffill()
irx = raw["^IRX"].reindex(idx).ffill().fillna(4.5) / 100.0
rf = irx / 252.0

def aligned(col):
    return raw[col].reindex(idx)

tlt, gld, efa, dbc = aligned("TLT"), aligned("GLD"), aligned("EFA"), aligned("DBC")
btc = raw["BTC-USD"].reindex(idx)   # 7-day asset sampled on trading days
tqqq = aligned("TQQQ")

r_qqq = qqq.pct_change().fillna(0.0)
r_spy = spy.pct_change().fillna(0.0)

SPREAD = 0.005 / 252.0
EXP = 0.0095 / 252.0

def synth(underlying, L, expense=EXP):
    r = underlying.pct_change()
    return (L * r - (L - 1) * (rf + SPREAD) - expense), ~r.isna()

r_ndx3, _ = synth(qqq, 3)
r_tqqq_real = tqqq.pct_change()
ov = r_tqqq_real.dropna().index
r_ndx3 = r_ndx3.fillna(0.0)
r_ndx3.loc[ov] = r_tqqq_real.loc[ov]
r_tlt3, v_tlt = synth(tlt, 3);  r_tlt3 = r_tlt3.fillna(0.0)
r_gld2, v_gld = synth(gld, 2);  r_gld2 = r_gld2.fillna(0.0)
r_efa2, v_efa = synth(efa, 2);  r_efa2 = r_efa2.fillna(0.0)
r_dbc2, v_dbc = synth(dbc, 2);  r_dbc2 = r_dbc2.fillna(0.0)
r_btc1 = (btc.pct_change() - 0.01 / 252.0).fillna(0.0)
v_btc = ~btc.pct_change().isna()

n = len(idx)

# ---------------- gates ----------------
def gate_binary(px, band=0.02):
    sma = px.rolling(200).mean().to_numpy(); c = px.to_numpy()
    g = np.zeros(n); state = 0.0
    for i in range(n):
        if np.isnan(sma[i]) or np.isnan(c[i]): state = 0.0
        elif c[i] > sma[i]: state = 1.0
        elif c[i] < (1 - band) * sma[i]: state = 0.0
        g[i] = state
    return g

def gate_ensemble(px):
    """Mean of sign(trailing return > 0) over 63/126/252 days -> {0,1/3,2/3,1}."""
    votes = []
    for L in (63, 126, 252):
        votes.append((px / px.shift(L) - 1 > 0).astype(float).where(~px.isna(), 0.0))
    return pd.concat(votes, axis=1).mean(axis=1).fillna(0.0).to_numpy()

def ewma_vol(px):
    r = px.pct_change().fillna(0.0)
    return np.sqrt(252.0 * (r ** 2).ewm(alpha=0.06, adjust=False).mean()).to_numpy()

s_q, s_t, s_g = ewma_vol(qqq), ewma_vol(tlt), ewma_vol(gld)
s_e, s_d, s_b = ewma_vol(efa), ewma_vol(dbc), ewma_vol(btc)

gb_q = gate_binary(qqq); gb_t = gate_binary(tlt); gb_g = gate_binary(gld)
ge_q = gate_ensemble(qqq); ge_t = gate_ensemble(tlt); ge_g = gate_ensemble(gld)
ge_e = gate_ensemble(efa); ge_d = gate_ensemble(dbc); ge_b = gate_ensemble(btc)

def sleeve_w(gate, budget, sigp, L, sigma, valid):
    w = gate * budget * sigp / (L * np.where(sigma > 1e-8, sigma, np.nan))
    return np.nan_to_num(np.where(valid, w, 0.0), nan=0.0)

def band_positions(WT, band=0.05):
    P = np.zeros_like(WT); held = np.zeros(WT.shape[1])
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
    ret = (Pl * R).sum(axis=1) + (1 - Pl.sum(axis=1)) * rf.to_numpy() - COST * turn
    return pd.Series(ret, index=idx), pd.Series(turn, index=idx)

def metrics(ret, start=None, end=None):
    r = ret.loc[start:end].dropna()
    eq = (1 + r).cumprod(); yrs = len(r) / 252.0
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    rfl = rf.loc[r.index]
    sharpe = ((r - rfl).mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0
    dd = (eq / eq.cummax() - 1)
    monthly = (1 + r).resample("ME").prod() - 1
    return dict(cagr=round(float(cagr), 4), vol=round(float(r.std() * np.sqrt(252)), 3),
                sharpe=round(float(sharpe), 2), max_dd=round(float(dd.min()), 3),
                calmar=round(float(cagr / abs(dd.min())), 2),
                worst_month=round(float(monthly.min()), 3))

TRUE = np.ones(n, dtype=bool)
def build(sleeves, sigp):
    """sleeves: list of (gate, budget, L, sigma, valid, returns)"""
    W = np.column_stack([sleeve_w(g, b, sigp, L, s, v) for g, b, L, s, v, _ in sleeves])
    tot = W.sum(axis=1)
    W *= np.where(tot > 1.0, 1.0 / tot, 1.0)[:, None]
    P = band_positions(W)
    R = np.column_stack([r.to_numpy() for *_, r in sleeves])
    return pnl(P, R) + (P,)

variants = {
 "A_trio_base": build([
    (gb_q, 0.60, 3, s_q, TRUE, r_ndx3), (gb_t, 0.20, 3, s_t, v_tlt.to_numpy(), r_tlt3),
    (gb_g, 0.20, 2, s_g, v_gld.to_numpy(), r_gld2)], 0.50),
 "B_trio_ens": build([
    (ge_q, 0.60, 3, s_q, TRUE, r_ndx3), (ge_t, 0.20, 3, s_t, v_tlt.to_numpy(), r_tlt3),
    (ge_g, 0.20, 2, s_g, v_gld.to_numpy(), r_gld2)], 0.50),
 "C_quint_ens": build([
    (ge_q, 0.45, 3, s_q, TRUE, r_ndx3), (ge_e, 0.10, 2, s_e, v_efa.to_numpy(), r_efa2),
    (ge_t, 0.15, 3, s_t, v_tlt.to_numpy(), r_tlt3), (ge_g, 0.20, 2, s_g, v_gld.to_numpy(), r_gld2),
    (ge_d, 0.10, 2, s_d, v_dbc.to_numpy(), r_dbc2)], 0.50),
 "D_six_ens": build([
    (ge_q, 0.40, 3, s_q, TRUE, r_ndx3), (ge_e, 0.10, 2, s_e, v_efa.to_numpy(), r_efa2),
    (ge_t, 0.15, 3, s_t, v_tlt.to_numpy(), r_tlt3), (ge_g, 0.15, 2, s_g, v_gld.to_numpy(), r_gld2),
    (ge_d, 0.10, 2, s_d, v_dbc.to_numpy(), r_dbc2), (ge_b, 0.10, 1, s_b, v_btc.to_numpy(), r_btc1)], 0.50),
 "E_six_ens_65": build([
    (ge_q, 0.40, 3, s_q, TRUE, r_ndx3), (ge_e, 0.10, 2, s_e, v_efa.to_numpy(), r_efa2),
    (ge_t, 0.15, 3, s_t, v_tlt.to_numpy(), r_tlt3), (ge_g, 0.15, 2, s_g, v_gld.to_numpy(), r_gld2),
    (ge_d, 0.10, 2, s_d, v_dbc.to_numpy(), r_dbc2), (ge_b, 0.10, 1, s_b, v_btc.to_numpy(), r_btc1)], 0.65),
}

last = idx[-1]
start3 = last - pd.DateOffset(years=3)
s1999 = idx[260]

res = {"windows": {}, "crises": {}, "turnover": {}, "expo": {}}
WINDOWS = [("full_1999", s1999, None), ("from_2007", "2007-01-01", None),
           ("from_2015", "2015-01-01", None), ("last_3y", start3, None)]
for name, (ret, turn, P) in variants.items():
    res["windows"][name] = {w: metrics(ret, a, b) for w, a, b in WINDOWS}
    res["turnover"][name] = round(float(turn.sum() / (n / 252)), 2)
    res["expo"][name] = [round(float(x), 3) for x in P.mean(axis=0)]
for w, a, b in WINDOWS:
    res["windows"].setdefault("bench_SPY", {})[w] = metrics(r_spy, a, b)
    res["windows"].setdefault("bench_QQQ", {})[w] = metrics(r_qqq, a, b)
    res["windows"].setdefault("bench_3xNDX", {})[w] = metrics(r_ndx3, a, b)

for lab, a, b in [("dotcom", "2000-03-01", "2003-03-31"),
                  ("gfc", "2007-10-01", "2009-06-30"),
                  ("y2022", "2022-01-01", "2022-12-31"),
                  ("tariff_2025", "2025-02-15", "2025-06-30")]:
    res["crises"][lab] = {name: metrics(v[0], a, b) for name, v in variants.items()}
    res["crises"][lab]["bench_SPY"] = metrics(r_spy, a, b)

# yearly returns for D vs A
def yearly(ret):
    y = (1 + ret.dropna()).resample("YE").prod() - 1
    return {str(k.year): round(float(v), 3) for k, v in y.items()}
res["yearly"] = {"A": yearly(variants["A_trio_base"][0].loc[s1999:]),
                 "D": yearly(variants["D_six_ens"][0].loc[s1999:]),
                 "E": yearly(variants["E_six_ens_65"][0].loc[s1999:])}

# equity series (monthly, log10) from 2015 (all sleeves live) and from 1999
for tag, start in (("m2015", "2015-01-01"), ("m1999", s1999)):
    eqs = {}
    for name in ("A_trio_base", "D_six_ens", "E_six_ens_65"):
        eqs[name] = (1 + variants[name][0].loc[start:]).cumprod() * 100
    eqs["spy"] = (1 + r_spy.loc[start:]).cumprod() * 100
    eqs["qqq"] = (1 + r_qqq.loc[start:]).cumprod() * 100
    mo = eqs["A_trio_base"].resample("ME").last().dropna()
    res[f"series_{tag}"] = {"dates": [str(d.date())[2:7].replace("-", "/") for d in mo.index]}
    for k, v in eqs.items():
        vm = v.resample("ME").last()
        res[f"series_{tag}"][k] = [round(float(np.log10(vm.get(d, np.nan))), 3) for d in mo.index]

# sleeve strategy-return correlations (D, gross sleeve pnl streams, from 2015)
P = variants["D_six_ens"][2]
R = np.column_stack([r_ndx3, r_efa2, r_tlt3, r_gld2, r_dbc2, r_btc1])
Pl = np.vstack([np.zeros((1, 6)), P[:-1]])
sleeve_pnl = pd.DataFrame(Pl * R, index=idx,
                          columns=["NDX", "EFA", "TLT", "GLD", "DBC", "BTC"]).loc["2015":]
res["sleeve_corr"] = sleeve_pnl.corr().round(2).to_dict()

with open(OUT, "w") as f:
    json.dump(res, f)
print(json.dumps({k: res[k] for k in ("windows", "crises", "turnover", "expo")}, indent=1))
print("YEARLY D:", res["yearly"]["D"])
print("CORR:", res["sleeve_corr"])
