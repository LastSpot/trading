"""Revised cash-account book vs the current live config (v3_cash.py variant F).

Proposed revisions, each tested separately and combined:
  1. sigma* 0.65 -> 0.50 (the 65% choice was in-sample; D-vs-E showed the extra
     return is bought almost purely with leverage and deeper drawdowns).
  2. Drop EFA (0.58 corr to NDX, little independent premium, largest capital
     consumer unlevered); redistribute its 10% budget to TLT and GLD.
  3. Replace DBC with a managed-futures trend sleeve (DBMF, proxied by AQMIX
     before DBMF's 2019-05 inception, spliced like real TQQQ into synthetic
     history). Long/short trend is the only sleeve type that profits in the
     stocks-and-bonds-down regime (2022, tariff 2025) that hurts this book.

Variants:
  F65_live   -- current live config (baseline)
  F50        -- live book at sigma* 0.50
  R65        -- revised book (no EFA, MF for DBC) at 0.65
  R50        -- revised book at 0.50
  R50d       -- keep DBC too (no EFA, MF sleeve added, TLT/GLD back to .15) at 0.50

Same engine as v3_cash.py: ensemble gates (63/126/252d votes) on underlyings,
EWMA vol sizing, cap at 100%, 5pt band, T+1, 5 bps per unit turnover, idle
cash at T-bills. MF sleeve enters 2010-01; windows before that are still
comparable because sleeves simply sit invalid (as DBC pre-2006, BTC pre-2014).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "output" / "v3_revised_results.json"

tick = ["QQQ", "SPY", "TLT", "GLD", "EFA", "DBC", "TQQQ", "BTC-USD",
        "AQMIX", "DBMF", "^IRX"]
raw = yf.download(tick, start="1999-03-10", auto_adjust=True, progress=False)["Close"]
qqq = raw["QQQ"].dropna(); idx = qqq.index; n = len(idx)
spy = raw["SPY"].reindex(idx).ffill()
irx = raw["^IRX"].reindex(idx).ffill().fillna(4.5) / 100.0
rf = irx / 252.0
def al(c): return raw[c].reindex(idx)
tlt, gld, efa, dbc, btc, tqqq = al("TLT"), al("GLD"), al("EFA"), al("DBC"), al("BTC-USD"), al("TQQQ")

SPREAD = 0.005 / 252; EXP = 0.0095 / 252
def synth(u, L):
    r = u.pct_change()
    return (L * r - (L - 1) * (rf + SPREAD) - EXP), ~r.isna()
r_ndx3, _ = synth(qqq, 3)
rtr = tqqq.pct_change(); ov = rtr.dropna().index
r_ndx3 = r_ndx3.fillna(0.0); r_ndx3.loc[ov] = rtr.loc[ov]
r_tlt3, v_tlt = synth(tlt, 3); r_tlt3 = r_tlt3.fillna(0.0)
r_gld2, v_gld = synth(gld, 2); r_gld2 = r_gld2.fillna(0.0)
r_efa1 = efa.pct_change().fillna(0.0); v_efa = ~efa.pct_change().isna()
r_dbc1 = dbc.pct_change().fillna(0.0); v_dbc = ~dbc.pct_change().isna()
r_btc1 = (btc.pct_change() - 0.01 / 252).fillna(0.0)
v_btc = ~btc.pct_change().isna()
r_spy = spy.pct_change().fillna(0.0)

# managed futures: AQMIX (mutual fund, 2010-01) spliced with DBMF (ETF, 2019-05)
r_mf = al("AQMIX").pct_change()
r_db = al("DBMF").pct_change(); ovm = r_db.dropna().index
r_mf.loc[ovm] = r_db.loc[ovm]
v_mf = ~r_mf.isna()
px_mf = (1 + r_mf.fillna(0.0)).cumprod().where(v_mf.cummax())  # NaN pre-inception
r_mf = r_mf.fillna(0.0)

def gate_ens(px):
    votes = [(px / px.shift(L) - 1 > 0).astype(float).where(~px.isna(), 0.0)
             for L in (63, 126, 252)]
    return pd.concat(votes, axis=1).mean(axis=1).fillna(0.0).to_numpy()
def evol(px):
    r = px.pct_change().fillna(0.0)
    return np.sqrt(252 * (r ** 2).ewm(alpha=0.06, adjust=False).mean()).to_numpy()

g = {k: gate_ens(v) for k, v in dict(q=qqq, t=tlt, gl=gld, e=efa, d=dbc, b=btc, m=px_mf).items()}
s = {k: evol(v) for k, v in dict(q=qqq, t=tlt, gl=gld, e=efa, d=dbc, b=btc, m=px_mf).items()}
TRUE = np.ones(n, dtype=bool)

def sleeve_w(gate, b, sigp, L, sig, valid):
    w = gate * b * sigp / (L * np.where(sig > 1e-8, sig, np.nan))
    return np.nan_to_num(np.where(valid, w, 0.0), nan=0.0)

def band(WT, bd=0.05):
    P = np.zeros_like(WT); held = np.zeros(WT.shape[1])
    for i in range(n):
        for j in range(WT.shape[1]):
            if abs(WT[i, j] - held[j]) > bd or (WT[i, j] == 0.0 and held[j] != 0.0):
                held[j] = WT[i, j]
            P[i, j] = held[j]
    return P

def pnl(P, R):
    Pl = np.vstack([np.zeros((1, P.shape[1])), P[:-1]])
    turn = np.abs(np.diff(np.vstack([np.zeros((1, P.shape[1])), Pl]), axis=0)).sum(axis=1)
    ret = pd.Series((Pl * R).sum(axis=1) + (1 - Pl.sum(axis=1)) * rf.to_numpy()
                    - 0.0005 * turn, index=idx)
    return ret, float(turn.sum() / (n / 252))

def met(ret, a=None, b=None):
    r = ret.loc[a:b].dropna()
    eq = (1 + r).cumprod(); yrs = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    rfl = rf.loc[r.index]
    sh = ((r - rfl).mean() / r.std()) * np.sqrt(252)
    dd = (eq / eq.cummax() - 1).min()
    monthly = (1 + r).resample("ME").prod() - 1
    return dict(cagr=round(float(cagr), 4), vol=round(float(r.std() * np.sqrt(252)), 3),
                sharpe=round(float(sh), 2), max_dd=round(float(dd), 3),
                calmar=round(float(cagr / abs(dd)), 2),
                worst_month=round(float(monthly.min()), 3))

def build(sleeves, sigp):
    W = np.column_stack([sleeve_w(gg, bb, sigp, L, ss, vv) for gg, bb, L, ss, vv, _ in sleeves])
    tot = W.sum(axis=1); W *= np.where(tot > 1.0, 1.0 / np.maximum(tot, 1e-12), 1.0)[:, None]
    P = band(W)
    R = np.column_stack([r.to_numpy() for *_, r in sleeves])
    ret, turn = pnl(P, R)
    return ret, turn, P

live_book = lambda: [
    (g['q'], .40, 3, s['q'], TRUE, r_ndx3), (g['e'], .10, 1, s['e'], v_efa.to_numpy(), r_efa1),
    (g['t'], .15, 3, s['t'], v_tlt.to_numpy(), r_tlt3), (g['gl'], .15, 2, s['gl'], v_gld.to_numpy(), r_gld2),
    (g['d'], .10, 1, s['d'], v_dbc.to_numpy(), r_dbc1), (g['b'], .10, 1, s['b'], v_btc.to_numpy(), r_btc1)]
rev_book = lambda: [
    (g['q'], .40, 3, s['q'], TRUE, r_ndx3),
    (g['t'], .20, 3, s['t'], v_tlt.to_numpy(), r_tlt3), (g['gl'], .20, 2, s['gl'], v_gld.to_numpy(), r_gld2),
    (g['m'], .10, 1, s['m'], v_mf.to_numpy(), r_mf), (g['b'], .10, 1, s['b'], v_btc.to_numpy(), r_btc1)]
rev_dbc_book = lambda: [
    (g['q'], .40, 3, s['q'], TRUE, r_ndx3),
    (g['t'], .15, 3, s['t'], v_tlt.to_numpy(), r_tlt3), (g['gl'], .15, 2, s['gl'], v_gld.to_numpy(), r_gld2),
    (g['d'], .10, 1, s['d'], v_dbc.to_numpy(), r_dbc1),
    (g['m'], .10, 1, s['m'], v_mf.to_numpy(), r_mf), (g['b'], .10, 1, s['b'], v_btc.to_numpy(), r_btc1)]

variants = {
    "F65_live": build(live_book(), 0.65),
    "F50": build(live_book(), 0.50),
    "R65": build(rev_book(), 0.65),
    "R50": build(rev_book(), 0.50),
    "R50d": build(rev_dbc_book(), 0.50),
}

last = idx[-1]; start3 = last - pd.DateOffset(years=3); s1999 = idx[260]
WINDOWS = [("full_1999", s1999, None), ("from_2010", "2010-01-01", None),
           ("from_2015", "2015-01-01", None), ("last_3y", start3, None)]
CRISES = [("y2022", "2022-01-01", "2022-12-31"),
          ("tariff_2025", "2025-02-15", "2025-06-30")]

res = {"windows": {}, "crises": {}, "turnover": {}, "capital": {}, "yearly": {}}
for name, (ret, turn, P) in variants.items():
    res["windows"][name] = {w: met(ret, a, b) for w, a, b in WINDOWS}
    res["turnover"][name] = round(turn, 2)
    tot = P.sum(axis=1)
    res["capital"][name] = {"avg_total_weight": round(float(tot.mean()), 3),
                            "pct_days_cap_binds": round(float((tot > 0.99).mean()), 3)}
    y = (1 + ret.loc["2010":].dropna()).resample("YE").prod() - 1
    res["yearly"][name] = {str(k.year): round(float(v), 3) for k, v in y.items()}
for w, a, b in WINDOWS:
    res["windows"].setdefault("bench_SPY", {})[w] = met(r_spy, a, b)
    res["windows"].setdefault("bench_QQQ", {})[w] = met(qqq.pct_change().fillna(0.0), a, b)
for lab, a, b in CRISES:
    res["crises"][lab] = {name: met(v[0], a, b) for name, v in variants.items()}
    res["crises"][lab]["bench_SPY"] = met(r_spy, a, b)

with open(OUT, "w") as f:
    json.dump(res, f)
print(json.dumps(res, indent=1))
