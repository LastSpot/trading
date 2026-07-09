"""Cash-account variant: EFA and DBC held unlevered at 2x weight (no margin).
Same risk contribution, no financing drag, more capital consumed."""
import json
import numpy as np
import pandas as pd
import yfinance as yf

tick = ["QQQ", "SPY", "TLT", "GLD", "EFA", "DBC", "TQQQ", "BTC-USD", "^IRX"]
raw = yf.download(tick, start="1999-03-10", auto_adjust=True, progress=False)["Close"]
qqq = raw["QQQ"].dropna(); idx = qqq.index; n = len(idx)
spy = raw["SPY"].reindex(idx).ffill()
irx = raw["^IRX"].reindex(idx).ffill().fillna(4.5) / 100.0
rf = irx / 252.0
def al(c): return raw[c].reindex(idx)
tlt, gld, efa, dbc, btc, tqqq = al("TLT"), al("GLD"), al("EFA"), al("DBC"), al("BTC-USD"), al("TQQQ")

SPREAD = 0.005/252; EXP = 0.0095/252
def synth(u, L):
    r = u.pct_change()
    return (L*r - (L-1)*(rf+SPREAD) - EXP), ~r.isna()
r_ndx3, _ = synth(qqq, 3)
rtr = tqqq.pct_change(); ov = rtr.dropna().index
r_ndx3 = r_ndx3.fillna(0.0); r_ndx3.loc[ov] = rtr.loc[ov]
r_tlt3, v_tlt = synth(tlt, 3); r_tlt3 = r_tlt3.fillna(0.0)
r_gld2, v_gld = synth(gld, 2); r_gld2 = r_gld2.fillna(0.0)
r_efa2, v_efa = synth(efa, 2); r_efa2 = r_efa2.fillna(0.0)
r_dbc2, v_dbc = synth(dbc, 2); r_dbc2 = r_dbc2.fillna(0.0)
r_efa1 = efa.pct_change().fillna(0.0)          # real fund, fees embedded
r_dbc1 = dbc.pct_change().fillna(0.0)
r_btc1 = (btc.pct_change() - 0.01/252).fillna(0.0)
v_btc = ~btc.pct_change().isna()
r_spy = spy.pct_change().fillna(0.0)

def gate_ens(px):
    votes = [(px/px.shift(L) - 1 > 0).astype(float).where(~px.isna(), 0.0) for L in (63,126,252)]
    return pd.concat(votes, axis=1).mean(axis=1).fillna(0.0).to_numpy()
def evol(px):
    r = px.pct_change().fillna(0.0)
    return np.sqrt(252*(r**2).ewm(alpha=0.06, adjust=False).mean()).to_numpy()

g = {k: gate_ens(v) for k, v in dict(q=qqq, t=tlt, gl=gld, e=efa, d=dbc, b=btc).items()}
s = {k: evol(v) for k, v in dict(q=qqq, t=tlt, gl=gld, e=efa, d=dbc, b=btc).items()}
TRUE = np.ones(n, dtype=bool)

def sleeve_w(gate, b, sigp, L, sig, valid):
    w = gate * b * sigp / (L * np.where(sig > 1e-8, sig, np.nan))
    return np.nan_to_num(np.where(valid, w, 0.0), nan=0.0)

def band(WT, bd=0.05):
    P = np.zeros_like(WT); held = np.zeros(WT.shape[1])
    for i in range(n):
        for j in range(WT.shape[1]):
            if abs(WT[i,j]-held[j]) > bd or (WT[i,j]==0.0 and held[j]!=0.0): held[j] = WT[i,j]
            P[i,j] = held[j]
    return P

def pnl(P, R):
    Pl = np.vstack([np.zeros((1,P.shape[1])), P[:-1]])
    turn = np.abs(np.diff(np.vstack([np.zeros((1,P.shape[1])), Pl]), axis=0)).sum(axis=1)
    return pd.Series((Pl*R).sum(axis=1) + (1-Pl.sum(axis=1))*rf.to_numpy() - 0.0005*turn, index=idx)

def met(ret, a=None, b=None):
    r = ret.loc[a:b].dropna()
    eq = (1+r).cumprod(); yrs = len(r)/252
    cagr = eq.iloc[-1]**(1/yrs)-1
    rfl = rf.loc[r.index]
    sh = ((r-rfl).mean()/r.std())*np.sqrt(252)
    dd = (eq/eq.cummax()-1).min()
    return dict(cagr=round(float(cagr),3), sharpe=round(float(sh),2),
                mdd=round(float(dd),3), calmar=round(float(cagr/abs(dd)),2))

def build(sleeves, sigp):
    W = np.column_stack([sleeve_w(gg, bb, sigp, L, ss, vv) for gg, bb, L, ss, vv, _ in sleeves])
    tot = W.sum(axis=1); W *= np.where(tot > 1.0, 1.0/tot, 1.0)[:, None]
    P = band(W)
    R = np.column_stack([r.to_numpy() for *_, r in sleeves])
    return pnl(P, R), P

# E: margin-style (2x synth for EFA/DBC).  F: cash account (1x at 2x weight).
E, PE = build([
  (g['q'],.40,3,s['q'],TRUE,r_ndx3), (g['e'],.10,2,s['e'],v_efa.to_numpy(),r_efa2),
  (g['t'],.15,3,s['t'],v_tlt.to_numpy(),r_tlt3), (g['gl'],.15,2,s['gl'],v_gld.to_numpy(),r_gld2),
  (g['d'],.10,2,s['d'],v_dbc.to_numpy(),r_dbc2), (g['b'],.10,1,s['b'],v_btc.to_numpy(),r_btc1)], 0.65)
F, PF = build([
  (g['q'],.40,3,s['q'],TRUE,r_ndx3), (g['e'],.10,1,s['e'],v_efa.to_numpy(),r_efa1),
  (g['t'],.15,3,s['t'],v_tlt.to_numpy(),r_tlt3), (g['gl'],.15,2,s['gl'],v_gld.to_numpy(),r_gld2),
  (g['d'],.10,1,s['d'],v_dbc.to_numpy(),r_dbc1), (g['b'],.10,1,s['b'],v_btc.to_numpy(),r_btc1)], 0.65)

last = idx[-1]; start3 = last - pd.DateOffset(years=3); s1999 = idx[260]
out = {}
for nm, ret, P in (("E_margin_style", E, PE), ("F_cash_account", F, PF)):
    out[nm] = {"full99": met(ret, s1999), "from2015": met(ret, "2015-01-01"),
               "yr3": met(ret, start3),
               "avg_total_weight": round(float(P.sum(axis=1).mean()), 3),
               "pct_days_cap_binds": round(float((P.sum(axis=1) > 0.99).mean()), 3)}
print(json.dumps(out, indent=1))
