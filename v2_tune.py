"""Spend the diversification dividend: raise sigma*_p on the multi-sleeve."""
import json
import numpy as np
import pandas as pd
import yfinance as yf

tick = ["QQQ", "SPY", "TLT", "GLD", "TQQQ", "^IRX"]
raw = yf.download(tick, start="1999-03-10", auto_adjust=True, progress=False)["Close"]
qqq = raw["QQQ"].dropna(); idx = qqq.index
spy = raw["SPY"].reindex(idx).ffill()
tlt = raw["TLT"].reindex(idx); gld = raw["GLD"].reindex(idx)
tqqq = raw["TQQQ"].reindex(idx)
irx = raw["^IRX"].reindex(idx).ffill().fillna(4.5) / 100.0
rf = irx / 252.0
r_qqq = qqq.pct_change().fillna(0.0)
r_tlt = tlt.pct_change(); r_gld = gld.pct_change()
r_tqqq_real = tqqq.pct_change()
SPREAD = 0.005/252; EXP3 = 0.0095/252
r_syn3q = 3*r_qqq - 2*(rf+SPREAD) - EXP3
ov = r_tqqq_real.dropna().index
r_ndx3 = r_syn3q.copy(); r_ndx3.loc[ov] = r_tqqq_real.loc[ov]
r_tlt3 = (3*r_tlt - 2*(rf+SPREAD) - EXP3).fillna(0.0)
r_gld2 = (2*r_gld - 1*(rf+SPREAD) - EXP3).fillna(0.0)
tlt_valid = ~r_tlt.isna(); gld_valid = ~r_gld.isna()
n = len(idx)

def gate_hyst(px, band=0.02):
    sma = px.rolling(200).mean().to_numpy(); c = px.to_numpy()
    g = np.zeros(len(px)); state = 0.0
    for i in range(len(px)):
        if np.isnan(sma[i]) or np.isnan(c[i]): state = 0.0
        elif c[i] > sma[i]: state = 1.0
        elif c[i] < (1-band)*sma[i]: state = 0.0
        g[i] = state
    return g
def ewma_vol(r): return np.sqrt(252*(r**2).ewm(alpha=0.06, adjust=False).mean()).to_numpy()

g_q, g_t, g_g = gate_hyst(qqq), gate_hyst(tlt), gate_hyst(gld)
s_q, s_t, s_g = ewma_vol(r_qqq), ewma_vol(r_tlt.fillna(0)), ewma_vol(r_gld.fillna(0))
RB = np.column_stack([r_ndx3.to_numpy(), r_tlt3.to_numpy(), r_gld2.to_numpy()])

def band_positions(WT, band=0.05):
    P = np.zeros_like(WT); held = np.zeros(WT.shape[1])
    for i in range(n):
        for j in range(WT.shape[1]):
            if abs(WT[i,j]-held[j]) > band or (WT[i,j]==0.0 and held[j]!=0.0):
                held[j] = WT[i,j]
            P[i,j] = held[j]
    return P

COST = 0.0005
def pnl(P, R):
    Pl = np.vstack([np.zeros((1,P.shape[1])), P[:-1]])
    turn = np.abs(np.diff(np.vstack([np.zeros((1,P.shape[1])), Pl]), axis=0)).sum(axis=1)
    ret = (Pl*R).sum(axis=1) + (1-Pl.sum(axis=1))*rf.to_numpy() - COST*turn
    return pd.Series(ret, index=idx)

def metrics(ret, start=None, end=None):
    r = ret.loc[start:end].dropna()
    eq = (1+r).cumprod(); yrs = len(r)/252
    cagr = eq.iloc[-1]**(1/yrs)-1
    rfl = rf.loc[r.index]
    sharpe = ((r-rfl).mean()/r.std())*np.sqrt(252)
    dd = (eq/eq.cummax()-1).min()
    return dict(cagr=round(float(cagr),3), vol=round(float(r.std()*np.sqrt(252)),3),
                sharpe=round(float(sharpe),2), mdd=round(float(dd),3),
                calmar=round(float(cagr/abs(dd)),2))

def build_B(sigp, budgets=(0.6,0.2,0.2), maxw=1.0):
    bN, bT, bG = budgets
    wN = g_q * bN*sigp/(3*np.where(s_q>1e-8,s_q,np.nan))
    wT = np.where(tlt_valid, g_t * bT*sigp/(3*np.where(s_t>1e-8,s_t,np.nan)), 0.0)
    wG = np.where(gld_valid, g_g * bG*sigp/(2*np.where(s_g>1e-8,s_g,np.nan)), 0.0)
    W = np.nan_to_num(np.column_stack([wN,wT,wG]), nan=0.0)
    tot = W.sum(axis=1); W *= np.where(tot>maxw, maxw/tot, 1.0)[:,None]
    return pnl(band_positions(W), RB)

last = idx[-1]; start3 = last - pd.DateOffset(years=3)
s1999 = idx[260]

out = {}
for sigp in (0.50, 0.65, 0.80, 1.00):
    for buds in ((0.6,0.2,0.2), (0.7,0.15,0.15)):
        r = build_B(sigp, buds)
        key = f"sig{int(sigp*100)}_b{int(buds[0]*100)}"
        out[key] = {"full99": metrics(r, s1999), "yr3": metrics(r, start3),
                    "dotcom": metrics(r, "2000-03-01","2003-03-31"),
                    "y2008": metrics(r, "2007-10-01","2009-06-30"),
                    "y2022": metrics(r, "2022-01-01","2022-12-31"),
                    "final_eq": round(float((1+r.loc[s1999:]).cumprod().iloc[-1]*100))}
print(json.dumps(out, indent=1))
