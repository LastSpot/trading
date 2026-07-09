"""
Round 3 research: maximize growth with survivable risk.

A) Splice synthetic 3x QQQ (1999-2010) with real TQQQ (2010+), validate synthetic.
B) Dot-com stress test: RG-VT on spliced series 1999-2026 vs buy-hold.
C) Multi-sleeve: 3x NDX + 3x TLT + 2x GLD, per-sleeve gate + vol target,
   risk budgets, total weight capped at 1.0 (no margin).
D) Same rules on SOXL (2010+) as an aggressiveness check.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "v2_results.json"

tick = ["QQQ", "SPY", "TLT", "GLD", "TQQQ", "SOXL", "SOXX", "^IRX"]
raw = yf.download(tick, start="1999-03-10", auto_adjust=True, progress=False)["Close"]
qqq = raw["QQQ"].dropna()
idx = qqq.index
spy = raw["SPY"].reindex(idx).ffill()
tlt = raw["TLT"].reindex(idx)
gld = raw["GLD"].reindex(idx)
tqqq = raw["TQQQ"].reindex(idx)
soxl = raw["SOXL"].reindex(idx)
soxx = raw["SOXX"].reindex(idx)
irx = raw["^IRX"].reindex(idx).ffill().fillna(4.5) / 100.0
rf = irx / 252.0

r_qqq = qqq.pct_change().fillna(0.0)
r_spy = spy.pct_change().fillna(0.0)
r_tlt = tlt.pct_change()
r_gld = gld.pct_change()
r_tqqq_real = tqqq.pct_change()
r_soxl = soxl.pct_change()

# ---- synthetic 3x QQQ: 3*r - 2*(rf+spread) - expense
SPREAD = 0.005 / 252.0   # ~50bps swap spread
EXP3 = 0.0095 / 252.0
r_syn3q = 3.0 * r_qqq - 2.0 * (rf + SPREAD) - EXP3

# validate on overlap
ov = r_tqqq_real.dropna().index
ovs = r_syn3q.loc[ov]
ovr = r_tqqq_real.loc[ov]
corr = float(np.corrcoef(ovs, ovr)[0, 1])
cagr_syn = float((1 + ovs).prod() ** (252 / len(ov)) - 1)
cagr_real = float((1 + ovr).prod() ** (252 / len(ov)) - 1)

# spliced 3x NDX return stream: synthetic before TQQQ inception, real after
r_ndx3 = r_syn3q.copy()
r_ndx3.loc[ov] = ovr

# synthetic 3x TLT and 2x GLD
r_tlt3 = (3.0 * r_tlt - 2.0 * (rf + SPREAD) - EXP3).fillna(0.0)
r_gld2 = (2.0 * r_gld - 1.0 * (rf + SPREAD) - EXP3).fillna(0.0)
tlt_valid = ~r_tlt.isna()
gld_valid = ~r_gld.isna()

# ---- shared machinery
def gate_hyst(px, band=0.02):
    sma = px.rolling(200).mean().to_numpy()
    c = px.to_numpy()
    g = np.zeros(len(px)); state = 0.0
    for i in range(len(px)):
        if np.isnan(sma[i]) or np.isnan(c[i]): state = 0.0
        elif c[i] > sma[i]: state = 1.0
        elif c[i] < (1 - band) * sma[i]: state = 0.0
        g[i] = state
    return g

def ewma_vol(r):
    return np.sqrt(252.0 * (r ** 2).ewm(alpha=0.06, adjust=False).mean()).to_numpy()

g_q, g_t, g_g = gate_hyst(qqq), gate_hyst(tlt), gate_hyst(gld)
s_q, s_t, s_g = ewma_vol(r_qqq), ewma_vol(r_tlt.fillna(0)), ewma_vol(r_gld.fillna(0))
n = len(idx)

def band_positions(WT, band=0.05):
    """WT: (n, k) target weights -> banded positions"""
    k = WT.shape[1]
    P = np.zeros_like(WT)
    held = np.zeros(k)
    for i in range(n):
        for j in range(k):
            if abs(WT[i, j] - held[j]) > band or (WT[i, j] == 0.0 and held[j] != 0.0):
                held[j] = WT[i, j]
            P[i, j] = held[j]
    return P

COST = 0.0005
def pnl_multi(P, R):
    """P: (n,k) positions, R: (n,k) sleeve returns -> daily strategy returns"""
    Pl = np.vstack([np.zeros((1, P.shape[1])), P[:-1]])   # shift 1 (T+1 exec)
    turn = np.abs(np.diff(np.vstack([np.zeros((1, P.shape[1])), Pl]), axis=0)).sum(axis=1)
    gross = (Pl * R).sum(axis=1)
    cash_w = 1.0 - Pl.sum(axis=1)
    ret = gross + cash_w * rf.to_numpy() - COST * turn
    return pd.Series(ret, index=idx), pd.Series(turn, index=idx)

def metrics(ret, start=None, end=None):
    r = ret.loc[start:end].dropna()
    eq = (1 + r).cumprod(); yrs = len(r) / 252.0
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    rfl = rf.loc[r.index]
    sharpe = ((r - rfl).mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0
    dd = (eq / eq.cummax() - 1)
    monthly = (1 + r).resample("ME").prod() - 1
    return dict(cagr=round(float(cagr), 4), vol=round(float(r.std() * np.sqrt(252)), 4),
                sharpe=round(float(sharpe), 2), max_dd=round(float(dd.min()), 4),
                calmar=round(float(cagr / abs(dd.min())), 2),
                total=round(float(eq.iloc[-1]) - 1, 2),
                worst_month=round(float(monthly.min()), 4))

# ---- Strategy A: TQQQ-only RG-VT on spliced series, 1999-2026
wA = (g_q * np.minimum(1.0, 0.50 / (3.0 * np.where(s_q > 1e-8, s_q, np.nan))))
wA = np.nan_to_num(wA, nan=0.0).reshape(-1, 1)
PA = band_positions(wA)
retA, turnA = pnl_multi(PA, r_ndx3.to_numpy().reshape(-1, 1))

# ---- Strategy B: multi-sleeve, budgets 0.6 / 0.2 / 0.2, sigma*_p = 50%
B_NDX, B_TLT, B_GLD = 0.60, 0.20, 0.20
SIG_P = 0.50
w_ndx = g_q * B_NDX * SIG_P / (3.0 * np.where(s_q > 1e-8, s_q, np.nan))
w_tlt = np.where(tlt_valid, g_t * B_TLT * SIG_P / (3.0 * np.where(s_t > 1e-8, s_t, np.nan)), 0.0)
w_gld = np.where(gld_valid, g_g * B_GLD * SIG_P / (2.0 * np.where(s_g > 1e-8, s_g, np.nan)), 0.0)
WB = np.nan_to_num(np.column_stack([w_ndx, w_tlt, w_gld]), nan=0.0)
tot = WB.sum(axis=1)
scale = np.where(tot > 1.0, 1.0 / tot, 1.0)
WB = WB * scale[:, None]
PB = band_positions(WB)
RB = np.column_stack([r_ndx3.to_numpy(), r_tlt3.to_numpy(), r_gld2.to_numpy()])
retB, turnB = pnl_multi(PB, RB)

# ---- Strategy D: same rules on SOXL (gate/vol on SOXX underlying), 2010+
g_sx = gate_hyst(soxx)
s_sx = ewma_vol(soxx.pct_change().fillna(0))
wD = np.nan_to_num(g_sx * np.minimum(1.0, 0.50 / (3.0 * np.where(s_sx > 1e-8, s_sx, np.nan))), nan=0.0)
wD = np.where(~r_soxl.isna().to_numpy(), wD, 0.0).reshape(-1, 1)
PD = band_positions(wD)
retD, turnD = pnl_multi(PD, r_soxl.fillna(0.0).to_numpy().reshape(-1, 1))

# benchmarks
eq_bh3 = (1 + r_ndx3).cumprod()

last = idx[-1]
start3 = last - pd.DateOffset(years=3)
s2004 = "2004-12-01"   # all three sleeves live (GLD inception Nov 2004)
s1999 = idx[260]        # after SMA200 warmup

res = {
 "synthetic_validation": {"corr_daily": round(corr, 4),
    "cagr_synthetic_overlap": round(cagr_syn, 4), "cagr_real_overlap": round(cagr_real, 4)},
 "windows": {},
}

def block(start, label, include_soxl=False):
    d = {
      "A_tqqq_rgvt": metrics(retA, start),
      "B_multisleeve": metrics(retB, start),
      "bh_3xNDX": metrics(r_ndx3, start),
      "bh_QQQ": metrics(r_qqq, start),
      "bh_SPY": metrics(r_spy, start),
    }
    if include_soxl:
        d["D_soxl_rgvt"] = metrics(retD, start)
    res["windows"][label] = d

block(s1999, "full_1999")
block(s2004, "from_2004")
block("2011-01-01", "from_2011", include_soxl=True)
block(start3, "last_3y", include_soxl=True)

# dot-com window specifically
res["dotcom_2000_2003"] = {
  "A_tqqq_rgvt": metrics(retA, "2000-03-01", "2003-03-31"),
  "B_multisleeve": metrics(retB, "2000-03-01", "2003-03-31"),
  "bh_3xNDX": metrics(r_ndx3, "2000-03-01", "2003-03-31"),
  "bh_QQQ": metrics(r_qqq, "2000-03-01", "2003-03-31"),
}
# 2008 and 2022
for lab, a, b in [("gfc_2008", "2007-10-01", "2009-06-30"), ("bear_2022", "2022-01-01", "2022-12-31")]:
    res[lab] = {
      "A_tqqq_rgvt": metrics(retA, a, b), "B_multisleeve": metrics(retB, a, b),
      "bh_3xNDX": metrics(r_ndx3, a, b), "bh_SPY": metrics(r_spy, a, b)}

res["avg_expo"] = {
  "A_full": round(float(PA.mean()), 3),
  "B_ndx": round(float(PB[:, 0].mean()), 3),
  "B_tlt": round(float(PB[:, 1].mean()), 3),
  "B_gld": round(float(PB[:, 2].mean()), 3),
  "B_cash": round(float(1 - PB.sum(axis=1).mean()), 3),
}
res["turnover_yr"] = {"A": round(float(turnA.sum() / (n / 252)), 2),
                      "B": round(float(turnB.sum() / (n / 252)), 2)}

# yearly returns for key years
def yearly(ret):
    y = (1 + ret.dropna()).resample("YE").prod() - 1
    return {str(k.year): round(float(v), 4) for k, v in y.items()}
res["yearly"] = {"A": yearly(retA.loc[s1999:]), "B": yearly(retB.loc[s1999:]),
                 "bh3x": yearly(r_ndx3.loc[s1999:]), "spy": yearly(r_spy.loc[s1999:])}

# monthly log10 equity series from 1999 for canvas
eqA = (1 + retA.loc[s1999:]).cumprod() * 100
eqB = (1 + retB.loc[s1999:]).cumprod() * 100
eq3 = (1 + r_ndx3.loc[s1999:]).cumprod() * 100
eqQ = (1 + r_qqq.loc[s1999:]).cumprod() * 100
eqS = (1 + r_spy.loc[s1999:]).cumprod() * 100
mo = eqA.resample("ME").last().dropna()
res["series_log"] = {
 "dates": [str(d.date())[2:7].replace("-", "/") for d in mo.index],
 "A": [round(float(np.log10(v)), 3) for v in mo],
 "B": [round(float(np.log10(eqB.resample('ME').last()[d])), 3) for d in mo.index],
 "bh3x": [round(float(np.log10(eq3.resample('ME').last()[d])), 3) for d in mo.index],
 "qqq": [round(float(np.log10(eqQ.resample('ME').last()[d])), 3) for d in mo.index],
 "spy": [round(float(np.log10(eqS.resample('ME').last()[d])), 3) for d in mo.index],
}
res["final_equity"] = {"A": round(float(eqA.iloc[-1])), "B": round(float(eqB.iloc[-1])),
                       "bh3x": round(float(eq3.iloc[-1])), "qqq": round(float(eqQ.iloc[-1])),
                       "spy": round(float(eqS.iloc[-1]))}

# B sleeve exposure over time (monthly) for canvas
expB = pd.DataFrame(PB, index=idx, columns=["ndx", "tlt", "gld"]).loc[s1999:]
expm = expB.resample("ME").mean().dropna() * 100
res["expo_series"] = {
 "dates": [str(d.date())[2:7].replace("-", "/") for d in expm.index],
 "ndx": [round(float(v), 1) for v in expm["ndx"]],
 "tlt": [round(float(v), 1) for v in expm["tlt"]],
 "gld": [round(float(v), 1) for v in expm["gld"]],
}

with open(OUT, "w") as f:
    json.dump(res, f)
print(json.dumps({k: v for k, v in res.items() if k not in ("series_log", "expo_series", "yearly")}, indent=1))
print("YEARLY A:", res["yearly"]["A"])
print("YEARLY B:", res["yearly"]["B"])
print("YEARLY bh3x:", res["yearly"]["bh3x"])
