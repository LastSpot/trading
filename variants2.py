"""Round 2: hybrid variants + sensitivity on finalist."""
import json
import numpy as np
import pandas as pd
import yfinance as yf

raw = yf.download(["TQQQ", "QQQ", "SPY", "^IRX"], start="2010-01-01",
                  auto_adjust=True, progress=False)["Close"]
px = raw[["TQQQ", "QQQ", "SPY"]].dropna()
irx = raw["^IRX"].reindex(px.index).ffill().fillna(0.0) / 100.0
rf_daily = irx / 252.0
n = len(px)
logq = np.log(px["QQQ"].to_numpy())
close_q = px["QQQ"].to_numpy()

def rolling_tstat(y, L):
    x = np.arange(L, dtype=float); xm = x - x.mean(); sxx = float((xm**2).sum())
    out = np.full(len(y), np.nan)
    for i in range(L - 1, len(y)):
        w = y[i - L + 1: i + 1]; ym = w - w.mean()
        slope = float((xm * ym).sum()) / sxx
        resid = ym - slope * xm
        s2 = float((resid**2).sum()) / (L - 2)
        se = np.sqrt(s2 / sxx)
        out[i] = slope / se if se > 1e-12 else 0.0
    return out

t63, t126 = (rolling_tstat(logq, L) for L in (63, 126))
score_slow = np.clip(np.nanmean([np.tanh(t/2) for t in (t63, t126)], axis=0), 0, 1)
tstat_pos = ((np.nan_to_num(t63) > 0) | (np.nan_to_num(t126) > 0)).astype(float)

sma200 = px["QQQ"].rolling(200).mean().to_numpy()
def hyst(band=0.02):
    g = np.zeros(n); state = 0.0
    for i in range(n):
        if np.isnan(sma200[i]): state = 0.0
        elif close_q[i] > sma200[i]: state = 1.0
        elif close_q[i] < (1 - band) * sma200[i]: state = 0.0
        g[i] = state
    return g
gate = hyst()

r_q = px["QQQ"].pct_change()
sigma_q = np.sqrt(252.0 * (r_q**2).ewm(alpha=0.06, adjust=False).mean()).to_numpy()
r_tqqq = px["TQQQ"].pct_change().fillna(0.0)
r_spy = px["SPY"].pct_change().fillna(0.0)

def apply_band(w_target, band=0.05):
    pos = np.zeros(n); held = 0.0
    for i in range(n):
        if abs(w_target[i] - held) > band or (w_target[i] == 0.0 and held != 0.0):
            held = w_target[i]
        pos[i] = held
    return pd.Series(pos, index=px.index)

def pnl(pos_s, shift=1, cost=0.0005):
    p = pos_s.shift(shift).fillna(0.0)
    turn = p.diff().abs().fillna(0.0)
    return p * r_tqqq + (1 - p) * rf_daily - cost * turn, turn

def metrics(ret, start=None):
    r = ret.loc[start:] if start is not None else ret
    eq = (1 + r).cumprod(); yrs = len(r) / 252.0
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    rf = rf_daily.loc[r.index]
    sharpe = ((r - rf).mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0
    dd = (eq / eq.cummax() - 1).min()
    return dict(cagr=round(float(cagr),3), vol=round(float(r.std()*np.sqrt(252)),3),
                sharpe=round(float(sharpe),2), mdd=round(float(dd),3),
                calmar=round(float(cagr/abs(dd)),2) if dd < 0 else None)

last = px.index[-1]; start3 = last - pd.DateOffset(years=3)
full_start = px.index[210]

def cap(st): return np.nan_to_num(np.minimum(1.0, st / (3.0 * sigma_q)), nan=0.0)

variants = {
  "V7_sma_vt55": gate * cap(0.55),
  "V8_sma_tstatgate_vt55": gate * tstat_pos * cap(0.55),
  "V9_sma_scorefloor_vt55": gate * (0.5 + 0.5 * score_slow) * cap(0.55),
  "V10_sma_vt45": gate * cap(0.45),
  "V11_sma_vt65": gate * cap(0.65),
  "V12_sma_tstatgate_vt45": gate * tstat_pos * cap(0.45),
}
out = {}
for name, wt in variants.items():
    pos_s = apply_band(wt)
    ret, turn = pnl(pos_s)
    out[name] = {"full": metrics(ret, full_start), "yr3": metrics(ret, start3),
                 "expo3y": round(float(pos_s.loc[start3:].mean()),2),
                 "to_yr": round(float(turn.loc[start3:].sum()/3),1)}

# sensitivity on V9
pos9 = apply_band(variants["V9_sma_scorefloor_vt55"])
for shift in (1, 2):
    for cost in (0.0005, 0.0015):
        ret, _ = pnl(pos9, shift=shift, cost=cost)
        out[f"V9_shift{shift}_cost{int(cost*1e4)}bp"] = {"yr3": metrics(ret, start3),
                                                          "full": metrics(ret, full_start)}
print(json.dumps(out, indent=1))
