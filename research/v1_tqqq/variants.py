"""Variant comparison for the TQQQ systematic strategy."""
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

t21, t63, t126 = (rolling_tstat(logq, L) for L in (21, 63, 126))

sma200 = px["QQQ"].rolling(200).mean().to_numpy()
def hysteresis_gate(band=0.02):
    g = np.zeros(n); state = 0.0
    for i in range(n):
        if np.isnan(sma200[i]): state = 0.0
        elif close_q[i] > sma200[i]: state = 1.0
        elif close_q[i] < (1 - band) * sma200[i]: state = 0.0
        g[i] = state
    return g
gate = hysteresis_gate()

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

COST = 0.0005
def pnl(pos_s, shift=1):
    p = pos_s.shift(shift).fillna(0.0)
    turn = p.diff().abs().fillna(0.0)
    return p * r_tqqq + (1 - p) * rf_daily - COST * turn, turn

def metrics(ret, start=None):
    r = ret.loc[start:] if start is not None else ret
    eq = (1 + r).cumprod(); yrs = len(r) / 252.0
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    rf = rf_daily.loc[r.index]
    sharpe = ((r - rf).mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0
    dd = (eq / eq.cummax() - 1).min()
    return dict(cagr=round(float(cagr),3), vol=round(float(vol),3),
                sharpe=round(float(sharpe),2), mdd=round(float(dd),3),
                calmar=round(float(cagr/abs(dd)),2) if dd < 0 else None)

last = px.index[-1]
start3 = last - pd.DateOffset(years=3)
full_start = px.index[210]

def cap(sig_target):
    return np.nan_to_num(np.minimum(1.0, sig_target / (3.0 * sigma_q)), nan=0.0)

score_all = np.clip(np.nanmean([np.tanh(t/2) for t in (t21, t63, t126)], axis=0), 0, 1)
score_slow = np.clip(np.nanmean([np.tanh(t/2) for t in (t63, t126)], axis=0), 0, 1)
vol_ok35 = (sigma_q <= 0.35).astype(float)
vol_ok40 = (sigma_q <= 0.40).astype(float)

variants = {
  "V0_buyhold_TQQQ": pd.Series(np.ones(n), index=px.index),
  "V1_SMA200_binary": apply_band(gate, band=0.01),
  "V2_SMA200_voltarget40": apply_band(gate * cap(0.40)),
  "V3_orig_score_vt30_kill35": apply_band(gate * vol_ok35 * score_all * cap(0.30)),
  "V4_score_vt40_kill40": apply_band(gate * vol_ok40 * score_all * cap(0.40)),
  "V5_slowscore_vt40_kill40": apply_band(gate * vol_ok40 * score_slow * cap(0.40)),
  "V6_slowscore_vt50": apply_band(gate * score_slow * cap(0.50)),
  "V7_SMA200_voltarget55": apply_band(gate * cap(0.55)),
}

rows = {}
for name, pos_s in variants.items():
    ret, turn = pnl(pos_s)
    rows[name] = {
        "full": metrics(ret, full_start),
        "yr3": metrics(ret, start3),
        "avg_expo_3y": round(float(pos_s.loc[start3:].mean()), 2),
        "turnover_yr": round(float(turn.loc[start3:].sum() / 3), 1),
    }

rows["SPY"] = {"full": metrics(r_spy, full_start), "yr3": metrics(r_spy, start3)}
rows["QQQ"] = {"full": metrics(r_q.fillna(0), full_start), "yr3": metrics(r_q.fillna(0), start3)}
print(json.dumps(rows, indent=1))
