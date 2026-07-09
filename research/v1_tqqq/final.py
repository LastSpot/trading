"""Final strategy: SMA200 hysteresis gate + vol targeting (sigma*=0.50) on TQQQ."""
import json
import numpy as np
import pandas as pd
import yfinance as yf

OUT = "/tmp/tqqq_backtest/final_results.json"

raw = yf.download(["TQQQ", "QQQ", "SPY", "^IRX"], start="2010-01-01",
                  auto_adjust=True, progress=False)["Close"]
px = raw[["TQQQ", "QQQ", "SPY"]].dropna()
irx = raw["^IRX"].reindex(px.index).ffill().fillna(0.0) / 100.0
rf_daily = irx / 252.0
n = len(px)
close_q = px["QQQ"].to_numpy()
logq = np.log(close_q)

def rolling_tstat(y, L):
    x = np.arange(L, dtype=float); xm = x - x.mean(); sxx = float((xm**2).sum())
    out = np.full(len(y), np.nan)
    for i in range(L - 1, len(y)):
        w = y[i - L + 1: i + 1]; ym = w - w.mean()
        slope = float((xm * ym).sum()) / sxx
        resid = ym - slope * xm
        se = np.sqrt((float((resid**2).sum()) / (L - 2)) / sxx)
        out[i] = slope / se if se > 1e-12 else 0.0
    return out

sma200 = px["QQQ"].rolling(200).mean().to_numpy()
def gate_fn(band):
    g = np.zeros(n); state = 0.0
    for i in range(n):
        if np.isnan(sma200[i]): state = 0.0
        elif close_q[i] > sma200[i]: state = 1.0
        elif close_q[i] < (1 - band) * sma200[i]: state = 0.0
        g[i] = state
    return g

r_q = px["QQQ"].pct_change()
sigma_q = np.sqrt(252.0 * (r_q**2).ewm(alpha=0.06, adjust=False).mean()).to_numpy()
r_tqqq = px["TQQQ"].pct_change().fillna(0.0)
r_spy = px["SPY"].pct_change().fillna(0.0)

def build_pos(sig_target=0.50, hyst=0.02, band=0.05):
    wt = np.nan_to_num(gate_fn(hyst) * np.minimum(1.0, sig_target / (3.0 * sigma_q)), nan=0.0)
    pos = np.zeros(n); held = 0.0
    for i in range(n):
        if abs(wt[i] - held) > band or (wt[i] == 0.0 and held != 0.0):
            held = wt[i]
        pos[i] = held
    return pd.Series(pos, index=px.index)

def pnl(pos_s, shift=1, cost=0.0005):
    p = pos_s.shift(shift).fillna(0.0)
    turn = p.diff().abs().fillna(0.0)
    return p * r_tqqq + (1 - p) * rf_daily - cost * turn, turn

def metrics(ret, start=None, end=None):
    r = ret.loc[start:end]
    eq = (1 + r).cumprod(); yrs = len(r) / 252.0
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    rf = rf_daily.loc[r.index]
    sharpe = ((r - rf).mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0
    downside = r[r < 0].std() * np.sqrt(252)
    sortino = ((r - rf).mean() * 252) / downside if downside > 0 else 0
    dd = (eq / eq.cummax() - 1)
    monthly = (1 + r).resample("ME").prod() - 1
    return dict(cagr=round(float(cagr),4), vol=round(float(r.std()*np.sqrt(252)),4),
                sharpe=round(float(sharpe),2), sortino=round(float(sortino),2),
                max_dd=round(float(dd.min()),4),
                calmar=round(float(cagr/abs(dd.min())),2),
                total=round(float(eq.iloc[-1])-1,4),
                win_months=round(float((monthly>0).mean()),3),
                best_month=round(float(monthly.max()),4),
                worst_month=round(float(monthly.min()),4))

pos = build_pos()
ret, turn = pnl(pos)
last = px.index[-1]; start3 = last - pd.DateOffset(years=3)
full_start = px.index[210]

res = {
 "data_start": str(px.index[0].date()), "data_end": str(last.date()),
 "start3": str(px.index[px.index.searchsorted(start3)].date()),
 "yr3": {"strategy": metrics(ret, start3), "tqqq": metrics(r_tqqq, start3),
         "spy": metrics(r_spy, start3), "qqq": metrics(r_q.fillna(0), start3)},
 "full": {"strategy": metrics(ret, full_start), "tqqq": metrics(r_tqqq, full_start),
          "spy": metrics(r_spy, full_start), "qqq": metrics(r_q.fillna(0), full_start)},
 "expo3y": round(float(pos.loc[start3:].mean()),3),
 "cash_days_3y": round(float((pos.loc[start3:]==0).mean()),3),
 "expo_full": round(float(pos.loc[full_start:].mean()),3),
 "trades_yr_3y": round(float((pos.loc[start3:].diff().abs()>0).sum()/3),1),
 "turnover_yr_3y": round(float(turn.loc[start3:].sum()/3),1),
}

def yearly(r):
    y = (1 + r).resample("YE").prod() - 1
    return {str(k.year): round(float(v),4) for k, v in y.items()}
res["yearly"] = {"strategy": yearly(ret.loc[full_start:]),
                 "spy": yearly(r_spy.loc[full_start:]),
                 "tqqq": yearly(r_tqqq.loc[full_start:]),
                 "qqq": yearly(r_q.fillna(0).loc[full_start:])}

# sensitivity grid
sens = {}
for st in (0.40, 0.50, 0.60):
    for hy in (0.01, 0.02, 0.03):
        p = build_pos(st, hy)
        rr, _ = pnl(p)
        m3, mf = metrics(rr, start3), metrics(rr, full_start)
        sens[f"vt{int(st*100)}_hyst{int(hy*100)}"] = {
            "cagr3": m3["cagr"], "mdd3": m3["max_dd"],
            "cagrF": mf["cagr"], "mddF": mf["max_dd"], "calmarF": mf["calmar"]}
for shift, cost in ((2, 0.0005), (1, 0.0015)):
    rr, _ = pnl(pos, shift=shift, cost=cost)
    m3, mf = metrics(rr, start3), metrics(rr, full_start)
    sens[f"shift{shift}_cost{int(cost*1e4)}bp"] = {
        "cagr3": m3["cagr"], "mdd3": m3["max_dd"],
        "cagrF": mf["cagr"], "mddF": mf["max_dd"], "calmarF": mf["calmar"]}
res["sensitivity"] = sens

# rejected variants summary (from research rounds)
t21, t63, t126 = (rolling_tstat(logq, L) for L in (21, 63, 126))
score_all = np.clip(np.nanmean([np.tanh(t/2) for t in (t21, t63, t126)], axis=0), 0, 1)
vol_ok35 = (sigma_q <= 0.35).astype(float)
def build_from_wt(wt, band=0.05):
    wt = np.nan_to_num(wt, nan=0.0)
    p = np.zeros(n); held = 0.0
    for i in range(n):
        if abs(wt[i] - held) > band or (wt[i] == 0.0 and held != 0.0):
            held = wt[i]
        p[i] = held
    return pd.Series(p, index=px.index)
g2 = gate_fn(0.02)
alt = {
 "ensemble_tstat_vt30_kill35": build_from_wt(g2 * vol_ok35 * score_all *
        np.minimum(1.0, 0.30/(3.0*sigma_q))),
 "sma200_binary": build_from_wt(g2, band=0.01),
}
res["rejected"] = {}
for name, p in alt.items():
    rr, _ = pnl(p)
    res["rejected"][name] = {"yr3": metrics(rr, start3), "full": metrics(rr, full_start)}

# series for canvas
def wk(s): return s.resample("W-FRI").last().dropna()
eq_s = (1 + ret.loc[start3:]).cumprod() * 100
eq_spy = (1 + r_spy.loc[start3:]).cumprod() * 100
eq_tq = (1 + r_tqqq.loc[start3:]).cumprod() * 100
eq_qq = (1 + r_q.fillna(0).loc[start3:]).cumprod() * 100
w_s, w_spy, w_tq, w_qq = wk(eq_s), wk(eq_spy), wk(eq_tq), wk(eq_qq)
common = w_s.index
dd_s = wk((eq_s / eq_s.cummax() - 1) * 100)
dd_tq = wk((eq_tq / eq_tq.cummax() - 1) * 100)
dd_spy = wk((eq_spy / eq_spy.cummax() - 1) * 100)
expo = pos.loc[start3:].resample("W-FRI").mean().dropna() * 100
volw = pd.Series(sigma_q, index=px.index).loc[start3:].resample("W-FRI").last().dropna() * 100

res["series"] = {
 "dates": [str(d.date()) for d in common],
 "strategy": [round(float(v),2) for v in w_s],
 "spy": [round(float(w_spy.get(d, np.nan)),2) for d in common],
 "tqqq": [round(float(w_tq.get(d, np.nan)),2) for d in common],
 "qqq": [round(float(w_qq.get(d, np.nan)),2) for d in common],
 "dd_strategy": [round(float(dd_s.get(d, np.nan)),2) for d in common],
 "dd_tqqq": [round(float(dd_tq.get(d, np.nan)),2) for d in common],
 "dd_spy": [round(float(dd_spy.get(d, np.nan)),2) for d in common],
 "exposure": [round(float(expo.get(d, np.nan)),1) for d in common],
 "qqq_vol": [round(float(volw.get(d, np.nan)),1) for d in common],
}

# full-sample equity (log scale) monthly sampled
eq_full_s = (1 + ret.loc[full_start:]).cumprod() * 100
eq_full_spy = (1 + r_spy.loc[full_start:]).cumprod() * 100
eq_full_tq = (1 + r_tqqq.loc[full_start:]).cumprod() * 100
mo = eq_full_s.resample("ME").last().dropna()
res["series_full"] = {
 "dates": [str(d.date()) for d in mo.index],
 "strategy": [round(float(v),1) for v in mo],
 "spy": [round(float(eq_full_spy.resample('ME').last().get(d, np.nan)),1) for d in mo.index],
 "tqqq": [round(float(eq_full_tq.resample('ME').last().get(d, np.nan)),1) for d in mo.index],
}

m_s = (1 + ret.loc[start3:]).resample("ME").prod() - 1
m_spy = (1 + r_spy.loc[start3:]).resample("ME").prod() - 1
res["monthly"] = {str(k.date())[:7]: [round(float(v)*100,1), round(float(m_spy[k])*100,1)]
                  for k, v in m_s.items() if k in m_spy.index}

with open(OUT, "w") as f:
    json.dump(res, f)
print(json.dumps({k: v for k, v in res.items()
                  if k not in ("series","series_full","monthly")}, indent=1))
print("wrote", OUT)
