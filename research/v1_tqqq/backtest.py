"""
Systematic TQQQ swing strategy backtest.

Strategy: "Regime-Filtered Trend, Volatility-Targeted" (RFT-VT)
  - All signals computed on QQQ (underlying, 3x better SNR than path-dependent TQQQ)
  - Trend: ensemble of OLS log-price regression slope t-statistics, L in {21, 63, 126},
    score_L = tanh(t_L / 2)  (saturates beyond 2 standard errors of significance)
  - Regime gate: QQQ > SMA200 to enter, exit below 0.98*SMA200 (2% hysteresis band)
  - Vol kill-switch: EWMA (RiskMetrics lambda=0.94) annualized QQQ vol > 35% -> flat
    (variance drag of 3x ETF ~ 3*sigma^2 ~ 37%/yr at sigma=35%: negative EV zone)
  - Sizing: w = regime * max(S,0) * min(1, sigma_target / (3*sigma_QQQ))
    sigma_target = 30% ~ half-Kelly (full-Kelly vol = Sharpe ~ 0.55-0.60 for NDX)
  - Execution: signal at close t, executed MOC same close is unrealistic for retail ->
    positions applied to NEXT day's return (shift 1 = trade at close t as MOC using
    3:55pm prices; we additionally report shift-2 sensitivity)
  - Costs: 5 bps on turnover; idle cash earns 13-week T-bill (^IRX)
"""
import json
import numpy as np
import pandas as pd
import yfinance as yf

OUT = "/tmp/tqqq_backtest/results.json"

# ---------------- data ----------------
raw = yf.download(["TQQQ", "QQQ", "SPY", "^IRX"], start="2010-01-01",
                  auto_adjust=True, progress=False)["Close"]
px = raw[["TQQQ", "QQQ", "SPY"]].dropna()
irx = raw["^IRX"].reindex(px.index).ffill().fillna(0.0) / 100.0
rf_daily = irx / 252.0

print("Data range:", px.index[0].date(), "->", px.index[-1].date(), " rows:", len(px))

logq = np.log(px["QQQ"].to_numpy())
n = len(px)

# ---------------- trend: rolling OLS slope t-stat ----------------
def rolling_tstat(y, L):
    x = np.arange(L, dtype=float)
    xm = x - x.mean()
    sxx = float((xm ** 2).sum())
    out = np.full(len(y), np.nan)
    for i in range(L - 1, len(y)):
        w = y[i - L + 1: i + 1]
        ym = w - w.mean()
        slope = float((xm * ym).sum()) / sxx
        resid = ym - slope * xm
        s2 = float((resid ** 2).sum()) / (L - 2)
        se = np.sqrt(s2 / sxx)
        out[i] = slope / se if se > 1e-12 else 0.0
    return out

LOOKBACKS = [21, 63, 126]
scores = np.nanmean(np.array([np.tanh(rolling_tstat(logq, L) / 2.0)
                              for L in LOOKBACKS]), axis=0)
S_plus = np.clip(scores, 0.0, 1.0)

# ---------------- regime gate with hysteresis ----------------
sma200 = px["QQQ"].rolling(200).mean().to_numpy()
close_q = px["QQQ"].to_numpy()
regime = np.zeros(n)
state = 0.0
for i in range(n):
    if np.isnan(sma200[i]):
        state = 0.0
    elif close_q[i] > sma200[i]:
        state = 1.0
    elif close_q[i] < 0.98 * sma200[i]:
        state = 0.0
    regime[i] = state

# ---------------- EWMA vol (RiskMetrics) ----------------
r_q = px["QQQ"].pct_change()
ewma_var = (r_q ** 2).ewm(alpha=1 - 0.94, adjust=False).mean()
sigma_q = np.sqrt(252.0 * ewma_var).to_numpy()          # annualized QQQ vol
VOL_KILL = 0.35
SIGMA_TARGET = 0.30

vol_ok = (sigma_q <= VOL_KILL).astype(float)
cap = np.minimum(1.0, SIGMA_TARGET / (3.0 * np.where(sigma_q > 1e-8, sigma_q, np.nan)))

w_target = regime * vol_ok * S_plus * cap
w_target = np.nan_to_num(w_target, nan=0.0)

# ---------------- rebalance band ----------------
BAND = 0.05
pos = np.zeros(n)
held = 0.0
for i in range(n):
    if abs(w_target[i] - held) > BAND or w_target[i] == 0.0:
        held = w_target[i]
    pos[i] = held

pos_s = pd.Series(pos, index=px.index)

# ---------------- P&L ----------------
COST = 0.0005
r_tqqq = px["TQQQ"].pct_change().fillna(0.0)
r_spy = px["SPY"].pct_change().fillna(0.0)

def run_pnl(shift):
    p = pos_s.shift(shift).fillna(0.0)
    turn = p.diff().abs().fillna(0.0)
    ret = p * r_tqqq + (1 - p) * rf_daily - COST * turn
    return ret, turn

ret1, turn1 = run_pnl(1)
ret2, _ = run_pnl(2)

# ---------------- metrics ----------------
def metrics(ret, idx0=None, idx1=None):
    r = ret.loc[idx0:idx1] if idx0 is not None else ret
    eq = (1 + r).cumprod()
    yrs = len(r) / 252.0
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    rf = rf_daily.loc[r.index]
    sharpe = ((r - rf).mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0.0
    downside = r[r < 0].std() * np.sqrt(252)
    sortino = ((r - rf).mean() * 252) / downside if downside > 0 else 0.0
    dd = eq / eq.cummax() - 1
    mdd = dd.min()
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    monthly = (1 + r).resample("ME").prod() - 1
    return dict(cagr=round(float(cagr), 4), vol=round(float(vol), 4),
                sharpe=round(float(sharpe), 2), sortino=round(float(sortino), 2),
                max_dd=round(float(mdd), 4), calmar=round(float(calmar), 2),
                total=round(float(eq.iloc[-1]) - 1, 4),
                win_months=round(float((monthly > 0).mean()), 3),
                best_month=round(float(monthly.max()), 4),
                worst_month=round(float(monthly.min()), 4))

last = px.index[-1]
start3 = last - pd.DateOffset(years=3)
# warmup: skip first 200 bars for "full" sample
full_start = px.index[210]

res = {
    "data_start": str(px.index[0].date()), "data_end": str(last.date()),
    "start3": str(px.index[px.index.searchsorted(start3)].date()),
    "full": {
        "strategy": metrics(ret1, full_start),
        "tqqq": metrics(r_tqqq, full_start),
        "spy": metrics(r_spy, full_start),
        "qqq": metrics(r_q.fillna(0), full_start),
    },
    "yr3": {
        "strategy": metrics(ret1, start3),
        "strategy_shift2": metrics(ret2, start3),
        "tqqq": metrics(r_tqqq, start3),
        "spy": metrics(r_spy, start3),
        "qqq": metrics(r_q.fillna(0), start3),
    },
    "exposure_3y": round(float(pos_s.loc[start3:].mean()), 3),
    "pct_days_in_cash_3y": round(float((pos_s.loc[start3:] == 0).mean()), 3),
    "trades_per_year_3y": round(float((pos_s.loc[start3:].diff().abs() > 0).sum() / 3.0), 1),
    "annual_turnover_3y": round(float(turn1.loc[start3:].sum() / 3.0), 1),
}

# yearly returns (calendar) for strategy vs spy vs tqqq, last ~4 calendar years
def yearly(ret):
    y = (1 + ret).resample("YE").prod() - 1
    return {str(k.year): round(float(v), 4) for k, v in y.items()}

res["yearly"] = {"strategy": yearly(ret1), "spy": yearly(r_spy), "tqqq": yearly(r_tqqq)}

# ---------------- series for canvas (weekly sampled, 3y) ----------------
def weekly_series(ret, start):
    eq = (1 + ret.loc[start:]).cumprod() * 100.0
    wk = eq.resample("W-FRI").last().dropna()
    return wk

wk_strat = weekly_series(ret1, start3)
wk_spy = weekly_series(r_spy, start3)
wk_tqqq = weekly_series(r_tqqq, start3)
common = wk_strat.index.intersection(wk_spy.index).intersection(wk_tqqq.index)

eq_strat_d = (1 + ret1.loc[start3:]).cumprod()
dd_strat = (eq_strat_d / eq_strat_d.cummax() - 1).resample("W-FRI").min().dropna()
eq_tqqq_d = (1 + r_tqqq.loc[start3:]).cumprod()
dd_tqqq = (eq_tqqq_d / eq_tqqq_d.cummax() - 1).resample("W-FRI").min().dropna()
expo_wk = pos_s.loc[start3:].resample("W-FRI").mean().dropna()

res["series"] = {
    "dates": [str(d.date()) for d in common],
    "strategy": [round(float(wk_strat[d]), 2) for d in common],
    "spy": [round(float(wk_spy[d]), 2) for d in common],
    "tqqq": [round(float(wk_tqqq[d]), 2) for d in common],
    "dd_strategy": [round(float(dd_strat.get(d, np.nan)) * 100, 2) for d in common],
    "dd_tqqq": [round(float(dd_tqqq.get(d, np.nan)) * 100, 2) for d in common],
    "exposure": [round(float(expo_wk.get(d, np.nan)) * 100, 1) for d in common],
}

# monthly returns strategy vs SPY, 3y, for table
m_strat = ((1 + ret1.loc[start3:]).resample("ME").prod() - 1)
m_spy = ((1 + r_spy.loc[start3:]).resample("ME").prod() - 1)
res["monthly"] = {str(k.date())[:7]: [round(float(v) * 100, 1), round(float(m_spy[k]) * 100, 1)]
                  for k, v in m_strat.items() if k in m_spy.index}

with open(OUT, "w") as f:
    json.dump(res, f, indent=1)

print(json.dumps({k: v for k, v in res.items() if k not in ("series", "monthly")}, indent=1))
print("wrote", OUT)
