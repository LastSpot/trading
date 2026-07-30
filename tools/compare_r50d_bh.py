"""Compare live R50d vs QQQ and 50/50 S&P–Nasdaq buy-and-hold."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

REPO = Path(__file__).resolve().parents[1]
res = json.loads((REPO / "output" / "v9_r50d_improve_results.json").read_text())

raw = yf.download(
    ["QQQ", "SPY", "VOO"],
    start="1999-03-10",
    auto_adjust=True,
    progress=False,
)["Close"]
qqq = raw["QQQ"].dropna()
idx = qqq.index
spy = raw["SPY"].reindex(idx).ffill()
voo = raw["VOO"].reindex(idx)
r_qqq = qqq.pct_change().fillna(0.0)
r_spy = spy.pct_change().fillna(0.0)
r_voo = voo.pct_change().fillna(0.0)
voo_start = voo.dropna().index[0]


def monthly_5050(ra: pd.Series, rb: pd.Series) -> pd.Series:
    month = idx.to_period("M")
    flags = np.zeros(len(idx), dtype=bool)
    flags[:-1] = month[:-1] != month[1:]
    flags[-1] = True
    P = np.zeros((len(idx), 2))
    w = np.array([0.5, 0.5])
    for i in range(len(idx)):
        if flags[i] or i == 0:
            w = np.array([0.5, 0.5])
        P[i] = w
    Pl = np.vstack([np.zeros((1, 2)), P[:-1]])
    r = np.column_stack([ra.to_numpy(), rb.to_numpy()])
    return pd.Series((Pl * r).sum(axis=1), index=idx)


r_5050_spy = monthly_5050(r_spy, r_qqq)
r_5050_voo = monthly_5050(r_voo, r_qqq)


def met(series: pd.Series, start, end=None) -> dict:
    s = series.loc[start:end].dropna()
    eq = (1 + s).cumprod()
    yrs = max((s.index[-1] - s.index[0]).days / 365.25, len(s) / 252)
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    dd = float((eq / eq.cummax() - 1).min())
    return {
        "cagr": cagr,
        "max_dd": dd,
        "calmar": cagr / abs(dd) if dd < 0 else None,
        "vol": float(s.std() * np.sqrt(252)),
        "terminal": float(100_000 * eq.iloc[-1]),
    }


eval_start = pd.Timestamp(res["meta"]["evaluation_start"])
windows = {
    "full_1999ish": (eval_start, None),
    "from_2010": (pd.Timestamp("2010-01-01"), None),
    "voo_era": (voo_start, None),
    "from_2015": (pd.Timestamp("2015-01-01"), None),
}

print("VOO inception:", voo_start.date())
print("R50d data end:", res["meta"]["data_end"])
for label, (a, b) in windows.items():
    print(f"\n== {label} ==")
    key = {
        "full_1999ish": "full",
        "from_2010": "from_2010",
        "from_2015": "from_2015",
        "voo_era": "from_2010",
    }[label]
    m = res["windows"][key]["R50d"]
    print(
        f"{'R50d':16s} cagr={m['cagr']*100:6.2f}%  mdd={m['drawdown']['max_dd']*100:7.2f}%  "
        f"calmar={m['calmar']:5.2f}  term=${m['terminal_100k']:,.0f}"
    )
    for name, series in [
        ("QQQ", r_qqq),
        ("50/50 SPY/QQQ", r_5050_spy),
        ("50/50 VOO/QQQ", r_5050_voo if label != "full_1999ish" else None),
    ]:
        if series is None:
            continue
        if label == "full_1999ish" and name == "50/50 VOO/QQQ":
            continue
        mm = met(series, a, b)
        print(
            f"{name:16s} cagr={mm['cagr']*100:6.2f}%  mdd={mm['max_dd']*100:7.2f}%  "
            f"calmar={mm['calmar']:5.2f}  term=${mm['terminal']:,.0f}"
        )
