"""
Round 6b: refine Dual Momentum as the high-turnover paper candidate.

v6 screen found DualMom as the only family that can beat R50d on CAGR with
clearly higher turnover. This pass:
  - sweeps L, K, sigma*, abs-gate, band
  - applies a proper position-level DD brake (not return-blend approx)
  - reports 2015 / 2019 / last-3y / crisis slices
  - compares to R50d baseline and TSMOM_L21 (best risk-adjusted alt)
  - proposes a deployable paper spec
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "output" / "v6_dualmom_refine_results.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

tick = [
    "QQQ", "SPY", "IWM", "EFA", "EEM",
    "TLT", "IEF", "GLD", "DBC", "DBMF",
    "BTC-USD", "TQQQ", "^IRX",
]
raw = yf.download(tick, start="2005-01-01", auto_adjust=True, progress=False)["Close"]
spy = raw["SPY"].dropna()
idx = spy.index
n = len(idx)


def al(c: str) -> pd.Series:
    return raw[c].reindex(idx)


qqq, iwm, efa, eem = al("QQQ"), al("IWM"), al("EFA"), al("EEM")
tlt, ief, gld, dbc, dbmf = al("TLT"), al("IEF"), al("GLD"), al("DBC"), al("DBMF")
btc, tqqq = al("BTC-USD"), al("TQQQ")
irx = al("^IRX").ffill().fillna(4.5) / 100.0
rf = irx / 252.0

ROT = ["QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "DBMF", "BTC"]
R = {
    "QQQ": qqq.pct_change().fillna(0.0),
    "IWM": iwm.pct_change().fillna(0.0),
    "EFA": efa.pct_change().fillna(0.0),
    "EEM": eem.pct_change().fillna(0.0),
    "TLT": tlt.pct_change().fillna(0.0),
    "IEF": ief.pct_change().fillna(0.0),
    "GLD": gld.pct_change().fillna(0.0),
    "DBC": dbc.pct_change().fillna(0.0),
    "DBMF": dbmf.pct_change().fillna(0.0),
    "BTC": (btc.pct_change() - 0.01 / 252.0).fillna(0.0),
    "SPY": spy.pct_change().fillna(0.0),
    "TQQQ": tqqq.pct_change().fillna(0.0),
}
PX = {
    "QQQ": qqq, "IWM": iwm, "EFA": efa, "EEM": eem,
    "TLT": tlt, "IEF": ief, "GLD": gld, "DBC": dbc, "DBMF": dbmf,
    "BTC": btc, "SPY": spy, "TQQQ": tqqq,
}
VALID = {k: (~PX[k].isna() & (PX[k] > 0)).to_numpy() for k in PX}
COST = 0.0005


def ewma_vol(px: pd.Series, alpha: float = 0.12) -> np.ndarray:
    r = px.pct_change().fillna(0.0)
    return np.sqrt(252.0 * (r**2).ewm(alpha=alpha, adjust=False).mean()).to_numpy()


VOL = {k: ewma_vol(PX[k]) for k in ROT + ["QQQ", "SPY", "TQQQ", "TLT", "GLD", "DBC", "DBMF", "BTC"]}


def mom_score(px: pd.Series, L: int) -> pd.Series:
    return px / px.shift(L) - 1


def band_positions(W: np.ndarray, band: float) -> np.ndarray:
    P = np.zeros_like(W)
    held = np.zeros(W.shape[1])
    for i in range(len(W)):
        for j in range(W.shape[1]):
            if abs(W[i, j] - held[j]) > band or (W[i, j] == 0.0 and held[j] != 0.0):
                held[j] = W[i, j]
            P[i, j] = held[j]
    return P


def weekly_hold(P: np.ndarray) -> np.ndarray:
    is_fri = idx.weekday == 4
    out = np.zeros_like(P)
    held = np.zeros(P.shape[1])
    for i in range(len(P)):
        if is_fri[i] or i == 0:
            held = P[i].copy()
        out[i] = held
    return out


def pnl(P: np.ndarray, rets: np.ndarray):
    Pl = np.vstack([np.zeros((1, P.shape[1])), P[:-1]])
    turn = np.abs(np.diff(np.vstack([np.zeros((1, P.shape[1])), Pl]), axis=0)).sum(axis=1)
    invested = Pl.sum(axis=1)
    ret = (Pl * rets).sum(axis=1) + (1.0 - invested) * rf.to_numpy() - COST * turn
    return pd.Series(ret, index=idx), pd.Series(turn, index=idx)


def metrics(ret: pd.Series, turn: pd.Series, start=None, end=None) -> dict:
    r = ret.loc[start:end].dropna()
    t = turn.reindex(r.index).fillna(0.0)
    if len(r) < 60:
        return dict(cagr=None, vol=None, sharpe=None, max_dd=None, calmar=None,
                    total=None, turnover_yr=None, worst_month=None, avg_invested=None)
    eq = (1 + r).cumprod()
    yrs = len(r) / 252.0
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    rfl = rf.loc[r.index]
    sharpe = float(((r - rfl).mean() / r.std()) * np.sqrt(252)) if r.std() > 0 else 0.0
    dd = eq / eq.cummax() - 1
    max_dd = float(dd.min())
    monthly = (1 + r).resample("ME").prod() - 1
    return dict(
        cagr=round(cagr, 4),
        vol=round(float(r.std() * np.sqrt(252)), 3),
        sharpe=round(sharpe, 2),
        max_dd=round(max_dd, 3),
        calmar=round(cagr / abs(max_dd), 2) if max_dd < 0 else None,
        total=round(float(eq.iloc[-1]) - 1, 2),
        turnover_yr=round(float(t.sum() / yrs), 2),
        worst_month=round(float(monthly.min()), 3),
    )


def build_dualmom(
    L=21,
    top_k=3,
    sigp=0.40,
    band=0.10,
    abs_gate=True,
    weekly=True,
    universe=None,
    dd_brake=False,
    dd_trig=-0.15,
    dd_floor=0.50,
):
    universe = universe or ROT
    scores = pd.DataFrame({s: mom_score(PX[s], L) for s in universe}).reindex(idx)
    W = np.zeros((n, len(universe)))

    for i, dt in enumerate(idx):
        row = scores.loc[dt]
        eligible = []
        for s in universe:
            if not VALID[s][i]:
                continue
            sc = row[s]
            if not np.isfinite(sc):
                continue
            if abs_gate and sc <= 0:
                continue
            eligible.append(s)
        if not eligible:
            continue
        ranked = sorted(eligible, key=lambda s: row[s], reverse=True)[:top_k]
        raw = np.zeros(len(universe))
        for s in ranked:
            raw[universe.index(s)] = 1.0 / len(ranked)
        # portfolio vol approx (ignore corr — retail simple)
        port_vol2 = sum((raw[universe.index(s)] * VOL[s][i]) ** 2 for s in ranked)
        port_vol = float(np.sqrt(port_vol2)) if port_vol2 > 0 else np.nan
        scale = min(sigp / port_vol, 1.0) if port_vol and port_vol > 1e-8 else 0.0
        W[i] = raw * scale

    P = band_positions(W, band=band)
    if weekly:
        P = weekly_hold(P)

    # Position-level DD brake: scale targets using pilot equity path
    if dd_brake:
        ret0, _ = pnl(P, np.column_stack([R[s].to_numpy() for s in universe]))
        eq = (1 + ret0).cumprod()
        peak = eq.cummax()
        dd = eq / peak - 1.0
        scale = np.ones(n)
        braked = False
        for i in range(n):
            if dd.iloc[i] <= dd_trig:
                braked = True
            if braked and abs(dd.iloc[i]) < 1e-12:
                braked = False
            scale[i] = dd_floor if braked else 1.0
        P = P * scale[:, None]

    rets = np.column_stack([R[s].to_numpy() for s in universe])
    return pnl(P, rets) + (P,)


def build_tsmom(L=21, sigp=0.30, band=0.08, weekly=True):
    core = ["QQQ", "TLT", "GLD", "DBC", "DBMF", "BTC"]
    bud = {"QQQ": 0.35, "TLT": 0.20, "GLD": 0.15, "DBC": 0.10, "DBMF": 0.10, "BTC": 0.10}
    cols = []
    for s in core:
        g = (mom_score(PX[s], L).fillna(0.0) > 0).astype(float).to_numpy()
        w = g * bud[s] * sigp / np.where(VOL[s] > 1e-8, VOL[s], np.nan)
        w = np.nan_to_num(np.where(VALID[s], w, 0.0), nan=0.0)
        cols.append(w)
    W = np.column_stack(cols)
    tot = W.sum(axis=1)
    W *= np.where(tot > 1.0, 1.0 / tot, 1.0)[:, None]
    P = band_positions(W, band=band)
    if weekly:
        P = weekly_hold(P)
    return pnl(P, np.column_stack([R[s].to_numpy() for s in core])) + (P,)


def build_r50d():
    from v6_high_turnover import build_r50d_baseline  # may fail if path; inline instead
    raise NotImplementedError


def build_r50d_inline():
    lookbacks = (63, 126, 252)
    sigp = 0.50

    def gate_ens(px, lbs):
        votes = [(px / px.shift(L) - 1 > 0).astype(float).where(~px.isna(), 0.0) for L in lbs]
        return pd.concat(votes, axis=1).mean(axis=1).fillna(0.0).to_numpy()

    def evol(px, alpha=0.06):
        r = px.pct_change().fillna(0.0)
        return np.sqrt(252 * (r**2).ewm(alpha=alpha, adjust=False).mean()).to_numpy()

    sleeves = [
        ("QQQ", 0.40, 3.0, R["TQQQ"].where(VALID["TQQQ"], 3 * R["QQQ"]).fillna(0.0)),
        ("TLT", 0.15, 3.0, 3 * R["TLT"] - 2 * (rf + 0.005 / 252)),
        ("GLD", 0.15, 2.0, 2 * R["GLD"] - 1 * (rf + 0.005 / 252)),
        ("DBC", 0.10, 1.0, R["DBC"]),
        ("DBMF", 0.10, 1.0, R["DBMF"]),
        ("BTC", 0.10, 1.0, R["BTC"]),
    ]
    cols, rets = [], []
    for sig, bud, lev, r in sleeves:
        g = gate_ens(PX[sig], lookbacks)
        vol = evol(PX[sig])
        w = g * bud * sigp / (lev * np.where(vol > 1e-8, vol, np.nan))
        w = np.nan_to_num(np.where(VALID[sig], w, 0.0), nan=0.0)
        cols.append(w)
        rets.append(r.to_numpy())
    W = np.column_stack(cols)
    tot = W.sum(axis=1)
    W *= np.where(tot > 1.0, 1.0 / tot, 1.0)[:, None]
    P = band_positions(W, band=0.05)
    return pnl(P, np.column_stack(rets)) + (P,)


# No-BTC universe (more traditional dual mom)
ROT_NO_BTC = [s for s in ROT if s != "BTC"]
# Equity+bond+gold only
ROT_CLASSIC = ["QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD"]

print("Building refine grid...")
variants = {}
variants["R50d_baseline"] = build_r50d_inline()
variants["TSMOM_L21_W_s30"] = build_tsmom(L=21, sigp=0.30)

# DualMom sweep
for L in (15, 21, 42, 63, 84):
    for k in (2, 3, 4):
        for sigp in (0.35, 0.45, 0.55):
            name = f"DM_L{L}_k{k}_s{int(sigp*100)}"
            variants[name] = build_dualmom(L=L, top_k=k, sigp=sigp, band=0.10, weekly=True)

# Risk overlays on best-looking region (L21/L42, k3)
for L in (21, 42):
    for k in (3, 4):
        for trig, floor in ((-0.12, 0.5), (-0.15, 0.5), (-0.15, 0.35), (-0.20, 0.5)):
            name = f"DM_L{L}_k{k}_s45_dd{int(abs(trig)*100)}_{int(floor*100)}"
            variants[name] = build_dualmom(
                L=L, top_k=k, sigp=0.45, band=0.10, weekly=True,
                dd_brake=True, dd_trig=trig, dd_floor=floor,
            )

# Universe ablations
variants["DM_L21_k3_s45_nobtc"] = build_dualmom(
    L=21, top_k=3, sigp=0.45, universe=ROT_NO_BTC
)
variants["DM_L21_k3_s45_classic"] = build_dualmom(
    L=21, top_k=3, sigp=0.45, universe=ROT_CLASSIC
)
variants["DM_L42_k3_s45_nobtc"] = build_dualmom(
    L=42, top_k=3, sigp=0.45, universe=ROT_NO_BTC
)

# No absolute gate (relative only) — usually worse
variants["DM_L21_k3_s45_noabs"] = build_dualmom(
    L=21, top_k=3, sigp=0.45, abs_gate=False
)

# Daily dual mom (higher turnover stress)
variants["DM_L21_k3_s45_daily"] = build_dualmom(
    L=21, top_k=3, sigp=0.45, weekly=False, band=0.08
)

windows = {
    "from_2015": "2015-01-01",
    "from_2019": "2019-06-01",
    "last_3y": str((idx[-1] - pd.DateOffset(years=3)).date()),
}

# Crisis windows
crises = {
    "covid_2020": ("2020-02-01", "2020-04-30"),
    "bear_2022": ("2022-01-01", "2022-12-31"),
    "tariff_2025": ("2025-01-01", "2025-06-30"),
}

res = {"windows": {}, "crises": {}, "ranking": {}, "deploy_spec": {}}

print(f"Evaluating {len(variants)} variants...")
for wname, start in windows.items():
    res["windows"][wname] = {}
    for name, (ret, turn, _) in variants.items():
        res["windows"][wname][name] = metrics(ret, turn, start=start)

for cname, (a, b) in crises.items():
    res["crises"][cname] = {}
    for name in (
        "R50d_baseline",
        "TSMOM_L21_W_s30",
        "DM_L21_k3_s45",
        "DM_L42_k3_s45",
        "DM_L21_k3_s45_dd15_50",
        "DM_L42_k4_s45_dd15_50",
        "DM_L21_k3_s45_nobtc",
    ):
        if name not in variants:
            continue
        ret, turn, _ = variants[name]
        r = ret.loc[a:b]
        if len(r) < 5:
            continue
        eq = (1 + r).cumprod()
        res["crises"][cname][name] = {
            "total": round(float(eq.iloc[-1] - 1), 3),
            "max_dd": round(float((eq / eq.cummax() - 1).min()), 3),
        }

base = res["windows"]["from_2015"]
# Rank: want CAGR >= R50d*0.85 OR beat R50d, with DD >= -30%, turn >= 15
r50 = base["R50d_baseline"]
ranked = []
for name, m in base.items():
    if name == "R50d_baseline" or m["cagr"] is None:
        continue
    if m["turnover_yr"] is None or m["turnover_yr"] < 12:
        continue
    if m["max_dd"] is None or m["max_dd"] < -0.30:
        continue
    ranked.append((name, m))

ranked.sort(key=lambda kv: (kv[1]["calmar"] or -999, kv[1]["cagr"] or -999), reverse=True)
res["ranking"]["from_2015_turn12_dd30"] = [{"name": n, **m} for n, m in ranked[:20]]

ranked_cagr = sorted(
    [(n, m) for n, m in ranked],
    key=lambda kv: kv[1]["cagr"],
    reverse=True,
)
res["ranking"]["from_2015_by_cagr"] = [{"name": n, **m} for n, m in ranked_cagr[:20]]

# Robustness: require also decent 2019+ and last_3y
robust = []
for name, m15 in base.items():
    if name.startswith("R50d") or m15["cagr"] is None:
        continue
    m19 = res["windows"]["from_2019"].get(name, {})
    m3 = res["windows"]["last_3y"].get(name, {})
    if not m19.get("cagr") or not m3.get("cagr"):
        continue
    if m15["turnover_yr"] < 12 or m15["max_dd"] < -0.30:
        continue
    if m15["sharpe"] < 0.8 or m19["sharpe"] < 0.7:
        continue
    # prefer not collapsing in last 3y
    if m3["cagr"] < 0.05:
        continue
    score = (
        0.4 * (m15["calmar"] or 0)
        + 0.3 * (m19["calmar"] or 0)
        + 0.3 * (m3["calmar"] or 0)
    )
    robust.append((name, score, m15, m19, m3))

robust.sort(key=lambda x: x[1], reverse=True)
res["ranking"]["robust"] = [
    {
        "name": n,
        "score": round(sc, 3),
        "from_2015": m15,
        "from_2019": m19,
        "last_3y": m3,
    }
    for n, sc, m15, m19, m3 in robust[:15]
]

# Deploy recommendation
if robust:
    best = robust[0][0]
else:
    best = ranked[0][0] if ranked else "DM_L21_k3_s45_dd15_50"

res["deploy_spec"] = {
    "recommended_name": best,
    "account_id_suggestion": "paper_dm21",
    "rebalance": "weekly (Friday close -> next session)",
    "universe": ROT,
    "signal": "dual momentum: abs return > 0 over L days, hold top-K by return",
    "sizing": "equal weight among winners, scale portfolio to sigma* with simple vol, 100% notional cap",
    "risk": "optional peak-DD brake",
    "costs_assumed_bps": 5,
    "rationale": (
        "Only high-turnover family in v6 that can compete with / beat R50d on CAGR; "
        "weekly cadence is retail-operable; diversifies vs R50d's slow trend book."
    ),
    "baseline_R50d_from_2015": r50,
}

OUT.write_text(json.dumps(res, indent=2))
print(f"Wrote {OUT}")

print("\n=== Baseline R50d ===")
for w in windows:
    m = res["windows"][w]["R50d_baseline"]
    print(f"  {w}: CAGR {m['cagr']:.1%} Sh {m['sharpe']} DD {m['max_dd']:.1%} turn {m['turnover_yr']} Calmar {m['calmar']}")

print("\n=== Robust ranking ===")
for row in res["ranking"]["robust"][:10]:
    m = row["from_2015"]
    print(
        f"  {row['name']:<36} score {row['score']:.2f}  "
        f"15y CAGR {m['cagr']:.1%} Sh {m['sharpe']} DD {m['max_dd']:.1%} turn {m['turnover_yr']}"
    )

print("\n=== Top Calmar (2015, turn>=12, DD>=-30%) ===")
for row in res["ranking"]["from_2015_turn12_dd30"][:10]:
    print(
        f"  {row['name']:<36} CAGR {row['cagr']:.1%} Sh {row['sharpe']} "
        f"DD {row['max_dd']:.1%} turn {row['turnover_yr']} Calmar {row['calmar']}"
    )

print("\n=== Top CAGR same filter ===")
for row in res["ranking"]["from_2015_by_cagr"][:10]:
    print(
        f"  {row['name']:<36} CAGR {row['cagr']:.1%} Sh {row['sharpe']} "
        f"DD {row['max_dd']:.1%} turn {row['turnover_yr']} Calmar {row['calmar']}"
    )

print("\n=== Crises ===")
for cname, rows in res["crises"].items():
    print(f"  {cname}:")
    for name, m in rows.items():
        print(f"    {name:<32} total {m['total']:+.1%}  DD {m['max_dd']:.1%}")

print(f"\nRecommended deploy: {res['deploy_spec']['recommended_name']}")
