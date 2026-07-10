"""Soft DD-brake research on the live R50d cash book.

Question: can a soft peak-to-trough drawdown brake improve live R50d risk
without giving up the ~27% CAGR the user wants to keep?

Brake (causal, live-realistic):
  - At each close t, compute strategy equity from positions held into t
    (T+1 P&L, costs, idle cash at T-bills) — same path a live account sees.
  - If peak-to-trough DD <= trigger, multiply ALL sleeve *targets* by floor
    until equity makes a new high, then restore full size.
  - 5pt rebalance band applies to the scaled targets.

This is single-pass causal (scale from the braked book's own equity), not the
two-pass unbraked-pilot approximation in v5_r50dd_sweep.py.

Book = live R50d: TQQQ/TMF/UGL/DBC/DBMF/BTC, ensemble 63/126/252, EWMA vol,
sigma*=50%, DBC/DBMF/BTC L=1, 100% notional cap. MF sleeve uses AQMIX→DBMF
splice (same as v3_revised / live research).

Soft grid (floors stay high so we don't donate CAGR):
  triggers: -12%, -15%, -18%, -20%, -22%, -25%
  floors:   0.70, 0.75, 0.80, 0.85, 0.90
  + raw R50d baseline

Objective: among variants within 1pp CAGR of R50d on the live DBMF window,
prefer better maxDD / ulcer / worst month. Flag Pareto improvements if any.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "output" / "v5_soft_dd_results.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

tick = [
    "QQQ", "SPY", "TLT", "GLD", "DBC", "TQQQ", "TMF", "UGL",
    "BTC-USD", "AQMIX", "DBMF", "^IRX",
]
raw = yf.download(tick, start="1999-03-10", auto_adjust=True, progress=False)["Close"]
qqq = raw["QQQ"].dropna()
idx = qqq.index
n = len(idx)


def al(c):
    return raw[c].reindex(idx)


spy = al("SPY").ffill()
tlt, gld, dbc, tqqq, tmf, ugl = al("TLT"), al("GLD"), al("DBC"), al("TQQQ"), al("TMF"), al("UGL")
btc = al("BTC-USD")
irx = al("^IRX").ffill().fillna(4.5) / 100.0
rf = irx / 252.0
rf_np = rf.to_numpy()

SPREAD = 0.005 / 252.0
EXP = 0.0095 / 252.0
COST = 0.0005
BAND = 0.05
SIGP = 0.50
TRUE = np.ones(n, dtype=bool)


def synth(u, L):
    r = u.pct_change()
    return (L * r - (L - 1) * (rf + SPREAD) - EXP), ~r.isna()


r_ndx3, _ = synth(qqq, 3)
ov = tqqq.pct_change().dropna().index
r_ndx3 = r_ndx3.fillna(0.0)
r_ndx3.loc[ov] = tqqq.pct_change().loc[ov]

r_tlt3, v_tlt = synth(tlt, 3)
r_tlt3 = r_tlt3.fillna(0.0)
ov_tmf = tmf.pct_change().dropna().index
r_tlt3.loc[ov_tmf] = tmf.pct_change().loc[ov_tmf]

r_gld2, v_gld = synth(gld, 2)
r_gld2 = r_gld2.fillna(0.0)
ov_ugl = ugl.pct_change().dropna().index
r_gld2.loc[ov_ugl] = ugl.pct_change().loc[ov_ugl]

r_dbc1 = dbc.pct_change().fillna(0.0)
v_dbc = ~dbc.pct_change().isna()

# managed futures: AQMIX spliced with DBMF (matches v3_revised / live research)
r_mf = al("AQMIX").pct_change()
r_db = al("DBMF").pct_change()
ovm = r_db.dropna().index
r_mf.loc[ovm] = r_db.loc[ovm]
v_mf = ~r_mf.isna()
px_mf = (1 + r_mf.fillna(0.0)).cumprod().where(v_mf.cummax())
r_mf = r_mf.fillna(0.0)

r_btc1 = (btc.pct_change() - 0.01 / 252.0).fillna(0.0)
v_btc = ~btc.pct_change().isna()
r_spy = spy.pct_change().fillna(0.0)
r_qqq = qqq.pct_change().fillna(0.0)


def gate_ens(px: pd.Series) -> np.ndarray:
    votes = [
        (px / px.shift(L) - 1 > 0).astype(float).where(~px.isna(), 0.0)
        for L in (63, 126, 252)
    ]
    return pd.concat(votes, axis=1).mean(axis=1).fillna(0.0).to_numpy()


def ewma_vol(px: pd.Series) -> np.ndarray:
    r = px.pct_change().fillna(0.0)
    return np.sqrt(252.0 * (r ** 2).ewm(alpha=0.06, adjust=False).mean()).to_numpy()


g = {
    "q": gate_ens(qqq),
    "t": gate_ens(tlt),
    "gl": gate_ens(gld),
    "d": gate_ens(dbc),
    "m": gate_ens(px_mf),
    "b": gate_ens(btc),
}
svol = {
    "q": ewma_vol(qqq),
    "t": ewma_vol(tlt),
    "gl": ewma_vol(gld),
    "d": ewma_vol(dbc),
    "m": ewma_vol(px_mf),
    "b": ewma_vol(btc),
}


def sleeve_w(gate, budget, sigp, L, sigma, valid):
    w = gate * budget * sigp / (L * np.where(sigma > 1e-8, sigma, np.nan))
    return np.nan_to_num(np.where(valid, w, 0.0), nan=0.0)


def raw_targets():
    """Live R50d target weights (pre-band), capped at 100% notional."""
    sleeves = [
        ("q", 0.40, 3, "q", TRUE, r_ndx3),
        ("t", 0.15, 3, "t", v_tlt.to_numpy(), r_tlt3),
        ("gl", 0.15, 2, "gl", v_gld.to_numpy(), r_gld2),
        ("d", 0.10, 1, "d", v_dbc.to_numpy(), r_dbc1),
        ("m", 0.10, 1, "m", v_mf.to_numpy(), r_mf),
        ("b", 0.10, 1, "b", v_btc.to_numpy(), r_btc1),
    ]
    W = np.column_stack([
        sleeve_w(g[gk], bud, SIGP, L, svol[sk], valid)
        for gk, bud, L, sk, valid, _ in sleeves
    ])
    tot = W.sum(axis=1)
    W *= np.where(tot > 1.0, 1.0 / np.maximum(tot, 1e-12), 1.0)[:, None]
    R = np.column_stack([r.to_numpy() for *_, r in sleeves])
    return W, R


def simulate_soft_brake(W, R, dd_trig=None, scale_floor=0.80, band=BAND):
    """Causal single-pass soft DD brake on targets.

    At close t, scale is decided from strategy equity through t-1 (no look-ahead).
    Banded targets P[t] are set at close t and earn R[t+1] (T+1, same as live).
    """
    k = W.shape[1]
    P = np.zeros_like(W)
    held = np.zeros(k)
    rets = np.zeros(n)
    turns = np.zeros(n)
    scales = np.ones(n)
    braked = False
    equity = 1.0
    peak = 1.0

    for i in range(n):
        dd = equity / peak - 1.0
        if dd_trig is not None:
            if dd <= dd_trig:
                braked = True
            if braked and abs(dd) < 1e-12:
                braked = False
            scales[i] = scale_floor if braked else 1.0

        target = W[i] * scales[i]
        for j in range(k):
            if abs(target[j] - held[j]) > band or (target[j] == 0.0 and held[j] != 0.0):
                held[j] = target[j]
            P[i, j] = held[j]

        if i == 0:
            rets[i] = rf_np[i]
            turns[i] = 0.0
        else:
            prev = P[i - 2] if i > 1 else np.zeros(k)
            turns[i] = float(np.abs(P[i - 1] - prev).sum())
            rets[i] = float(
                (P[i - 1] * R[i]).sum()
                + (1.0 - P[i - 1].sum()) * rf_np[i]
                - COST * turns[i]
            )

        equity *= 1.0 + rets[i]
        if equity > peak:
            peak = equity

    return (
        pd.Series(rets, index=idx),
        pd.Series(turns, index=idx),
        P,
        pd.Series(scales, index=idx),
        float((scales < 1.0 - 1e-12).mean()),
    )


def metrics(ret, start=None, end=None):
    r = ret.loc[start:end].dropna()
    empty = {
        "cagr": None, "vol": None, "sharpe": None, "max_dd": None, "calmar": None,
        "total": None, "worst_month": None, "p5_roll12": None, "ulcer": None,
        "final_100k": None, "yrs": None,
    }
    if len(r) < 60:
        return empty
    eq = (1 + r).cumprod()
    yrs = len(r) / 252.0
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    rfl = rf.loc[r.index]
    vol = float(r.std() * np.sqrt(252))
    sharpe = float(((r - rfl).mean() / r.std()) * np.sqrt(252)) if r.std() > 0 else 0.0
    dd = eq / eq.cummax() - 1
    mdd = float(dd.min())
    monthly = (1 + r).resample("ME").prod() - 1
    roll12 = (1 + r).rolling(252).apply(lambda x: np.prod(1 + x) - 1, raw=True)
    p5 = float(roll12.quantile(0.05)) if roll12.notna().sum() > 20 else None
    ulcer = float(np.sqrt((dd.clip(upper=0) ** 2).mean()))
    return dict(
        cagr=round(cagr, 4),
        vol=round(vol, 3),
        sharpe=round(sharpe, 2),
        max_dd=round(mdd, 3),
        calmar=round(cagr / abs(mdd), 2) if mdd < 0 else None,
        total=round(float(eq.iloc[-1] - 1), 2),
        worst_month=round(float(monthly.min()), 3),
        p5_roll12=round(p5, 3) if p5 is not None else None,
        ulcer=round(ulcer, 3),
        final_100k=round(float(100_000 * eq.iloc[-1]), 0),
        yrs=round(yrs, 1),
    )


# ---- build ----
W, R = raw_targets()

TRIGGERS = (-0.12, -0.15, -0.18, -0.20, -0.22, -0.25)
FLOORS = (0.70, 0.75, 0.80, 0.85, 0.90)

specs = [("R50d", None, 1.0)]
for trig in TRIGGERS:
    for floor in FLOORS:
        tlab = abs(int(round(trig * 100)))
        flab = int(round(floor * 100))
        specs.append((f"soft_dd{tlab}_{flab}", trig, floor))

built = {}
for name, trig, floor in specs:
    ret, turn, P, scales, pct = simulate_soft_brake(W, R, dd_trig=trig, scale_floor=floor)
    built[name] = dict(ret=ret, turn=turn, P=P, scales=scales, pct_braked=pct)

s_dbmf = al("DBMF").dropna().index.min()
s_live = max(pd.Timestamp("2019-06-01"), s_dbmf)
s2015 = "2015-01-01"
start3 = idx[-1] - pd.DateOffset(years=3)
WINDOWS = [
    ("live_dbmf", s_live, None),
    ("from_2015", s2015, None),
    ("last_3y", start3, None),
]
CRISES = [
    ("covid", "2020-02-15", "2020-04-30"),
    ("y2022", "2022-01-01", "2022-12-31"),
    ("tariff_2025", "2025-02-15", "2025-06-30"),
]

res = {
    "meta": {
        "book": "live R50d (TQQQ/TMF/UGL/DBC/DBMF/BTC, sigma*=50%, cash cap)",
        "brake": "causal single-pass position scale; release at new equity high",
        "live_start": str(pd.Timestamp(s_live).date()),
        "data_end": str(idx[-1].date()),
        "objective": "keep CAGR within ~1pp of R50d; improve DD/ulcer if possible",
    },
    "windows": {},
    "crises": {},
    "brake_stats": {},
    "yearly_live": {},
}

for name, b in built.items():
    res["windows"][name] = {w: metrics(b["ret"], a, c) for w, a, c in WINDOWS}
    yrs = n / 252.0
    res["brake_stats"][name] = {
        "pct_days_braked": round(b["pct_braked"], 4),
        "turnover_yr": round(float(b["turn"].sum() / yrs), 2),
        "avg_gross": round(float(b["P"].sum(axis=1).mean()), 3),
        "avg_scale": round(float(b["scales"].mean()), 4),
    }

for w, a, c in WINDOWS:
    res["windows"].setdefault("bench_SPY", {})[w] = metrics(r_spy, a, c)
    res["windows"].setdefault("bench_QQQ", {})[w] = metrics(r_qqq, a, c)

for lab, a, c in CRISES:
    res["crises"][lab] = {
        name: {
            "max_dd": metrics(b["ret"], a, c)["max_dd"],
            "cagr": metrics(b["ret"], a, c)["cagr"],
            "total": metrics(b["ret"], a, c)["total"],
        }
        for name, b in built.items()
    }
    res["crises"][lab]["bench_SPY"] = {
        "max_dd": metrics(r_spy, a, c)["max_dd"],
        "cagr": metrics(r_spy, a, c)["cagr"],
        "total": metrics(r_spy, a, c)["total"],
    }


def yearly(ret, start):
    y = (1 + ret.loc[start:].dropna()).resample("YE").prod() - 1
    return {str(k.year): round(float(v), 3) for k, v in y.items()}


# Yearlies for baseline + a few soft candidates
for name in ["R50d"] + [s[0] for s in specs[1:] if s[0].endswith(("_80", "_85", "_90"))]:
    if name in built:
        res["yearly_live"][name] = yearly(built[name]["ret"], s_live)

base = res["windows"]["R50d"]["live_dbmf"]
res["vs_R50d_live"] = {}
for name in built:
    m = res["windows"][name]["live_dbmf"]
    if m["cagr"] is None or base["cagr"] is None:
        continue
    res["vs_R50d_live"][name] = {
        "cagr_delta_pp": round((m["cagr"] - base["cagr"]) * 100, 2),
        "mdd_delta_pp": round((m["max_dd"] - base["max_dd"]) * 100, 2),
        "calmar_delta": round(m["calmar"] - base["calmar"], 2),
        "ulcer_delta": round(m["ulcer"] - base["ulcer"], 4),
        "final_delta_$": round(m["final_100k"] - base["final_100k"], 0),
        "worst_month_delta_pp": round((m["worst_month"] - base["worst_month"]) * 100, 2),
        "pct_braked": res["brake_stats"][name]["pct_days_braked"],
    }


def board(window, sort_key="cagr"):
    rows = []
    for name in built:
        m = res["windows"][name][window]
        if m["cagr"] is None:
            continue
        rows.append({
            "name": name,
            "cagr": m["cagr"],
            "max_dd": m["max_dd"],
            "calmar": m["calmar"],
            "sharpe": m["sharpe"],
            "ulcer": m["ulcer"],
            "worst_month": m["worst_month"],
            "final_100k": m["final_100k"],
            "pct_braked": res["brake_stats"][name]["pct_days_braked"],
            **res["vs_R50d_live"].get(name, {}),
        })
    rows.sort(key=lambda x: (x.get(sort_key) if x.get(sort_key) is not None else -999), reverse=True)
    return rows


res["scoreboard_cagr"] = board("live_dbmf", "cagr")
res["scoreboard_calmar"] = board("live_dbmf", "calmar")

# Keep-CAGR picks: within 1pp and within 2pp of R50d
def keep_cagr_picks(tol_pp):
    near = [
        r for r in res["scoreboard_cagr"]
        if r["cagr"] is not None and r["cagr"] >= base["cagr"] - tol_pp / 100.0
    ]
    by_dd = sorted(near, key=lambda x: x["max_dd"], reverse=True)
    by_ulcer = sorted(near, key=lambda x: x["ulcer"])
    by_cal = sorted(near, key=lambda x: x["calmar"], reverse=True)
    return {
        "tolerance_cagr_pp": tol_pp,
        "n_candidates": len(near),
        "best_max_dd": by_dd[0] if by_dd else None,
        "best_ulcer": by_ulcer[0] if by_ulcer else None,
        "best_calmar": by_cal[0] if by_cal else None,
        "candidates": near,
    }


res["keep_cagr_1pp"] = keep_cagr_picks(1.0)
res["keep_cagr_2pp"] = keep_cagr_picks(2.0)

# Pareto: strictly better CAGR and better (less negative) maxDD than R50d
pareto = []
for name, d in res["vs_R50d_live"].items():
    if name == "R50d":
        continue
    if d["cagr_delta_pp"] > 0 and d["mdd_delta_pp"] > 0:
        pareto.append({"name": name, **d})
res["pareto_vs_R50d"] = pareto

# Recommended soft configs: high floor, modest trigger, small CAGR tax
soft_ranked = [
    r for r in res["scoreboard_cagr"]
    if r["name"] != "R50d" and r["cagr"] is not None
]
# score: penalize CAGR loss heavily, reward DD improvement
for r in soft_ranked:
    r["score"] = round(
        -3.0 * abs(min(0.0, r.get("cagr_delta_pp", 0.0)))  # punish CAGR loss
        + 1.0 * max(0.0, r.get("mdd_delta_pp", 0.0))       # reward DD improvement
        - 50.0 * max(0.0, r.get("ulcer_delta", 0.0) or 0.0)
        + 10.0 * max(0.0, r.get("calmar_delta", 0.0) or 0.0),
        3,
    )
soft_ranked.sort(key=lambda x: x["score"], reverse=True)
res["soft_ranked"] = soft_ranked[:15]

# Verdict
best = soft_ranked[0] if soft_ranked else None
base_cagr = base["cagr"]
verdict = {
    "baseline_cagr": base_cagr,
    "baseline_max_dd": base["max_dd"],
    "baseline_calmar": base["calmar"],
    "pareto_count": len(pareto),
    "top_soft": best,
}
if best is None:
    verdict["recommendation"] = "no soft brake beat the scoring rule; keep raw R50d"
elif best["cagr_delta_pp"] >= -1.0 and best["mdd_delta_pp"] > 0.5:
    verdict["recommendation"] = (
        f"optional soft brake {best['name']}: keeps CAGR within 1pp and improves DD; "
        "not required — raw R50d remains the growth default"
    )
elif best["cagr_delta_pp"] >= -1.0 and best["mdd_delta_pp"] <= 0.5:
    verdict["recommendation"] = (
        f"soft brakes within 1pp CAGR barely move DD ({best['name']}); "
        "do not change live — insurance value is mostly out-of-sample"
    )
else:
    verdict["recommendation"] = (
        f"best soft score {best['name']} costs >1pp CAGR; "
        "keep live R50d if you want to preserve ~27% growth"
    )
res["verdict"] = verdict

with open(OUT, "w") as f:
    json.dump(res, f, indent=1)

print("=== META ===")
print(json.dumps(res["meta"], indent=1))
print(f"\nBaseline R50d live: CAGR {base['cagr']*100:.2f}%  DD {base['max_dd']*100:.1f}%  "
      f"Calmar {base['calmar']}  ulcer {base['ulcer']}  $100k -> ${base['final_100k']:,.0f}")

print("\n=== TOP 12 by CAGR (live DBMF) ===")
for r in res["scoreboard_cagr"][:12]:
    print(
        f"{r['name']:<18} CAGR {r['cagr']*100:5.2f}% ({r.get('cagr_delta_pp', 0):+5.2f}pp)  "
        f"DD {r['max_dd']*100:6.1f}% ({r.get('mdd_delta_pp', 0):+5.1f}pp)  "
        f"Calmar {r['calmar']:4.2f}  ulcer {r['ulcer']:.3f}  "
        f"braked {100*(r['pct_braked'] or 0):4.1f}%"
    )

print("\n=== Keep-CAGR within 1pp (best DD / ulcer / calmar) ===")
print(json.dumps({
    k: res["keep_cagr_1pp"][k]
    for k in ("n_candidates", "best_max_dd", "best_ulcer", "best_calmar")
}, indent=1))

print("\n=== Soft ranked (CAGR-preserving score) top 8 ===")
for r in res["soft_ranked"][:8]:
    print(
        f"{r['name']:<18} score {r['score']:+7.2f}  "
        f"CAGR {r.get('cagr_delta_pp', 0):+5.2f}pp  "
        f"DD {r.get('mdd_delta_pp', 0):+5.1f}pp  "
        f"ulcer {r.get('ulcer_delta', 0):+.4f}  braked {100*(r['pct_braked'] or 0):4.1f}%"
    )

print("\n=== Pareto vs R50d (higher CAGR AND better DD) ===")
print(res["pareto_vs_R50d"] or "none")

print("\n=== Crisis max DD (R50d vs top soft) ===")
focus = ["R50d"] + [r["name"] for r in res["soft_ranked"][:3]]
for lab, d in res["crises"].items():
    print(lab, {k: d[k]["max_dd"] for k in focus if k in d})

print("\n=== VERDICT ===")
print(json.dumps(res["verdict"], indent=1))
print("\nwrote", OUT)
