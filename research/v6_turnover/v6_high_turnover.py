"""
Round 6: high-turnover paper-book search (independent of live R50d).

Goal
----
Find a *separate* retail-deployable strategy with meaningfully higher turnover
than the multi-month R50d book, aiming for more return *if* an edge survives
costs, with survivable drawdowns for limited capital.

Context (not the thing to improve)
----------------------------------
Live R50d = six-sleeve cash book, 63/126/252d ensemble gates, sigma*=50%,
daily post-close rebalance, ~low-moderate turnover. Included only as baseline.

Candidates (all long-only, cash-account friendly, liquid ETFs)
-------------------------------------------------------------
1) FastTrend_W  — weekly rebalance; ensemble gates 10/21/42d; multi-sleeve
                  vol-target; 1x underlyings (less levered-ETF decay under churn)
2) FastTrend_D  — same signals, daily rebalance (higher turnover stress test)
3) TSMOM_W      — classic time-series momentum: sign(return over L) * vol target
                  on a diversified sleeve set; weekly
4) DualMom_W    — absolute + relative momentum rotation into top-K sleeves weekly
5) Breakout_W   — Donchian/channel breakout (20/55) with ATR vol sizing; weekly
6) MR_SPY_D     — short-horizon mean-reversion on SPY (2-5d z-score); daily
                  (known retail style; often dies to costs — included as check)
7) Hybrid_W     — FastTrend core + DualMom satellite (80/20 risk budgets)

Risk stack (applied as overlays where noted)
--------------------------------------------
- Vol targeting (sigma* grid)
- Soft DD brake (halve risk after -15% peak DD until new high)
- Chop filter on FastTrend (skip weak |ret|/vol regimes)
- Costs: 5 bps per unit turnover; idle cash at T-bills
- Execution: T+1 (signal close t -> earn return t+1)

Ranking
-------
Primary: Calmar, then CAGR, subject to maxDD not much worse than ~-30% on the
common live window. Prefer annual turnover >> R50d baseline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "output" / "v6_high_turnover_results.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
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

# Sleeve returns (1x; BTC haircut for frictions)
R = {
    "QQQ": qqq.pct_change().fillna(0.0),
    "SPY": spy.pct_change().fillna(0.0),
    "IWM": iwm.pct_change().fillna(0.0),
    "EFA": efa.pct_change().fillna(0.0),
    "EEM": eem.pct_change().fillna(0.0),
    "TLT": tlt.pct_change().fillna(0.0),
    "IEF": ief.pct_change().fillna(0.0),
    "GLD": gld.pct_change().fillna(0.0),
    "DBC": dbc.pct_change().fillna(0.0),
    "DBMF": dbmf.pct_change().fillna(0.0),
    "BTC": (btc.pct_change() - 0.01 / 252.0).fillna(0.0),
    "TQQQ": tqqq.pct_change().fillna(0.0),
}
PX = {
    "QQQ": qqq, "SPY": spy, "IWM": iwm, "EFA": efa, "EEM": eem,
    "TLT": tlt, "IEF": ief, "GLD": gld, "DBC": dbc, "DBMF": dbmf,
    "BTC": btc, "TQQQ": tqqq,
}
VALID = {k: (~PX[k].isna() & (PX[k] > 0)).to_numpy() for k in PX}

COST = 0.0005  # 5 bps per unit turnover


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ewma_vol(px: pd.Series, alpha: float = 0.12) -> np.ndarray:
    """Faster than R50d's 0.06 — better match to weekly/short horizons."""
    r = px.pct_change().fillna(0.0)
    return np.sqrt(252.0 * (r**2).ewm(alpha=alpha, adjust=False).mean()).to_numpy()


def gate_ens(px: pd.Series, lookbacks: tuple[int, ...]) -> np.ndarray:
    votes = [
        (px / px.shift(L) - 1 > 0).astype(float).where(~px.isna(), 0.0)
        for L in lookbacks
    ]
    return pd.concat(votes, axis=1).mean(axis=1).fillna(0.0).to_numpy()


def tsmom_sign(px: pd.Series, L: int) -> np.ndarray:
    ret = px / px.shift(L) - 1
    return np.sign(ret.fillna(0.0)).clip(lower=0.0).to_numpy()  # long-only


def mom_score(px: pd.Series, L: int) -> pd.Series:
    return (px / px.shift(L) - 1).fillna(-np.inf)


def donchian_breakout(px: pd.Series, L: int) -> np.ndarray:
    """1 when close makes L-day high (exclude today from window)."""
    prior_high = px.shift(1).rolling(L).max()
    return (px >= prior_high).astype(float).fillna(0.0).to_numpy()


def chop_mult(px: pd.Series, L: int = 21, thresh: float = 0.5) -> np.ndarray:
    rL = px / px.shift(L) - 1
    volL = px.pct_change().rolling(L).std() * np.sqrt(252)
    strength = (rL.abs() / volL.replace(0, np.nan)).fillna(0.0)
    return (strength >= thresh).astype(float).to_numpy()


def zscore_mr(px: pd.Series, L: int = 5) -> np.ndarray:
    """Mean-reversion score in [0,1]: buy when short-term z is low."""
    r = px.pct_change()
    mu = r.rolling(L).mean()
    sd = r.rolling(L).std()
    z = ((r - mu) / sd.replace(0, np.nan)).fillna(0.0)
    # map z: more negative -> higher long weight; clip
    score = (-z).clip(lower=0.0, upper=2.0) / 2.0
    return score.to_numpy()


def band_positions(W: np.ndarray, band: float = 0.05) -> np.ndarray:
    P = np.zeros_like(W)
    held = np.zeros(W.shape[1])
    for i in range(len(W)):
        for j in range(W.shape[1]):
            if abs(W[i, j] - held[j]) > band or (W[i, j] == 0.0 and held[j] != 0.0):
                held[j] = W[i, j]
            P[i, j] = held[j]
    return P


def weekly_hold(P: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    """Freeze positions between Fridays (use last Friday's target through week)."""
    is_fri = dates.weekday == 4
    out = np.zeros_like(P)
    held = np.zeros(P.shape[1])
    for i in range(len(P)):
        if is_fri[i] or i == 0:
            held = P[i].copy()
        out[i] = held
    return out


def pnl(P: np.ndarray, rets: np.ndarray) -> tuple[pd.Series, pd.Series]:
    """T+1: yesterday's position earns today's return; cash earns rf."""
    Pl = np.vstack([np.zeros((1, P.shape[1])), P[:-1]])
    turn = np.abs(np.diff(np.vstack([np.zeros((1, P.shape[1])), Pl]), axis=0)).sum(axis=1)
    invested = Pl.sum(axis=1)
    ret = (Pl * rets).sum(axis=1) + (1.0 - invested) * rf.to_numpy() - COST * turn
    return pd.Series(ret, index=idx), pd.Series(turn, index=idx)


def apply_dd_brake(ret: pd.Series, scale_floor: float = 0.5, dd_trig: float = -0.15):
    eq = (1 + ret).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    scale = pd.Series(1.0, index=ret.index)
    braked = False
    for i, _dt in enumerate(ret.index):
        if dd.iloc[i] <= dd_trig:
            braked = True
        if braked and abs(dd.iloc[i]) < 1e-12:
            braked = False
        scale.iloc[i] = scale_floor if braked else 1.0
    return rf + scale * (ret - rf), scale


def metrics(ret: pd.Series, turn: pd.Series, start=None, end=None) -> dict:
    r = ret.loc[start:end].dropna()
    t = turn.reindex(r.index).fillna(0.0)
    if len(r) < 60:
        return dict(cagr=None, vol=None, sharpe=None, max_dd=None, calmar=None,
                    total=None, turnover_yr=None, worst_month=None)
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


def cap_row(W: np.ndarray, cap: float = 1.0) -> np.ndarray:
    tot = W.sum(axis=1)
    scale = np.where(tot > cap, cap / np.where(tot > 0, tot, 1.0), 1.0)
    return W * scale[:, None]


# ---------------------------------------------------------------------------
# Strategy builders
# ---------------------------------------------------------------------------
# Diversified 1x book used by several candidates
CORE = ["QQQ", "TLT", "GLD", "DBC", "DBMF", "BTC"]
CORE_BUDGET = {"QQQ": 0.35, "TLT": 0.20, "GLD": 0.15, "DBC": 0.10, "DBMF": 0.10, "BTC": 0.10}

# Broader rotation universe for DualMom
ROT = ["QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "DBMF", "BTC"]

VOL = {k: ewma_vol(PX[k], alpha=0.12) for k in set(CORE + ROT + ["SPY", "TQQQ"])}


def build_fast_trend(
    lookbacks=(10, 21, 42),
    sigp=0.35,
    band=0.08,
    weekly=True,
    use_chop=True,
    budgets=None,
    sleeves=None,
):
    sleeves = sleeves or CORE
    budgets = budgets or CORE_BUDGET
    cols = []
    rets = []
    for sym in sleeves:
        g = gate_ens(PX[sym], lookbacks)
        if use_chop:
            g = g * chop_mult(PX[sym], L=lookbacks[1], thresh=0.45)
        w = g * budgets[sym] * sigp / np.where(VOL[sym] > 1e-8, VOL[sym], np.nan)
        w = np.nan_to_num(np.where(VALID[sym], w, 0.0), nan=0.0)
        cols.append(w)
        rets.append(R[sym].to_numpy())
    W = cap_row(np.column_stack(cols))
    P = band_positions(W, band=band)
    if weekly:
        P = weekly_hold(P, idx)
    return pnl(P, np.column_stack(rets)) + (P,)


def build_tsmom(L=21, sigp=0.35, band=0.08, weekly=True):
    cols, rets = [], []
    for sym in CORE:
        g = tsmom_sign(PX[sym], L)
        w = g * CORE_BUDGET[sym] * sigp / np.where(VOL[sym] > 1e-8, VOL[sym], np.nan)
        w = np.nan_to_num(np.where(VALID[sym], w, 0.0), nan=0.0)
        cols.append(w)
        rets.append(R[sym].to_numpy())
    W = cap_row(np.column_stack(cols))
    P = band_positions(W, band=band)
    if weekly:
        P = weekly_hold(P, idx)
    return pnl(P, np.column_stack(rets)) + (P,)


def build_dual_mom(L=21, top_k=3, sigp=0.40, band=0.10, weekly=True, abs_gate=True):
    """Equal-risk among top-K absolute+relative winners; cash if none pass abs gate."""
    scores = {sym: mom_score(PX[sym], L) for sym in ROT}
    score_df = pd.DataFrame(scores).reindex(idx)
    # absolute momentum: require positive L-return
    abs_ok = score_df > 0 if abs_gate else pd.DataFrame(True, index=idx, columns=ROT)

    W = np.zeros((n, len(ROT)))
    for i, dt in enumerate(idx):
        row = score_df.loc[dt]
        ok = abs_ok.loc[dt]
        # only valid sleeves
        eligible = [
            s for s in ROT
            if ok[s] and VALID[s][i] and np.isfinite(row[s]) and row[s] > -1e8
        ]
        if not eligible:
            continue
        ranked = sorted(eligible, key=lambda s: row[s], reverse=True)[:top_k]
        # equal capital among winners, then vol-scale toward sigp
        raw = np.zeros(len(ROT))
        for s in ranked:
            j = ROT.index(s)
            raw[j] = 1.0 / len(ranked)
        # vol scale portfolio: scale so dollar-vol ~ sigp
        port_vol = 0.0
        for s in ranked:
            j = ROT.index(s)
            port_vol += (raw[j] * VOL[s][i]) ** 2
        port_vol = float(np.sqrt(port_vol)) if port_vol > 0 else np.nan
        scale = sigp / port_vol if port_vol and port_vol > 1e-8 else 0.0
        scale = min(scale, 1.0)  # cash-account notional cap
        W[i] = raw * scale

    P = band_positions(W, band=band)
    if weekly:
        P = weekly_hold(P, idx)
    rets = np.column_stack([R[s].to_numpy() for s in ROT])
    return pnl(P, rets) + (P,)


def build_breakout(L_fast=20, L_slow=55, sigp=0.35, band=0.10, weekly=True):
    cols, rets = [], []
    for sym in CORE:
        # require both channels (stronger) OR fast only — use OR of fast/slow highs
        g = np.maximum(donchian_breakout(PX[sym], L_fast), donchian_breakout(PX[sym], L_slow) * 0.5)
        # exit-ish: if below 20d high for a while gate fades via g already 0/1
        w = g * CORE_BUDGET[sym] * sigp / np.where(VOL[sym] > 1e-8, VOL[sym], np.nan)
        w = np.nan_to_num(np.where(VALID[sym], w, 0.0), nan=0.0)
        cols.append(w)
        rets.append(R[sym].to_numpy())
    W = cap_row(np.column_stack(cols))
    P = band_positions(W, band=band)
    if weekly:
        P = weekly_hold(P, idx)
    return pnl(P, np.column_stack(rets)) + (P,)


def build_mr_spy(L=5, sigp=0.25, band=0.05, vol_kill=0.30):
    """Daily mean-reversion on SPY; flat in high vol."""
    score = zscore_mr(spy, L=L)
    vol = VOL["SPY"]
    w = score * sigp / np.where(vol > 1e-8, vol, np.nan)
    w = np.nan_to_num(w, nan=0.0)
    w = np.where(vol > vol_kill, 0.0, w)
    w = np.clip(w, 0.0, 1.0)
    W = w.reshape(-1, 1)
    P = band_positions(W, band=band)
    return pnl(P, R["SPY"].to_numpy().reshape(-1, 1)) + (P,)


def build_r50d_baseline():
    """Approximate live R50d on 1x/levered mix for turnover/return context only."""
    # Use same structure as live: TQQQ/TLT/GLD/DBC/DBMF/BTC with long gates
    sleeves = [
        ("QQQ", "TQQQ", 0.40, 3.0),
        ("TLT", "TLT", 0.15, 3.0),  # approx with 1x TLT * 3 via sizing
        ("GLD", "GLD", 0.15, 2.0),
        ("DBC", "DBC", 0.10, 1.0),
        ("DBMF", "DBMF", 0.10, 1.0),
        ("BTC", "BTC", 0.10, 1.0),
    ]
    # For baseline fidelity use synthetic levered where needed
    lookbacks = (63, 126, 252)
    sigp = 0.50
    alpha = 0.06
    cols, rets = [], []
    for sig, traded, budget, lev in sleeves:
        g = gate_ens(PX[sig], lookbacks)
        vol = ewma_vol(PX[sig], alpha=alpha)
        w = g * budget * sigp / (lev * np.where(vol > 1e-8, vol, np.nan))
        w = np.nan_to_num(np.where(VALID[sig], w, 0.0), nan=0.0)
        cols.append(w)
        if traded == "TQQQ":
            # prefer real TQQQ when available else 3x QQQ approx
            r = R["TQQQ"].where(VALID["TQQQ"], 3.0 * R["QQQ"]).fillna(0.0)
            # crude financing drag already partly in TQQQ; ok for baseline
            rets.append(r.to_numpy())
        elif traded == "TLT" and lev == 3.0:
            # approx 3x TLT return minus rough drag
            rets.append((3.0 * R["TLT"] - 2.0 * (rf + 0.005 / 252)).to_numpy())
        elif traded == "GLD" and lev == 2.0:
            rets.append((2.0 * R["GLD"] - 1.0 * (rf + 0.005 / 252)).to_numpy())
        else:
            rets.append(R[traded].to_numpy())
    W = cap_row(np.column_stack(cols))
    P = band_positions(W, band=0.05)
    return pnl(P, np.column_stack(rets)) + (P,)


def build_hybrid(sigp_core=0.30, sigp_sat=0.40):
    """80% FastTrend weekly + 20% DualMom weekly (capital mix after each is built)."""
    r1, t1, P1 = build_fast_trend(sigp=sigp_core, weekly=True, use_chop=True)
    r2, t2, P2 = build_dual_mom(sigp=sigp_sat, weekly=True)
    # Combine by capital: 0.8 * book1 + 0.2 * book2 (positions live in different spaces)
    # Approximate via return blend + turnover blend (good enough for screening)
    ret = 0.8 * r1 + 0.2 * r2
    turn = 0.8 * t1 + 0.2 * t2
    return ret, turn, None


# ---------------------------------------------------------------------------
# Run grid
# ---------------------------------------------------------------------------
print("Building strategies...")

variants: dict[str, tuple] = {}

# Baseline
variants["R50d_baseline"] = build_r50d_baseline()

# Fast trend grid
for lb_name, lbs in {
    "ft_5_10_21": (5, 10, 21),
    "ft_10_21_42": (10, 21, 42),
    "ft_21_42_63": (21, 42, 63),
}.items():
    for sigp in (0.25, 0.35, 0.45):
        for weekly, tag in ((True, "W"), (False, "D")):
            name = f"FastTrend_{lb_name}_{tag}_s{int(sigp*100)}"
            variants[name] = build_fast_trend(
                lookbacks=lbs, sigp=sigp, weekly=weekly, use_chop=True, band=0.08
            )

# Fast trend without chop (ablation)
variants["FastTrend_ft_10_21_42_W_s35_nochop"] = build_fast_trend(
    lookbacks=(10, 21, 42), sigp=0.35, weekly=True, use_chop=False, band=0.08
)

# TSMOM
for L in (10, 21, 42, 63):
    for sigp in (0.30, 0.40):
        variants[f"TSMOM_L{L}_W_s{int(sigp*100)}"] = build_tsmom(L=L, sigp=sigp, weekly=True)

# Dual momentum
for L in (21, 42, 63):
    for k in (2, 3, 4):
        variants[f"DualMom_L{L}_k{k}_W"] = build_dual_mom(L=L, top_k=k, sigp=0.40, weekly=True)

# Breakout
for Lf, Ls in ((20, 55), (10, 20), (55, 100)):
    variants[f"Breakout_{Lf}_{Ls}_W"] = build_breakout(L_fast=Lf, L_slow=Ls, sigp=0.35, weekly=True)

# Mean reversion
for L in (3, 5, 10):
    variants[f"MR_SPY_L{L}_D"] = build_mr_spy(L=L, sigp=0.25)

# Hybrid
variants["Hybrid_FT_DM_W"] = build_hybrid()

# Levered fast-trend stress (TQQQ-heavy) — expect high return *and* high DD
def build_fast_tqqq(sigp=0.30, weekly=True):
    g = gate_ens(qqq, (10, 21, 42)) * chop_mult(qqq, L=21, thresh=0.45)
    vol = ewma_vol(qqq, alpha=0.12)
    w = g * sigp / (3.0 * np.where(vol > 1e-8, vol, np.nan))
    w = np.nan_to_num(np.clip(w, 0.0, 1.0), nan=0.0)
    P = band_positions(w.reshape(-1, 1), band=0.08)
    if weekly:
        P = weekly_hold(P, idx)
    r = R["TQQQ"].where(VALID["TQQQ"], 3.0 * R["QQQ"]).fillna(0.0)
    return pnl(P, r.to_numpy().reshape(-1, 1)) + (P,)


variants["FastTQQQ_W_s30"] = build_fast_tqqq(0.30, weekly=True)
variants["FastTQQQ_D_s30"] = build_fast_tqqq(0.30, weekly=False)

# Windows
# DBMF live-ish from mid-2019; use 2015+ as main retail window, plus last 3y
windows = {
    "from_2015": ("2015-01-01", None),
    "from_2019": ("2019-06-01", None),  # DBMF live
    "last_3y": (str((idx[-1] - pd.DateOffset(years=3)).date()), None),
    "full": (None, None),
}

res = {"windows": {}, "with_dd_brake": {}, "ranking": {}, "notes": {}}

print(f"Evaluating {len(variants)} variants...")

for wname, (start, end) in windows.items():
    res["windows"][wname] = {}
    for name, pack in variants.items():
        ret, turn, _P = pack
        res["windows"][wname][name] = metrics(ret, turn, start=start, end=end)

# DD brake on promising weekly candidates (recompute metrics on braked returns)
brake_names = [
    n for n in variants
    if any(k in n for k in ("FastTrend_ft_10_21_42_W", "DualMom_L21", "TSMOM_L21_W",
                            "Hybrid", "Breakout_20_55", "FastTQQQ_W"))
]
res["with_dd_brake"]["from_2015"] = {}
for name in brake_names:
    ret, turn, _ = variants[name]
    braked, _ = apply_dd_brake(ret, scale_floor=0.5, dd_trig=-0.15)
    # turnover approx unchanged (brake scales risk via returns approx)
    res["with_dd_brake"]["from_2015"][name] = metrics(braked, turn, start="2015-01-01")

# Ranking on from_2015: require turnover_yr >= 4, max_dd > -0.35, prefer calmar then cagr
base = res["windows"]["from_2015"]
ranked = []
for name, m in base.items():
    if m["cagr"] is None or m["turnover_yr"] is None:
        continue
    if name == "R50d_baseline":
        continue
    if m["turnover_yr"] < 3.0:  # must be higher-turnover than slow trend
        continue
    if m["max_dd"] is not None and m["max_dd"] < -0.35:
        continue
    ranked.append((name, m))

ranked.sort(key=lambda kv: (
    kv[1]["calmar"] if kv[1]["calmar"] is not None else -999,
    kv[1]["cagr"] if kv[1]["cagr"] is not None else -999,
), reverse=True)

res["ranking"]["from_2015_high_turn_dd35"] = [
    {"name": n, **m} for n, m in ranked[:15]
]

# Also rank by CAGR among those with sharpe>=0.7 and dd>-30%
ranked2 = [
    (n, m) for n, m in base.items()
    if n != "R50d_baseline"
    and m["cagr"] is not None
    and m["turnover_yr"] is not None
    and m["turnover_yr"] >= 3.0
    and m["sharpe"] is not None and m["sharpe"] >= 0.7
    and m["max_dd"] is not None and m["max_dd"] >= -0.30
]
ranked2.sort(key=lambda kv: kv[1]["cagr"], reverse=True)
res["ranking"]["from_2015_cagr_sharpe07_dd30"] = [
    {"name": n, **m} for n, m in ranked2[:15]
]

# Baseline snapshot
res["baseline_R50d"] = {
    w: base_m
    for w, base_m in (
        (wn, res["windows"][wn].get("R50d_baseline")) for wn in windows
    )
}

res["notes"] = {
    "objective": "separate high-turnover paper strategy for retail limited capital",
    "cost_bps": 5,
    "execution": "T+1",
    "notional_cap": "100% cash-account style",
    "recommendation_rule": (
        "Prefer weekly FastTrend or DualMom with DD brake if Calmar competitive "
        "and turnover clearly above R50d; reject MR if costs kill edge."
    ),
}

OUT.write_text(json.dumps(res, indent=2))
print(f"Wrote {OUT}")

# Console summary
print("\n=== R50d baseline (context) ===")
for w in ("from_2015", "from_2019", "last_3y"):
    m = res["windows"][w]["R50d_baseline"]
    print(
        f"  {w}: CAGR {m['cagr']:.1%}  Sharpe {m['sharpe']}  "
        f"DD {m['max_dd']:.1%}  turn/yr {m['turnover_yr']}  Calmar {m['calmar']}"
    )

print("\n=== Top by Calmar (2015+, turn/yr>=3, DD>=-35%) ===")
for row in res["ranking"]["from_2015_high_turn_dd35"][:10]:
    print(
        f"  {row['name']:<40} CAGR {row['cagr']:.1%}  Sh {row['sharpe']}  "
        f"DD {row['max_dd']:.1%}  turn {row['turnover_yr']}  Calmar {row['calmar']}"
    )

print("\n=== Top by CAGR (2015+, Sh>=0.7, DD>=-30%, turn>=3) ===")
for row in res["ranking"]["from_2015_cagr_sharpe07_dd30"][:10]:
    print(
        f"  {row['name']:<40} CAGR {row['cagr']:.1%}  Sh {row['sharpe']}  "
        f"DD {row['max_dd']:.1%}  turn {row['turnover_yr']}  Calmar {row['calmar']}"
    )

print("\n=== DD-brake overlay (2015+) on shortlist ===")
for name, m in sorted(
    res["with_dd_brake"]["from_2015"].items(),
    key=lambda kv: (kv[1]["calmar"] or -999),
    reverse=True,
)[:12]:
    print(
        f"  {name:<40} CAGR {m['cagr']:.1%}  Sh {m['sharpe']}  "
        f"DD {m['max_dd']:.1%}  turn {m['turnover_yr']}  Calmar {m['calmar']}"
    )
