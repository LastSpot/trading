"""R50d_cash: live R50d core + idle-cash premia overlay (MR dip-buy + VRP).

Paper A/B candidate from research/v10_orthogonal (v10c `R50d_cash`,
DESIGN_C.md). The six-sleeve R50d core is unchanged; two overlay slots are
funded strictly from headroom = 1 - sum(core targets), MR first:

  - MR (QQQ): enter when Wilder RSI(2) < 10 while QQQ > SMA200 and the VIX
    curve is not in backwardation (VIX3M/VIX > 1.00); exit at RSI(2) > 60 or
    after 10 sessions. Weight min(0.30, headroom).
  - VRP (SVXY): on when VIX3M/VIX >= 1.05, off when <= 1.00 (hysteresis);
    sized 0.15 x 0.50 / EWMA vol, capped at 0.225 and at remaining headroom.

Ops notes:
  - VIX/VIX3M come from CBOE's free daily CSVs (Alpaca has no index data and
    Yahoo's ^VIX3M lags). A stale feed beyond 5 sessions aborts the run.
  - Overlay state (MR in/out + sessions held, VRP hysteresis) is rebuilt each
    run by walking the fetched history, like r50d_asym.
  - The engine's MAX_MOVE data-sanity rail also halts the run on an extreme
    SVXY day; the failure email then calls for human review before trading.
"""
from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

NAME = "r50d_cash"
DESCRIPTION = (
    "R50d core + idle-cash overlay: QQQ dip-buy (RSI2, backwardation veto) "
    "and SVXY carry (VIX3M/VIX hysteresis), never displacing the core"
)


@dataclass(frozen=True)
class Sleeve:
    signal: str      # ticker whose closes drive the gate and vol estimate
    budget: float    # share of the portfolio risk budget
    leverage: float  # leverage embedded in the traded product
    crypto: bool = False


CORE_SLEEVES: dict[str, Sleeve] = {
    "TQQQ": Sleeve("QQQ", 0.40, 3.0),
    "TMF": Sleeve("TLT", 0.15, 3.0),
    "UGL": Sleeve("GLD", 0.15, 2.0),
    "DBC": Sleeve("DBC", 0.10, 1.0),
    "DBMF": Sleeve("DBMF", 0.10, 1.0),
    "BTC/USD": Sleeve("BTC/USD", 0.10, 1.0, crypto=True),
}
# Overlay sleeves are listed so the engine fetches, reconciles, and trades them.
SLEEVES: dict[str, Sleeve] = {
    **CORE_SLEEVES,
    "QQQ": Sleeve("QQQ", 0.30, 1.0),
    "SVXY": Sleeve("SVXY", 0.15, 1.0),
}
SIGMA_TARGET = 0.50
LOOKBACKS = (63, 126, 252)
EWMA_ALPHA = 0.06
TRADE_BAND = 0.05
HISTORY_DAYS = 600
MAX_MOVE = 0.25

MR_CAP = 0.30
MR_ENTRY = 10.0
MR_EXIT = 60.0
MR_MAX_HOLD = 10
VRP_BUDGET = 0.15
VRP_SIZE_CAP = 1.5 * VRP_BUDGET
VRP_ENTER = 1.05
VRP_EXIT = 1.00

CBOE_URLS = {
    "VIX": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
    "VIX3M": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv",
}
MAX_RATIO_STALE_SESSIONS = 5


def ensemble_gate(px: pd.Series) -> float:
    return float(np.mean([px.iloc[-1] > px.iloc[-1 - L] for L in LOOKBACKS]))


def ewma_vol(px: pd.Series) -> float:
    r = px.pct_change().dropna()
    var = (r**2).ewm(alpha=EWMA_ALPHA, adjust=False).mean().iloc[-1]
    return float(np.sqrt(252.0 * var))


def _fetch_cboe_closes(url: str) -> pd.Series:
    with urllib.request.urlopen(url, timeout=30) as resp:
        raw = resp.read()
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip().upper() for c in df.columns]
    s = pd.Series(
        df["CLOSE"].to_numpy(dtype=float),
        index=[d.date() for d in pd.to_datetime(df["DATE"])],
    )
    return s[~s.index.duplicated(keep="last")].sort_index()


def fetch_vix_ratio(index: pd.Index) -> pd.Series:
    """VIX3M/VIX aligned to the closes index (ffilled, staleness-guarded)."""
    vix = _fetch_cboe_closes(CBOE_URLS["VIX"])
    vix3m = _fetch_cboe_closes(CBOE_URLS["VIX3M"])
    last_cboe: date = min(vix.index[-1], vix3m.index[-1])
    stale_sessions = int(sum(1 for d in index if d > last_cboe))
    if stale_sessions > MAX_RATIO_STALE_SESSIONS:
        raise RuntimeError(
            f"CBOE VIX/VIX3M stale: last {last_cboe}, "
            f"{stale_sessions} sessions behind Alpaca closes"
        )
    ratio = (vix3m / vix).reindex(index).ffill()
    if not np.isfinite(ratio.iloc[-1]):
        raise RuntimeError("VIX3M/VIX ratio is not finite after alignment")
    return ratio


def rsi2(px: pd.Series) -> pd.Series:
    """Wilder RSI, period 2 (matches research/v10_orthogonal)."""
    d = px.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    au = up.ewm(alpha=0.5, adjust=False).mean()
    ad = dn.ewm(alpha=0.5, adjust=False).mean()
    rsi = 100.0 - 100.0 / (1.0 + au / ad.where(ad > 0))
    return rsi.where(ad > 0, np.where(au > 0, 100.0, 50.0))


def mr_state(qqq: pd.Series, ratio: pd.Series) -> tuple[bool, dict]:
    """Walk history so the live MR state matches the v10c backtest machine."""
    r = rsi2(qqq).to_numpy()
    sma = qqq.rolling(200).mean().to_numpy()
    p = qqq.to_numpy()
    veto = (ratio.reindex(qqq.index).to_numpy() <= VRP_EXIT + 1e-12)
    in_pos = False
    held = 0
    for i in range(len(qqq)):
        if in_pos:
            held += 1
            if (np.isfinite(r[i]) and r[i] > MR_EXIT) or held >= MR_MAX_HOLD:
                in_pos = False
                held = 0
        else:
            if (
                np.isfinite(r[i])
                and r[i] < MR_ENTRY
                and np.isfinite(sma[i])
                and p[i] > sma[i]
                and not veto[i]
            ):
                in_pos = True
                held = 0
    return in_pos, {
        "rsi2": float(r[-1]) if np.isfinite(r[-1]) else None,
        "above_sma200": bool(np.isfinite(sma[-1]) and p[-1] > sma[-1]),
        "vetoed": bool(veto[-1]),
        "sessions_held": held if in_pos else 0,
    }


def vrp_hysteresis(ratio: pd.Series) -> bool:
    on = False
    for x in ratio.to_numpy():
        if not np.isfinite(x):
            continue
        if on and x <= VRP_EXIT + 1e-12:
            on = False
        elif not on and x >= VRP_ENTER - 1e-12:
            on = True
    return on


def target_weights(closes: pd.DataFrame) -> tuple[dict[str, float], dict[str, dict]]:
    raw, diag = {}, {}
    for sym, s in CORE_SLEEVES.items():
        px = closes[s.signal].dropna()
        gate, vol = ensemble_gate(px), ewma_vol(px)
        raw[sym] = gate * s.budget * SIGMA_TARGET / (s.leverage * vol)
        diag[sym] = {"gate": gate, "vol": vol}
    total = sum(raw.values())
    scale = 1.0 / total if total > 1.0 else 1.0
    targets = {k: v * scale for k, v in raw.items()}
    headroom = max(0.0, 1.0 - sum(targets.values()))

    qqq = closes["QQQ"].dropna()
    ratio = fetch_vix_ratio(closes.index)
    mr_on, mr_diag = mr_state(qqq, ratio)
    w_mr = min(MR_CAP, headroom) if mr_on else 0.0

    svxy = closes["SVXY"].dropna()
    svxy_vol = ewma_vol(svxy)
    vrp_on = vrp_hysteresis(ratio)
    w_size = min(VRP_BUDGET * SIGMA_TARGET / svxy_vol, VRP_SIZE_CAP)
    w_vrp = min(w_size, headroom - w_mr) if vrp_on else 0.0

    targets["QQQ"] = w_mr
    targets["SVXY"] = w_vrp
    diag["QQQ"] = {
        "gate": 1.0 if mr_on else 0.0,
        "vol": ewma_vol(qqq),
        **mr_diag,
    }
    diag["SVXY"] = {
        "gate": 1.0 if vrp_on else 0.0,
        "vol": svxy_vol,
        "ts_ratio": float(ratio.iloc[-1]),
        "headroom": headroom,
    }
    return targets, diag
