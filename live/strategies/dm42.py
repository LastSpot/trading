"""DM42: weekly dual-momentum book (paper high-turnover satellite).

Research pick DM_L42_k4_s45_dd20_50 from research/v6_turnover:
  42d absolute + relative momentum, top-4 equal weight, sigma*=45%,
  peak-DD brake at -20% -> 50% scale until new equity high.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

NAME = "dm42"
DESCRIPTION = (
    "Weekly dual momentum (QQQ/IWM/EFA/EEM/TLT/IEF/GLD/DBC/DBMF/BTC), "
    "top-4, EWMA vol targeting at 45%, DD brake -20%/50%"
)

STATE_DIR = Path(__file__).resolve().parents[1] / "state"


@dataclass(frozen=True)
class Sleeve:
    signal: str      # ticker whose closes drive the signal and vol estimate
    budget: float    # unused for dual-mom (equal weight among winners); kept for parity
    leverage: float  # leverage embedded in the traded product
    crypto: bool = False


SLEEVES: dict[str, Sleeve] = {
    "QQQ": Sleeve("QQQ", 0.0, 1.0),
    "IWM": Sleeve("IWM", 0.0, 1.0),
    "EFA": Sleeve("EFA", 0.0, 1.0),
    "EEM": Sleeve("EEM", 0.0, 1.0),
    "TLT": Sleeve("TLT", 0.0, 1.0),
    "IEF": Sleeve("IEF", 0.0, 1.0),
    "GLD": Sleeve("GLD", 0.0, 1.0),
    "DBC": Sleeve("DBC", 0.0, 1.0),
    "DBMF": Sleeve("DBMF", 0.0, 1.0),
    "BTC/USD": Sleeve("BTC/USD", 0.0, 1.0, crypto=True),
}

LOOKBACKS = (42,)
MOM_LOOKBACK = 42
TOP_K = 4
SIGMA_TARGET = 0.45
EWMA_ALPHA = 0.12
TRADE_BAND = 0.10
HISTORY_DAYS = 600
MAX_MOVE = 0.25

DD_TRIG = -0.20
DD_FLOOR = 0.50


def ewma_vol(px: pd.Series) -> float:
    r = px.pct_change().dropna()
    if r.empty:
        return float("nan")
    var = (r**2).ewm(alpha=EWMA_ALPHA, adjust=False).mean().iloc[-1]
    return float(np.sqrt(252.0 * var))


def mom_score(px: pd.Series, lookback: int = MOM_LOOKBACK) -> float:
    if len(px) <= lookback:
        return float("nan")
    a, b = float(px.iloc[-1]), float(px.iloc[-1 - lookback])
    if b == 0.0 or np.isnan(a) or np.isnan(b):
        return float("nan")
    return a / b - 1.0


def target_weights(closes: pd.DataFrame) -> tuple[dict[str, float], dict[str, dict]]:
    scores: dict[str, float] = {}
    vols: dict[str, float] = {}
    diag: dict[str, dict] = {}

    for sym, s in SLEEVES.items():
        px = closes[s.signal].dropna()
        mom = mom_score(px)
        vol = ewma_vol(px)
        scores[sym] = mom
        vols[sym] = vol
        # "gate" is the momentum score (engine email table expects this key)
        diag[sym] = {"gate": mom if np.isfinite(mom) else 0.0, "vol": vol, "selected": False}

    eligible = [
        sym for sym, mom in scores.items()
        if np.isfinite(mom) and mom > 0.0 and np.isfinite(vols[sym]) and vols[sym] > 1e-8
    ]
    ranked = sorted(eligible, key=lambda sym: scores[sym], reverse=True)[:TOP_K]

    raw = {sym: 0.0 for sym in SLEEVES}
    if ranked:
        ew = 1.0 / len(ranked)
        for sym in ranked:
            raw[sym] = ew
            diag[sym]["selected"] = True
        port_vol2 = sum((raw[sym] * vols[sym]) ** 2 for sym in ranked)
        port_vol = float(np.sqrt(port_vol2)) if port_vol2 > 0 else float("nan")
        scale = min(SIGMA_TARGET / port_vol, 1.0) if port_vol and port_vol > 1e-8 else 0.0
    else:
        scale = 0.0
        port_vol = 0.0

    targets = {sym: raw[sym] * scale for sym in SLEEVES}
    for sym in SLEEVES:
        diag[sym]["scale"] = scale
        diag[sym]["port_vol"] = port_vol
    return targets, diag


def _state_path(state_key: str) -> Path:
    return STATE_DIR / f"{state_key}.json"


def _load_state(state_key: str) -> dict:
    path = _state_path(state_key)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_state(state_key: str, state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = _state_path(state_key)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def scale_for_risk(
    targets: dict[str, float],
    equity: float,
    *,
    state_key: str,
) -> tuple[dict[str, float], dict]:
    """Peak-to-trough DD brake using live broker equity (persisted per account)."""
    if equity <= 0:
        raise RuntimeError(f"invalid equity {equity} for DD brake")

    state = _load_state(state_key)
    peak = float(state.get("peak_equity", equity))
    braked = bool(state.get("braked", False))

    if equity > peak:
        peak = equity
    dd = equity / peak - 1.0
    if dd <= DD_TRIG:
        braked = True
    if braked and abs(dd) < 1e-12:
        braked = False

    scale = DD_FLOOR if braked else 1.0
    scaled = {sym: w * scale for sym, w in targets.items()}

    _save_state(
        state_key,
        {
            "peak_equity": peak,
            "braked": braked,
            "last_equity": equity,
            "last_dd": dd,
            "last_scale": scale,
        },
    )
    return scaled, {
        "peak_equity": peak,
        "dd": dd,
        "braked": braked,
        "scale": scale,
    }
