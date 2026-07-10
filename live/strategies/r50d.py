"""R50d: six-sleeve cash-account book (live paper baseline).

Variant R50d from research/v3_extended/v3_revised.py — v3_cash variant F with
EFA dropped, a managed-futures trend sleeve added, and sigma* = 50%.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NAME = "r50d"
DESCRIPTION = (
    "Six-sleeve trend book (TQQQ/TMF/UGL/DBC/DBMF/BTC), ensemble gates, "
    "EWMA vol targeting at 50%, 100% notional cap"
)


@dataclass(frozen=True)
class Sleeve:
    signal: str      # ticker whose closes drive the gate and vol estimate
    budget: float    # share of the portfolio risk budget
    leverage: float  # leverage embedded in the traded product
    crypto: bool = False


SLEEVES: dict[str, Sleeve] = {
    "TQQQ": Sleeve("QQQ", 0.40, 3.0),
    "TMF": Sleeve("TLT", 0.15, 3.0),
    "UGL": Sleeve("GLD", 0.15, 2.0),
    "DBC": Sleeve("DBC", 0.10, 1.0),
    "DBMF": Sleeve("DBMF", 0.10, 1.0),
    "BTC/USD": Sleeve("BTC/USD", 0.10, 1.0, crypto=True),
}
SIGMA_TARGET = 0.50
LOOKBACKS = (63, 126, 252)
EWMA_ALPHA = 0.06
TRADE_BAND = 0.05
HISTORY_DAYS = 600
MAX_MOVE = 0.25


def ensemble_gate(px: pd.Series) -> float:
    return float(np.mean([px.iloc[-1] > px.iloc[-1 - L] for L in LOOKBACKS]))


def ewma_vol(px: pd.Series) -> float:
    r = px.pct_change().dropna()
    var = (r**2).ewm(alpha=EWMA_ALPHA, adjust=False).mean().iloc[-1]
    return float(np.sqrt(252.0 * var))


def target_weights(closes: pd.DataFrame) -> tuple[dict[str, float], dict[str, dict]]:
    raw, diag = {}, {}
    for sym, s in SLEEVES.items():
        px = closes[s.signal].dropna()
        gate, vol = ensemble_gate(px), ewma_vol(px)
        raw[sym] = gate * s.budget * SIGMA_TARGET / (s.leverage * vol)
        diag[sym] = {"gate": gate, "vol": vol}
    total = sum(raw.values())
    scale = 1.0 / total if total > 1.0 else 1.0
    return {k: v * scale for k, v in raw.items()}, diag
