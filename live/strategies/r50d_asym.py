"""R50d_asym: live R50d with asymmetric TQQQ entry/exit hysteresis.

Paper A/B candidate from research/v9_r50d_improve (R50d_asym). Same six-sleeve
book and sizing as r50d, except the equity sleeve uses:
  - risk-on when QQQ ensemble >= 2/3
  - risk-off when QQQ ensemble <= 1/3
  - hold prior state between those levels
  - when risk-on, size with the continuous ensemble gate
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NAME = "r50d_asym"
DESCRIPTION = (
    "R50d + asymmetric TQQQ gate (enter >=2/3, exit <=1/3 hysteresis), "
    "EWMA vol targeting at 50%, 100% notional cap"
)


@dataclass(frozen=True)
class Sleeve:
    signal: str
    budget: float
    leverage: float
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
ENTER_GATE = 2.0 / 3.0
EXIT_GATE = 1.0 / 3.0


def ensemble_gate(px: pd.Series) -> float:
    return float(np.mean([px.iloc[-1] > px.iloc[-1 - L] for L in LOOKBACKS]))


def ensemble_gate_series(px: pd.Series) -> pd.Series:
    votes = [
        (px / px.shift(L) - 1 > 0).astype(float).where(~px.isna(), 0.0)
        for L in LOOKBACKS
    ]
    return pd.concat(votes, axis=1).mean(axis=1).fillna(0.0)


def asymmetric_equity_gate(gate: pd.Series) -> tuple[float, bool]:
    """Walk history so live hysteresis matches the v9 backtest state machine."""
    on = False
    last = 0.0
    for g in gate.to_numpy():
        if on:
            if g <= EXIT_GATE + 1e-12:
                on = False
        else:
            if g >= ENTER_GATE - 1e-12:
                on = True
        last = float(g) if on else 0.0
    return last, on


def ewma_vol(px: pd.Series) -> float:
    r = px.pct_change().dropna()
    if r.empty:
        return float("nan")
    var = (r**2).ewm(alpha=EWMA_ALPHA, adjust=False).mean().iloc[-1]
    return float(np.sqrt(252.0 * var))


def target_weights(closes: pd.DataFrame) -> tuple[dict[str, float], dict[str, dict]]:
    raw, diag = {}, {}
    qqq = closes["QQQ"].dropna()
    equity_gate, equity_on = asymmetric_equity_gate(ensemble_gate_series(qqq))

    for sym, s in SLEEVES.items():
        px = closes[s.signal].dropna()
        if sym == "TQQQ":
            gate = equity_gate
        else:
            gate = ensemble_gate(px)
        vol = ewma_vol(px)
        raw[sym] = gate * s.budget * SIGMA_TARGET / (s.leverage * vol)
        diag[sym] = {
            "gate": gate,
            "vol": vol,
            "equity_on": equity_on if sym == "TQQQ" else None,
        }

    total = sum(raw.values())
    scale = 1.0 / total if total > 1.0 else 1.0
    return {k: v * scale for k, v in raw.items()}, diag
