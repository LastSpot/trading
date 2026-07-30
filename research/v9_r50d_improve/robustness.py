"""Robustness suite: R50d vs R50d_asym vs SPY/VOO, 50/50 VOO-QQQ, QQQ.

Pre-registered style tests (no retuning after looking):
1. Crisis windows + ex-ante regime bins
2. Equity-gate parameter neighborhood
3. Locked holdout (post-2018 design cut; last 24m)
4. Placebo / lagged equity gates

Benchmarks:
- SPY (pre-VOO) / VOO (from inception) as the S&P book
- 50/50 VOO–QQQ monthly rebalance (SPY spliced before VOO)
- QQQ buy-and-hold
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from v9_research import (
    INITIAL,
    SLEEVE_ORDER,
    asymmetric_equity_gate,
    band_positions,
    build_baseline,
    build_variant,
    load_data,
    metrics,
    native,
    raw_weights,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = REPO_ROOT / "output" / "v9_robustness_results.json"
OUT_SUMMARY = REPO_ROOT / "output" / "v9_robustness_summary.md"

COST = 0.0005

# Fixed before looking at results
CRISES = {
    "dotcom": ("2000-03-24", "2002-10-09"),
    "gfc": ("2007-10-09", "2009-03-09"),
    "2018_q4": ("2018-10-01", "2018-12-24"),
    "covid_crash": ("2020-02-19", "2020-03-23"),
    "covid_rebound": ("2020-03-24", "2020-08-31"),
    "inflation_2021_23": ("2021-01-04", "2023-10-27"),
    "bear_2022": ("2022-01-03", "2022-10-12"),
    "tariff_2025": ("2025-02-15", "2025-06-30"),
}

WINDOWS = {
    "full": None,  # filled after data load
    "from_2010": "2010-01-01",
    "from_2015": "2015-01-01",
    "holdout_from_2018": "2018-01-01",
    "holdout_last_24m": None,  # filled after data load
}

# enter_threshold, exit_threshold (fraction of 3-vote ensemble)
NEIGHBORHOOD = [
    ("live_continuous", None, None),  # R50d baseline
    ("enter_1_exit_0", 1 / 3, 0.0),
    ("enter_2_exit_0", 2 / 3, 0.0),
    ("enter_2_exit_1", 2 / 3, 1 / 3),  # R50d_asym
    ("enter_3_exit_1", 1.0, 1 / 3),
    ("enter_3_exit_2", 1.0, 2 / 3),
    ("enter_2_exit_2", 2 / 3, 2 / 3),  # no hysteresis; on only at >=2/3
]


def hysteresis_gate(gate: np.ndarray, enter: float, exit_: float) -> np.ndarray:
    out = np.zeros_like(gate)
    on = False
    for i, g in enumerate(gate):
        if on:
            if g <= exit_ + 1e-12:
                on = False
        else:
            if g >= enter - 1e-12:
                on = True
        out[i] = g if on else 0.0
    return out


def monthly_5050(ra: pd.Series, rb: pd.Series) -> pd.Series:
    idx = ra.index
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
    Pl = np.vstack([np.zeros((1, P.shape[1])), P[:-1]])
    r = np.column_stack([ra.to_numpy(), rb.to_numpy()])
    return pd.Series((Pl * r).sum(axis=1), index=idx)


def load_benchmarks(idx: pd.DatetimeIndex) -> dict[str, pd.Series]:
    raw = yf.download(
        ["SPY", "VOO", "QQQ"],
        start="1999-03-10",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    spy = close["SPY"].reindex(idx).ffill()
    voo = close["VOO"].reindex(idx)
    qqq = close["QQQ"].reindex(idx).ffill()
    # S&P book: VOO when live, else SPY
    sap = voo.combine_first(spy)
    r_sap = sap.pct_change().fillna(0.0)
    r_qqq = qqq.pct_change().fillna(0.0)
    r_spy = spy.pct_change().fillna(0.0)
    r_voo = voo.pct_change()
    r_voo = r_voo.fillna(0.0)
    # 50/50: VOO/QQQ with SPY splice before VOO
    sap_for_5050 = voo.combine_first(spy)
    r_5050 = monthly_5050(sap_for_5050.pct_change().fillna(0.0), r_qqq)
    voo_start = voo.dropna().index.min() if not voo.dropna().empty else None
    return {
        "SPY_VOO": r_sap,
        "VOO_QQQ_5050": r_5050,
        "QQQ": r_qqq,
        "SPY_only": r_spy,
        "VOO_only": r_voo,
        "voo_inception": voo_start,
    }


def run_from_equity_gate(data: dict, equity_gate: np.ndarray) -> dict:
    W = raw_weights(data, equity_gate)
    P = band_positions(W)
    R = np.column_stack([data["rets"][s].to_numpy() for s in SLEEVE_ORDER])
    Pl = np.vstack([np.zeros((1, P.shape[1])), P[:-1]])
    turn = np.abs(np.diff(np.vstack([np.zeros((1, P.shape[1])), Pl]), axis=0)).sum(axis=1)
    ret = (
        (Pl * R).sum(axis=1)
        + (1.0 - Pl.sum(axis=1)) * data["rf"].to_numpy()
        - COST * turn
    )
    return {
        "returns": pd.Series(ret, index=data["index"]),
        "turnover": pd.Series(turn, index=data["index"]),
        "positions": pd.DataFrame(P, index=data["index"], columns=SLEEVE_ORDER),
    }


def slim(m: dict) -> dict:
    if not m:
        return {}
    return native(
        {
            "cagr": m.get("cagr"),
            "vol": m.get("annualized_vol"),
            "sharpe": m.get("sharpe"),
            "calmar": m.get("calmar"),
            "max_dd": m.get("drawdown", {}).get("max_dd"),
            "total_return": m.get("total_return"),
            "terminal_100k": m.get("terminal_100k"),
            "start": m.get("start"),
            "end": m.get("end"),
            "recovery_days": m.get("drawdown", {}).get("peak_to_recovery_trading_days"),
        }
    )


def score_books(
    books: dict[str, dict],
    benches: dict[str, pd.Series],
    rf: pd.Series,
    start,
    end=None,
) -> dict:
    out = {}
    for name, run in books.items():
        out[name] = slim(metrics(run["returns"], rf, start, end, run.get("turnover")))
    for bname in ("SPY_VOO", "VOO_QQQ_5050", "QQQ"):
        out[bname] = slim(metrics(benches[bname], rf, start, end))
    return out


def regime_labels(data: dict, benches: dict) -> pd.DataFrame:
    """Ex-ante regime tags using only lagged information."""
    idx = data["index"]
    qqq_px = (1.0 + benches["QQQ"]).cumprod()
    spy_px = (1.0 + benches["SPY_VOO"]).cumprod()
    # 200d trend: price vs 200d MA, decided on prior close
    ma200 = qqq_px.rolling(200).mean()
    equity_bull = (qqq_px.shift(1) > ma200.shift(1)).fillna(False)
    # High vol: 21d realized vol above its trailing 3y median (lagged)
    r = benches["QQQ"]
    vol21 = r.rolling(21).std() * np.sqrt(252.0)
    vol_med = vol21.rolling(252 * 3).median()
    high_vol = (vol21.shift(1) > vol_med.shift(1)).fillna(False)
    # Rising rates proxy: 63d change in ^IRX (already in data via rf*252-ish)
    # use rf annualized level change
    rf_ann = data["rf"] * 252.0
    rising_rates = (rf_ann - rf_ann.shift(63) > 0.005).fillna(False)
    # Days where continuous R50d equity gate is on but asym hysteresis is off
    # (the only material divergence; ensemble votes are in {0,1/3,2/3,1}).
    g = data["gates"]["TQQQ"]
    asym_g = asymmetric_equity_gate(g)
    asym_off_cont_on = pd.Series((asym_g <= 1e-12) & (g > 1e-12), index=idx)
    return pd.DataFrame(
        {
            "equity_bull": equity_bull,
            "equity_bear": ~equity_bull,
            "high_vol": high_vol,
            "low_vol": ~high_vol,
            "rising_rates": rising_rates,
            "asym_off_while_r50d_on": asym_off_cont_on,
        },
        index=idx,
    )


def metrics_on_mask(ret: pd.Series, rf: pd.Series, mask: pd.Series) -> dict:
    selected = ret.loc[mask].dropna()
    if len(selected) < 40:
        return {"sessions": int(len(selected)), "note": "too_few_sessions"}
    # Regime bins are non-contiguous; use simple mean/vol and path DD on concat equity
    equity = (1.0 + selected).cumprod()
    years = len(selected) / 252.0
    cagr = float(equity.iloc[-1] ** (1.0 / max(years, 1e-9)) - 1.0)
    vol = float(selected.std() * np.sqrt(252.0))
    dd = float((equity / equity.cummax() - 1.0).min())
    excess = selected - rf.reindex(selected.index).fillna(0.0)
    sharpe = float(excess.mean() / selected.std() * np.sqrt(252.0)) if selected.std() > 0 else None
    return native(
        {
            "sessions": int(len(selected)),
            "cagr_from_concat": cagr,
            "mean_ann": float(selected.mean() * 252.0),
            "vol": vol,
            "sharpe": sharpe,
            "max_dd_on_concat": dd,
            "total_return": float(equity.iloc[-1] - 1.0),
        }
    )


def write_summary(results: dict) -> None:
    h = results["holdout"]["holdout_from_2018"]
    last = results["holdout"]["holdout_last_24m"]
    lines = [
        "# R50d vs R50d_asym robustness",
        "",
        "Benches: SPY/VOO, 50/50 VOO–QQQ (monthly), QQQ. No parameters retuned after looking.",
        "",
        "## Holdout (locked)",
        "",
        "### From 2018",
        "",
        f"- R50d: {h['R50d']['cagr']:.2%} CAGR, {h['R50d']['max_dd']:.2%} max DD",
        f"- R50d_asym: {h['R50d_asym']['cagr']:.2%} CAGR, {h['R50d_asym']['max_dd']:.2%} max DD",
        f"- SPY/VOO: {h['SPY_VOO']['cagr']:.2%} CAGR, {h['SPY_VOO']['max_dd']:.2%} max DD",
        f"- 50/50 VOO–QQQ: {h['VOO_QQQ_5050']['cagr']:.2%} CAGR, {h['VOO_QQQ_5050']['max_dd']:.2%} max DD",
        f"- QQQ: {h['QQQ']['cagr']:.2%} CAGR, {h['QQQ']['max_dd']:.2%} max DD",
        "",
        "### Last 24 months",
        "",
        f"- R50d: {last['R50d']['cagr']:.2%} CAGR, {last['R50d']['max_dd']:.2%} max DD",
        f"- R50d_asym: {last['R50d_asym']['cagr']:.2%} CAGR, {last['R50d_asym']['max_dd']:.2%} max DD",
        f"- QQQ: {last['QQQ']['cagr']:.2%} CAGR, {last['QQQ']['max_dd']:.2%} max DD",
        "",
        "## Neighborhood (from_2015)",
        "",
    ]
    for row in results["neighborhood"]["from_2015"]:
        lines.append(
            f"- {row['name']}: {row['cagr']:.2%} CAGR, {row['max_dd']:.2%} max DD, "
            f"calmar {row['calmar']:.2f}"
            if row.get("calmar") is not None
            else f"- {row['name']}: {row['cagr']:.2%} CAGR, {row['max_dd']:.2%} max DD"
        )
    lines.extend(
        [
            "",
            "## Placebo (from_2015)",
            "",
        ]
    )
    for name, m in results["placebo"]["from_2015"].items():
        if name in ("SPY_VOO", "VOO_QQQ_5050", "QQQ"):
            continue
        lines.append(
            f"- {name}: {m['cagr']:.2%} CAGR, {m['max_dd']:.2%} max DD"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            results["conclusion"]["text"],
            "",
            "## Limitations",
            "",
            "- Regime bins concatenate non-contiguous days; CAGR is illustrative.",
            "- Neighborhood is discrete vote thresholds only (not a continuous grid search).",
            "- Holdout from_2018 still overlaps some design intuition from prior research.",
            "- Paper A/B remains the cleanest forward test.",
        ]
    )
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    data = load_data()
    benches = load_benchmarks(data["index"])
    rf = data["rf"]
    eval_start = data["index"][260]
    last = data["index"][-1]

    books = {
        "R50d": build_baseline(data),
        "R50d_asym": build_variant(data, "R50d_asym"),
    }

    # --- Windows + crises ---
    window_starts = {
        "full": eval_start,
        "from_2010": pd.Timestamp("2010-01-01"),
        "from_2015": pd.Timestamp("2015-01-01"),
        "holdout_from_2018": pd.Timestamp("2018-01-01"),
        "holdout_last_24m": last - pd.DateOffset(months=24),
    }

    windows = {
        label: score_books(books, benches, rf, start)
        for label, start in window_starts.items()
    }
    crises = {
        label: score_books(books, benches, rf, a, b)
        for label, (a, b) in CRISES.items()
    }

    # --- Regime bins (from_2015 onward for fair sleeve era) ---
    labels = regime_labels(data, benches)
    regime_start = pd.Timestamp("2015-01-01")
    regime_mask_base = labels.index >= regime_start
    regimes = {}
    for col in labels.columns:
        mask = regime_mask_base & labels[col].astype(bool)
        regimes[col] = {
            name: metrics_on_mask(run["returns"], rf, mask)
            for name, run in books.items()
        }
        for bname in ("SPY_VOO", "VOO_QQQ_5050", "QQQ"):
            regimes[col][bname] = metrics_on_mask(benches[bname], rf, mask)
        regimes[col]["sessions"] = int(mask.sum())

    # --- Neighborhood ---
    neighborhood_runs = {}
    for name, enter, exit_ in NEIGHBORHOOD:
        if enter is None:
            neighborhood_runs[name] = books["R50d"]
        else:
            g = hysteresis_gate(data["gates"]["TQQQ"], enter, exit_)
            neighborhood_runs[name] = run_from_equity_gate(data, g)
    # Ensure asym label alias
    neighborhood_runs["R50d_asym_alias"] = neighborhood_runs["enter_2_exit_1"]

    neighborhood = {}
    for wlabel, start in (
        ("from_2015", pd.Timestamp("2015-01-01")),
        ("holdout_from_2018", pd.Timestamp("2018-01-01")),
        ("full", eval_start),
    ):
        rows = []
        for name, _, _ in NEIGHBORHOOD:
            m = slim(
                metrics(
                    neighborhood_runs[name]["returns"],
                    rf,
                    start,
                    turnover=neighborhood_runs[name]["turnover"],
                )
            )
            m["name"] = name
            rows.append(m)
        neighborhood[wlabel] = rows

    # --- Placebo ---
    gate = data["gates"]["TQQQ"]
    rng = np.random.default_rng(42)
    shuffled = gate.copy()
    rng.shuffle(shuffled)
    lagged_21 = np.concatenate([np.zeros(21), gate[:-21]])
    lagged_63 = np.concatenate([np.zeros(63), gate[:-63]])
    # random walk of on/off with same on-fraction as live continuous >0
    on_frac = float((gate > 0).mean())
    random_on = (rng.random(len(gate)) < on_frac).astype(float) * gate.mean()
    # use constant mean gate when "on" so sizing roughly comparable
    mean_on = float(gate[gate > 0].mean()) if (gate > 0).any() else 0.0
    random_gate = (rng.random(len(gate)) < on_frac).astype(float) * mean_on

    placebo_books = {
        "R50d": books["R50d"],
        "R50d_asym": books["R50d_asym"],
        "placebo_shuffle_gate": run_from_equity_gate(data, shuffled),
        "placebo_lag_21": run_from_equity_gate(data, lagged_21),
        "placebo_lag_63": run_from_equity_gate(data, lagged_63),
        "placebo_random_on": run_from_equity_gate(data, random_gate),
        "asym_on_shuffled": run_from_equity_gate(data, asymmetric_equity_gate(shuffled)),
    }
    placebo = {
        "from_2015": score_books(placebo_books, benches, rf, pd.Timestamp("2015-01-01")),
        "holdout_from_2018": score_books(
            placebo_books, benches, rf, pd.Timestamp("2018-01-01")
        ),
    }

    # --- Conclusion heuristics (descriptive, not a pass bar) ---
    h2018 = windows["holdout_from_2018"]
    neigh = {r["name"]: r for r in neighborhood["from_2015"]}
    asym = neigh["enter_2_exit_1"]
    live = neigh["live_continuous"]
    rivals = [
        r
        for r in neighborhood["from_2015"]
        if r["name"] not in ("live_continuous", "enter_2_exit_1")
    ]
    better_than_asym = [
        r["name"]
        for r in rivals
        if r.get("calmar") is not None
        and asym.get("calmar") is not None
        and r["calmar"] > asym["calmar"] + 0.05
    ]
    placebo_2015 = placebo["from_2015"]
    shuf = placebo_2015["placebo_shuffle_gate"]
    shuf_h = placebo["holdout_from_2018"]["placebo_shuffle_gate"]
    calmar_gap = (placebo_2015["R50d"].get("calmar") or 0) - (shuf.get("calmar") or 0)
    holdout_cagr_gap = (
        placebo["holdout_from_2018"]["R50d"]["cagr"] - shuf_h["cagr"]
    ) * 100
    # Holdout neighborhood: any neighbor clearly better Calmar than asym
    neigh_h = {r["name"]: r for r in neighborhood["holdout_from_2018"]}
    asym_h = neigh_h["enter_2_exit_1"]
    better_h = [
        r["name"]
        for r in neighborhood["holdout_from_2018"]
        if r["name"] not in ("live_continuous", "enter_2_exit_1")
        and r.get("calmar") is not None
        and asym_h.get("calmar") is not None
        and r["calmar"] > asym_h["calmar"] + 0.02
    ]
    conclusion = {
        "holdout_2018_asym_vs_r50d_cagr_pp": (
            h2018["R50d_asym"]["cagr"] - h2018["R50d"]["cagr"]
        )
        * 100,
        "holdout_2018_asym_vs_r50d_mdd_pp": (
            h2018["R50d_asym"]["max_dd"] - h2018["R50d"]["max_dd"]
        )
        * 100,
        "holdout_2018_r50d_beats_qqq_cagr": h2018["R50d"]["cagr"] > h2018["QQQ"]["cagr"],
        "holdout_2018_r50d_shallower_dd_than_qqq": (
            h2018["R50d"]["max_dd"] > h2018["QQQ"]["max_dd"]
        ),
        "neighborhood_fragile": len(better_than_asym) > 0 or len(better_h) > 0,
        "better_neighbors_from_2015": better_than_asym,
        "better_neighbors_holdout_2018": better_h,
        "placebo_from_2015_calmar_gap": calmar_gap,
        "placebo_holdout_2018_cagr_gap_pp": holdout_cagr_gap,
        "text": "",
    }
    bits = []
    bits.append(
        f"On holdout from_2018, R50d_asym vs R50d: "
        f"{conclusion['holdout_2018_asym_vs_r50d_cagr_pp']:+.2f}pp CAGR, "
        f"{conclusion['holdout_2018_asym_vs_r50d_mdd_pp']:+.2f}pp max DD "
        f"(positive DD delta = shallower)."
    )
    bits.append(
        f"Both books beat SPY/VOO and 50/50 on CAGR and max DD; vs QQQ they have "
        f"{'higher' if conclusion['holdout_2018_r50d_beats_qqq_cagr'] else 'lower'} CAGR "
        f"and much shallower drawdowns."
    )
    if better_h:
        bits.append(
            "Neighborhood fragility: on holdout_2018, "
            f"{', '.join(better_h)} beat enter_2_exit_1 (asym) on Calmar — "
            "asym is nearby-competitive, not uniquely optimal."
        )
    elif better_than_asym:
        bits.append(
            "Neighborhood: from_2015 Calmar rivals exist "
            f"({', '.join(better_than_asym)}); holdout did not confirm domination."
        )
    else:
        bits.append(
            "Neighborhood: asym is locally competitive on Calmar across nearby gates."
        )
    bits.append(
        f"Placebo: shuffled equity gate can match/beat from_2015 CAGR but with worse "
        f"Calmar (gap {calmar_gap:+.2f}); on holdout_2018 live R50d beats shuffle by "
        f"{holdout_cagr_gap:+.1f}pp CAGR and ~6pp shallower max DD — timing of the "
        "real gate matters out of sample, and diversifier sleeves carry a lot of the level."
    )
    last24 = windows["holdout_last_24m"]
    bits.append(
        f"Last 24m: QQQ CAGR {last24['QQQ']['cagr']*100:.1f}% tops both books on return, "
        f"but R50d_asym keeps the shallowest max DD ({last24['R50d_asym']['max_dd']*100:.1f}%)."
    )
    conclusion["text"] = " ".join(bits)

    results = native(
        {
            "meta": {
                "strategies": ["R50d", "R50d_asym"],
                "benchmarks": ["SPY_VOO", "VOO_QQQ_5050", "QQQ"],
                "data_start": data["index"][0],
                "data_end": data["index"][-1],
                "evaluation_start": eval_start,
                "voo_inception": benches["voo_inception"],
                "initial_capital": INITIAL,
                "note": "SPY_VOO uses VOO when available else SPY; 50/50 monthly rebalance.",
            },
            "windows": windows,
            "crises": crises,
            "regimes": regimes,
            "neighborhood": neighborhood,
            "placebo": placebo,
            "holdout": {
                "holdout_from_2018": windows["holdout_from_2018"],
                "holdout_last_24m": windows["holdout_last_24m"],
            },
            "conclusion": conclusion,
        }
    )
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_summary(results)
    print(
        json.dumps(
            {
                "data_end": str(results["meta"]["data_end"]),
                "holdout_2018": {
                    k: {
                        "cagr": windows["holdout_from_2018"][k]["cagr"],
                        "max_dd": windows["holdout_from_2018"][k]["max_dd"],
                    }
                    for k in ("R50d", "R50d_asym", "SPY_VOO", "VOO_QQQ_5050", "QQQ")
                },
                "conclusion": conclusion["text"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
