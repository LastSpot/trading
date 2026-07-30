"""V10b: MR sleeve deployed only from R50d idle cash (pre-registered revision).

Candidate: R50d_mr (live R50d untouched + RSI(2) dip-buy on QQQ from headroom)
Diagnostics: headroom/fill stats, VRP-on-idle-cash, MR entry neighborhood,
cost stress.

Pre-registered in research/v10_orthogonal/DESIGN_B.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_research import (  # noqa: E402
    BAND,
    COST_CORE,
    COST_NEW,
    MR_ENTRY,
    REPO_ROOT,
    SLEEVE_ORDER,
    band_positions,
    load_data,
    metrics,
    mr_positions,
    native,
    r50d_weights,
    vrp_state,
    yearly,
)

OUT_JSON = REPO_ROOT / "output" / "v10b_mr_cash_results.json"
OUT_SUMMARY = REPO_ROOT / "output" / "v10b_mr_cash_summary.md"

B_M_MAX = 0.50


def overlay_positions(sig: np.ndarray, headroom: np.ndarray, cap: float = B_M_MAX) -> np.ndarray:
    """Banded, headroom-capped overlay positions (targets as of each close)."""
    P = np.zeros(len(sig))
    held = 0.0
    for i in range(len(sig)):
        target = (min(cap, headroom[i]) if sig[i] > 0 else 0.0)
        if target == 0.0 and held != 0.0:
            held = 0.0
        elif abs(target - held) > BAND or (held == 0.0 and target > 0.0):
            held = target
        held = min(held, headroom[i])
        P[i] = held
    return P


def build_book(data: dict, sig: np.ndarray, r_overlay: np.ndarray, cost_overlay: float) -> dict:
    """R50d core (unchanged) + overlay column funded from core headroom."""
    W6 = r50d_weights(data)
    P6 = band_positions(W6)
    headroom = np.maximum(0.0, 1.0 - P6.sum(axis=1))
    Pm = overlay_positions(sig, headroom)

    P6l = np.vstack([np.zeros((1, 6)), P6[:-1]])
    Pml = np.concatenate([[0.0], Pm[:-1]])
    turn6 = np.abs(np.diff(np.vstack([np.zeros((1, 6)), P6l]), axis=0)).sum(axis=1)
    turnm = np.abs(np.diff(np.concatenate([[0.0], Pml])))

    R6 = np.column_stack([data["rets"][s].to_numpy() for s in SLEEVE_ORDER])
    ret = (
        (P6l * R6).sum(axis=1)
        + Pml * r_overlay
        + (1.0 - P6l.sum(axis=1) - Pml) * data["rf"].to_numpy()
        - COST_CORE * turn6
        - cost_overlay * turnm
    )
    core_ret = (
        (P6l * R6).sum(axis=1)
        + (1.0 - P6l.sum(axis=1)) * data["rf"].to_numpy()
        - COST_CORE * turn6
    )
    return {
        "returns": pd.Series(ret, index=data["index"]),
        "core_returns": pd.Series(core_ret, index=data["index"]),
        "turnover": pd.Series(turn6 + turnm, index=data["index"]),
        "overlay_pos": pd.Series(Pm, index=data["index"]),
        "headroom": pd.Series(headroom, index=data["index"]),
        "sig": pd.Series(sig, index=data["index"]),
    }


def main() -> None:
    data = load_data()

    mr_sig = mr_positions(data["qqq_px"], entry_th=MR_ENTRY)
    r_qqq = data["qqq"].to_numpy()
    books = {"R50d_mr": build_book(data, mr_sig, r_qqq, COST_NEW)}
    books["R50d"] = {
        "returns": books["R50d_mr"]["core_returns"],
        "turnover": books["R50d_mr"]["turnover"] * 0 + 0.0,  # replaced below
    }
    # baseline turnover: rebuild core-only turnover for reporting
    W6 = r50d_weights(data)
    P6 = band_positions(W6)
    P6l = np.vstack([np.zeros((1, 6)), P6[:-1]])
    turn6 = np.abs(np.diff(np.vstack([np.zeros((1, 6)), P6l]), axis=0)).sum(axis=1)
    books["R50d"]["turnover"] = pd.Series(turn6, index=data["index"])

    # diagnostics
    vrp_sig = vrp_state(data["ts_ratio"], 1.05, 1.00)
    sv_valid = data["svxy_valid"].astype(float)
    books["R50d_vrpcash"] = build_book(
        data, vrp_sig * sv_valid, data["r_svxy_mod"].to_numpy(), COST_NEW
    )
    for th in (5.0, 15.0):
        sig = mr_positions(data["qqq_px"], entry_th=th)
        books[f"R50d_mr{int(th)}"] = build_book(data, sig, r_qqq, COST_NEW)
    books["R50d_mr_cost2x"] = build_book(data, mr_sig, r_qqq, COST_NEW * 2.0)

    eval_start = data["index"][260]
    last = data["index"][-1]
    windows = {
        "full": (eval_start, None),
        "from_2010": (pd.Timestamp("2010-01-01"), None),
        "from_2015": (pd.Timestamp("2015-01-01"), None),
        "from_2018": (pd.Timestamp("2018-01-01"), None),
        "last_3y": (last - pd.DateOffset(years=3), None),
    }
    crises = {
        "gfc": ("2007-10-09", "2009-03-09"),
        "covid_crash": ("2020-02-19", "2020-03-23"),
        "inflation_bear_2022": ("2022-01-03", "2022-10-12"),
        "tariff_2025": ("2025-02-15", "2025-06-30"),
    }

    window_results = {
        label: {
            name: metrics(run["returns"], data["rf"], a, b, run["turnover"])
            for name, run in books.items()
        }
        for label, (a, b) in windows.items()
    }
    for label, (a, b) in windows.items():
        window_results[label]["bench_QQQ"] = metrics(data["qqq"], data["rf"], a, b)
    crisis_results = {
        label: {
            name: metrics(run["returns"], data["rf"], a, b, run["turnover"])
            for name, run in books.items()
        }
        for label, (a, b) in crises.items()
    }
    yearly_map = {name: yearly(run["returns"], eval_start) for name, run in books.items()}

    # headroom / fill diagnostics while MR signal is on (T+1 aligned to holdings)
    cand = books["R50d_mr"]
    sig_on = cand["sig"].to_numpy() > 0
    head_on = cand["headroom"].to_numpy()[sig_on]
    pos_on = cand["overlay_pos"].to_numpy()[sig_on]
    fill = {
        "signal_days": int(sig_on.sum()),
        "avg_headroom_when_signal": float(head_on.mean()) if sig_on.any() else None,
        "median_headroom_when_signal": float(np.median(head_on)) if sig_on.any() else None,
        "avg_deployed_when_signal": float(pos_on.mean()) if sig_on.any() else None,
        "pct_signals_deploying_10pct": float((pos_on >= 0.10).mean()) if sig_on.any() else None,
        "vrpcash_avg_deployed_when_on": float(
            books["R50d_vrpcash"]["overlay_pos"][books["R50d_vrpcash"]["sig"] > 0].mean()
        ),
        "mr_contribution_by_year": {
            str(yr): float(v)
            for yr, v in (
                (cand["overlay_pos"].shift(1).fillna(0.0) * data["qqq"])
                .loc[str(eval_start.year) :]
                .resample("YE")
                .sum()
                .items()
            )
        },
    }

    base = window_results["from_2015"]["R50d"]
    c = window_results["from_2015"]["R50d_mr"]
    y_base = yearly_map["R50d"].get("2022")
    y_cand = yearly_map["R50d_mr"].get("2022")
    checks = {
        "cagr_up_05pp": c["cagr"] >= base["cagr"] + 0.005,
        "max_dd_within_05pp": c["drawdown"]["max_dd"] >= base["drawdown"]["max_dd"] - 0.005,
        "sharpe_not_lower": c["sharpe"] >= base["sharpe"],
        "y2022_not_worse_05pp": (
            y_cand is not None and y_base is not None and y_cand >= y_base - 0.005
        ),
    }
    passed = all(checks.values())
    failed = [k for k, ok in checks.items() if not ok]
    conclusion = {
        "verdict": "pass" if passed else "reject",
        "checks": checks,
        "deltas_from_2015": {
            "cagr_pp": (c["cagr"] - base["cagr"]) * 100,
            "max_dd_pp": (c["drawdown"]["max_dd"] - base["drawdown"]["max_dd"]) * 100,
            "sharpe": c["sharpe"] - base["sharpe"],
            "y2022_pp": None if y_cand is None or y_base is None else (y_cand - y_base) * 100,
        },
        "reason": (
            "R50d_mr passed all pre-registered from_2015 checks versus live R50d."
            if passed
            else "Failed pre-registered check(s): " + ", ".join(failed) + "."
        ),
    }

    results = native(
        {
            "meta": {
                "strategy": "V10b MR on idle cash",
                "primary": "R50d_mr",
                "baseline": "R50d",
                "data_start": data["index"][0],
                "data_end": data["index"][-1],
                "design": "research/v10_orthogonal/DESIGN_B.md",
                "params": {"B_M_MAX": B_M_MAX, "mr_entry": MR_ENTRY},
            },
            "windows": window_results,
            "crises": crisis_results,
            "yearly": yearly_map,
            "fill_diagnostics": fill,
            "conclusion": conclusion,
        }
    )
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        "# V10b MR-on-Idle-Cash Research",
        "",
        f"Data: {results['meta']['data_start']} through {results['meta']['data_end']}.",
        "Primary window: from_2015. Design: research/v10_orthogonal/DESIGN_B.md.",
        "",
        "## Candidate vs live R50d (from_2015)",
        "",
        f"- Live R50d: {base['cagr']:.2%} CAGR, Sharpe {base['sharpe']:.2f}, "
        f"{base['drawdown']['max_dd']:.2%} max DD, Calmar {base['calmar']:.2f}",
        f"- R50d_mr:  {c['cagr']:.2%} CAGR, Sharpe {c['sharpe']:.2f}, "
        f"{c['drawdown']['max_dd']:.2%} max DD, Calmar {c['calmar']:.2f}",
        f"- Deltas: CAGR {(c['cagr'] - base['cagr']) * 100:+.2f} pp (need >= +0.50), "
        f"max DD {(c['drawdown']['max_dd'] - base['drawdown']['max_dd']) * 100:+.2f} pp "
        f"(need >= -0.50), Sharpe {c['sharpe'] - base['sharpe']:+.03f} (need >= 0)",
        f"- 2022: R50d {y_base:.2%}, R50d_mr {y_cand:.2%} (need >= base - 0.5pp)",
        "",
        "## Other windows (R50d -> R50d_mr, CAGR / max DD)",
        "",
    ]
    for label in ("full", "from_2010", "from_2018", "last_3y"):
        b_, c_ = window_results[label]["R50d"], window_results[label]["R50d_mr"]
        lines.append(
            f"- {label}: {b_['cagr']:.2%} / {b_['drawdown']['max_dd']:.2%} -> "
            f"{c_['cagr']:.2%} / {c_['drawdown']['max_dd']:.2%}"
        )
    lines += [
        "",
        "## Fill diagnostics",
        "",
        f"```\n{json.dumps(results['fill_diagnostics'], indent=2)}\n```",
        "",
        "## Neighborhood and stress (from_2015, CAGR / max DD)",
        "",
    ]
    for name in ("R50d_mr5", "R50d_mr15", "R50d_mr_cost2x", "R50d_vrpcash"):
        m = window_results["from_2015"][name]
        lines.append(f"- {name}: {m['cagr']:.2%} / {m['drawdown']['max_dd']:.2%}")
    lines += [
        "",
        "## Decision",
        "",
        f"**{conclusion['verdict'].upper()}** — {conclusion['reason']}",
        "",
        "## Limitations",
        "",
        "- Headroom is computed from backtest core positions; live must compute it",
        "  from actual broker positions after the core rebalance.",
        "- Pre-2010 core history uses synthetic levered sleeves as in v9.",
        "- No live account was changed.",
    ]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "verdict": conclusion["verdict"],
                "checks": checks,
                "deltas_from_2015": conclusion["deltas_from_2015"],
                "from_2015": {
                    "R50d": {"cagr": base["cagr"], "mdd": base["drawdown"]["max_dd"], "sharpe": base["sharpe"]},
                    "R50d_mr": {"cagr": c["cagr"], "mdd": c["drawdown"]["max_dd"], "sharpe": c["sharpe"]},
                },
                "fill": {k: v for k, v in fill.items() if k != "mr_contribution_by_year"},
                "holdout_2018": {
                    "R50d": window_results["from_2018"]["R50d"]["cagr"],
                    "R50d_mr": window_results["from_2018"]["R50d_mr"]["cagr"],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
