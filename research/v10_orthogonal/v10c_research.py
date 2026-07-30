"""V10c: MR + VRP funded strictly from R50d idle cash (final iteration).

Candidate: R50d_cash (core untouched; MR with backwardation veto takes
headroom first, vol-targeted VRP takes the remainder)
Ablations: R50d_mrveto, R50d_vrponly, CAP_M neighborhood, cost stress.

Pre-registered in research/v10_orthogonal/DESIGN_C.md. Pass earns a paper
A/B recommendation only (third pass over the same history).
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
    MR_EXIT,
    MR_MAX_HOLD,
    REPO_ROOT,
    SLEEVE_ORDER,
    band_positions,
    load_data,
    metrics,
    native,
    r50d_weights,
    rsi2,
    vrp_state,
    vrp_weights,
    yearly,
)

OUT_JSON = REPO_ROOT / "output" / "v10c_idle_cash_results.json"
OUT_SUMMARY = REPO_ROOT / "output" / "v10c_idle_cash_summary.md"

CAP_M = 0.30


def mr_signal_veto(px: pd.Series, veto: np.ndarray, entry_th: float = MR_ENTRY) -> np.ndarray:
    """V10 MR state machine; `veto` blocks new entries but not open positions."""
    r = rsi2(px).to_numpy()
    sma = px.rolling(200).mean().to_numpy()
    p = px.to_numpy()
    pos = np.zeros(len(px))
    in_pos = False
    held = 0
    for i in range(len(px)):
        if in_pos:
            held += 1
            if (np.isfinite(r[i]) and r[i] > MR_EXIT) or held >= MR_MAX_HOLD:
                in_pos = False
                held = 0
        else:
            if (
                np.isfinite(r[i])
                and r[i] < entry_th
                and np.isfinite(sma[i])
                and p[i] > sma[i]
                and not veto[i]
            ):
                in_pos = True
                held = 0
        pos[i] = 1.0 if in_pos else 0.0
    return pos


def banded(held: float, target: float) -> float:
    if target == 0.0 and held != 0.0:
        return 0.0
    if abs(target - held) > BAND or (held == 0.0 and target > 0.0):
        return target
    return held


def build_book(
    data: dict,
    sig_m: np.ndarray | None,
    sig_v: np.ndarray | None,
    cap_m: float = CAP_M,
    cost_overlay: float = COST_NEW,
) -> dict:
    n = data["n"]
    W6 = r50d_weights(data)
    P6 = band_positions(W6)
    headroom = np.maximum(0.0, 1.0 - P6.sum(axis=1))

    w_v_size = vrp_weights(data, gated=False)  # vol-targeted size, cap 1.5 x 0.15
    Pm = np.zeros(n)
    Pv = np.zeros(n)
    hm = hv = 0.0
    for i in range(n):
        tm = min(cap_m, headroom[i]) if (sig_m is not None and sig_m[i] > 0) else 0.0
        hm = min(banded(hm, tm), headroom[i])
        rem = headroom[i] - hm
        tv = min(w_v_size[i], rem) if (sig_v is not None and sig_v[i] > 0) else 0.0
        hv = min(banded(hv, tv), rem)
        Pm[i], Pv[i] = hm, hv

    P6l = np.vstack([np.zeros((1, 6)), P6[:-1]])
    Pml = np.concatenate([[0.0], Pm[:-1]])
    Pvl = np.concatenate([[0.0], Pv[:-1]])
    turn6 = np.abs(np.diff(np.vstack([np.zeros((1, 6)), P6l]), axis=0)).sum(axis=1)
    turn_o = np.abs(np.diff(np.concatenate([[0.0], Pml]))) + np.abs(
        np.diff(np.concatenate([[0.0], Pvl]))
    )

    R6 = np.column_stack([data["rets"][s].to_numpy() for s in SLEEVE_ORDER])
    r_qqq = data["qqq"].to_numpy()
    r_v = data["r_svxy_mod"].to_numpy()
    core = (P6l * R6).sum(axis=1) - COST_CORE * turn6
    ret = (
        core
        + Pml * r_qqq
        + Pvl * r_v
        + (1.0 - P6l.sum(axis=1) - Pml - Pvl) * data["rf"].to_numpy()
        - cost_overlay * turn_o
    )
    core_ret = core + (1.0 - P6l.sum(axis=1)) * data["rf"].to_numpy()
    return {
        "returns": pd.Series(ret, index=data["index"]),
        "core_returns": pd.Series(core_ret, index=data["index"]),
        "turnover": pd.Series(turn6 + turn_o, index=data["index"]),
        "mr_pos": pd.Series(Pm, index=data["index"]),
        "vrp_pos": pd.Series(Pv, index=data["index"]),
    }


def main() -> None:
    data = load_data()
    ratio = data["ts_ratio"].to_numpy()
    backwardation = np.where(np.isfinite(ratio), ratio <= 1.00, False)
    sig_m = mr_signal_veto(data["qqq_px"], backwardation)
    sig_v = vrp_state(data["ts_ratio"], 1.05, 1.00) * data["svxy_valid"].astype(float)

    books = {
        "R50d_cash": build_book(data, sig_m, sig_v),
        "R50d_mrveto": build_book(data, sig_m, None),
        "R50d_vrponly": build_book(data, None, sig_v),
        "R50d_cash_cap20": build_book(data, sig_m, sig_v, cap_m=0.20),
        "R50d_cash_cap40": build_book(data, sig_m, sig_v, cap_m=0.40),
        "R50d_cash_cost2x": build_book(data, sig_m, sig_v, cost_overlay=COST_NEW * 2.0),
    }
    cand = books["R50d_cash"]
    books["R50d"] = {"returns": cand["core_returns"], "turnover": cand["turnover"] * 0.0}
    W6 = r50d_weights(data)
    P6 = band_positions(W6)
    P6l = np.vstack([np.zeros((1, 6)), P6[:-1]])
    books["R50d"]["turnover"] = pd.Series(
        np.abs(np.diff(np.vstack([np.zeros((1, 6)), P6l]), axis=0)).sum(axis=1),
        index=data["index"],
    )

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
        "aug_2015_flash": ("2015-08-01", "2016-02-29"),
        "volmageddon_2018": ("2018-01-26", "2018-04-30"),
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

    contrib = {
        "mr_by_year": {
            str(d.year): float(v)
            for d, v in (cand["mr_pos"].shift(1).fillna(0.0) * data["qqq"]).resample("YE").sum().items()
            if abs(v) > 1e-12
        },
        "vrp_by_year": {
            str(d.year): float(v)
            for d, v in (cand["vrp_pos"].shift(1).fillna(0.0) * data["r_svxy_mod"]).resample("YE").sum().items()
            if abs(v) > 1e-12
        },
        "avg_mr_pos_when_on": float(cand["mr_pos"][cand["mr_pos"] > 0].mean()),
        "avg_vrp_pos_when_on": float(cand["vrp_pos"][cand["vrp_pos"] > 0].mean()),
        "pct_days_vrp_on": float((cand["vrp_pos"] > 0).mean()),
        "pct_days_mr_on": float((cand["mr_pos"] > 0).mean()),
    }

    p15_b = window_results["from_2015"]["R50d"]
    p15_c = window_results["from_2015"]["R50d_cash"]
    p18_b = window_results["from_2018"]["R50d"]
    p18_c = window_results["from_2018"]["R50d_cash"]
    y_base = yearly_map["R50d"].get("2022")
    y_cand = yearly_map["R50d_cash"].get("2022")
    checks = {
        "cagr_up_1pp": p15_c["cagr"] >= p15_b["cagr"] + 0.01,
        "max_dd_within_1pp": p15_c["drawdown"]["max_dd"] >= p15_b["drawdown"]["max_dd"] - 0.01,
        "sharpe_not_lower": p15_c["sharpe"] >= p15_b["sharpe"],
        "y2022_not_worse_05pp": (
            y_cand is not None and y_base is not None and y_cand >= y_base - 0.005
        ),
        "holdout_2018_cagr_up_and_dd_within_1pp": (
            p18_c["cagr"] > p18_b["cagr"]
            and p18_c["drawdown"]["max_dd"] >= p18_b["drawdown"]["max_dd"] - 0.01
        ),
    }
    passed = all(checks.values())
    failed = [k for k, ok in checks.items() if not ok]
    conclusion = {
        "verdict": "pass_paper_ab_only" if passed else "reject",
        "checks": checks,
        "deltas_from_2015": {
            "cagr_pp": (p15_c["cagr"] - p15_b["cagr"]) * 100,
            "max_dd_pp": (p15_c["drawdown"]["max_dd"] - p15_b["drawdown"]["max_dd"]) * 100,
            "sharpe": p15_c["sharpe"] - p15_b["sharpe"],
            "y2022_pp": None if y_cand is None or y_base is None else (y_cand - y_base) * 100,
        },
        "deltas_from_2018": {
            "cagr_pp": (p18_c["cagr"] - p18_b["cagr"]) * 100,
            "max_dd_pp": (p18_c["drawdown"]["max_dd"] - p18_b["drawdown"]["max_dd"]) * 100,
        },
        "reason": (
            "R50d_cash passed all five pre-registered checks; eligible for a paper "
            "A/B account only (third pass over the same history)."
            if passed
            else "Failed pre-registered check(s): " + ", ".join(failed) + "."
        ),
    }

    results = native(
        {
            "meta": {
                "strategy": "V10c idle-cash premia overlay (MR veto + VRP)",
                "primary": "R50d_cash",
                "baseline": "R50d",
                "data_start": data["index"][0],
                "data_end": data["index"][-1],
                "design": "research/v10_orthogonal/DESIGN_C.md",
                "params": {
                    "CAP_M": CAP_M,
                    "mr_entry": MR_ENTRY,
                    "mr_veto": "VIX3M/VIX <= 1.00 blocks new entries",
                    "vrp": "hysteresis 1.05/1.00, vol-target 0.15 x 0.50 / EWMA, cap 0.225",
                },
            },
            "windows": window_results,
            "crises": crisis_results,
            "yearly": yearly_map,
            "contribution": contrib,
            "conclusion": conclusion,
        }
    )
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        "# V10c Idle-Cash Premia Overlay Research",
        "",
        f"Data: {results['meta']['data_start']} through {results['meta']['data_end']}.",
        "Primary window: from_2015 with a from_2018 holdout gate.",
        "Design: research/v10_orthogonal/DESIGN_C.md.",
        "",
        "## Candidate vs live R50d",
        "",
        f"- from_2015 R50d:      {p15_b['cagr']:.2%} CAGR, Sharpe {p15_b['sharpe']:.2f}, "
        f"{p15_b['drawdown']['max_dd']:.2%} max DD, Calmar {p15_b['calmar']:.2f}",
        f"- from_2015 R50d_cash: {p15_c['cagr']:.2%} CAGR, Sharpe {p15_c['sharpe']:.2f}, "
        f"{p15_c['drawdown']['max_dd']:.2%} max DD, Calmar {p15_c['calmar']:.2f}",
        f"- from_2018 R50d:      {p18_b['cagr']:.2%} CAGR, {p18_b['drawdown']['max_dd']:.2%} max DD",
        f"- from_2018 R50d_cash: {p18_c['cagr']:.2%} CAGR, {p18_c['drawdown']['max_dd']:.2%} max DD",
        f"- 2022: R50d {y_base:.2%}, R50d_cash {y_cand:.2%}",
        "",
        "## Ablations and stress (from_2015, CAGR / max DD / Sharpe)",
        "",
    ]
    for name in ("R50d_mrveto", "R50d_vrponly", "R50d_cash_cap20", "R50d_cash_cap40", "R50d_cash_cost2x"):
        m = window_results["from_2015"][name]
        lines.append(
            f"- {name}: {m['cagr']:.2%} / {m['drawdown']['max_dd']:.2%} / {m['sharpe']:.2f}"
        )
    lines += [
        "",
        "## Sleeve contribution",
        "",
        f"```\n{json.dumps(results['contribution'], indent=2)}\n```",
        "",
        "## Decision",
        "",
        f"**{conclusion['verdict'].upper()}** — {conclusion['reason']}",
        "",
        "## Limitations",
        "",
        "- Third pass over the same history: from_2015 numbers are partially",
        "  burned; only paper trading provides true out-of-sample evidence.",
        "- SVXY pre-2018 is the half-leverage proxy of the -1x fund.",
        "- Yahoo ^VIX3M lags; live requires the CBOE daily CSV fallback.",
        "- Headroom must be computed from broker positions live.",
        "- No live account was changed.",
    ]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "verdict": conclusion["verdict"],
                "checks": checks,
                "from_2015": {
                    "R50d": {"cagr": p15_b["cagr"], "mdd": p15_b["drawdown"]["max_dd"], "sharpe": p15_b["sharpe"]},
                    "R50d_cash": {"cagr": p15_c["cagr"], "mdd": p15_c["drawdown"]["max_dd"], "sharpe": p15_c["sharpe"]},
                },
                "from_2018": {
                    "R50d": {"cagr": p18_b["cagr"], "mdd": p18_b["drawdown"]["max_dd"]},
                    "R50d_cash": {"cagr": p18_c["cagr"], "mdd": p18_c["drawdown"]["max_dd"]},
                },
                "y2022": {"R50d": y_base, "R50d_cash": y_cand},
                "ablations_from_2015": {
                    name: {
                        "cagr": window_results["from_2015"][name]["cagr"],
                        "mdd": window_results["from_2015"][name]["drawdown"]["max_dd"],
                    }
                    for name in ("R50d_mrveto", "R50d_vrponly", "R50d_cash_cap20", "R50d_cash_cap40", "R50d_cash_cost2x")
                },
                "contribution": {k: v for k, v in contrib.items() if not k.endswith("_by_year")},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
