"""Monte Carlo A/B test: paper R50d vs paper R50d_asym.

Builds historical daily returns for both live paper strategies, then runs a
paired stationary block bootstrap so each simulated path uses the same resampled
calendar for both books. That answers: how often does asym beat r50d on terminal
wealth, max drawdown, and CAGR over fixed forward horizons.

Also reports an i.i.d. bootstrap as a stress diagnostic (destroys serial dependence).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from v9_research import (
    INITIAL,
    build_baseline,
    build_variant,
    load_data,
    native,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = REPO_ROOT / "output" / "v9_monte_carlo_results.json"
OUT_SUMMARY = REPO_ROOT / "output" / "v9_monte_carlo_summary.md"

RNG_SEED = 42
N_PATHS = 8_000
BLOCK_LEN = 21  # ~1 trading month; preserves short-run dependence
HORIZONS = {
    "1y": 252,
    "3y": 756,
    "5y": 1_260,
    "10y": 2_520,
}
EVAL_START = "2015-01-01"  # fairer paper-era sample; full sample also reported
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def path_metrics(returns: np.ndarray) -> dict[str, float]:
    equity = np.cumprod(1.0 + returns)
    years = max(len(returns) / 252.0, 1e-9)
    cagr = float(equity[-1] ** (1.0 / years) - 1.0)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    max_dd = float(dd.min())
    return {
        "cagr": cagr,
        "max_dd": max_dd,
        "terminal_multiple": float(equity[-1]),
        "terminal_100k": float(INITIAL * equity[-1]),
        "vol": float(np.std(returns, ddof=1) * np.sqrt(252.0)) if len(returns) > 1 else 0.0,
    }


def stationary_bootstrap_indices(
    n: int,
    horizon: int,
    block_len: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Politis–Romano stationary bootstrap index path of length `horizon`."""
    p = 1.0 / block_len
    out = np.empty(horizon, dtype=np.int64)
    out[0] = rng.integers(0, n)
    for t in range(1, horizon):
        if rng.random() < p:
            out[t] = rng.integers(0, n)
        else:
            out[t] = (out[t - 1] + 1) % n
    return out


def iid_bootstrap_indices(n: int, horizon: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=horizon)


def summarize(samples: np.ndarray) -> dict:
    return native(
        {
            "mean": float(np.mean(samples)),
            "std": float(np.std(samples, ddof=1)),
            "quantiles": {str(q): float(np.quantile(samples, q)) for q in QUANTILES},
        }
    )


def run_monte_carlo(
    r_a: np.ndarray,
    r_b: np.ndarray,
    *,
    name_a: str,
    name_b: str,
    horizon: int,
    n_paths: int,
    block_len: int,
    method: str,
    seed: int,
) -> dict:
    assert len(r_a) == len(r_b)
    n = len(r_a)
    rng = np.random.default_rng(seed)

    cagr_a = np.empty(n_paths)
    cagr_b = np.empty(n_paths)
    dd_a = np.empty(n_paths)
    dd_b = np.empty(n_paths)
    term_a = np.empty(n_paths)
    term_b = np.empty(n_paths)
    beat_term = 0
    beat_cagr = 0
    better_dd = 0  # less deep (higher algebraically)

    for i in range(n_paths):
        if method == "block":
            idx = stationary_bootstrap_indices(n, horizon, block_len, rng)
        else:
            idx = iid_bootstrap_indices(n, horizon, rng)
        ma = path_metrics(r_a[idx])
        mb = path_metrics(r_b[idx])
        cagr_a[i] = ma["cagr"]
        cagr_b[i] = mb["cagr"]
        dd_a[i] = ma["max_dd"]
        dd_b[i] = mb["max_dd"]
        term_a[i] = ma["terminal_100k"]
        term_b[i] = mb["terminal_100k"]
        if mb["terminal_100k"] > ma["terminal_100k"]:
            beat_term += 1
        if mb["cagr"] > ma["cagr"]:
            beat_cagr += 1
        if mb["max_dd"] > ma["max_dd"]:  # e.g. -0.15 > -0.20
            better_dd += 1

    delta_term = term_b - term_a
    delta_cagr = cagr_b - cagr_a
    delta_dd = dd_b - dd_a

    return native(
        {
            "method": method,
            "horizon_sessions": horizon,
            "n_paths": n_paths,
            "block_len": block_len if method == "block" else None,
            "prob_b_beats_a_terminal": beat_term / n_paths,
            "prob_b_beats_a_cagr": beat_cagr / n_paths,
            "prob_b_shallower_max_dd": better_dd / n_paths,
            name_a: {
                "cagr": summarize(cagr_a),
                "max_dd": summarize(dd_a),
                "terminal_100k": summarize(term_a),
            },
            name_b: {
                "cagr": summarize(cagr_b),
                "max_dd": summarize(dd_b),
                "terminal_100k": summarize(term_b),
            },
            "delta_b_minus_a": {
                "cagr": summarize(delta_cagr),
                "max_dd": summarize(delta_dd),
                "terminal_100k": summarize(delta_term),
            },
            # compact histogram edges for canvas (terminal multiples of 100k)
            "hist_terminal_100k": {
                name_a: _hist(term_a),
                name_b: _hist(term_b),
            },
            "hist_max_dd": {
                name_a: _hist(dd_a, bins=40),
                name_b: _hist(dd_b, bins=40),
            },
        }
    )


def _hist(samples: np.ndarray, bins: int = 30) -> dict:
    counts, edges = np.histogram(samples, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "centers": [float(x) for x in centers],
        "counts": [int(c) for c in counts],
    }


def historical_stats(returns: pd.Series) -> dict:
    r = returns.dropna().to_numpy()
    m = path_metrics(r)
    return native(
        {
            "sessions": len(r),
            "start": returns.index[0],
            "end": returns.index[-1],
            **m,
        }
    )


def write_summary(results: dict) -> None:
    block = results["monte_carlo"]["from_2015"]["block"]["5y"]
    lines = [
        "# Monte Carlo: paper R50d vs paper R50d_asym",
        "",
        f"Paths: {N_PATHS:,}. Primary method: stationary block bootstrap "
        f"(block length {BLOCK_LEN} sessions). Seed {RNG_SEED}.",
        "",
        "## Historical sample (from_2015)",
        "",
        f"- R50d: CAGR {results['historical']['from_2015']['R50d']['cagr']:.2%}, "
        f"max DD {results['historical']['from_2015']['R50d']['max_dd']:.2%}",
        f"- R50d_asym: CAGR {results['historical']['from_2015']['R50d_asym']['cagr']:.2%}, "
        f"max DD {results['historical']['from_2015']['R50d_asym']['max_dd']:.2%}",
        "",
        "## 5-year forward block bootstrap (paired)",
        "",
        f"- P(asym terminal > R50d terminal): "
        f"**{block['prob_b_beats_a_terminal']:.1%}**",
        f"- P(asym CAGR > R50d CAGR): "
        f"**{block['prob_b_beats_a_cagr']:.1%}**",
        f"- P(asym max DD shallower): "
        f"**{block['prob_b_shallower_max_dd']:.1%}**",
        f"- Median terminal $100k — R50d: "
        f"${block['R50d']['terminal_100k']['quantiles']['0.5']:,.0f}; "
        f"asym: ${block['R50d_asym']['terminal_100k']['quantiles']['0.5']:,.0f}",
        f"- 5th pct terminal $100k — R50d: "
        f"${block['R50d']['terminal_100k']['quantiles']['0.05']:,.0f}; "
        f"asym: ${block['R50d_asym']['terminal_100k']['quantiles']['0.05']:,.0f}",
        f"- Median max DD — R50d: "
        f"{block['R50d']['max_dd']['quantiles']['0.5']:.2%}; "
        f"asym: {block['R50d_asym']['max_dd']['quantiles']['0.5']:.2%}",
        "",
        "## Interpretation",
        "",
        results["conclusion"]["text"],
        "",
        "## Limitations",
        "",
        "- Bootstrap resamples realized strategy returns; it does not re-simulate "
        "markets or change regime structure beyond historical blocks.",
        "- from_2015 omits pre-2015 stress that still matters for R50d design.",
        "- Paper trading remains the true out-of-sample test.",
    ]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    data = load_data()
    baseline = build_baseline(data)
    asym = build_variant(data, "R50d_asym")

    series = {
        "R50d": baseline["returns"],
        "R50d_asym": asym["returns"],
    }

    windows = {
        "from_2015": pd.Timestamp(EVAL_START),
        "full": pd.Timestamp(data["index"][260]),
    }

    historical = {}
    aligned = {}
    for label, start in windows.items():
        historical[label] = {
            name: historical_stats(ret.loc[start:]) for name, ret in series.items()
        }
        a = series["R50d"].loc[start:].dropna()
        b = series["R50d_asym"].reindex(a.index).fillna(0.0)
        aligned[label] = (
            a.to_numpy(dtype=float),
            b.to_numpy(dtype=float),
        )

    mc: dict = {}
    for label, (ra, rb) in aligned.items():
        mc[label] = {"block": {}, "iid": {}}
        for method in ("block", "iid"):
            for h_name, h_len in HORIZONS.items():
                if h_len > len(ra):
                    continue
                mc[label][method][h_name] = run_monte_carlo(
                    ra,
                    rb,
                    name_a="R50d",
                    name_b="R50d_asym",
                    horizon=h_len,
                    n_paths=N_PATHS,
                    block_len=BLOCK_LEN,
                    method=method,
                    seed=RNG_SEED + (0 if method == "block" else 10_000) + h_len,
                )

    # Decision helper: asym "favored" if 5y block P(beat terminal) > 55% and
    # P(shallower DD) > 50%; otherwise inconclusive / R50d preferred.
    primary = mc["from_2015"]["block"]["5y"]
    p_term = primary["prob_b_beats_a_terminal"]
    p_dd = primary["prob_b_shallower_max_dd"]
    if p_term >= 0.55 and p_dd >= 0.50:
        verdict = "asym_favored"
        text = (
            f"Over simulated 5-year block paths from the 2015+ return sample, "
            f"R50d_asym beat R50d on terminal wealth in {p_term:.1%} of paths and "
            f"had a shallower max drawdown in {p_dd:.1%}. That supports continuing "
            f"the paper A/B, not replacing live R50d yet."
        )
    elif p_term <= 0.45 and p_dd <= 0.50:
        verdict = "r50d_favored"
        text = (
            f"Over simulated 5-year block paths, R50d_asym beat R50d on terminal "
            f"wealth in only {p_term:.1%} of paths. Keep paper_r50d as the core book."
        )
    else:
        verdict = "inconclusive"
        text = (
            f"5-year block results are mixed (P(asym terminal win)={p_term:.1%}, "
            f"P(shallower DD)={p_dd:.1%}). Edge is small relative to sampling noise; "
            f"keep both paper accounts running."
        )

    results = native(
        {
            "meta": {
                "strategies": ["R50d", "R50d_asym"],
                "paper_accounts": ["paper_r50d", "paper_r50d_asym"],
                "n_paths": N_PATHS,
                "block_len": BLOCK_LEN,
                "seed": RNG_SEED,
                "initial_capital": INITIAL,
                "data_end": data["index"][-1],
                "methods": ["stationary_block_bootstrap", "iid_bootstrap"],
            },
            "historical": historical,
            "monte_carlo": mc,
            "conclusion": {"verdict": verdict, "text": text},
        }
    )
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_summary(results)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "p_asym_terminal_5y_block": p_term,
                "p_asym_shallower_dd_5y_block": p_dd,
                "median_terminal_r50d": primary["R50d"]["terminal_100k"]["quantiles"]["0.5"],
                "median_terminal_asym": primary["R50d_asym"]["terminal_100k"]["quantiles"]["0.5"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
