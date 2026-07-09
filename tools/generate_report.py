#!/usr/bin/env python3
"""Generate a polished, interactive HTML dashboard from backtest result JSON.

Handles every result format produced by the backtests (v1, v2, v3, final)
by normalizing the raw JSON into a common shape, then rendering only the
sections for which data exists.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Constants / labels
# ---------------------------------------------------------------------------

WINDOW_RE = re.compile(r"^(full|from_|last_|yr\d|start\d?|since)", re.I)

WINDOW_LABELS = {
    "full": "Full Sample",
    "full_1999": "Full Sample (1999\u2013)",
    "from_2004": "From 2004",
    "from_2007": "From 2007",
    "from_2011": "From 2011",
    "from_2015": "From 2015",
    "yr3": "Last 3 Years",
    "last_3y": "Last 3 Years",
}

CRISIS_LABELS = {
    "dotcom": "Dot-Com Bust (2000\u201303)",
    "dotcom_2000_2003": "Dot-Com Bust (2000\u201303)",
    "gfc": "Global Financial Crisis (2008)",
    "gfc_2008": "Global Financial Crisis (2008)",
    "bear_2022": "2022 Bear Market",
    "y2022": "2022 Bear Market",
    "tariff_2025": "2025 Tariff Shock",
}
CRISIS_KEYS = tuple(CRISIS_LABELS)

STRATEGY_LABELS = {
    "strategy": "Strategy",
    "spy": "SPY (B&H)",
    "qqq": "QQQ (B&H)",
    "tqqq": "TQQQ (B&H)",
    "bh3x": "3x NDX (B&H)",
}

SCALAR_LABELS = {
    "exposure_3y": ("Avg Exposure (3y)", "pct"),
    "expo3y": ("Avg Exposure (3y)", "pct"),
    "expo_full": ("Avg Exposure (Full)", "pct"),
    "pct_days_in_cash_3y": ("Days in Cash (3y)", "pct"),
    "cash_days_3y": ("Days in Cash (3y)", "pct"),
    "trades_per_year_3y": ("Trades / Year (3y)", "num"),
    "trades_yr_3y": ("Trades / Year (3y)", "num"),
    "annual_turnover_3y": ("Turnover / Year (3y)", "num"),
    "turnover_yr_3y": ("Turnover / Year (3y)", "num"),
}

# (key, label, kind, direction)  direction: 1 = higher better, -1 = lower better, 0 = neutral
METRICS = (
    ("cagr", "CAGR", "pct", 1),
    ("vol", "Volatility", "pct", -1),
    ("sharpe", "Sharpe", "num", 1),
    ("sortino", "Sortino", "num", 1),
    ("max_dd", "Max DD", "pct", 1),
    ("calmar", "Calmar", "num", 1),
    ("total", "Total Return", "total", 1),
    ("win_months", "Win Months", "pct0", 0),
    ("best_month", "Best Month", "pct", 0),
    ("worst_month", "Worst Month", "pct", 1),
)

STRAT_COLORS = ("#38bdf8", "#a78bfa", "#34d399", "#f472b6", "#fbbf24", "#22d3ee", "#c084fc")
BENCH_COLORS = ("#8b98ab", "#f97316", "#697586", "#b45309", "#a8a29e")
POS = "#34d399"
NEG = "#f87171"

PLOTLY_CONFIG = {"displaylogo": False, "responsive": True,
                 "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}

SLEEVE_GUESS = {
    3: ("NDX", "TLT", "GLD"),
    5: ("NDX", "EFA", "TLT", "GLD", "DBC"),
    6: ("NDX", "EFA", "TLT", "GLD", "DBC", "BTC"),
}


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _is_metrics(obj: Any) -> bool:
    return isinstance(obj, dict) and "cagr" in obj and "sharpe" in obj


def is_benchmark(name: str) -> bool:
    low = name.lower()
    return low.startswith(("bh_", "bh3", "bench_")) or low in ("spy", "qqq", "tqqq", "bh3x")


def strategy_label(name: str) -> str:
    if name in STRATEGY_LABELS:
        return STRATEGY_LABELS[name]
    low = name.lower()
    if low.startswith("bh_"):
        return f"{name[3:]} (B&H)"
    if low.startswith("bench_"):
        return f"{name[6:]} (B&H)"
    return name


def window_label(name: str) -> str:
    return WINDOW_LABELS.get(name, name.replace("_", " ").title())


def window_sort_key(name: str) -> tuple[int, int]:
    low = name.lower()
    year = re.search(r"(19|20)\d{2}", low)
    if low.startswith("full"):
        return (0, int(year.group()) if year else 0)
    if low.startswith(("last", "yr", "start")):
        return (2, 0)
    return (1, int(year.group()) if year else 9999)


def sorted_strategies(names: list[str]) -> list[str]:
    return sorted(names, key=lambda n: (is_benchmark(n), n.lower()))


def _extract_windows(data: dict[str, Any]) -> dict[str, dict[str, dict]]:
    """Return window -> strategy -> metrics regardless of input orientation."""
    raw = data.get("windows")
    if isinstance(raw, dict) and raw:
        outer = list(raw.keys())
        inner: list[str] = []
        for v in raw.values():
            if isinstance(v, dict):
                inner.extend(v.keys())
        outer_score = sum(bool(WINDOW_RE.match(k)) for k in outer) / max(len(outer), 1)
        inner_score = sum(bool(WINDOW_RE.match(k)) for k in inner) / max(len(inner), 1)
        out: dict[str, dict[str, dict]] = {}
        if outer_score >= inner_score:  # window -> strategy
            for window, block in raw.items():
                if isinstance(block, dict):
                    out[window] = {s: m for s, m in block.items() if _is_metrics(m)}
        else:  # strategy -> window: invert
            for strategy, block in raw.items():
                if isinstance(block, dict):
                    for window, m in block.items():
                        if _is_metrics(m):
                            out.setdefault(window, {})[strategy] = m
        return {w: s for w, s in out.items() if s}

    # v1 / final style: flat top-level blocks
    out = {}
    for key in ("full", "yr3", "start3"):
        block = data.get(key)
        if isinstance(block, dict) and block and all(_is_metrics(v) for v in block.values()):
            out[key] = block
    return out


def _extract_crises(data: dict[str, Any]) -> dict[str, dict[str, dict]]:
    raw = data.get("crises")
    if isinstance(raw, dict) and raw and all(isinstance(v, dict) for v in raw.values()):
        return raw
    out: dict[str, dict[str, dict]] = {}
    for key in CRISIS_KEYS:
        block = data.get(key)
        if isinstance(block, dict) and block and all(_is_metrics(v) for v in block.values()):
            out[key] = block
    return out


def _extract_series_blocks(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Find every {dates: [...], name: [...]} block and split equity / drawdown /
    exposure lines. Log-scale equity (log10 values) is converted to levels."""
    blocks = []
    for key, block in data.items():
        if not isinstance(block, dict) or "dates" not in block:
            continue
        dates = block["dates"]
        if not isinstance(dates, list) or not dates:
            continue
        lines = {k: v for k, v in block.items()
                 if k != "dates" and isinstance(v, list) and len(v) == len(dates)}
        if not lines:
            continue
        is_expo_block = "expo" in key.lower()
        equity: dict[str, list] = {}
        drawdown: dict[str, list] = {}
        exposure: dict[str, list] = {}
        for name, vals in lines.items():
            if name.startswith("dd_"):
                drawdown[name[3:]] = vals
            elif is_expo_block or name in ("exposure", "expo"):
                exposure[name] = vals
            elif name.endswith("_vol") or name == "qqq_vol":
                continue  # auxiliary series, not equity
            else:
                equity[name] = vals
        if equity:
            numeric = [v for vals in equity.values() for v in vals if isinstance(v, (int, float))]
            if numeric and max(numeric) < 12:  # log10-scale equity
                equity = {n: [10 ** v if v is not None else None for v in vals]
                          for n, vals in equity.items()}
        blocks.append({
            "name": key, "dates": dates, "equity": equity,
            "drawdown": drawdown, "exposure": exposure,
            "start": dates[0], "end": dates[-1], "points": len(dates),
        })
    blocks.sort(key=lambda b: b["start"])
    return blocks


def normalize(data: dict[str, Any]) -> dict[str, Any]:
    turnover = {}
    for key in ("turnover", "turnover_yr"):
        if isinstance(data.get(key), dict):
            turnover = {k: float(v) for k, v in data[key].items()
                        if isinstance(v, (int, float))}
            break
    if not turnover and isinstance(data.get("annual_turnover_3y"), (int, float)):
        turnover = {"strategy": float(data["annual_turnover_3y"])}

    scalars = {}
    for key, (label, kind) in SCALAR_LABELS.items():
        if isinstance(data.get(key), (int, float)):
            scalars[label] = (data[key], kind)

    return {
        "windows": _extract_windows(data),
        "crises": _extract_crises(data),
        "yearly": data.get("yearly") if isinstance(data.get("yearly"), dict) else {},
        "monthly": data.get("monthly") if isinstance(data.get("monthly"), dict) else {},
        "series_blocks": _extract_series_blocks(data),
        "turnover": turnover,
        "expo": data.get("expo") if isinstance(data.get("expo"), dict) else {},
        "avg_expo": data.get("avg_expo") if isinstance(data.get("avg_expo"), dict) else {},
        "sleeve_corr": data.get("sleeve_corr") if isinstance(data.get("sleeve_corr"), dict) else {},
        "sensitivity": data.get("sensitivity") if isinstance(data.get("sensitivity"), dict) else {},
        "rejected": data.get("rejected") if isinstance(data.get("rejected"), dict) else {},
        "synthetic": data.get("synthetic_validation") if isinstance(data.get("synthetic_validation"), dict) else {},
        "scalars": scalars,
        "data_start": data.get("data_start"),
        "data_end": data.get("data_end"),
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_pct(v: Any, digits: int = 1) -> str:
    if not isinstance(v, (int, float)):
        return "\u2013"
    return f"{v * 100:,.{digits}f}%"


def fmt_num(v: Any, digits: int = 2) -> str:
    if not isinstance(v, (int, float)):
        return "\u2013"
    return f"{v:,.{digits}f}"


def fmt_total(v: Any) -> str:
    if not isinstance(v, (int, float)):
        return "\u2013"
    if v >= 3:
        return f"{v + 1:,.1f}\u00d7"
    return fmt_pct(v, 0)


def fmt_metric(v: Any, kind: str) -> str:
    if kind == "pct":
        return fmt_pct(v)
    if kind == "pct0":
        return fmt_pct(v, 0)
    if kind == "total":
        return fmt_total(v)
    return fmt_num(v)


def sign_class(v: Any) -> str:
    if not isinstance(v, (int, float)) or v == 0:
        return ""
    return "pos" if v > 0 else "neg"


# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------

def style_fig(fig: go.Figure, height: int = 460) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="'Inter', 'Segoe UI', system-ui, sans-serif", color="#b8c4d4", size=12),
        margin=dict(l=54, r=18, t=18, b=42),
        height=height,
        hoverlabel=dict(bgcolor="#1c2940", bordercolor="#31415c",
                        font=dict(color="#e2e8f0", size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#22304a", zerolinecolor="#31415c", linecolor="#31415c")
    fig.update_yaxes(gridcolor="#22304a", zerolinecolor="#31415c", linecolor="#31415c")
    return fig


def fig_html(fig: go.Figure, height: int = 460) -> str:
    style_fig(fig, height)
    return fig.to_html(full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG,
                       default_width="100%", default_height=height)


def color_map(names: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    si = bi = 0
    for name in names:
        if is_benchmark(name):
            out[name] = BENCH_COLORS[bi % len(BENCH_COLORS)]
            bi += 1
        else:
            out[name] = STRAT_COLORS[si % len(STRAT_COLORS)]
            si += 1
    return out


# ---------------------------------------------------------------------------
# HTML table builder
# ---------------------------------------------------------------------------

def metrics_table_html(strategies: dict[str, dict], turnover: dict[str, float] | None = None) -> str:
    names = sorted_strategies(list(strategies.keys()))
    cols = [m for m in METRICS if any(m[0] in strategies[n] for n in names)]
    show_turnover = bool(turnover) and any(n in turnover for n in names)

    # best value per directional column (across all rows)
    best: dict[str, float] = {}
    for key, _, _, direction in cols:
        if direction == 0:
            continue
        vals = [strategies[n][key] for n in names
                if isinstance(strategies[n].get(key), (int, float))]
        if vals:
            best[key] = max(vals) if direction > 0 else min(vals)

    head = "<th class='name'>Strategy</th>" + "".join(
        f"<th>{label}</th>" for _, label, _, _ in cols)
    if show_turnover:
        head += "<th>Turnover/yr</th>"

    rows = []
    for name in names:
        m = strategies[name]
        row_cls = "bench" if is_benchmark(name) else "strat"
        cells = [f"<td class='name'>{html.escape(strategy_label(name))}</td>"]
        for key, _, kind, direction in cols:
            v = m.get(key)
            classes = []
            if kind in ("pct", "num", "total"):
                classes.append(sign_class(v))
            if direction != 0 and isinstance(v, (int, float)) and key in best and v == best[key]:
                classes.append("best")
            cls = f" class='{' '.join(c for c in classes if c)}'" if any(classes) else ""
            cells.append(f"<td{cls}>{fmt_metric(v, kind)}</td>")
        if show_turnover:
            t = turnover.get(name) if turnover else None
            cells.append(f"<td>{fmt_num(t, 1) if t is not None else '\u2013'}</td>")
        rows.append(f"<tr class='{row_cls}'>{''.join(cells)}</tr>")

    return (f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


# ---------------------------------------------------------------------------
# Section builders — each returns inner HTML or "" to skip
# ---------------------------------------------------------------------------

def main_window(norm: dict[str, Any]) -> str | None:
    windows = norm["windows"]
    if not windows:
        return None
    return min(windows.keys(), key=window_sort_key)


def primary_strategy(strategies: dict[str, dict]) -> str | None:
    candidates = [n for n in strategies if not is_benchmark(n)] or list(strategies)
    if not candidates:
        return None
    return max(candidates, key=lambda n: strategies[n].get("sharpe") or float("-inf"))


def build_overview(norm: dict[str, Any]) -> str:
    parts = []
    window = main_window(norm)
    if window:
        strategies = norm["windows"][window]
        primary = primary_strategy(strategies)
        if primary:
            m = strategies[primary]
            cards = []
            for key, label, kind, _ in METRICS:
                if key in ("sortino", "total", "win_months", "best_month"):
                    continue
                if key not in m:
                    continue
                cls = sign_class(m.get(key)) if kind != "pct0" else ""
                cards.append(
                    f"<div class='kpi'><div class='kpi-value {cls}'>"
                    f"{fmt_metric(m.get(key), kind)}</div>"
                    f"<div class='kpi-label'>{label}</div></div>")
            parts.append(
                f"<p class='kpi-caption'>{html.escape(strategy_label(primary))} "
                f"&middot; {html.escape(window_label(window))}</p>"
                f"<div class='kpi-grid'>{''.join(cards)}</div>")

    chips = []
    for label, (value, kind) in norm["scalars"].items():
        text = fmt_pct(value) if kind == "pct" else fmt_num(value, 1)
        chips.append(f"<div class='chip'><span>{label}</span><strong>{text}</strong></div>")
    syn = norm["synthetic"]
    if syn:
        for key, label, kind in (
            ("corr_daily", "Synthetic vs Real Corr (daily)", "num"),
            ("cagr_synthetic_overlap", "Synthetic CAGR (overlap)", "pct"),
            ("cagr_real_overlap", "Real CAGR (overlap)", "pct"),
        ):
            if key in syn:
                text = fmt_pct(syn[key]) if kind == "pct" else fmt_num(syn[key], 4)
                chips.append(f"<div class='chip'><span>{label}</span><strong>{text}</strong></div>")
    if chips:
        parts.append(f"<div class='chip-row'>{''.join(chips)}</div>")

    return "".join(parts)


def _equity_fig(block: dict[str, Any], selector: bool = True) -> go.Figure:
    names = sorted_strategies(list(block["equity"].keys()))
    colors = color_map(names)
    fig = go.Figure()
    for name in names:
        bench = is_benchmark(name)
        fig.add_trace(go.Scatter(
            x=block["dates"], y=block["equity"][name], mode="lines",
            name=strategy_label(name),
            line=dict(color=colors[name], width=1.6 if bench else 2.4,
                      dash="dot" if bench else "solid"),
            hovertemplate="%{y:,.0f}<extra>" + strategy_label(name) + "</extra>",
        ))
    fig.update_yaxes(type="log", title="Growth of $100 (log scale)")
    if selector:
        fig.update_xaxes(
            rangeslider=dict(visible=True, thickness=0.06, bgcolor="#16233a"),
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=3, label="3Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(count=10, label="10Y", step="year", stepmode="backward"),
                    dict(step="all", label="All"),
                ],
                bgcolor="#1c2940", activecolor="#2d4368", font=dict(color="#dbe4f0"),
            ),
        )
    return fig


def build_equity(norm: dict[str, Any]) -> str:
    blocks = [b for b in norm["series_blocks"] if b["equity"]]
    if not blocks:
        return ""
    blocks = sorted(blocks, key=lambda b: b["start"])
    main = blocks[0]

    parts = [f"<h3>{main['start']} \u2192 {main['end']}</h3>",
             fig_html(_equity_fig(main), 520)]

    # extra charts only when they add resolution or coverage beyond the main one
    def res(b: dict) -> float:
        span = max((_days_between(b["start"], b["end"])), 1)
        return b["points"] / span

    for b in blocks[1:]:
        if b["start"] <= main["start"] and b["end"] >= main["end"]:
            continue
        if res(b) > 2 * res(main):
            parts.append(f"<h3>Detail: {b['start']} \u2192 {b['end']}</h3>")
            parts.append(fig_html(_equity_fig(b, selector=False), 420))
    return "".join(parts)


def _days_between(a: str, b: str) -> int:
    try:
        return (datetime.fromisoformat(b) - datetime.fromisoformat(a)).days
    except ValueError:
        return 1


def _compute_drawdown(values: list) -> list[float | None]:
    out: list[float | None] = []
    peak = float("-inf")
    for v in values:
        if not isinstance(v, (int, float)):
            out.append(None)
            continue
        peak = max(peak, v)
        out.append((v / peak - 1.0) * 100 if peak > 0 else None)
    return out


def build_drawdown(norm: dict[str, Any]) -> str:
    blocks = norm["series_blocks"]
    dd_block = next((b for b in blocks if b["drawdown"]), None)
    if dd_block is not None:
        dates = dd_block["dates"]
        series = dict(dd_block["drawdown"])
        # values may be fractions or percents
        flat = [v for vals in series.values() for v in vals if isinstance(v, (int, float))]
        if flat and min(flat) > -1.5:
            series = {n: [v * 100 if isinstance(v, (int, float)) else None for v in vals]
                      for n, vals in series.items()}
    else:
        eq_block = next((b for b in sorted(blocks, key=lambda b: b["start"]) if b["equity"]), None)
        if eq_block is None:
            return ""
        dates = eq_block["dates"]
        series = {n: _compute_drawdown(vals) for n, vals in eq_block["equity"].items()}

    names = sorted_strategies(list(series.keys()))
    colors = color_map(names)
    fig = go.Figure()
    for name in names:
        bench = is_benchmark(name)
        fig.add_trace(go.Scatter(
            x=dates, y=series[name], mode="lines", name=strategy_label(name),
            line=dict(color=colors[name], width=1.4 if bench else 2,
                      dash="dot" if bench else "solid"),
            fill=None if bench else "tozeroy",
            fillcolor=None if bench else _hex_alpha(colors[name], 0.12),
            hovertemplate="%{y:.1f}%<extra>" + strategy_label(name) + "</extra>",
        ))
    fig.update_yaxes(title="Drawdown (%)", ticksuffix="%")
    return fig_html(fig, 380)


def _hex_alpha(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def build_windows(norm: dict[str, Any]) -> str:
    windows = norm["windows"]
    if not windows:
        return ""
    parts = []
    for window in sorted(windows.keys(), key=window_sort_key):
        parts.append(f"<h3>{html.escape(window_label(window))}</h3>")
        parts.append(metrics_table_html(windows[window]))
    return "".join(parts)


def build_crises(norm: dict[str, Any]) -> str:
    crises = norm["crises"]
    if not crises:
        return ""
    parts = []

    all_names: list[str] = []
    for block in crises.values():
        for n in block:
            if n not in all_names:
                all_names.append(n)
    all_names = sorted_strategies(all_names)
    colors = color_map(all_names)
    labels = [CRISIS_LABELS.get(c, c.replace("_", " ").title()) for c in crises]

    fig = go.Figure()
    for name in all_names:
        fig.add_trace(go.Bar(
            x=labels,
            y=[(crises[c].get(name, {}).get("max_dd") or 0) * 100 for c in crises],
            name=strategy_label(name), marker_color=colors[name],
            hovertemplate="%{y:.1f}%<extra>" + strategy_label(name) + "</extra>",
        ))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Max Drawdown (%)", ticksuffix="%")
    parts.append(fig_html(fig, 380))

    for crisis, block in crises.items():
        parts.append(f"<h3>{html.escape(CRISIS_LABELS.get(crisis, crisis.replace('_', ' ').title()))}</h3>")
        parts.append(metrics_table_html(block))
    return "".join(parts)


def build_yearly(norm: dict[str, Any]) -> str:
    yearly = norm["yearly"]
    if not yearly:
        return ""
    names = sorted_strategies(list(yearly.keys()))
    colors = color_map(names)
    years = sorted({y for per in yearly.values() if isinstance(per, dict) for y in per})
    fig = go.Figure()
    for name in names:
        per = yearly.get(name) or {}
        fig.add_trace(go.Bar(
            x=years,
            y=[(per.get(y) or 0) * 100 for y in years],
            name=strategy_label(name), marker_color=colors[name],
            hovertemplate="%{y:.1f}%<extra>" + strategy_label(name) + "</extra>",
        ))
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_yaxes(title="Return (%)", ticksuffix="%")
    fig.add_hline(y=0, line_width=1, line_color="#4a5c78")
    return fig_html(fig, 420)


def build_monthly(norm: dict[str, Any]) -> str:
    """Monthly returns heatmap (strategy = first element of each month's list)."""
    monthly = norm["monthly"]
    if not monthly:
        return ""
    points: dict[tuple[int, int], float] = {}
    for month_key, vals in monthly.items():
        m = re.match(r"^(\d{4})-(\d{2})$", str(month_key))
        if not m or not isinstance(vals, list) or not vals:
            continue
        if isinstance(vals[0], (int, float)):
            points[(int(m.group(1)), int(m.group(2)))] = float(vals[0])
    if not points:
        return ""
    years = sorted({y for y, _ in points}, reverse=True)
    z = [[points.get((y, mo)) for mo in range(1, 13)] for y in years]
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    zmax = max(abs(v) for v in points.values()) or 1
    fig = go.Figure(go.Heatmap(
        z=z, x=month_names, y=[str(y) for y in years],
        colorscale=[[0, "#ef4444"], [0.5, "#16233a"], [1, "#22c55e"]],
        zmid=0, zmin=-zmax, zmax=zmax,
        texttemplate="%{z:.1f}", textfont=dict(size=11),
        hovertemplate="%{y} %{x}: %{z:.1f}%<extra></extra>",
        colorbar=dict(ticksuffix="%", outlinewidth=0),
        hoverongaps=False,
    ))
    fig.update_yaxes(autorange="reversed")
    return fig_html(fig, 90 + 42 * len(years))


def build_exposure(norm: dict[str, Any]) -> str:
    parts = []

    # time-series exposure (v1/final: single line; v2: stacked sleeves)
    for block in norm["series_blocks"]:
        if not block["exposure"]:
            continue
        names = list(block["exposure"].keys())
        flat = [v for vals in block["exposure"].values()
                for v in vals if isinstance(v, (int, float))]
        scale = 1 if (flat and max(flat) > 1.5) else 100
        fig = go.Figure()
        stacked = len(names) > 1
        for i, name in enumerate(names):
            vals = [v * scale if isinstance(v, (int, float)) else None
                    for v in block["exposure"][name]]
            color = STRAT_COLORS[i % len(STRAT_COLORS)]
            fig.add_trace(go.Scatter(
                x=block["dates"], y=vals, mode="lines", name=name.upper(),
                line=dict(color=color, width=1.2),
                stackgroup="one" if stacked else None,
                fill=None if stacked else "tozeroy",
                fillcolor=None if stacked else _hex_alpha(color, 0.15),
                hovertemplate="%{y:.0f}%<extra>" + name.upper() + "</extra>",
            ))
        fig.update_yaxes(title="Exposure (%)", ticksuffix="%", rangemode="tozero")
        parts.append(f"<h3>Exposure over time ({block['start']} \u2192 {block['end']})</h3>")
        parts.append(fig_html(fig, 320))

    # static average allocation per strategy (v3 `expo`)
    expo = norm["expo"]
    alloc_rows = {name: weights for name, weights in expo.items()
                  if isinstance(weights, list) and weights}
    if alloc_rows:
        strategies = list(alloc_rows.keys())
        max_len = max(len(w) for w in alloc_rows.values())
        sleeve_names = list(SLEEVE_GUESS.get(max_len, tuple(f"Sleeve {i + 1}" for i in range(max_len))))
        fig = go.Figure()
        palette = STRAT_COLORS + BENCH_COLORS
        for i, sleeve in enumerate(sleeve_names):
            ys, xs = [], []
            for s in strategies:
                w = alloc_rows[s]
                guess = SLEEVE_GUESS.get(len(w), tuple(f"Sleeve {j + 1}" for j in range(len(w))))
                if sleeve in guess:
                    xs.append(w[guess.index(sleeve)] * 100)
                else:
                    xs.append(0)
                ys.append(strategy_label(s))
            fig.add_trace(go.Bar(
                y=ys, x=xs, name=sleeve, orientation="h",
                marker_color=palette[i % len(palette)],
                hovertemplate="%{x:.1f}%<extra>" + sleeve + "</extra>",
            ))
        cash = [max(0.0, 1 - sum(alloc_rows[s])) * 100 for s in strategies]
        if any(c > 0.5 for c in cash):
            fig.add_trace(go.Bar(
                y=[strategy_label(s) for s in strategies], x=cash, name="Cash",
                orientation="h", marker_color="#3b4a63",
                hovertemplate="%{x:.1f}%<extra>Cash</extra>",
            ))
        fig.update_layout(barmode="stack")
        fig.update_xaxes(title="Average allocation (%)", ticksuffix="%")
        parts.append("<h3>Average allocation by strategy</h3>")
        parts.append(fig_html(fig, 120 + 44 * len(strategies)))

    # v2 avg_expo scalar map
    avg_expo = norm["avg_expo"]
    if avg_expo and not alloc_rows:
        chips = "".join(
            f"<div class='chip'><span>{html.escape(k.replace('_', ' ').upper())}</span>"
            f"<strong>{fmt_pct(v)}</strong></div>"
            for k, v in avg_expo.items() if isinstance(v, (int, float)))
        parts.append(f"<h3>Average exposure</h3><div class='chip-row'>{chips}</div>")

    return "".join(parts)


def build_correlation(norm: dict[str, Any]) -> str:
    corr = norm["sleeve_corr"]
    if not corr:
        return ""
    names = list(corr.keys())
    z = [[corr[r].get(c) for c in names] for r in names]
    fig = go.Figure(go.Heatmap(
        z=z, x=names, y=names,
        colorscale=[[0, "#3b82f6"], [0.5, "#16233a"], [1, "#ef4444"]],
        zmin=-1, zmax=1,
        texttemplate="%{z:.2f}", textfont=dict(size=12),
        hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
        colorbar=dict(outlinewidth=0),
    ))
    fig.update_yaxes(autorange="reversed")
    return fig_html(fig, 90 + 52 * len(names))


def build_sensitivity(norm: dict[str, Any]) -> str:
    sens = norm["sensitivity"]
    if not sens:
        return ""
    parts = []

    # heatmap of full-sample Calmar over the vol-target x hysteresis grid
    grid: dict[tuple[int, int], float] = {}
    for key, m in sens.items():
        match = re.match(r"^vt(\d+)_hyst(\d+)$", key)
        if match and isinstance(m, dict) and isinstance(m.get("calmarF"), (int, float)):
            grid[(int(match.group(1)), int(match.group(2)))] = m["calmarF"]
    if len(grid) >= 4:
        vts = sorted({vt for vt, _ in grid})
        hysts = sorted({h for _, h in grid})
        z = [[grid.get((vt, h)) for vt in vts] for h in hysts]
        fig = go.Figure(go.Heatmap(
            z=z, x=[f"VT {vt}%" for vt in vts], y=[f"Hyst {h}" for h in hysts],
            colorscale=[[0, "#16233a"], [1, "#22c55e"]],
            texttemplate="%{z:.2f}", textfont=dict(size=13),
            hovertemplate="%{x}, %{y}: Calmar %{z:.2f}<extra></extra>",
            colorbar=dict(outlinewidth=0),
        ))
        parts.append("<h3>Full-sample Calmar across parameter grid</h3>")
        parts.append(fig_html(fig, 110 + 60 * len(hysts)))

    cols = (("cagr3", "CAGR (3y)", "pct"), ("mdd3", "Max DD (3y)", "pct"),
            ("cagrF", "CAGR (full)", "pct"), ("mddF", "Max DD (full)", "pct"),
            ("calmarF", "Calmar (full)", "num"))
    head = "<th class='name'>Variant</th>" + "".join(f"<th>{label}</th>" for _, label, _ in cols)
    rows = []
    for key, m in sens.items():
        if not isinstance(m, dict):
            continue
        cells = [f"<td class='name'>{html.escape(key)}</td>"]
        for ckey, _, kind in cols:
            v = m.get(ckey)
            cls = sign_class(v)
            cells.append(f"<td class='{cls}'>{fmt_metric(v, kind)}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    parts.append("<h3>All sensitivity runs</h3>")
    parts.append(f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead>"
                 f"<tbody>{''.join(rows)}</tbody></table></div>")
    return "".join(parts)


def build_ladder(norm: dict[str, Any]) -> str:
    window = main_window(norm)
    if not window:
        return ""
    strategies = norm["windows"][window]
    if len(strategies) < 2:
        return ""
    turnover = norm["turnover"]

    ranked = sorted(strategies.items(),
                    key=lambda kv: -(kv[1].get("sharpe") or float("-inf")))
    head = ("<th>#</th><th class='name'>Strategy</th><th>CAGR</th><th>Sharpe</th>"
            "<th>Max DD</th><th>Calmar</th><th>Turnover/yr</th>")
    rows = []
    for i, (name, m) in enumerate(ranked, 1):
        t = turnover.get(name)
        row_cls = "bench" if is_benchmark(name) else "strat"
        rows.append(
            f"<tr class='{row_cls}'><td>{i}</td>"
            f"<td class='name'>{html.escape(strategy_label(name))}</td>"
            f"<td class='{sign_class(m.get('cagr'))}'>{fmt_pct(m.get('cagr'))}</td>"
            f"<td class='{sign_class(m.get('sharpe'))}'>{fmt_num(m.get('sharpe'))}</td>"
            f"<td class='{sign_class(m.get('max_dd'))}'>{fmt_pct(m.get('max_dd'))}</td>"
            f"<td class='{sign_class(m.get('calmar'))}'>{fmt_num(m.get('calmar'))}</td>"
            f"<td>{fmt_num(t, 1) if t is not None else '\u2013'}</td></tr>")
    return (f"<p class='section-note'>Ranked by Sharpe &middot; {html.escape(window_label(window))}</p>"
            f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def build_rejected(norm: dict[str, Any]) -> str:
    rejected = norm["rejected"]
    if not rejected:
        return ""
    flat: dict[str, dict] = {}
    for name, block in rejected.items():
        if _is_metrics(block):
            flat[name] = block
        elif isinstance(block, dict):
            for window, m in block.items():
                if _is_metrics(m):
                    flat[f"{name} \u00b7 {window_label(window)}"] = m
    if not flat:
        return ""
    return metrics_table_html(flat)


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------

SECTIONS = (
    ("overview", "Overview", build_overview),
    ("equity", "Equity Curve", build_equity),
    ("drawdown", "Drawdown", build_drawdown),
    ("windows", "Performance by Window", build_windows),
    ("crises", "Crisis Stress Tests", build_crises),
    ("yearly", "Yearly Returns", build_yearly),
    ("monthly", "Monthly Returns", build_monthly),
    ("exposure", "Exposure & Allocation", build_exposure),
    ("correlation", "Sleeve Correlation", build_correlation),
    ("sensitivity", "Parameter Sensitivity", build_sensitivity),
    ("ladder", "Variant Ladder", build_ladder),
    ("rejected", "Rejected Variants", build_rejected),
)

CSS = """
:root {
  --bg: #0b1220; --panel: #111a2c; --panel-border: #1e2c45;
  --text: #dbe4f0; --muted: #8595ab; --accent: #38bdf8;
  --pos: #34d399; --neg: #f87171;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; scroll-padding-top: 76px; }
body {
  margin: 0; background: var(--bg); color: var(--text);
  font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
  font-size: 14px; line-height: 1.55;
}
.topbar {
  position: sticky; top: 0; z-index: 50;
  background: rgba(11, 18, 32, 0.92); backdrop-filter: blur(8px);
  border-bottom: 1px solid var(--panel-border);
  padding: 14px 28px 0;
}
.topbar-row { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; }
.topbar h1 { margin: 0; font-size: 1.25rem; font-weight: 700; letter-spacing: -0.01em; }
.topbar .meta { color: var(--muted); font-size: 0.8rem; }
nav { display: flex; gap: 2px; overflow-x: auto; padding: 8px 0 0; }
nav a {
  color: var(--muted); text-decoration: none; font-size: 0.8rem; font-weight: 600;
  padding: 6px 12px 10px; border-bottom: 2px solid transparent; white-space: nowrap;
}
nav a:hover { color: var(--text); }
main { max-width: 1200px; margin: 0 auto; padding: 28px 28px 60px; }
section {
  background: var(--panel); border: 1px solid var(--panel-border);
  border-radius: 12px; padding: 22px 24px; margin-bottom: 22px;
}
section > h2 {
  margin: 0 0 14px; font-size: 1.02rem; font-weight: 700;
  letter-spacing: 0.01em; color: #eef3fa;
}
section h3 {
  margin: 20px 0 8px; font-size: 0.85rem; font-weight: 600;
  color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em;
}
section h3:first-of-type { margin-top: 4px; }
.section-note { color: var(--muted); font-size: 0.8rem; margin: 0 0 10px; }
.kpi-caption { color: var(--muted); font-size: 0.82rem; margin: 0 0 12px; }
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; }
.kpi {
  background: #16233a; border: 1px solid #22304a; border-radius: 10px;
  padding: 14px 16px;
}
.kpi-value { font-size: 1.45rem; font-weight: 700; letter-spacing: -0.02em; }
.kpi-label { color: var(--muted); font-size: 0.74rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.07em; margin-top: 4px; }
.chip-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }
.chip {
  display: flex; gap: 8px; align-items: baseline;
  background: #16233a; border: 1px solid #22304a; border-radius: 999px;
  padding: 6px 14px; font-size: 0.8rem;
}
.chip span { color: var(--muted); }
.chip strong { color: var(--text); }
.table-wrap { overflow-x: auto; border: 1px solid #22304a; border-radius: 10px; }
table { border-collapse: collapse; width: 100%; font-size: 0.84rem; }
thead th {
  background: #16233a; color: var(--muted); font-weight: 600; text-align: right;
  padding: 10px 14px; text-transform: uppercase; font-size: 0.7rem;
  letter-spacing: 0.06em; white-space: nowrap;
}
thead th.name { text-align: left; }
tbody td {
  padding: 9px 14px; text-align: right; border-top: 1px solid #1c2940;
  font-variant-numeric: tabular-nums; white-space: nowrap;
}
tbody td.name { text-align: left; font-weight: 600; }
tbody tr:hover { background: #16233a; }
tr.strat td.name { color: var(--accent); }
tr.bench td.name { color: var(--muted); font-weight: 500; }
td.pos { color: var(--pos); }
td.neg { color: var(--neg); }
td.best { font-weight: 700; }
td.best::after { content: ' \\2605'; font-size: 0.7em; color: #fbbf24; vertical-align: super; }
footer { text-align: center; color: var(--muted); font-size: 0.75rem; padding: 10px 0 30px; }
.js-plotly-plot { width: 100% !important; }
"""


def build_report(data: dict[str, Any], source: str) -> str:
    norm = normalize(data)

    rendered: list[tuple[str, str, str]] = []
    for anchor, title, builder in SECTIONS:
        body = builder(norm)
        if body:
            rendered.append((anchor, title, body))

    nav = "".join(f"<a href='#{anchor}'>{title}</a>" for anchor, title, _ in rendered)
    sections_html = "".join(
        f"<section id='{anchor}'><h2>{title}</h2>{body}</section>"
        for anchor, title, body in rendered)

    title = f"Backtest Report \u2014 {Path(source).stem}"
    meta_bits = []
    if norm["data_start"] and norm["data_end"]:
        meta_bits.append(f"{norm['data_start']} \u2192 {norm['data_end']}")
    meta_bits.append(html.escape(Path(source).name))
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(title)}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{CSS}</style>
</head>
<body>
<header class="topbar">
  <div class="topbar-row">
    <h1>{html.escape(title)}</h1>
    <span class="meta">{' &middot; '.join(meta_bits)}</span>
  </div>
  <nav>{nav}</nav>
</header>
<main>
{sections_html}
</main>
<footer>Generated {generated}</footer>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an interactive HTML report from backtest result JSON.")
    parser.add_argument("input", type=Path, help="Path to result JSON file")
    parser.add_argument("-o", "--output", type=Path,
                        help="Output HTML path (default: <input_stem>_report.html)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = args.input.resolve()
    if not input_path.is_file():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        return 1

    output_path = args.output or input_path.with_name(f"{input_path.stem}_report.html")

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    html_out = build_report(data, str(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_out, encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
