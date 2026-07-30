"""V8: independent alpha ensemble for a retail cash account.

Rules are fixed in DESIGN.md. The strategy combines an always-on equity core,
cross-asset relative momentum, and a short-horizon panic-reversal sleeve.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = REPO_ROOT / "output" / "v8_ensemble_results.json"
OUT_SUMMARY = REPO_ROOT / "output" / "v8_ensemble_summary.md"

START = "1999-03-10"
TICKERS = [
    "SPY", "QQQ", "IWM", "EFA", "EEM", "IEF", "TLT", "GLD", "DBC",
    "DBMF", "TQQQ", "^IRX",
]
UNDERLYINGS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "IEF", "TLT", "GLD", "DBC", "DBMF"]
MOMENTUM_UNIVERSE = ["QQQ", "IWM", "EFA", "EEM", "IEF", "TLT", "GLD", "DBC", "DBMF"]
TRADE_ASSETS = [*UNDERLYINGS, "TQQQ"]

INITIAL_CAPITAL = 100_000.0
CORE_SPY = 0.25
CORE_QQQ = 0.25
MOMENTUM_ALLOCATION = 0.30
REVERSAL_ALLOCATION = 0.20
PORTFOLIO_VOL_TARGET = 0.18
MAX_SCALE = 2.0
MAX_SINGLE_WEIGHT = 0.35
MAX_NON_QQQ = 0.75
MAX_TQQQ = 0.25
VOL_SPAN = 63
TRADE_BAND = 0.05
BASE_COST = 0.001
STRESS_COST = 0.002
SYNTH_SPREAD = 0.005 / 252.0
SYNTH_EXPENSE = 0.0095 / 252.0

CRISES = {
    "dotcom": ("2000-03-24", "2002-10-09"),
    "gfc": ("2007-10-09", "2009-03-09"),
    "covid_crash": ("2020-02-19", "2020-03-23"),
    "inflation_bear_2022": ("2022-01-03", "2022-10-12"),
    "tariff_2025": ("2025-02-15", "2025-06-30"),
}


def native(value):
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value).date())
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    return value


def annualized_vol(returns: pd.Series) -> pd.Series:
    return np.sqrt(
        252.0
        * returns.pow(2).ewm(
            alpha=2.0 / (VOL_SPAN + 1.0),
            adjust=False,
            min_periods=20,
        ).mean()
    )


def last_session_flags(index: pd.DatetimeIndex, frequency: str) -> np.ndarray:
    periods = index.to_period(frequency)
    flags = np.zeros(len(index), dtype=bool)
    flags[:-1] = periods[:-1] != periods[1:]
    flags[-1] = (
        index[-1].is_month_end if frequency == "M" else index[-1].weekday() == 4
    )
    return flags


def load_data() -> dict:
    downloaded = yf.download(
        TICKERS,
        start=START,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if downloaded.empty:
        raise RuntimeError("Yahoo Finance returned no data")
    close = downloaded["Close"] if isinstance(downloaded.columns, pd.MultiIndex) else downloaded
    missing = [ticker for ticker in TICKERS if ticker not in close.columns]
    if missing:
        raise RuntimeError(f"Yahoo Finance omitted required tickers: {missing}")

    index = close["QQQ"].dropna().index
    prices = close.reindex(index)
    rf_daily = prices["^IRX"].ffill().div(100.0).fillna(0.0).div(252.0)

    actual_returns = prices[UNDERLYINGS].pct_change(fill_method=None)
    qqq_return = actual_returns["QQQ"]
    synthetic_tqqq = (
        3.0 * qqq_return
        - 2.0 * (rf_daily + SYNTH_SPREAD)
        - SYNTH_EXPENSE
    )
    actual_tqqq = prices["TQQQ"].pct_change(fill_method=None)
    tqqq_return = synthetic_tqqq.copy()
    tqqq_return.loc[actual_tqqq.notna()] = actual_tqqq.loc[actual_tqqq.notna()]
    actual_returns["TQQQ"] = tqqq_return

    volatility = actual_returns[UNDERLYINGS].apply(annualized_vol)
    inception = {
        ticker: (
            str(prices[ticker].dropna().index[0].date())
            if not prices[ticker].dropna().empty
            else None
        )
        for ticker in TICKERS
    }
    return {
        "index": index,
        "prices": prices,
        "actual_returns": actual_returns,
        "pnl_returns": actual_returns[TRADE_ASSETS].fillna(0.0),
        "volatility": volatility,
        "rf_daily": rf_daily,
        "synthetic_tqqq": synthetic_tqqq,
        "actual_tqqq": actual_tqqq,
        "inception": inception,
    }


def component_targets(
    data: dict,
    include_core: bool,
    include_momentum: bool,
    include_reversal: bool,
) -> tuple[pd.DataFrame, dict]:
    index = data["index"]
    prices = data["prices"]
    vol = data["volatility"]
    components = pd.DataFrame(0.0, index=index, columns=UNDERLYINGS)

    if include_core:
        components["SPY"] += CORE_SPY
        components["QQQ"] += CORE_QQQ

    momentum_positions = pd.DataFrame(0.0, index=index, columns=UNDERLYINGS)
    if include_momentum:
        month_end = last_session_flags(index, "M")
        held = {ticker: 0.0 for ticker in MOMENTUM_UNIVERSE}
        for i in range(len(index)):
            if month_end[i]:
                candidates = []
                for ticker in MOMENTUM_UNIVERSE:
                    px = prices[ticker]
                    if px.iloc[: i + 1].notna().sum() < 253:
                        continue
                    ret_126 = px.iloc[i] / px.iloc[i - 126] - 1.0
                    ret_252 = px.iloc[i] / px.iloc[i - 252] - 1.0
                    sigma = vol[ticker].iloc[i]
                    if (
                        pd.notna(ret_126)
                        and pd.notna(ret_252)
                        and ret_126 > 0
                        and ret_252 > 0
                        and pd.notna(sigma)
                        and sigma > 0
                    ):
                        candidates.append((ticker, (ret_126 + ret_252) / 2.0, sigma))
                selected = sorted(candidates, key=lambda item: item[1], reverse=True)[:3]
                held = {ticker: 0.0 for ticker in MOMENTUM_UNIVERSE}
                if selected:
                    inverse_vol = np.array([1.0 / item[2] for item in selected])
                    weights = MOMENTUM_ALLOCATION * inverse_vol / inverse_vol.sum()
                    for (ticker, _, _), weight in zip(selected, weights):
                        held[ticker] = float(weight)
            for ticker in MOMENTUM_UNIVERSE:
                momentum_positions.iloc[i, momentum_positions.columns.get_loc(ticker)] = held[ticker]
        components += momentum_positions

    reversal = pd.Series(0.0, index=index)
    reversal_entries = 0
    if include_reversal:
        qqq = prices["QQQ"]
        ret_5 = qqq.div(qqq.shift(5)).sub(1.0)
        daily_vol_20 = qqq.pct_change(fill_method=None).rolling(20).std()
        sma_20 = qqq.rolling(20).mean()
        sma_200 = qqq.rolling(200).mean()
        active = False
        days_held = 0
        for i in range(len(index)):
            exited = False
            if active and (days_held >= 5 or qqq.iloc[i] >= sma_20.iloc[i]):
                active = False
                days_held = 0
                exited = True
            threshold = -2.0 * daily_vol_20.iloc[i] * np.sqrt(5.0)
            if (
                not active
                and not exited
                and pd.notna(threshold)
                and pd.notna(sma_200.iloc[i])
                and ret_5.iloc[i] <= threshold
                and qqq.iloc[i] > sma_200.iloc[i]
            ):
                active = True
                days_held = 0
                reversal_entries += 1
            if active:
                reversal.iloc[i] = REVERSAL_ALLOCATION
                days_held += 1
        components["QQQ"] += reversal

    diagnostics = {
        "momentum_positions": momentum_positions,
        "reversal_signal": reversal,
        "reversal_entries": reversal_entries,
    }
    return components, diagnostics


def constrained_weights(raw: pd.Series, covariance: pd.DataFrame) -> tuple[dict, float]:
    active = raw[raw > 0]
    if active.empty:
        return {ticker: 0.0 for ticker in TRADE_ASSETS}, 0.0
    cov = covariance.loc[active.index, active.index].to_numpy()
    raw_values = active.to_numpy()
    raw_vol = float(np.sqrt(max(raw_values @ cov @ raw_values, 0.0)))
    scale = min(MAX_SCALE, PORTFOLIO_VOL_TARGET / raw_vol) if raw_vol > 0 else 0.0
    desired = raw.mul(scale)

    final = {ticker: 0.0 for ticker in TRADE_ASSETS}
    non_qqq = [ticker for ticker in UNDERLYINGS if ticker != "QQQ"]
    for ticker in non_qqq:
        final[ticker] = min(MAX_SINGLE_WEIGHT, max(0.0, float(desired[ticker])))

    non_qqq_total = sum(final[ticker] for ticker in non_qqq)
    if non_qqq_total > MAX_NON_QQQ:
        reduction = MAX_NON_QQQ / non_qqq_total
        for ticker in non_qqq:
            final[ticker] *= reduction
        non_qqq_total = MAX_NON_QQQ

    cash_capacity = 1.0 - non_qqq_total
    qqq_equivalent = max(0.0, float(desired["QQQ"]))
    max_qqq_equivalent = cash_capacity + 2.0 * min(MAX_TQQQ, cash_capacity)
    qqq_equivalent = min(qqq_equivalent, max_qqq_equivalent)
    if qqq_equivalent <= cash_capacity:
        final["QQQ"] = qqq_equivalent
    else:
        final["TQQQ"] = min(MAX_TQQQ, (qqq_equivalent - cash_capacity) / 2.0)
        final["QQQ"] = qqq_equivalent - 3.0 * final["TQQQ"]

    total = sum(final.values())
    if total > 1.0 + 1e-10:
        raise AssertionError(f"capital constraint violated: {total}")
    return final, scale


def build_positions(
    data: dict,
    include_core: bool = True,
    include_momentum: bool = True,
    include_reversal: bool = True,
) -> dict:
    raw, diagnostics = component_targets(
        data,
        include_core,
        include_momentum,
        include_reversal,
    )
    returns = data["actual_returns"][UNDERLYINGS]
    index = data["index"]
    targets = pd.DataFrame(0.0, index=index, columns=TRADE_ASSETS)
    scales = pd.Series(0.0, index=index)

    for i in range(62, len(index)):
        active = raw.iloc[i][raw.iloc[i] > 0].index.tolist()
        if not active:
            continue
        window = returns[active].iloc[i - 62 : i + 1].dropna()
        if len(window) < 40:
            continue
        covariance = window.cov() * 252.0
        weights, scale = constrained_weights(raw.iloc[i], covariance)
        targets.iloc[i] = [weights[ticker] for ticker in TRADE_ASSETS]
        scales.iloc[i] = scale

    positions = pd.DataFrame(0.0, index=index, columns=TRADE_ASSETS)
    held = {ticker: 0.0 for ticker in TRADE_ASSETS}
    for i in range(len(index)):
        for ticker in TRADE_ASSETS:
            target = float(targets[ticker].iloc[i])
            if (
                abs(target - held[ticker]) > TRADE_BAND
                or (target == 0.0 and held[ticker] != 0.0)
            ):
                held[ticker] = target
        positions.iloc[i] = [held[ticker] for ticker in TRADE_ASSETS]

    diagnostics["raw_targets"] = raw
    diagnostics["scales"] = scales
    diagnostics["targets"] = targets
    return {"positions": positions, "diagnostics": diagnostics}


def portfolio_returns(
    positions: pd.DataFrame,
    data: dict,
    cost: float = BASE_COST,
    shift: int = 1,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    values = positions.to_numpy()
    lagged = np.zeros_like(values)
    lagged[shift:] = values[:-shift]
    effective = pd.DataFrame(lagged, index=positions.index, columns=positions.columns)
    changes = np.abs(np.diff(np.vstack([np.zeros((1, len(TRADE_ASSETS))), lagged]), axis=0))
    turnover = pd.Series(changes.sum(axis=1), index=positions.index)
    asset_returns = data["pnl_returns"][TRADE_ASSETS].to_numpy()
    invested = lagged.sum(axis=1)
    result = (
        (lagged * asset_returns).sum(axis=1)
        + (1.0 - invested) * data["rf_daily"].to_numpy()
        - cost * turnover.to_numpy()
    )
    return pd.Series(result, index=positions.index), turnover, effective


def run_strategy(
    data: dict,
    include_core: bool = True,
    include_momentum: bool = True,
    include_reversal: bool = True,
    cost: float = BASE_COST,
    shift: int = 1,
) -> dict:
    built = build_positions(data, include_core, include_momentum, include_reversal)
    returns, turnover, effective = portfolio_returns(built["positions"], data, cost, shift)
    return {
        **built,
        "returns": returns,
        "turnover": turnover,
        "effective_positions": effective,
    }


def monthly_6040(data: dict) -> tuple[pd.Series, pd.Series]:
    index = data["index"]
    month_end = last_session_flags(index, "M")
    positions = pd.DataFrame(0.0, index=index, columns=["SPY", "IEF"])
    held = np.array([0.60, 0.0])
    for i in range(len(index)):
        if month_end[i]:
            held = np.array([0.60, 0.40 if pd.notna(data["prices"]["IEF"].iloc[i]) else 0.0])
        positions.iloc[i] = held
    lagged = np.vstack([np.zeros((1, 2)), positions.to_numpy()[:-1]])
    turnover = pd.Series(
        np.abs(np.diff(np.vstack([np.zeros((1, 2)), lagged]), axis=0)).sum(axis=1),
        index=index,
    )
    returns = data["actual_returns"][["SPY", "IEF"]].fillna(0.0).to_numpy()
    invested = lagged.sum(axis=1)
    result = (
        (lagged * returns).sum(axis=1)
        + (1.0 - invested) * data["rf_daily"].to_numpy()
        - BASE_COST * turnover.to_numpy()
    )
    return pd.Series(result, index=index), turnover


def drawdown_episode(equity: pd.Series) -> dict:
    peak = equity.cummax()
    drawdown = equity.div(peak).sub(1.0)
    trough_date = drawdown.idxmin()
    peak_value = float(peak.loc[trough_date])
    peak_date = equity.loc[:trough_date][equity.loc[:trough_date] >= peak_value - 1e-12].index[-1]
    recovered = equity.loc[trough_date:][equity.loc[trough_date:] >= peak_value]
    recovery_date = recovered.index[0] if not recovered.empty else None
    peak_loc = equity.index.get_loc(peak_date)
    trough_loc = equity.index.get_loc(trough_date)
    recovery_loc = equity.index.get_loc(recovery_date) if recovery_date is not None else None

    longest_days = 0
    longest_peak = equity.index[0]
    longest_recovery = equity.index[0]
    active_peak = equity.index[0]
    active_value = float(equity.iloc[0])
    underwater = False
    for i in range(1, len(equity)):
        value = float(equity.iloc[i])
        if value >= active_value:
            if underwater and i - equity.index.get_loc(active_peak) > longest_days:
                longest_days = i - equity.index.get_loc(active_peak)
                longest_peak = active_peak
                longest_recovery = equity.index[i]
            active_peak = equity.index[i]
            active_value = value
            underwater = False
        else:
            underwater = True
    longest_recovered = True
    if underwater and len(equity) - 1 - equity.index.get_loc(active_peak) > longest_days:
        longest_days = len(equity) - 1 - equity.index.get_loc(active_peak)
        longest_peak = active_peak
        longest_recovery = None
        longest_recovered = False

    depth = float(drawdown.loc[trough_date])
    return native(
        {
            "max_dd": depth,
            "peak_date": peak_date,
            "trough_date": trough_date,
            "recovery_date": recovery_date,
            "peak_to_trough_trading_days": trough_loc - peak_loc,
            "trough_to_recovery_trading_days": (
                recovery_loc - trough_loc if recovery_loc is not None else None
            ),
            "peak_to_recovery_trading_days": (
                recovery_loc - peak_loc if recovery_loc is not None else None
            ),
            "max_dd_dollars_on_100k_path": abs(depth) * peak_value * INITIAL_CAPITAL,
            "time_underwater_pct": float((drawdown < 0).mean()),
            "longest_underwater": {
                "peak_date": longest_peak,
                "recovery_date": longest_recovery,
                "trading_days": longest_days,
                "recovered": longest_recovered,
            },
        }
    )


def rolling_stats(returns: pd.Series) -> dict:
    equity = (1.0 + returns).cumprod()
    result = {}
    for years, sessions in ((1, 252), (3, 756), (5, 1260)):
        values = equity.div(equity.shift(sessions)).pow(252.0 / sessions).sub(1.0).dropna()
        if not values.empty:
            result[f"{years}y"] = {
                "minimum_annualized": float(values.min()),
                "median_annualized": float(values.median()),
                "negative_pct": float((values < 0).mean()),
            }
    return native(result)


def metrics(
    returns: pd.Series,
    rf_daily: pd.Series,
    start,
    end=None,
    turnover: pd.Series | None = None,
    positions: pd.DataFrame | None = None,
) -> dict:
    selected = returns.loc[start:end].dropna()
    if selected.empty:
        return {}
    equity = (1.0 + selected).cumprod()
    years = max((selected.index[-1] - selected.index[0]).days / 365.25, len(selected) / 252.0)
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0)
    vol = float(selected.std() * np.sqrt(252.0))
    excess = selected - rf_daily.reindex(selected.index).fillna(0.0)
    sharpe = float(excess.mean() / selected.std() * np.sqrt(252.0)) if selected.std() > 0 else None
    downside = excess[excess < 0].std()
    sortino = float(excess.mean() / downside * np.sqrt(252.0)) if downside > 0 else None
    episode = drawdown_episode(equity)
    position_stats = {}
    if positions is not None:
        window_positions = positions.reindex(selected.index).fillna(0.0)
        invested = window_positions.sum(axis=1)
        position_stats = {
            "average_invested": float(invested.mean()),
            "pct_fully_invested": float((invested >= 0.99).mean()),
            "average_tqqq_weight": float(window_positions.get("TQQQ", pd.Series(0.0, index=selected.index)).mean()),
            "pct_days_tqqq": float((window_positions.get("TQQQ", pd.Series(0.0, index=selected.index)) > 0).mean()),
            "max_tqqq_weight": float(window_positions.get("TQQQ", pd.Series(0.0, index=selected.index)).max()),
        }
    return native(
        {
            "start": selected.index[0],
            "end": selected.index[-1],
            "cagr": cagr,
            "annualized_vol": vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": cagr / abs(episode["max_dd"]) if episode["max_dd"] < 0 else None,
            "total_return": float(equity.iloc[-1] - 1.0),
            "terminal_100k": float(INITIAL_CAPITAL * equity.iloc[-1]),
            "worst_month": float(((1.0 + selected).resample("ME").prod().sub(1.0)).min()),
            "annual_turnover": (
                float(turnover.reindex(selected.index).sum() / years)
                if turnover is not None
                else 0.0
            ),
            "drawdown": episode,
            "rolling": rolling_stats(selected),
            "positions": position_stats,
        }
    )


def yearly_returns(series_by_name: dict[str, pd.Series], start) -> dict:
    return native(
        {
            name: {
                str(date.year): float(value)
                for date, value in ((1.0 + returns.loc[start:]).resample("YE").prod().sub(1.0)).items()
            }
            for name, returns in series_by_name.items()
        }
    )


def monthly_curves(series_by_name: dict[str, pd.Series], start) -> dict:
    result = {}
    for name, returns in series_by_name.items():
        equity = INITIAL_CAPITAL * (1.0 + returns.loc[start:]).cumprod()
        drawdown = equity.div(equity.cummax()).sub(1.0)
        monthly_equity = equity.resample("ME").last()
        monthly_dd = drawdown.resample("ME").min()
        result[name] = [
            {
                "date": str(date.date()),
                "equity": float(value),
                "drawdown": float(monthly_dd.loc[date]),
            }
            for date, value in monthly_equity.items()
        ]
    return native(result)


def write_summary(results: dict) -> None:
    full = results["windows"]["full"]
    development = results["windows"]["development"]["ensemble"]
    validation = results["windows"]["validation"]["ensemble"]
    holdout = results["windows"]["holdout"]
    ensemble = full["ensemble"]
    qqq = full["QQQ"]
    dd = ensemble["drawdown"]
    full_diagnostics = results["diagnostics"]["full"]
    lines = [
        "# V8 Independent Alpha Ensemble",
        "",
        f"Data begins {results['meta']['data_start']}; evaluation begins "
        f"{results['meta']['evaluation_start']} after warmup and ends "
        f"{results['meta']['data_end']}.",
        "",
        "## Full-history result",
        "",
        f"- CAGR: **{ensemble['cagr']:.2%}**",
        f"- Annualized volatility: **{ensemble['annualized_vol']:.2%}**",
        f"- Maximum drawdown: **{dd['max_dd']:.2%}**",
        f"- Drawdown peak / trough / recovery: **{dd['peak_date']} / "
        f"{dd['trough_date']} / {dd['recovery_date'] or 'unrecovered'}**",
        f"- Peak-to-recovery: **{dd['peak_to_recovery_trading_days'] or 'unrecovered'} sessions**",
        f"- Terminal value from $100,000: **${ensemble['terminal_100k']:,.0f}**",
        f"- Calmar: **{ensemble['calmar']:.2f}**",
        "",
        f"QQQ returned {qqq['cagr']:.2%} with a {qqq['drawdown']['max_dd']:.2%} "
        f"maximum drawdown over the same period.",
        "",
        "## Chronological result",
        "",
        f"- Development label (2000–2009): {development['cagr']:.2%} CAGR, "
        f"{development['drawdown']['max_dd']:.2%} max drawdown",
        f"- Validation label (2010–2017): {validation['cagr']:.2%} CAGR, "
        f"{validation['drawdown']['max_dd']:.2%} max drawdown",
        f"- Holdout label (2018–latest): {holdout['ensemble']['cagr']:.2%} CAGR, "
        f"{holdout['ensemble']['drawdown']['max_dd']:.2%} max drawdown",
        "",
        "## Holdout (2018–latest)",
        "",
        f"- Ensemble: {holdout['ensemble']['cagr']:.2%} CAGR, "
        f"{holdout['ensemble']['drawdown']['max_dd']:.2%} max drawdown, "
        f"{holdout['ensemble']['calmar']:.2f} Calmar",
        f"- QQQ: {holdout['QQQ']['cagr']:.2%} CAGR, "
        f"{holdout['QQQ']['drawdown']['max_dd']:.2%} max drawdown, "
        f"{holdout['QQQ']['calmar']:.2f} Calmar",
        "",
        "## Fixed diagnostics",
        "",
    ]
    for name, values in results["diagnostics"]["full"].items():
        lines.append(
            f"- {name}: {values['cagr']:.2%} CAGR, "
            f"{values['drawdown']['max_dd']:.2%} max drawdown, "
            f"{values['calmar']:.2f} Calmar"
        )
    lines.extend(
        [
            "",
            "The momentum sleeve supplied the strongest standalone drawdown "
            "efficiency. The reversal sleeve was low-risk but sparse: adding it "
            f"changed ensemble CAGR from {full_diagnostics['without_reversal']['cagr']:.2%} "
            f"to {ensemble['cagr']:.2%} and max drawdown from "
            f"{full_diagnostics['without_reversal']['drawdown']['max_dd']:.2%} to "
            f"{dd['max_dd']:.2%}. It did not solve the recovery problem.",
            "",
            "## Decision",
            "",
            f"**{results['conclusion']['verdict'].upper()}** — "
            f"{results['conclusion']['reason']}",
            "",
            "No diagnostic replaces the pre-registered primary ensemble.",
            "",
            "## Limitations",
            "",
            "- TQQQ returns before 2010 are synthetic.",
            "- The fixed ETF universe is selected with hindsight, although every "
            "fund enters only after actual inception.",
            "- Yahoo adjusted closes do not model intraday gaps, taxes, or variable spreads.",
            "- The reversal sleeve is a sparse historical pattern and requires live paper evidence.",
        ]
    )
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    data = load_data()
    primary = run_strategy(data)
    core_only = run_strategy(data, include_momentum=False, include_reversal=False)
    momentum_only = run_strategy(data, include_core=False, include_reversal=False)
    reversal_only = run_strategy(data, include_core=False, include_momentum=False)
    no_reversal = run_strategy(data, include_reversal=False)
    cost_stress_returns, cost_stress_turnover, cost_stress_positions = portfolio_returns(
        primary["positions"], data, cost=STRESS_COST
    )
    delay_returns, delay_turnover, delay_positions = portfolio_returns(
        primary["positions"], data, shift=2
    )
    sixty_forty, sixty_forty_turnover = monthly_6040(data)

    benchmarks = {
        "SPY": data["actual_returns"]["SPY"].fillna(0.0),
        "QQQ": data["actual_returns"]["QQQ"].fillna(0.0),
        "60_40": sixty_forty,
    }
    benchmark_turnover = {
        "SPY": pd.Series(0.0, index=data["index"]),
        "QQQ": pd.Series(0.0, index=data["index"]),
        "60_40": sixty_forty_turnover,
    }
    series = {"ensemble": primary["returns"], **benchmarks}
    turnovers = {"ensemble": primary["turnover"], **benchmark_turnover}

    evaluation_start = data["index"][260]
    windows = {
        "full": (evaluation_start, None),
        "development": (evaluation_start, pd.Timestamp("2009-12-31")),
        "validation": (pd.Timestamp("2010-01-01"), pd.Timestamp("2017-12-31")),
        "holdout": (pd.Timestamp("2018-01-01"), None),
        "actual_tqqq": (data["prices"]["TQQQ"].dropna().index[0], None),
        "from_2015": (pd.Timestamp("2015-01-01"), None),
    }
    window_results = {}
    for label, (start, end) in windows.items():
        window_results[label] = {}
        for name, returns in series.items():
            positions = primary["effective_positions"] if name == "ensemble" else None
            window_results[label][name] = metrics(
                returns,
                data["rf_daily"],
                start,
                end,
                turnovers[name],
                positions,
            )

    crisis_results = {
        crisis: {
            name: metrics(returns, data["rf_daily"], start, end, turnovers[name])
            for name, returns in series.items()
        }
        for crisis, (start, end) in CRISES.items()
    }

    diagnostic_runs = {
        "core_only": core_only,
        "momentum_only": momentum_only,
        "reversal_only": reversal_only,
        "without_reversal": no_reversal,
        "cost_20bps": {
            "returns": cost_stress_returns,
            "turnover": cost_stress_turnover,
            "effective_positions": cost_stress_positions,
        },
        "execution_t2": {
            "returns": delay_returns,
            "turnover": delay_turnover,
            "effective_positions": delay_positions,
        },
    }
    diagnostics = {"full": {}, "holdout": {}}
    for label, start in (("full", evaluation_start), ("holdout", pd.Timestamp("2018-01-01"))):
        diagnostics[label] = {
            name: metrics(
                run["returns"],
                data["rf_daily"],
                start,
                turnover=run["turnover"],
                positions=run["effective_positions"],
            )
            for name, run in diagnostic_runs.items()
        }

    full_ensemble = window_results["full"]["ensemble"]
    full_qqq = window_results["full"]["QQQ"]
    holdout_ensemble = window_results["holdout"]["ensemble"]
    holdout_qqq = window_results["holdout"]["QQQ"]
    checks = {
        "full_cagr_qqq_plus_2pp": full_ensemble["cagr"] >= full_qqq["cagr"] + 0.02,
        "holdout_calmar_above_qqq": holdout_ensemble["calmar"] > holdout_qqq["calmar"],
        "max_dd_recovery_within_756_sessions": (
            full_ensemble["drawdown"]["peak_to_recovery_trading_days"] is not None
            and full_ensemble["drawdown"]["peak_to_recovery_trading_days"] <= 756
        ),
        "stress_holdout_cagr_positive": (
            diagnostics["holdout"]["cost_20bps"]["cagr"] > 0
            and diagnostics["holdout"]["execution_t2"]["cagr"] > 0
        ),
    }
    passed = all(checks.values())
    failed = [name for name, passed_check in checks.items() if not passed_check]
    conclusion = {
        "verdict": "pass" if passed else "reject",
        "checks": checks,
        "reason": (
            "All pre-registered checks passed."
            if passed
            else "Failed pre-registered check(s): " + ", ".join(failed) + "."
        ),
    }

    overlap = pd.concat(
        [data["synthetic_tqqq"], data["actual_tqqq"]],
        axis=1,
        keys=["synthetic", "actual"],
    ).dropna()
    results = native(
        {
            "meta": {
                "strategy": "V8 Independent Alpha Ensemble",
                "data_source": "Yahoo Finance adjusted daily close",
                "data_start": data["index"][0],
                "data_end": data["index"][-1],
                "evaluation_start": evaluation_start,
                "initial_capital": INITIAL_CAPITAL,
                "base_cost_bps": BASE_COST * 10_000,
                "inception": data["inception"],
                "design": "research/v8_ensemble/DESIGN.md",
                "reversal_entries": primary["diagnostics"]["reversal_entries"],
            },
            "windows": window_results,
            "crises": crisis_results,
            "diagnostics": diagnostics,
            "yearly": yearly_returns(
                {
                    "ensemble": primary["returns"],
                    "QQQ": benchmarks["QQQ"],
                    "SPY": benchmarks["SPY"],
                },
                evaluation_start,
            ),
            "curves": monthly_curves(
                {
                    "ensemble": primary["returns"],
                    "QQQ": benchmarks["QQQ"],
                    "without_reversal": no_reversal["returns"],
                },
                evaluation_start,
            ),
            "synthetic_validation": {
                "start": overlap.index[0],
                "end": overlap.index[-1],
                "daily_correlation": float(overlap.corr().iloc[0, 1]),
                "annualized_tracking_error": float(
                    (overlap["actual"] - overlap["synthetic"]).std() * np.sqrt(252.0)
                ),
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
                "data_end": results["meta"]["data_end"],
                "cagr": full_ensemble["cagr"],
                "max_dd": full_ensemble["drawdown"]["max_dd"],
                "recovery_sessions": full_ensemble["drawdown"]["peak_to_recovery_trading_days"],
                "holdout_cagr": holdout_ensemble["cagr"],
                "holdout_calmar": holdout_ensemble["calmar"],
                "verdict": conclusion["verdict"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
