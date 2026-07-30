"""V7: independent regime-adaptive Nasdaq strategy research.

The fixed rules are documented in DESIGN.md. This script downloads adjusted
daily data, runs a causal weekly backtest, and writes machine-readable results
plus a concise research summary. It does not import or compare the live books.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = REPO_ROOT / "output" / "v7_regime_results.json"
OUT_SUMMARY = REPO_ROOT / "output" / "v7_regime_summary.md"

TICKERS = ["QQQ", "SPY", "IEF", "TLT", "GLD", "DBC", "DBMF", "TQQQ", "^IRX"]
ASSETS = ["TQQQ", "QQQ", "IEF", "TLT", "GLD", "DBC", "DBMF"]
DEFENSIVE = ["IEF", "TLT", "GLD", "DBC", "DBMF"]
CONSTANT_DEFENSIVE = ["IEF", "TLT", "GLD"]

START = "1999-03-10"
BASE_SMA = 200
MOM_FAST = 63
MOM_SLOW = 126
VOL_SPAN = 63
HYSTERESIS = 0.02
BULL_VOL_TARGET = 0.30
NEUTRAL_VOL_TARGET = 0.15
BEAR_VOL_TARGET = 0.12
TRADE_BAND = 0.05
BASE_COST = 0.0010
STRESS_COST = 0.0020
SYNTH_SPREAD = 0.005 / 252.0
SYNTH_EXPENSE = 0.0095 / 252.0
INITIAL_CAPITAL = 100_000.0

CRISES = {
    "dotcom": ("2000-03-24", "2002-10-09"),
    "gfc": ("2007-10-09", "2009-03-09"),
    "covid_crash": ("2020-02-19", "2020-03-23"),
    "inflation_bear_2022": ("2022-01-03", "2022-10-12"),
    "tariff_2025": ("2025-02-15", "2025-06-30"),
}


def native(value):
    """Convert numpy/pandas values into JSON-compatible Python values."""
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
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    return value


def annualized_vol(returns: pd.Series) -> pd.Series:
    squared = returns.pow(2)
    return np.sqrt(
        252.0
        * squared.ewm(
            alpha=2.0 / (VOL_SPAN + 1.0),
            adjust=False,
            min_periods=20,
        ).mean()
    )


def last_session_flags(index: pd.DatetimeIndex, frequency: str) -> np.ndarray:
    """Flag completed week/month endings, excluding an incomplete final period."""
    periods = index.to_period(frequency)
    flags = np.zeros(len(index), dtype=bool)
    flags[:-1] = periods[:-1] != periods[1:]
    if frequency == "W-FRI":
        flags[-1] = index[-1].weekday() == 4
    elif frequency == "M":
        flags[-1] = index[-1].is_month_end
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
    rf_annual = prices["^IRX"].ffill().div(100.0).fillna(0.0)
    rf_daily = rf_annual.div(252.0)

    actual_returns = pd.DataFrame(index=index)
    for ticker in ["QQQ", "SPY", "IEF", "TLT", "GLD", "DBC", "DBMF"]:
        actual_returns[ticker] = prices[ticker].pct_change(fill_method=None)

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

    pnl_returns = actual_returns[ASSETS].fillna(0.0)
    volatility = actual_returns[ASSETS].apply(annualized_vol)
    momentum = pd.DataFrame(index=index)
    for ticker in ["QQQ", *DEFENSIVE]:
        momentum[ticker] = prices[ticker].div(prices[ticker].shift(MOM_SLOW)).sub(1.0)
    momentum["QQQ_63"] = prices["QQQ"].div(prices["QQQ"].shift(MOM_FAST)).sub(1.0)

    first_prices = {
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
        "pnl_returns": pnl_returns,
        "volatility": volatility,
        "momentum": momentum,
        "rf_daily": rf_daily,
        "synthetic_tqqq": synthetic_tqqq,
        "actual_tqqq": actual_tqqq,
        "inception": first_prices,
    }


def defensive_weights(
    i: int,
    universe: list[str],
    data: dict,
) -> dict[str, float]:
    prices = data["prices"]
    momentum = data["momentum"]
    vol = data["volatility"]
    actual_returns = data["actual_returns"]

    eligible = []
    for ticker in universe:
        history = prices[ticker].iloc[: i + 1].notna().sum()
        score = momentum[ticker].iloc[i]
        sigma = vol[ticker].iloc[i]
        if history >= MOM_SLOW + 1 and pd.notna(score) and score > 0 and sigma > 0:
            eligible.append((ticker, float(score)))
    selected = [ticker for ticker, _ in sorted(eligible, key=lambda x: x[1], reverse=True)[:2]]
    if not selected:
        return {}

    inverse_vol = np.array([1.0 / float(vol[ticker].iloc[i]) for ticker in selected])
    base = inverse_vol / inverse_vol.sum()

    if len(selected) == 1:
        portfolio_vol = float(vol[selected[0]].iloc[i])
    else:
        window = actual_returns[selected].iloc[max(0, i - 62) : i + 1].dropna()
        if len(window) < 40:
            return {}
        covariance = window.cov().to_numpy() * 252.0
        portfolio_vol = float(np.sqrt(max(base @ covariance @ base, 0.0)))

    scale = min(1.0, BEAR_VOL_TARGET / portfolio_vol) if portfolio_vol > 0 else 0.0
    return {ticker: float(weight * scale) for ticker, weight in zip(selected, base)}


def build_targets(
    data: dict,
    sma_length: int = BASE_SMA,
    defensive_universe: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    defensive_universe = defensive_universe or DEFENSIVE
    index = data["index"]
    prices = data["prices"]
    vol = data["volatility"]
    momentum = data["momentum"]
    week_end = last_session_flags(index, "W-FRI")
    sma = prices["SPY"].rolling(sma_length, min_periods=sma_length).mean()

    positions = pd.DataFrame(0.0, index=index, columns=ASSETS)
    regimes = pd.Series("warmup", index=index, dtype="object")
    held = {ticker: 0.0 for ticker in ASSETS}
    risk_on = False
    prior_regime = "warmup"

    for i, date in enumerate(index):
        if week_end[i] and pd.notna(sma.iloc[i]):
            spy_price = float(prices["SPY"].iloc[i])
            spy_sma = float(sma.iloc[i])
            if risk_on:
                risk_on = spy_price >= spy_sma * (1.0 - HYSTERESIS)
            else:
                risk_on = spy_price > spy_sma * (1.0 + HYSTERESIS)

            qqq_63 = momentum["QQQ_63"].iloc[i]
            qqq_126 = momentum["QQQ"].iloc[i]
            if risk_on and qqq_63 > 0 and qqq_126 > 0:
                regime = "bull"
                sigma = vol["TQQQ"].iloc[i]
                targets = {
                    "TQQQ": min(1.0, BULL_VOL_TARGET / float(sigma))
                    if pd.notna(sigma) and sigma > 0
                    else 0.0
                }
            elif not risk_on and qqq_126 <= 0:
                regime = "bear"
                targets = defensive_weights(i, defensive_universe, data)
            else:
                regime = "neutral"
                sigma = vol["QQQ"].iloc[i]
                targets = {
                    "QQQ": min(1.0, NEUTRAL_VOL_TARGET / float(sigma))
                    if pd.notna(sigma) and sigma > 0
                    else 0.0
                }

            target = {ticker: targets.get(ticker, 0.0) for ticker in ASSETS}
            total = sum(target.values())
            if total > 1.0:
                target = {ticker: weight / total for ticker, weight in target.items()}

            if regime != prior_regime:
                held = target
            else:
                for ticker in ASSETS:
                    if (
                        abs(target[ticker] - held[ticker]) > TRADE_BAND
                        or (target[ticker] == 0.0 and held[ticker] != 0.0)
                    ):
                        held[ticker] = target[ticker]
            prior_regime = regime

        regimes.iloc[i] = prior_regime
        positions.iloc[i] = [held[ticker] for ticker in ASSETS]

    return positions, regimes


def portfolio_returns(
    positions: pd.DataFrame,
    data: dict,
    cost: float = BASE_COST,
    shift: int = 1,
) -> tuple[pd.Series, pd.Series]:
    values = positions.to_numpy()
    lagged = np.zeros_like(values)
    lagged[shift:] = values[:-shift]
    executed = np.abs(np.diff(np.vstack([np.zeros((1, lagged.shape[1])), lagged]), axis=0))
    turnover = pd.Series(executed.sum(axis=1), index=positions.index)
    asset_return = data["pnl_returns"][ASSETS].to_numpy()
    invested = lagged.sum(axis=1)
    result = (
        (lagged * asset_return).sum(axis=1)
        + (1.0 - invested) * data["rf_daily"].to_numpy()
        - cost * turnover.to_numpy()
    )
    return pd.Series(result, index=positions.index), turnover


def build_strategy(
    data: dict,
    sma_length: int = BASE_SMA,
    defensive_universe: list[str] | None = None,
    cost: float = BASE_COST,
    shift: int = 1,
) -> dict:
    positions, signal_regime = build_targets(data, sma_length, defensive_universe)
    returns, turnover = portfolio_returns(positions, data, cost, shift)
    effective_regime = signal_regime.shift(shift).fillna("warmup")
    return {
        "returns": returns,
        "turnover": turnover,
        "positions": positions,
        "regime": effective_regime,
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
    running_peak = equity.cummax()
    drawdown = equity.div(running_peak).sub(1.0)
    trough_date = drawdown.idxmin()
    peak_value = float(running_peak.loc[trough_date])
    peak_candidates = equity.loc[:trough_date]
    peak_date = peak_candidates[peak_candidates >= peak_value - 1e-12].index[-1]
    post_trough = equity.loc[trough_date:]
    recovered = post_trough[post_trough >= peak_value]
    recovery_date = recovered.index[0] if not recovered.empty else None

    peak_loc = equity.index.get_loc(peak_date)
    trough_loc = equity.index.get_loc(trough_date)
    recovery_loc = equity.index.get_loc(recovery_date) if recovery_date is not None else None
    depth = float(drawdown.loc[trough_date])

    longest = {
        "peak_date": equity.index[0],
        "recovery_date": equity.index[0],
        "trading_days": 0,
        "calendar_days": 0,
        "recovered": True,
    }
    active_peak_date = equity.index[0]
    active_peak_value = float(equity.iloc[0])
    underwater = False
    for i in range(1, len(equity)):
        value = float(equity.iloc[i])
        if value >= active_peak_value:
            if underwater:
                duration = i - equity.index.get_loc(active_peak_date)
                if duration > longest["trading_days"]:
                    longest = {
                        "peak_date": active_peak_date,
                        "recovery_date": equity.index[i],
                        "trading_days": duration,
                        "calendar_days": (equity.index[i] - active_peak_date).days,
                        "recovered": True,
                    }
            active_peak_date = equity.index[i]
            active_peak_value = value
            underwater = False
        else:
            underwater = True
    if underwater:
        duration = len(equity) - 1 - equity.index.get_loc(active_peak_date)
        if duration > longest["trading_days"]:
            longest = {
                "peak_date": active_peak_date,
                "recovery_date": None,
                "trading_days": duration,
                "calendar_days": (equity.index[-1] - active_peak_date).days,
                "recovered": False,
            }

    return native(
        {
            "max_dd": depth,
            "peak_date": peak_date,
            "trough_date": trough_date,
            "recovery_date": recovery_date,
            "peak_to_trough_trading_days": trough_loc - peak_loc,
            "peak_to_trough_calendar_days": (trough_date - peak_date).days,
            "trough_to_recovery_trading_days": (
                recovery_loc - trough_loc if recovery_loc is not None else None
            ),
            "trough_to_recovery_calendar_days": (
                (recovery_date - trough_date).days if recovery_date is not None else None
            ),
            "peak_to_recovery_trading_days": (
                recovery_loc - peak_loc if recovery_loc is not None else None
            ),
            "max_dd_dollars_on_100k_path": abs(depth) * peak_value * INITIAL_CAPITAL,
            "time_underwater_pct": float((drawdown < 0).mean()),
            "longest_underwater": longest,
        }
    )


def rolling_stats(returns: pd.Series) -> dict:
    equity = (1.0 + returns).cumprod()
    result = {}
    for years, sessions in ((1, 252), (3, 756), (5, 1260)):
        rolling = equity.div(equity.shift(sessions)).pow(252.0 / sessions).sub(1.0).dropna()
        if rolling.empty:
            continue
        result[f"{years}y"] = {
            "minimum_annualized": float(rolling.min()),
            "median_annualized": float(rolling.median()),
            "negative_pct": float((rolling < 0).mean()),
        }
    return native(result)


def metrics(
    returns: pd.Series,
    rf_daily: pd.Series,
    start,
    end=None,
    turnover: pd.Series | None = None,
) -> dict:
    selected = returns.loc[start:end].dropna()
    if selected.empty:
        return {}
    equity = (1.0 + selected).cumprod()
    years = max((selected.index[-1] - selected.index[0]).days / 365.25, len(selected) / 252.0)
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0)
    annual_vol = float(selected.std() * np.sqrt(252.0))
    excess = selected - rf_daily.reindex(selected.index).fillna(0.0)
    sharpe = float(excess.mean() / selected.std() * np.sqrt(252.0)) if selected.std() > 0 else None
    downside = excess[excess < 0].std()
    sortino = float(excess.mean() / downside * np.sqrt(252.0)) if downside > 0 else None
    episode = drawdown_episode(equity)
    max_dd = episode["max_dd"]
    turnover_year = (
        float(turnover.reindex(selected.index).sum() / years) if turnover is not None else 0.0
    )
    return native(
        {
            "start": selected.index[0],
            "end": selected.index[-1],
            "cagr": cagr,
            "annualized_vol": annual_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": cagr / abs(max_dd) if max_dd < 0 else None,
            "total_return": float(equity.iloc[-1] - 1.0),
            "terminal_100k": float(INITIAL_CAPITAL * equity.iloc[-1]),
            "worst_month": float(((1.0 + selected).resample("ME").prod().sub(1.0)).min()),
            "annual_turnover": turnover_year,
            "drawdown": episode,
            "rolling": rolling_stats(selected),
        }
    )


def regime_statistics(strategy: dict, start) -> dict:
    returns = strategy["returns"].loc[start:]
    regimes = strategy["regime"].reindex(returns.index)
    total_log_return = float(np.log1p(returns).sum())
    result = {}
    for regime in ["bull", "neutral", "bear"]:
        mask = regimes == regime
        count = int(mask.sum())
        if count == 0:
            continue
        regime_return = returns[mask]
        log_return = float(np.log1p(regime_return).sum())
        result[regime] = {
            "sessions": count,
            "occupancy": count / len(returns),
            "annualized_return_when_active": float(
                np.exp(log_return * 252.0 / count) - 1.0
            ),
            "log_return_contribution": log_return,
            "share_of_total_log_return": (
                log_return / total_log_return if total_log_return != 0 else None
            ),
        }
    return native(result)


def yearly_returns(series_by_name: dict[str, pd.Series], start) -> dict:
    result = {}
    for name, returns in series_by_name.items():
        annual = (1.0 + returns.loc[start:]).resample("YE").prod().sub(1.0)
        result[name] = {str(date.year): float(value) for date, value in annual.items()}
    return native(result)


def monthly_curves(series_by_name: dict[str, pd.Series], start) -> dict:
    result = {}
    for name, returns in series_by_name.items():
        daily_equity = INITIAL_CAPITAL * (1.0 + returns.loc[start:]).cumprod()
        daily_drawdown = daily_equity.div(daily_equity.cummax()).sub(1.0)
        monthly_equity = daily_equity.resample("ME").last()
        monthly_drawdown = daily_drawdown.resample("ME").min()
        result[name] = [
            {
                "date": str(date.date()),
                "equity": float(value),
                "drawdown": float(monthly_drawdown.loc[date]),
            }
            for date, value in monthly_equity.items()
        ]
    return native(result)


def write_summary(results: dict) -> None:
    full = results["windows"]["full_1999"]
    strategy = full["strategy"]
    qqq = full["QQQ"]
    actual = results["windows"]["actual_tqqq_era"]
    dd = strategy["drawdown"]
    verdict = results["conclusion"]
    robust = results["robustness"]

    lines = [
        "# V7 Regime-Adaptive Nasdaq Research",
        "",
        f"Data: {results['meta']['data_start']} through {results['meta']['data_end']}. "
        f"Evaluation begins {results['meta']['full_evaluation_start']} after signal warmup. "
        "Initial capital: $100,000.",
        "",
        "## Result",
        "",
        f"- Annualized return: **{strategy['cagr']:.2%}**",
        f"- Annualized volatility: **{strategy['annualized_vol']:.2%}**",
        f"- Maximum drawdown: **{dd['max_dd']:.2%}**",
        f"- Drawdown peak: **{dd['peak_date']}**",
        f"- Drawdown trough: **{dd['trough_date']}**",
        f"- Recovery date: **{dd['recovery_date'] or 'not recovered'}**",
        f"- Peak-to-recovery: **{dd['peak_to_recovery_trading_days'] or 'unrecovered'} trading sessions**",
        f"- Maximum dollar loss on the simulated $100k path: **${dd['max_dd_dollars_on_100k_path']:,.0f}**",
        f"- Terminal value: **${strategy['terminal_100k']:,.0f}**",
        f"- Calmar ratio: **{strategy['calmar']:.2f}**",
        "",
        "## Independent benchmark",
        "",
        f"QQQ returned {qqq['cagr']:.2%} annualized with a "
        f"{qqq['drawdown']['max_dd']:.2%} maximum drawdown and a "
        f"{qqq['calmar']:.2f} Calmar ratio.",
        "",
        "## Actual-TQQQ era",
        "",
        f"From {actual['strategy']['start']}, the strategy returned "
        f"{actual['strategy']['cagr']:.2%} annualized versus "
        f"{actual['QQQ']['cagr']:.2%} for QQQ. Maximum drawdowns were "
        f"{actual['strategy']['drawdown']['max_dd']:.2%} and "
        f"{actual['QQQ']['drawdown']['max_dd']:.2%}, respectively. The strategy "
        "therefore delivered lower return without a meaningful drawdown improvement "
        "during the non-synthetic period.",
        "",
        "## Decision",
        "",
        f"**{verdict['verdict'].upper()}** — {verdict['reason']}",
        "",
        "The result was evaluated against the rules fixed in `DESIGN.md`; no "
        "post-result parameter was promoted.",
        "",
        "## Robustness",
        "",
    ]
    for name, values in robust.items():
        lines.append(
            f"- {name}: CAGR {values['cagr']:.2%}, max drawdown "
            f"{values['drawdown']['max_dd']:.2%}, Calmar {values['calmar']:.2f}"
        )
    lines.extend(
        [
            "",
            "## Important limitations",
            "",
            "- TQQQ history before its 2010 inception is synthetic and depends on "
            "the stated financing and expense assumptions.",
            "- Yahoo adjusted closes are a research data source, not guaranteed "
            "execution prices.",
            "- The ETF universe is selected with hindsight. DBC and DBMF enter "
            "only after actual inception; the constant-universe diagnostic shows "
            "their effect.",
            "- Taxes, bid/ask variation, market impact, and intraday gaps are not "
            "fully modeled.",
        ]
    )
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    data = load_data()
    primary = build_strategy(data)
    sixty_forty, sixty_forty_turnover = monthly_6040(data)

    benchmarks = {
        "SPY": data["actual_returns"]["SPY"].fillna(0.0),
        "QQQ": data["actual_returns"]["QQQ"].fillna(0.0),
        "60_40": sixty_forty,
        "TQQQ_buy_hold": data["actual_returns"]["TQQQ"].fillna(0.0),
    }
    turnover = {
        "strategy": primary["turnover"],
        "SPY": pd.Series(0.0, index=data["index"]),
        "QQQ": pd.Series(0.0, index=data["index"]),
        "60_40": sixty_forty_turnover,
        "TQQQ_buy_hold": pd.Series(0.0, index=data["index"]),
    }
    all_series = {"strategy": primary["returns"], **benchmarks}

    actual_tqqq_start = data["prices"]["TQQQ"].dropna().index[0]
    full_start = data["index"][max(BASE_SMA, MOM_SLOW, VOL_SPAN) + 60]
    windows = {
        "full_1999": (full_start, None),
        "actual_tqqq_era": (actual_tqqq_start, None),
        "from_2015": (pd.Timestamp("2015-01-01"), None),
        "last_3y": (data["index"][-1] - pd.DateOffset(years=3), None),
    }
    window_results = {}
    for window, (start, end) in windows.items():
        window_results[window] = {
            name: metrics(series, data["rf_daily"], start, end, turnover[name])
            for name, series in all_series.items()
        }

    crisis_results = {}
    for crisis, (start, end) in CRISES.items():
        crisis_results[crisis] = {
            name: metrics(series, data["rf_daily"], start, end, turnover[name])
            for name, series in all_series.items()
        }

    robustness_builds = {
        "cost_20bps": build_strategy(data, cost=STRESS_COST),
        "execution_t2": build_strategy(data, shift=2),
        "sma_150": build_strategy(data, sma_length=150),
        "sma_250": build_strategy(data, sma_length=250),
        "defense_IEF_TLT_GLD": build_strategy(
            data,
            defensive_universe=CONSTANT_DEFENSIVE,
        ),
    }
    robustness = {
        name: metrics(
            run["returns"],
            data["rf_daily"],
            full_start,
            turnover=run["turnover"],
        )
        for name, run in robustness_builds.items()
    }

    overlap = pd.concat(
        [data["synthetic_tqqq"], data["actual_tqqq"]],
        axis=1,
        keys=["synthetic", "actual"],
    ).dropna()
    synthetic_validation = {
        "start": overlap.index[0],
        "end": overlap.index[-1],
        "daily_correlation": float(overlap.corr().iloc[0, 1]),
        "annualized_tracking_error": float(
            (overlap["actual"] - overlap["synthetic"]).std() * np.sqrt(252.0)
        ),
    }

    full_strategy = window_results["full_1999"]["strategy"]
    full_qqq = window_results["full_1999"]["QQQ"]
    checks = {
        "calmar_above_qqq": full_strategy["calmar"] > full_qqq["calmar"],
        "cagr_qqq_plus_3pp": full_strategy["cagr"] >= full_qqq["cagr"] + 0.03,
        "max_dd_recovery_within_756_sessions": (
            full_strategy["drawdown"]["peak_to_recovery_trading_days"] is not None
            and full_strategy["drawdown"]["peak_to_recovery_trading_days"] <= 756
        ),
    }
    passed = all(checks.values())
    failed = [name for name, ok in checks.items() if not ok]
    conclusion = {
        "verdict": "pass" if passed else "reject",
        "checks": checks,
        "reason": (
            "All pre-registered return, drawdown-efficiency, and recovery checks passed."
            if passed
            else "Failed pre-registered check(s): " + ", ".join(failed) + "."
        ),
    }

    results = native(
        {
            "meta": {
                "strategy": "V7 Regime-Adaptive Nasdaq",
                "data_source": "Yahoo Finance adjusted daily close",
                "data_start": data["index"][0],
                "data_end": data["index"][-1],
                "full_evaluation_start": full_start,
                "initial_capital": INITIAL_CAPITAL,
                "base_cost_bps": BASE_COST * 10_000,
                "execution_delay_sessions": 1,
                "inception": data["inception"],
                "design": "research/v7_regime/DESIGN.md",
            },
            "windows": window_results,
            "crises": crisis_results,
            "robustness": robustness,
            "regime": regime_statistics(primary, full_start),
            "yearly": yearly_returns(all_series, full_start),
            "curves": monthly_curves(all_series, full_start),
            "synthetic_validation": synthetic_validation,
            "conclusion": conclusion,
        }
    )
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_summary(results)

    full = results["windows"]["full_1999"]["strategy"]
    print(
        json.dumps(
            {
                "data_end": results["meta"]["data_end"],
                "cagr": full["cagr"],
                "max_dd": full["drawdown"]["max_dd"],
                "peak": full["drawdown"]["peak_date"],
                "trough": full["drawdown"]["trough_date"],
                "recovery": full["drawdown"]["recovery_date"],
                "calmar": full["calmar"],
                "verdict": results["conclusion"]["verdict"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
