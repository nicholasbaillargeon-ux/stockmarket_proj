"""Statistical analysis: returns, risk metrics, and portfolio math."""

import numpy as np
import pandas as pd

TRADING_DAYS = 252
DEFAULT_RF = 0.0525  # approximate risk-free rate


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def cumulative_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return (1 + daily_returns(prices)).cumprod() - 1


def annualized_return(returns: pd.Series) -> float:
    total = (1 + returns).prod()
    n = len(returns)
    if n < 2:
        return float("nan")
    return float(total ** (TRADING_DAYS / n) - 1)


def annualized_volatility(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, rf: float = DEFAULT_RF) -> float:
    excess = returns - rf / TRADING_DAYS
    if returns.std() == 0:
        return float("nan")
    return float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns: pd.Series, rf: float = DEFAULT_RF) -> float:
    excess = returns - rf / TRADING_DAYS
    downside = excess[excess < 0].std()
    if downside == 0:
        return float("nan")
    return float(excess.mean() / downside * np.sqrt(TRADING_DAYS))


def max_drawdown(prices: pd.Series) -> float:
    roll_max = prices.cummax()
    dd = (prices - roll_max) / roll_max
    return float(dd.min())


def rolling_volatility(returns: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


def rolling_sharpe(returns: pd.DataFrame, window: int = 63, rf: float = DEFAULT_RF) -> pd.DataFrame:
    daily_rf = rf / TRADING_DAYS
    excess = returns - daily_rf
    roll_mean = excess.rolling(window).mean()
    roll_std = excess.rolling(window).std()
    return (roll_mean / roll_std) * np.sqrt(TRADING_DAYS)


def portfolio_returns(returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Compute weighted portfolio daily returns. Weights are normalized to sum=1."""
    tickers = [t for t in weights if t in returns.columns]
    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()
    return returns[tickers].dot(w).rename("Portfolio")


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()


def beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return float("nan")
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return float(cov[0, 1] / cov[1, 1])


def summary_stats(prices: pd.DataFrame, benchmark_prices: pd.Series | None = None, rf: float = DEFAULT_RF) -> pd.DataFrame:
    rets = daily_returns(prices)
    rows = []
    for col in prices.columns:
        r = rets[col].dropna()
        p = prices[col].dropna()
        b = beta(r, daily_returns(benchmark_prices.to_frame()).iloc[:, 0]) if benchmark_prices is not None else float("nan")
        rows.append({
            "Ticker": col,
            "Ann. Return": annualized_return(r),
            "Ann. Volatility": annualized_volatility(r),
            "Sharpe": sharpe_ratio(r, rf),
            "Sortino": sortino_ratio(r, rf),
            "Max Drawdown": max_drawdown(p),
            "Beta (vs SPY)": b,
        })
    return pd.DataFrame(rows).set_index("Ticker")
