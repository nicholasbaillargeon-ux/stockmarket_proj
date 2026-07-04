"""Statistical analysis: returns, risk metrics, and portfolio math."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

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


def value_at_risk(returns: pd.Series, level: float = 0.95) -> float:
    """Historical Value at Risk: the daily-return quantile at (1 - level).

    Returned as a negative number (a loss). e.g. -0.023 == a 2.3% one-day
    loss is exceeded only (1 - level) of the time.
    """
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    return float(np.percentile(r, (1 - level) * 100))


def conditional_var(returns: pd.Series, level: float = 0.95) -> float:
    """Conditional VaR (expected shortfall): the mean loss in the worst
    (1 - level) tail of daily returns. Returned as a negative number.
    """
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    var = np.percentile(r, (1 - level) * 100)
    tail = r[r <= var]
    if tail.empty:
        return float(var)
    return float(tail.mean())


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


# ── Portfolio optimization ──────────────────────────────────────────────────
def _annualized_moments(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (annualized mean vector, annualized covariance matrix)."""
    mu = returns.mean().values * TRADING_DAYS
    cov = returns.cov().values * TRADING_DAYS
    return mu, cov


def _portfolio_perf(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float) -> tuple[float, float, float]:
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    sharpe = (ret - rf) / vol if vol > 0 else float("nan")
    return ret, vol, sharpe


def optimize_portfolio(
    returns: pd.DataFrame,
    objective: str = "sharpe",
    rf: float = DEFAULT_RF,
    max_weight: float = 1.0,
) -> dict:
    """Long-only optimization. objective is 'sharpe' (max) or 'min_vol'.

    Returns {'weights': {ticker: w}, 'return', 'volatility', 'sharpe'}.
    """
    cols = list(returns.columns)
    n = len(cols)
    if n == 0:
        return {"weights": {}, "return": float("nan"), "volatility": float("nan"), "sharpe": float("nan")}
    if n == 1:
        mu, cov = _annualized_moments(returns)
        ret, vol, sharpe = _portfolio_perf(np.array([1.0]), mu, cov, rf)
        return {"weights": {cols[0]: 1.0}, "return": ret, "volatility": vol, "sharpe": sharpe}

    mu, cov = _annualized_moments(returns)
    bounds = tuple((0.0, max_weight) for _ in range(n))
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    x0 = np.full(n, 1.0 / n)

    if objective == "min_vol":
        obj = lambda w: w @ cov @ w
    else:  # max sharpe → minimize negative sharpe
        def obj(w):
            vol = np.sqrt(w @ cov @ w)
            return -(w @ mu - rf) / vol if vol > 0 else 1e9

    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"maxiter": 500, "ftol": 1e-9})
    w = res.x if res.success else x0
    w = np.clip(w, 0, None)
    w = w / w.sum() if w.sum() > 0 else x0
    ret, vol, sharpe = _portfolio_perf(w, mu, cov, rf)
    return {
        "weights": {c: float(wi) for c, wi in zip(cols, w)},
        "return": ret,
        "volatility": vol,
        "sharpe": sharpe,
    }


def efficient_frontier(returns: pd.DataFrame, n_points: int = 40, rf: float = DEFAULT_RF) -> pd.DataFrame:
    """Sample the long-only efficient frontier.

    Returns a DataFrame with columns ['return', 'volatility', 'sharpe'],
    one row per target-return level. Empty if fewer than 2 assets.
    """
    cols = list(returns.columns)
    n = len(cols)
    if n < 2:
        return pd.DataFrame(columns=["return", "volatility", "sharpe"])

    mu, cov = _annualized_moments(returns)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    x0 = np.full(n, 1.0 / n)
    targets = np.linspace(mu.min(), mu.max(), n_points)

    rows = []
    for target in targets:
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: w @ mu - t},
        )
        res = minimize(lambda w: w @ cov @ w, x0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"maxiter": 500, "ftol": 1e-9})
        if not res.success:
            continue
        ret, vol, sharpe = _portfolio_perf(res.x, mu, cov, rf)
        rows.append({"return": ret, "volatility": vol, "sharpe": sharpe})
    return pd.DataFrame(rows)


def summary_stats(
    prices: pd.DataFrame,
    benchmark_prices: pd.Series | None = None,
    rf: float = DEFAULT_RF,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Per-asset risk/return table. Adds a 'Portfolio' row when weights given.

    The beta column header reflects the benchmark's name (falls back to
    'Benchmark') so it is correct for SPY, QQQ, or any custom benchmark.
    """
    rets = daily_returns(prices)
    bm_rets = daily_returns(benchmark_prices.to_frame()).iloc[:, 0] if benchmark_prices is not None else None
    bm_name = getattr(benchmark_prices, "name", None) or "Benchmark"
    beta_col = f"Beta (vs {bm_name})"

    def _row(name, r, p):
        return {
            "Ticker": name,
            "Ann. Return": annualized_return(r),
            "Ann. Volatility": annualized_volatility(r),
            "Sharpe": sharpe_ratio(r, rf),
            "Sortino": sortino_ratio(r, rf),
            "Max Drawdown": max_drawdown(p),
            "VaR 95%": value_at_risk(r),
            "CVaR 95%": conditional_var(r),
            beta_col: beta(r, bm_rets) if bm_rets is not None else float("nan"),
        }

    rows = [_row(col, rets[col].dropna(), prices[col].dropna()) for col in prices.columns]

    if weights and any(weights.get(t, 0) for t in prices.columns):
        port_ret = portfolio_returns(rets, weights)
        port_prices = (1 + port_ret).cumprod()
        rows.append(_row("Portfolio", port_ret, port_prices))

    return pd.DataFrame(rows).set_index("Ticker")
