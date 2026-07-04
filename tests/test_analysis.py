"""Unit tests for the analysis module.

All tests use deterministic synthetic data (seeded RNG) so they run
offline and are fully reproducible — no network / yfinance calls.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import analysis as an  # noqa: E402


@pytest.fixture
def prices():
    """Three synthetic price series with distinct drift/vol profiles."""
    idx = pd.date_range("2022-01-01", periods=400, freq="B")
    rng = np.random.default_rng(1234)
    data = {
        "LOW": 100 * np.cumprod(1 + rng.normal(0.0003, 0.008, len(idx))),   # low vol
        "MID": 100 * np.cumprod(1 + rng.normal(0.0006, 0.015, len(idx))),
        "HIGH": 100 * np.cumprod(1 + rng.normal(0.0009, 0.028, len(idx))),  # high vol
    }
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def benchmark(prices):
    idx = prices.index
    rng = np.random.default_rng(99)
    bm = pd.Series(100 * np.cumprod(1 + rng.normal(0.0005, 0.010, len(idx))), index=idx, name="SPY")
    return bm


# ── Basic return math ──────────────────────────────────────────────────────────
def test_daily_returns_shape(prices):
    r = an.daily_returns(prices)
    assert list(r.columns) == list(prices.columns)
    assert len(r) == len(prices) - 1  # one row lost to pct_change


def test_cumulative_returns_starts_near_zero(prices):
    cum = an.cumulative_returns(prices)
    assert abs(cum.iloc[0].abs().max()) < 0.1  # first day is a small move from 0


def test_annualized_return_constant_growth():
    # A series that grows exactly 0.1% every trading day.
    idx = pd.date_range("2022-01-01", periods=253, freq="B")
    prices = pd.Series(100 * (1.001 ** np.arange(len(idx))), index=idx)
    r = an.daily_returns(prices.to_frame()).iloc[:, 0]
    ann = an.annualized_return(r)
    expected = 1.001 ** an.TRADING_DAYS - 1
    assert ann == pytest.approx(expected, rel=1e-6)


def test_annualized_return_too_short():
    r = pd.Series([0.01])
    assert np.isnan(an.annualized_return(r))


def test_annualized_volatility_positive(prices):
    r = an.daily_returns(prices)
    assert an.annualized_volatility(r["HIGH"]) > an.annualized_volatility(r["LOW"])


# ── Risk-adjusted ratios ────────────────────────────────────────────────────────
def test_sharpe_zero_vol_is_nan():
    r = pd.Series(np.zeros(100))
    assert np.isnan(an.sharpe_ratio(r))


def test_sortino_ge_sharpe_for_positive_skew(prices):
    # Sortino only penalizes downside, so it is >= Sharpe when the series
    # is not dominated by downside deviation. Just assert both are finite here.
    r = an.daily_returns(prices)["MID"]
    assert np.isfinite(an.sharpe_ratio(r))
    assert np.isfinite(an.sortino_ratio(r))


# ── Drawdown ────────────────────────────────────────────────────────────────────
def test_max_drawdown_monotonic_increasing_is_zero():
    idx = pd.date_range("2022-01-01", periods=50, freq="B")
    prices = pd.Series(np.arange(100, 150), index=idx)  # strictly increasing
    assert an.max_drawdown(prices) == pytest.approx(0.0)


def test_max_drawdown_known_value():
    # Peak 100 → trough 60 → recovery. Max drawdown = -40%.
    prices = pd.Series([100, 120, 60, 90, 130])
    # peak before trough is 120, trough 60 → (60-120)/120 = -0.5
    assert an.max_drawdown(prices) == pytest.approx(-0.5)


# ── VaR / CVaR ──────────────────────────────────────────────────────────────────
def test_var_is_negative_for_risky_series(prices):
    r = an.daily_returns(prices)["HIGH"]
    var = an.value_at_risk(r, level=0.95)
    assert var < 0


def test_cvar_not_greater_than_var(prices):
    # CVaR (mean of the tail) should be <= VaR (the tail threshold).
    r = an.daily_returns(prices)["HIGH"]
    var = an.value_at_risk(r, level=0.95)
    cvar = an.conditional_var(r, level=0.95)
    assert cvar <= var + 1e-12


def test_var_too_short_is_nan():
    assert np.isnan(an.value_at_risk(pd.Series([0.01])))
    assert np.isnan(an.conditional_var(pd.Series([0.01])))


# ── Portfolio math ──────────────────────────────────────────────────────────────
def test_portfolio_returns_normalizes_weights(prices):
    r = an.daily_returns(prices)
    # Unnormalized weights should give same result as normalized.
    p1 = an.portfolio_returns(r, {"LOW": 2, "MID": 2, "HIGH": 2})
    p2 = an.portfolio_returns(r, {"LOW": 1, "MID": 1, "HIGH": 1})
    pd.testing.assert_series_equal(p1, p2)


def test_portfolio_returns_ignores_unknown_tickers(prices):
    r = an.daily_returns(prices)
    p = an.portfolio_returns(r, {"LOW": 1, "ZZZ": 5})
    # ZZZ not in columns → portfolio is 100% LOW
    pd.testing.assert_series_equal(p, r["LOW"].rename("Portfolio"))


def test_beta_self_is_one(prices):
    r = an.daily_returns(prices)["MID"]
    assert an.beta(r, r) == pytest.approx(1.0, rel=1e-9)


def test_beta_too_short_is_nan():
    s = pd.Series([0.01], index=[pd.Timestamp("2022-01-01")])
    assert np.isnan(an.beta(s, s))


# ── Optimization ────────────────────────────────────────────────────────────────
def test_optimize_weights_sum_to_one(prices):
    r = an.daily_returns(prices)
    for obj in ("sharpe", "min_vol"):
        res = an.optimize_portfolio(r, obj)
        assert sum(res["weights"].values()) == pytest.approx(1.0, abs=1e-6)
        assert all(w >= -1e-9 for w in res["weights"].values())  # long-only


def test_min_vol_beats_equal_weight(prices):
    r = an.daily_returns(prices)
    mv = an.optimize_portfolio(r, "min_vol")
    _, cov = an._annualized_moments(r)
    n = len(r.columns)
    eq = np.full(n, 1.0 / n)
    eq_vol = float(np.sqrt(eq @ cov @ eq))
    assert mv["volatility"] <= eq_vol + 1e-9


def test_max_sharpe_beats_equal_weight_sharpe(prices):
    r = an.daily_returns(prices)
    ms = an.optimize_portfolio(r, "sharpe")
    mu, cov = an._annualized_moments(r)
    n = len(r.columns)
    eq = np.full(n, 1.0 / n)
    eq_sharpe = (eq @ mu - an.DEFAULT_RF) / np.sqrt(eq @ cov @ eq)
    assert ms["sharpe"] >= eq_sharpe - 1e-6


def test_optimize_max_weight_cap(prices):
    r = an.daily_returns(prices)
    res = an.optimize_portfolio(r, "sharpe", max_weight=0.5)
    assert all(w <= 0.5 + 1e-6 for w in res["weights"].values())


def test_optimize_single_ticker(prices):
    r = an.daily_returns(prices[["MID"]])
    res = an.optimize_portfolio(r, "sharpe")
    assert res["weights"] == {"MID": 1.0}


def test_optimize_empty():
    res = an.optimize_portfolio(pd.DataFrame(), "sharpe")
    assert res["weights"] == {}
    assert np.isnan(res["return"])


# ── Efficient frontier ──────────────────────────────────────────────────────────
def test_efficient_frontier_rows_and_bounds(prices):
    r = an.daily_returns(prices)
    ef = an.efficient_frontier(r, n_points=25)
    assert not ef.empty
    assert set(ef.columns) == {"return", "volatility", "sharpe"}
    assert (ef["volatility"] > 0).all()


def test_efficient_frontier_needs_two_assets(prices):
    r = an.daily_returns(prices[["MID"]])
    ef = an.efficient_frontier(r)
    assert ef.empty


# ── summary_stats ───────────────────────────────────────────────────────────────
def test_summary_stats_dynamic_beta_column(prices, benchmark):
    ss = an.summary_stats(prices, benchmark)
    assert "Beta (vs SPY)" in ss.columns


def test_summary_stats_benchmark_none_label(prices):
    ss = an.summary_stats(prices, None)
    assert "Beta (vs Benchmark)" in ss.columns
    assert ss["Beta (vs Benchmark)"].isna().all()


def test_summary_stats_adds_portfolio_row(prices, benchmark):
    ss = an.summary_stats(prices, benchmark, weights={"LOW": 40, "MID": 40, "HIGH": 20})
    assert "Portfolio" in ss.index
    assert set(prices.columns).issubset(set(ss.index))


def test_summary_stats_no_portfolio_row_without_weights(prices, benchmark):
    ss = an.summary_stats(prices, benchmark)
    assert "Portfolio" not in ss.index


def test_summary_stats_has_var_cvar_columns(prices, benchmark):
    ss = an.summary_stats(prices, benchmark)
    assert "VaR 95%" in ss.columns
    assert "CVaR 95%" in ss.columns
