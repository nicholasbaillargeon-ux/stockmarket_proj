# Portfolio Analyzer

An interactive [Dash](https://dash.plotly.com/) dashboard for analyzing a
portfolio of equities using live market data and standard quantitative
risk/return metrics — including **mean-variance portfolio optimization**.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)

## Features

- **Live market data** via `yfinance`, cached in-memory with a short TTL.
- **Custom weights** — set each holding's weight, or let the optimizer pick.
- **Portfolio optimization** (long-only, via `scipy.optimize`):
  - **Max Sharpe** and **Min Volatility** portfolios.
  - **Efficient frontier** plotted against your holdings and current mix.
- **Risk & return metrics:** CAGR, annualized volatility, Sharpe, Sortino,
  max drawdown, and historical **VaR / CVaR (95%)** — for each asset *and*
  the blended portfolio.
- **Visuals:** cumulative returns vs. a benchmark, allocation donut, rolling
  volatility, rolling Sharpe, drawdown curves, and a return-correlation heatmap.
- **Per-stock detail:** candlestick chart with 20/50-day moving averages and a
  volume subplot, plus a fundamentals panel (market cap, P/E, dividend yield,
  52-week range).
- **CSV export** of the per-asset statistics table.

## Project layout

| File            | Responsibility                                              |
| --------------- | ----------------------------------------------------------- |
| `data.py`       | Market-data fetching (`yfinance`) with a TTL cache.         |
| `analysis.py`   | Pure quant functions: returns, risk metrics, optimization.  |
| `app.py`        | Dash UI — layout, figures, and callbacks.                   |
| `tests/`        | Offline unit tests for `analysis.py` (deterministic).       |

`analysis.py` has **no Dash or network dependencies**, so the math is unit-tested
in isolation with seeded synthetic data.

## Getting started

```bash
# 1. Install dependencies (a virtualenv is recommended)
pip install -r requirements.txt

# 2. Run the app
python app.py

# 3. Open the dashboard
#    http://localhost:8050
```

### Usage

1. Enter tickers (e.g. `AAPL MSFT NVDA`), pick a period and benchmark.
2. Set portfolio weights manually, or click **★ Max Sharpe** / **◆ Min Vol** to
   have the optimizer fill them in, then press **Analyze**.
3. Inspect the efficient frontier — the ✕ marks your current mix relative to the
   optimal frontier and the individual assets.

## Development

```bash
pip install -r requirements-dev.txt
pytest -q
```

The test suite is fully offline (no `yfinance` calls) and covers the return
math, risk metrics, VaR/CVaR, beta, optimization (including weight-sum,
long-only, and per-asset caps), the efficient frontier, and `summary_stats`.

## Notes & caveats

- The risk-free rate is a fixed approximation (`DEFAULT_RF` in `analysis.py`);
  adjust it there if you want current-market accuracy.
- Optimization is **long-only** (`0 ≤ wᵢ ≤ max_weight`) and based on historical
  moments over the selected period — it is descriptive, **not** investment advice.
- Market data is provided by Yahoo Finance via `yfinance` and may be delayed or
  incomplete for some symbols; tickers that return no data are excluded with a
  warning.
