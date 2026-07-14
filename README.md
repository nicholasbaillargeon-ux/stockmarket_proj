# Portfolio Analyzer

An interactive [Dash](https://dash.plotly.com/) dashboard for analyzing a
portfolio of equities using live market data and standard quantitative
risk/return metrics — including **mean-variance portfolio optimization**.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)

## Features

- **Live market data** via `yfinance`, cached in Redis with a short TTL — the cache
  is shared across all workers, so one fetch serves every request.
- **Custom weights** — set each holding's weight, or let the optimizer pick.
- **Editable risk-free rate** — drives Sharpe, Sortino, and the optimizer live.
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
| `data.py`       | Market-data fetching (`yfinance`) with a Redis TTL cache.   |
| `analysis.py`   | Pure quant functions: returns, risk metrics, optimization.  |
| `app.py`        | Dash UI — layout, figures, and callbacks.                   |
| `tests/`        | Offline unit tests for `analysis.py` and the cache layer.   |

`analysis.py` has **no Dash or network dependencies**, so the math is unit-tested
in isolation with seeded synthetic data.

## Getting started

The app ships as a two-container stack (app + Redis):

```bash
docker compose up -d --build     # start; http://localhost:8050
docker compose logs -f           # live logs
docker compose down              # stop
```

To run it directly instead, without Docker:

```bash
pip install -r requirements.txt
python app.py                    # http://localhost:8050
```

With no `REDIS_URL` set the cache falls back to an in-process dict, so the app runs
fine standalone — you just lose cache sharing between workers.

### Configuration

| Variable    | Default | Effect                                                |
| ----------- | ------- | ----------------------------------------------------- |
| `REDIS_URL` | *unset* | Redis to cache into. Unset ⇒ in-memory fallback.      |
| `HOST`      | `0.0.0.0` | Bind address (direct runs).                         |
| `PORT`      | `8050`  | Bind port.                                            |
| `DEBUG`     | *unset* | `1` enables the Dash dev server with hot reload.      |

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

The test suite is fully offline — no `yfinance` calls and no running Redis. It covers
the return math, risk metrics, VaR/CVaR, beta, optimization (including weight-sum,
long-only, and per-asset caps), the efficient frontier, `summary_stats`, and the
cache layer's serialization round-trip and degradation when Redis is unreachable.

## Notes & caveats

- The risk-free rate defaults to `DEFAULT_RF` (`analysis.py`) but is editable in the
  UI; the input is clamped to a sane `[0, 25%]` range.
- Cache entries are serialized as **JSON, never pickle**, so a tampered or corrupted
  entry cannot execute code when it is read back.
- Optimization is **long-only** (`0 ≤ wᵢ ≤ max_weight`) and based on historical
  moments over the selected period — it is descriptive, **not** investment advice.
- Market data is provided by Yahoo Finance via `yfinance` and may be delayed or
  incomplete for some symbols; tickers that return no data are excluded with a
  warning.
