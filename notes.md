# Portfolio Analyzer — Technical Overview

A single-page quantitative dashboard that turns a list of tickers into live risk
analytics, correlation structure, and a mean–variance optimized portfolio. Built
as three clean layers so the math is testable in isolation.

| | |
|---|---|
| **Stack** | Dash · Plotly · pandas · NumPy · SciPy · yfinance · Redis |
| **Python** | 3.14 |
| **Size** | ~1,120 LOC across `data.py` (163) · `analysis.py` (238) · `app.py` (721) |
| **Tests** | 41 passing (`pytest -q`), fully offline |
| **Served** | `0.0.0.0:8050` (plain HTTP) — gunicorn in Docker, app + Redis |

---

## 1. The idea

You search up symbols like `AAPL MSFT NVDA`, choose a lookback window, a benchmark, and
a risk-free rate. The app pulls adjusted daily prices from Yahoo Finance, computes the
standard suite of return and risk statistics for **every holding *and* the blended
portfolio**, then solves for the portfolios that maximize risk-adjusted return or
minimize variance — plotting them against the full efficient frontier.

**What goes in**

- **Tickers** — any number, searched by symbol or company name against Yahoo's
  lookup endpoint, de-duplicated automatically
- **Period** — 6M · 1Y · 2Y · 5Y lookback
- **Benchmark** — SPY, QQQ, or none (drives the beta column)
- **Risk-free %** — feeds Sharpe, Sortino, and the optimizer
- **Weights** — set by hand or auto-filled by the optimizer

**What comes out**

- **8 metric cards** — Portfolio CAGR, volatility, Sharpe, Sortino, max drawdown, VaR, CVaR, avg total return
- **7 chart panels** — cumulative returns, allocation, rolling risk, drawdown, correlation, efficient frontier, candlesticks
- **Stats table** — per-asset rows plus a highlighted Portfolio row
- **Stock detail** — candlesticks, moving averages, fundamentals
- **CSV export** — the full statistics table, one click

---

## 2. Everything it contains

Grouped by responsibility. Tags mark what was **[new]**, **[fixed]**, or **[wired]**
(dead code brought to life) in the recent round of work.

### I. Market data (`data.py`)

- **Live prices** — adjusted daily closes via `yfinance`
- **Shared TTL cache** — Redis-backed, 5 min for prices / 10 min for info. Every
  gunicorn worker reads the same entries, so one fetch serves the whole app **[new]**
- **JSON-only serialization** — never pickle, so a tampered entry can't execute code **[new]**
- **Fallback** — no `REDIS_URL`, or Redis unreachable, degrades to a per-process
  dict rather than failing the request **[new]**
- **OHLCV** — per-ticker candles for the detail chart
- **Fundamentals** — name, sector, market cap, P/E, dividend yield, 52-week range **[wired]**
- **Resilience** — failed tickers are dropped with a visible warning **[new]**

### II. Risk & return analytics (`analysis.py`)

- **Core metrics** — CAGR, annualized volatility, Sharpe, Sortino, max drawdown
- **Tail risk** — historical VaR 95% and CVaR 95% (expected shortfall) **[new]**
- **Market sensitivity** — beta vs the chosen benchmark; the column self-labels **[fixed]**
- **Portfolio row** — the blend sits alongside its holdings in the stats table **[new]**
- **Rolling views** — 30-day volatility, 63-day Sharpe

### III. Optimization (`analysis.py`)

- **Max Sharpe** — long-only tangency portfolio via SciPy SLSQP **[new]**
- **Min variance** — lowest-risk feasible mix **[new]**
- **Efficient frontier** — 40 solved points, with assets, optimal, and current portfolio overlaid **[new]**
- **Configurable risk-free rate** — the `rf` input drives the objective live **[new]**
- **Weight cap** — optional per-asset ceiling for diversification **[new]**

### IV. Interface & delivery (`app.py`)

- **Dynamic weights** — per-ticker inputs generated from the ticker box
- **Ticker search** — symbol or company name, resolved against Yahoo's lookup **[new]**
- **Saved portfolios** — named setups recalled from `localStorage` **[new]**
- **Optimize buttons** — Equal / Max Sharpe / Min Vol fill in the weights **[new]**
- **CSV export** — download the stats table **[new]**
- **Dark theme + tooltips** — Bootstrap DARKLY, hover explanations on every metric
- **Risk-free input** — editable in the UI, clamped to `[0, 25%]`, drives Sharpe,
  Sortino, and the optimizer live **[new]**
- **Dockerized** — gunicorn + Redis via `docker compose`, restarts on reboot **[new]**

---

## 3. The design — three layers, one rule

The system is deliberately split so that **`analysis.py` has no dependency on Dash or
the network**. That single constraint is what makes the quant logic unit-testable
offline with seeded random data. Data flows one direction, top to bottom:

```
┌─────────────┐   prices in,        ┌──────────────┐   metrics in,     ┌─────────┐
│  data.py    │   DataFrames out    │ analysis.py  │   figures out     │ app.py  │
│  fetch +    │ ──────────────────► │  pure quant  │ ────────────────► │ Dash UI │
│  cache      │                     │  core        │                   │         │
│  (163 loc)  │                     │  (238 loc)   │                   │(721 loc)│
└─────────────┘                     └──────────────┘                   └─────────┘
 only network layer          NumPy · pandas · SciPy only         layout + 5 callbacks
       │
       └── Redis (shared TTL cache, all workers)
```

**`data.py` — fetch & cache.** Wraps `yfinance` behind a Redis TTL cache so repeated
views don't re-hit the API — and so the *other* gunicorn worker doesn't either. The
only layer that touches the network.
`fetch_prices` · `fetch_ohlcv` · `fetch_info` · `_cached` · `_encode` · `_decode` · `clear_cache`

**`analysis.py` — pure quant core.** Nothing but NumPy, pandas, and SciPy. Every
statistic and the optimizer live here — no I/O, no framework, fully deterministic.
19 functions including `daily_returns` · `sharpe_ratio` · `sortino_ratio` ·
`max_drawdown` · `value_at_risk` · `conditional_var` · `beta` · `optimize_portfolio` ·
`efficient_frontier` · `summary_stats`.

**`app.py` — Dash UI.** Layout, Plotly figure builders, and 5 callbacks. Imports the
two layers below it and never re-implements their logic. `build_dashboard` ·
`build_candle_figure` · `build_fundamentals` · `update_dashboard` ·
`apply_optimization` · `export_stats`.

---

## 4. The metrics, defined

Every figure traces back to one of these. Returns are daily; annualization uses **252**
trading days. `r` is the return series, `r_f` the daily risk-free rate, `σ` its
standard deviation.

| Metric | Definition | Reads as |
|---|---|---|
| Ann. return (CAGR) | `∏(1+r)^(252/n) − 1` | Geometric growth rate, compounded |
| Ann. volatility | `σ(r) · √252` | Dispersion of returns, scaled to a year |
| Sharpe ratio | `(r̄ − r_f)/σ · √252` | Excess return per unit of *total* risk |
| Sortino ratio | `(r̄ − r_f)/σ_down · √252` | Same, but penalizes only downside moves |
| Max drawdown | `min((P_t − peak)/peak)` | Worst peak-to-trough loss (lower is worse) |
| VaR 95% | `percentile_5(r)` | Daily loss exceeded only 5% of days |
| CVaR 95% | `mean(r ≤ VaR)` | Average loss *within* that worst 5% tail |
| Beta | `cov(r, r_bm)/var(r_bm)` | Sensitivity to benchmark (1.0 = market) |
| Efficient frontier | `min wᵀΣw s.t. wᵀμ = t` | Lowest risk per target return, `0 ≤ w ≤ cap` |

---

## 5. Decisions worth calling out

- **Pure core, testable in the dark.** Because `analysis.py` imports no Dash and no
  network, all 41 tests run offline against seeded synthetic prices — the optimizer,
  VaR, and beta are verified without ever calling Yahoo, and the cache layer is
  verified without a running Redis.

- **Convex, long-only optimization.** SciPy's SLSQP minimizes portfolio variance (or
  negative Sharpe) subject to `Σw = 1` and `0 ≤ wᵢ ≤ cap`. Corner solutions are
  expected and correct when one asset dominates on a risk-adjusted basis.

- **One source of truth for weights.** The optimize buttons don't secretly re-route the
  portfolio — they *write* the solved weights back into the visible inputs, so what you
  see is always what's analyzed.

- **Graceful degradation everywhere.** Single ticker, a failed fetch, no benchmark, a
  blank risk-free field — each path is handled: the frontier hides below two assets, the
  beta column self-labels, and `parse_rf` clamps to a sane `[0, 25%]` range.

- **Cache the network, not the math.** The TTL cache in `data.py` is the only state the
  server keeps. Analytics are recomputed on every Analyze — cheap, and it keeps results
  honest when inputs change.

- **Saved portfolios stay on the client.** They live in a `storage_type="local"` store,
  i.e. the browser's `localStorage` — no accounts, no database, nothing to migrate, and
  no server-side state to contradict the point above. The trade is that they follow the
  device rather than the person. Recalled weights are read back off the *selection*
  (`portfolio-select` + the store) inside `update_weight_inputs`, rather than being
  pushed into a staging store by the load callback. Both would work, but the selection is
  already committed before the recalled tickers arrive, so this version depends on no
  ordering between two callbacks racing to populate the same inputs.

- **The cache lives outside the process.** Under gunicorn the in-memory dict meant each
  worker kept its own copy and re-fetched the same ticker. It now lives in Redis, so a
  fetch by one worker serves them all. Two constraints fell out of that: entries are
  **JSON, never pickle** (a tampered entry must not be able to execute code on read),
  and column dtypes are carried explicitly — `read_json` would otherwise infer `int64`
  for a float column that happens to hold whole numbers, so a cached frame would come
  back subtly unlike the one that was stored. A dead Redis degrades to a local dict
  instead of a 500.

- **Pattern-matching callbacks.** Per-ticker weight fields are generated on the fly and
  addressed with Dash `ALL` pattern IDs, so the UI scales to any number of holdings
  without hard-coded slots.

---

## 6. Running it

```bash
# dev — hot reload on, in-memory cache (no Redis needed)
DEBUG=1 python3 app.py

# the real thing — gunicorn + Redis, restarts on reboot
docker compose up -d --build   # also the command to deploy code changes
docker compose logs -f         # live logs
docker compose ps              # health
```

Two containers: the app (gunicorn, `app:server`, 2 workers × 4 threads) and Redis
(256 MB cap, `allkeys-lru`, persistence off — it's a cache, nothing in it is precious).
Redis publishes **no port**; it is reachable only from the app over the compose network.
The app reads `REDIS_URL`, `HOST`, `PORT`, and `DEBUG` from the environment.

> The old systemd unit still exists in `deploy/` but is **stopped and disabled** — it
> competes for port 8050. Never run both.

**Access:** http://192.168.1.149:8050 (plain HTTP — use `http://`, not `https://`).

---

## Caveats

- The risk-free rate defaults to ~5.25% but is now editable in the UI.
- Optimization is **long-only** and based on historical moments over the selected
  period — it is descriptive, **not** investment advice.
- Market data comes from Yahoo Finance via `yfinance` and may be delayed or incomplete
  for some symbols; unresolved tickers are excluded with a warning.

---

*There is also a styled HTML version of this overview at `docs/overview.html`.*
