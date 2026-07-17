"""Portfolio Analyzer — Dash dashboard with real market data and statistical analysis."""

import logging
import math
import re
from pathlib import Path

import flask
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import ALL, Dash, Input, Output, State, callback, ctx, dcc, html, no_update, dash_table
from dash.exceptions import PreventUpdate

import analysis as an
import data as dt

log = logging.getLogger(__name__)

# ── App init ──────────────────────────────────────────────────────────────────
# The same two faces the landing page uses, so / and /app/ read as one product.
GOOGLE_FONTS = (
    "https://fonts.googleapis.com/css2"
    "?family=Inter:wght@400;500;600;700"
    "&family=JetBrains+Mono:wght@400;500;600&display=swap"
)

# The dashboard lives under /app/ so the marketing landing page can own /.
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY, GOOGLE_FONTS],
    title="Portfolio Analyzer",
    suppress_callback_exceptions=True,
    url_base_pathname="/app/",
)

# WSGI entry point for gunicorn (`gunicorn app:server`)
server = app.server

LANDING_DIR = Path(__file__).parent / "landing"


@server.route("/")
def landing():
    return flask.send_from_directory(LANDING_DIR, "index.html")


DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
DEFAULT_PERIOD = "1y"
PERIODS = {"6M": "6mo", "1Y": "1y", "2Y": "2y", "5Y": "5y"}

REPO_URL = "https://github.com/nicholasbaillargeon-ux/stockmarket_proj"

# ── Color palette ─────────────────────────────────────────────────────────────
# Shared with landing/index.html — change both together.
BG = "#1a1a2e"
CARD_BG = "#16213e"
PLOT_BG = "#0f3460"
ACCENT = "#e94560"
TEXT = "#eaeaea"
MUTED = "#98a1bb"
BORDER = "#2a2a4a"

# Reserved status colors: never reused as a series hue.
UP = "#2ecc71"
DOWN = ACCENT

FONT_SANS = "Inter, system-ui, -apple-system, sans-serif"
FONT_MONO = "'JetBrains Mono', ui-monospace, monospace"

# Categorical series hues, applied in this fixed order. Chosen in OKLCH to clear
# the reserved red/green status zones, then checked against the PLOT_BG surface:
# adjacent worst ΔE 42.6 (lines/bars), all-pairs worst 9.2 under deuteranopia —
# the floor band, which the frontier scatter offsets with a direct label per point.
COLORS = [
    "#c06f91",  # rose
    "#727ef1",  # violet
    "#008f84",  # teal
    "#3392db",  # blue
    "#d86800",  # orange
    "#00a2cd",  # cyan
    "#8b933e",  # olive
    "#a876b7",  # magenta
]

CHART_LAYOUT = dict(
    paper_bgcolor=CARD_BG,
    plot_bgcolor=PLOT_BG,
    font=dict(color=TEXT, size=12, family=FONT_SANS),
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    xaxis=dict(gridcolor=BORDER, showgrid=True),
    yaxis=dict(gridcolor=BORDER, showgrid=True),
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_pct(v, decimals=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v * 100:+.{decimals}f}%"


def fmt_float(v, decimals=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{decimals}f}"


def fmt_money(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(v) >= div:
            return f"${v / div:.2f}{unit}"
    return f"${v:,.0f}"


def metric_card(title, value, color="#eaeaea", tooltip=None):
    body = html.Div([
        html.P(title, className="text-muted mb-1", style={"fontSize": "0.75rem"}),
        html.H5(value, style={"color": color, "fontWeight": "bold", "marginBottom": 0}),
    ], title=tooltip or "")
    return dbc.Card(
        dbc.CardBody(body),
        style={"backgroundColor": CARD_BG, "border": "1px solid #2a2a4a"},
    )


def app_header():
    """The landing page's header, rebuilt in Dash so / and /app/ share one chrome.

    The wordmark links home; the diamond mark and spacing mirror landing/index.html.
    """
    nav_link = {"color": MUTED, "textDecoration": "none", "fontSize": "14px"}
    return html.Header(
        style={
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "space-between",
            "padding": "18px 8px",
            "marginBottom": "22px",
            "borderBottom": f"1px solid {BORDER}",
        },
        children=[
            html.A(
                href="/",
                style={"display": "flex", "alignItems": "center", "gap": "12px",
                       "textDecoration": "none", "color": TEXT},
                children=[
                    html.Div(style={"width": "12px", "height": "12px", "background": ACCENT,
                                    "borderRadius": "2px", "transform": "rotate(45deg)"}),
                    html.Span("Portfolio Analyzer",
                              style={"fontWeight": 700, "letterSpacing": "-.01em", "fontSize": "17px"}),
                ],
            ),
            html.Nav(
                style={"display": "flex", "alignItems": "center", "gap": "28px"},
                children=[
                    html.Span("Real-time market data · Statistical risk analysis",
                              style={"color": MUTED, "fontSize": "12px",
                                     "letterSpacing": ".04em", "fontFamily": FONT_MONO}),
                    html.A("← Home", href="/", style=nav_link),
                    html.A("GitHub", href=REPO_URL, style=nav_link),
                ],
            ),
        ],
    )


def portfolio_bar():
    """Save / recall named portfolios, held in the browser's localStorage.

    Nothing here reaches the server: the store below is per-browser, so saved
    portfolios follow the device, not the user.
    """
    field = {"backgroundColor": PLOT_BG, "color": TEXT, "borderColor": BORDER}
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Label("Saved portfolios", style={"color": TEXT}),
                dcc.Dropdown(
                    id="portfolio-select",
                    options=[],
                    placeholder="Nothing saved yet",
                    className="pa-dropdown",
                ),
            ], md=4),
            dbc.Col([
                dbc.Label("Save current as…", style={"color": TEXT}),
                dbc.Input(id="portfolio-name", placeholder="e.g. Core holdings", style=field),
            ], md=4),
            dbc.Col([
                dbc.Label(" ", style={"display": "block"}),
                dbc.Button("Save", id="save-portfolio-btn", color="secondary",
                           outline=True, n_clicks=0, className="w-100"),
            ], md=2),
            dbc.Col([
                dbc.Label(" ", style={"display": "block"}),
                dbc.Button("Delete", id="delete-portfolio-btn", color="secondary",
                           outline=True, n_clicks=0, className="w-100"),
            ], md=2),
        ], className="g-2"),
        html.Small(id="portfolio-status", className="text-muted mt-1",
                   style={"display": "block"}),
    ])


def parse_rf(value) -> float:
    """Convert the risk-free % input to a decimal rate, clamped to [0, 0.25].

    Falls back to the module default when the field is blank, invalid, or a
    non-finite float (NaN/inf), any of which would otherwise slip past the clamp.
    """
    try:
        rf = float(value) / 100.0
    except (TypeError, ValueError):
        return an.DEFAULT_RF
    if not math.isfinite(rf):
        return an.DEFAULT_RF
    return min(max(rf, 0.0), 0.25)


# A ticker is 1–10 chars, starts with a letter, and may carry a class/exchange
# suffix (BRK.B, BF-B, ^GSPC). Anything else is dropped so it can't reach a
# yfinance call or collide with the '_'-joined cache key.
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.^-]{0,9}$")


def parse_tickers(raw: str | list[str] | None) -> list[str]:
    """Normalize a ticker selection into a de-duplicated, validated list.

    Takes the search bar's list of symbols, or a free-text string (space- or
    comma-separated) so a typed/pasted "AAPL, MSFT" still parses.
    """
    if not raw:
        return []
    if isinstance(raw, str):
        raw = raw.replace(",", " ").split()
    seen, out = set(), []
    for t in raw:
        t = str(t).strip().upper()
        if t and t not in seen and _TICKER_RE.match(t):
            seen.add(t)
            out.append(t)
    return out


def build_candle_figure(ticker: str, period: str) -> go.Figure:
    """Candlestick + MA overlays + volume subplot for one ticker."""
    period_map = {"6mo": "6mo", "1y": "1y", "2y": "2y", "5y": "2y"}
    ohlcv_period = period_map.get(period, "1y")
    try:
        df = dt.fetch_ohlcv(ticker, period=ohlcv_period)
    except Exception as exc:
        return go.Figure().update_layout(**CHART_LAYOUT, title=f"Error: {exc}")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name=ticker,
        increasing=dict(line=dict(color=UP), fillcolor=UP),
        decreasing=dict(line=dict(color=DOWN), fillcolor=DOWN),
        hovertext=ticker,
    ))
    for w, c in [(20, "#f39c12"), (50, "#9b59b6")]:
        if len(df) >= w:
            ma = df["Close"].rolling(w).mean()
            fig.add_trace(go.Scatter(
                x=ma.index, y=ma, name=f"{w}-day MA",
                line=dict(color=c, width=1.2, dash="dot"),
                hovertemplate=f"{w}-day MA: %{{y:.2f}}<extra></extra>",
            ))
    colors = [UP if c >= o else DOWN for o, c in zip(df["Open"], df["Close"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume",
        marker_color=colors, opacity=0.4, yaxis="y2",
        hovertemplate="Vol: %{y:,.0f}<extra></extra>",
    ))
    base = {k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")}
    fig.update_layout(
        **base,
        title=f"{ticker} — Price & Volume",
        xaxis=dict(gridcolor="#2a2a4a", showgrid=True, rangeslider=dict(visible=False)),
        yaxis=dict(domain=[0.25, 1.0], gridcolor="#2a2a4a"),
        yaxis2=dict(domain=[0.0, 0.2], showgrid=False, showticklabels=False),
    )
    return fig


def build_fundamentals(ticker: str):
    """Row of fundamentals fetched via data.fetch_info; degrades gracefully."""
    try:
        info = dt.fetch_info(ticker)
    except Exception:
        info = {}
    dy = info.get("dividend_yield")
    # yfinance reports dividend yield inconsistently; normalize small fractions to %.
    dy_str = "—" if dy is None else (f"{dy:.2f}%" if dy > 1 else f"{dy * 100:.2f}%")
    hi, lo = info.get("52w_high"), info.get("52w_low")
    rng = f"{lo:.2f} – {hi:.2f}" if hi and lo else "—"
    fields = [
        ("Name", info.get("name", ticker)),
        ("Sector", info.get("sector", "—")),
        ("Market Cap", fmt_money(info.get("market_cap"))),
        ("P/E (TTM)", fmt_float(info.get("pe_ratio"))),
        ("Div Yield", dy_str),
        ("52W Range", rng),
    ]
    return dbc.Row([
        dbc.Col(html.Div([
            html.Div(label, className="text-muted", style={"fontSize": "0.7rem"}),
            html.Div(str(value), style={"color": TEXT, "fontWeight": "bold", "fontSize": "0.9rem"}),
        ]), xs=6, md=2)
        for label, value in fields
    ], className="mb-3 g-2")


# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = dbc.Container(
    fluid=True,
    style={"backgroundColor": BG, "minHeight": "100vh", "padding": "20px"},
    children=[
        app_header(),

        # Controls
        dbc.Card(
            dbc.CardBody([
                portfolio_bar(),
                html.Hr(style={"borderColor": BORDER, "margin": "16px 0"}),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Tickers — search by symbol or company name", style={"color": TEXT}),
                        dcc.Dropdown(
                            id="ticker-input",
                            options=[{"label": t, "value": t} for t in DEFAULT_TICKERS],
                            value=list(DEFAULT_TICKERS),
                            multi=True,
                            placeholder="Search e.g. NVDA or Nvidia…",
                            # Keep Yahoo's relevance order; 'index' would re-sort
                            # matches by the client-side index instead.
                            search_order="original",
                            className="pa-dropdown",
                        ),
                    ], md=4),
                    dbc.Col([
                        dbc.Label("Period", style={"color": TEXT}),
                        dbc.RadioItems(
                            id="period-select",
                            options=[{"label": k, "value": v} for k, v in PERIODS.items()],
                            value=DEFAULT_PERIOD,
                            inline=True,
                            style={"color": TEXT},
                            inputCheckedClassName="text-danger",
                        ),
                    ], md=3),
                    dbc.Col([
                        dbc.Label("Benchmark", style={"color": TEXT}),
                        dbc.Select(
                            id="benchmark-select",
                            options=[
                                {"label": "S&P 500 (SPY)", "value": "SPY"},
                                {"label": "Nasdaq (QQQ)", "value": "QQQ"},
                                {"label": "None", "value": "none"},
                            ],
                            value="SPY",
                            style={"backgroundColor": PLOT_BG, "color": TEXT, "borderColor": "#2a2a4a"},
                        ),
                    ], md=2),
                    dbc.Col([
                        dbc.Label(
                            html.Span("Risk-free % (0–25)",
                                      title="Annual risk-free rate used for Sharpe, Sortino, and optimization"),
                            style={"color": TEXT}),
                        dbc.Input(
                            id="rf-input",
                            type="number",
                            value=round(an.DEFAULT_RF * 100, 2),
                            # step="any" and no min/max: browser validation would send
                            # NaN (→ null → None) for any off-step or out-of-range entry,
                            # silently reverting to DEFAULT_RF. parse_rf clamps instead.
                            step="any",
                            style={"backgroundColor": PLOT_BG, "color": TEXT, "borderColor": "#2a2a4a"},
                        ),
                    ], md=2),
                    dbc.Col([
                        dbc.Label(" ", style={"display": "block"}),
                        dbc.Button("Analyze", id="analyze-btn", color="danger", n_clicks=0, className="w-100"),
                    ], md=1),
                ]),
                html.Div(id="weights-section", className="mt-3"),
            ]),
            style={"backgroundColor": CARD_BG, "border": "1px solid #2a2a4a", "marginBottom": "16px"},
        ),

        dcc.Loading(
            id="loading",
            type="circle",
            color=ACCENT,
            children=html.Div(id="dashboard-content"),
        ),

        # Store for processed data
        dcc.Store(id="prices-store"),
        dcc.Store(id="tickers-store"),

        # Saved portfolios: {name: {tickers, weights, period, benchmark, rf}}.
        # storage_type="local" persists to localStorage, so this survives a
        # reload and a restart — it is the only state in the app that does.
        dcc.Store(id="portfolios-store", storage_type="local"),
    ],
)


# ── Dashboard builder ─────────────────────────────────────────────────────────
def build_dashboard(prices: pd.DataFrame, benchmark: pd.Series | None, tickers: list[str], weights: dict[str, float] | None = None, period: str = DEFAULT_PERIOD, rf: float = an.DEFAULT_RF):
    rets = an.daily_returns(prices)
    cum_rets = an.cumulative_returns(prices)

    if weights is None or not any(weights.get(t, 0) for t in tickers):
        weights = {t: 1.0 for t in tickers}
    else:
        weights = {t: max(weights.get(t, 0.0), 0.0) for t in tickers}

    stats = an.summary_stats(prices, benchmark, rf=rf, weights=weights)

    port_ret = an.portfolio_returns(rets, weights)
    port_prices = (1 + port_ret).cumprod()

    # ── Metric cards ──────────────────────────────────────────────────────────
    port_sharpe = an.sharpe_ratio(port_ret, rf)
    port_sortino = an.sortino_ratio(port_ret, rf)
    port_vol = an.annualized_volatility(port_ret)
    port_dd = an.max_drawdown(port_prices)
    port_cagr = an.annualized_return(port_ret)
    port_var = an.value_at_risk(port_ret)
    port_cvar = an.conditional_var(port_ret)
    total_ret = float(cum_rets.iloc[-1].mean())

    def color_for(v, good_positive=True):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return TEXT
        return "#2ecc71" if (v > 0) == good_positive else ACCENT

    cards_row = dbc.Row([
        dbc.Col(metric_card("Portfolio CAGR", fmt_pct(port_cagr), color_for(port_cagr),
                            "Annualized (geometric) portfolio return"), xs=6, md=3),
        dbc.Col(metric_card("Ann. Volatility", fmt_pct(port_vol, 1), TEXT,
                            "Annualized standard deviation of returns"), xs=6, md=3),
        dbc.Col(metric_card("Sharpe Ratio", fmt_float(port_sharpe), color_for(port_sharpe),
                            "Excess return per unit of total risk"), xs=6, md=3),
        dbc.Col(metric_card("Sortino Ratio", fmt_float(port_sortino), color_for(port_sortino),
                            "Excess return per unit of downside risk"), xs=6, md=3),
        dbc.Col(metric_card("Max Drawdown", fmt_pct(port_dd, 1), color_for(port_dd, good_positive=False),
                            "Largest peak-to-trough decline"), xs=6, md=3),
        dbc.Col(metric_card("VaR 95% (1d)", fmt_pct(port_var, 2), color_for(port_var, good_positive=False),
                            "Daily loss exceeded only 5% of the time (historical)"), xs=6, md=3),
        dbc.Col(metric_card("CVaR 95% (1d)", fmt_pct(port_cvar, 2), color_for(port_cvar, good_positive=False),
                            "Average loss on the worst 5% of days"), xs=6, md=3),
        dbc.Col(metric_card("Avg Total Return", fmt_pct(total_ret), color_for(total_ret),
                            "Mean cumulative return across holdings"), xs=6, md=3),
    ], className="mb-3 g-2")

    # ── Cumulative returns chart ───────────────────────────────────────────────
    fig_cum = go.Figure()
    for i, col in enumerate(cum_rets.columns):
        fig_cum.add_trace(go.Scatter(
            x=cum_rets.index, y=cum_rets[col] * 100,
            name=col, line=dict(color=COLORS[i % len(COLORS)], width=2),
            hovertemplate=f"<b>{col}</b><br>%{{x|%b %d, %Y}}<br>Return: %{{y:.1f}}%<extra></extra>",
        ))
    if benchmark is not None:
        bm_cum = an.cumulative_returns(benchmark.to_frame()).iloc[:, 0]
        bm_cum = bm_cum.reindex(cum_rets.index, method="ffill")
        fig_cum.add_trace(go.Scatter(
            x=bm_cum.index, y=bm_cum * 100,
            name=benchmark.name, line=dict(color="#888", width=1.5, dash="dash"),
            hovertemplate=f"<b>{benchmark.name}</b><br>%{{x|%b %d, %Y}}<br>Return: %{{y:.1f}}%<extra></extra>",
        ))
    fig_cum.update_layout(**CHART_LAYOUT, title="Cumulative Returns (%)", yaxis_ticksuffix="%")

    # ── Allocation donut ──────────────────────────────────────────────────────
    alloc_values = [weights.get(t, 0.0) for t in tickers]
    fig_alloc = go.Figure(go.Pie(
        labels=tickers,
        values=alloc_values,
        hole=0.55,
        marker=dict(colors=COLORS[:len(tickers)]),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Weight: %{percent}<extra></extra>",
    ))
    fig_alloc.update_layout(
        **CHART_LAYOUT,
        title="Portfolio Allocation",
        showlegend=False,
    )

    # ── Rolling volatility ────────────────────────────────────────────────────
    roll_vol = an.rolling_volatility(rets, window=30).dropna()
    fig_vol = go.Figure()
    for i, col in enumerate(roll_vol.columns):
        fig_vol.add_trace(go.Scatter(
            x=roll_vol.index, y=roll_vol[col] * 100,
            name=col, fill="tozeroy" if len(roll_vol.columns) == 1 else None,
            line=dict(color=COLORS[i % len(COLORS)], width=1.5),
            hovertemplate=f"<b>{col}</b><br>%{{x|%b %d, %Y}}<br>Vol: %{{y:.1f}}%<extra></extra>",
        ))
    fig_vol.update_layout(**CHART_LAYOUT, title="30-Day Rolling Volatility (annualized)", yaxis_ticksuffix="%")

    # ── Rolling Sharpe ────────────────────────────────────────────────────────
    roll_sharpe = an.rolling_sharpe(rets, window=63, rf=rf).dropna()
    fig_rs = go.Figure()
    for i, col in enumerate(roll_sharpe.columns):
        fig_rs.add_trace(go.Scatter(
            x=roll_sharpe.index, y=roll_sharpe[col],
            name=col, line=dict(color=COLORS[i % len(COLORS)], width=1.5),
            hovertemplate=f"<b>{col}</b><br>%{{x|%b %d, %Y}}<br>Sharpe: %{{y:.2f}}<extra></extra>",
        ))
    fig_rs.add_hline(y=1.0, line_dash="dot", line_color="#888", annotation_text="Sharpe = 1")
    fig_rs.update_layout(**CHART_LAYOUT, title="63-Day Rolling Sharpe Ratio")

    # ── Drawdown chart ────────────────────────────────────────────────────────
    fig_dd = go.Figure()
    for i, col in enumerate(prices.columns):
        p = prices[col].dropna()
        dd = (p - p.cummax()) / p.cummax() * 100
        fig_dd.add_trace(go.Scatter(
            x=dd.index, y=dd,
            name=col, fill="tozeroy",
            line=dict(color=COLORS[i % len(COLORS)], width=1),
            hovertemplate=f"<b>{col}</b><br>%{{x|%b %d, %Y}}<br>DD: %{{y:.1f}}%<extra></extra>",
        ))
    fig_dd.update_layout(**CHART_LAYOUT, title="Drawdown (%)", yaxis_ticksuffix="%")

    # ── Correlation heatmap ───────────────────────────────────────────────────
    corr = an.correlation_matrix(rets)
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu_r",
        zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        hovertemplate="<b>%{x} vs %{y}</b><br>Correlation: %{z:.2f}<extra></extra>",
        colorbar=dict(tickfont=dict(color=TEXT)),
    ))
    fig_corr.update_layout(**CHART_LAYOUT, title="Return Correlation Matrix")

    # ── Efficient frontier ────────────────────────────────────────────────────
    fig_ef = go.Figure()
    opt_summary = html.Div()
    if len(prices.columns) >= 2:
        ef = an.efficient_frontier(rets, n_points=40, rf=rf)
        max_sharpe = an.optimize_portfolio(rets, "sharpe", rf=rf)
        min_vol = an.optimize_portfolio(rets, "min_vol", rf=rf)

        if not ef.empty:
            fig_ef.add_trace(go.Scatter(
                x=ef["volatility"] * 100, y=ef["return"] * 100,
                mode="lines", name="Efficient frontier",
                line=dict(color="#4ea8de", width=2),
                hovertemplate="Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
            ))
        # Individual assets
        for i, col in enumerate(prices.columns):
            fig_ef.add_trace(go.Scatter(
                x=[stats.loc[col, "Ann. Volatility"] * 100],
                y=[stats.loc[col, "Ann. Return"] * 100],
                mode="markers+text", name=col, text=[col], textposition="top center",
                textfont=dict(size=9, color=TEXT),
                marker=dict(color=COLORS[i % len(COLORS)], size=9),
                hovertemplate=f"<b>{col}</b><br>Vol: %{{x:.1f}}%<br>Return: %{{y:.1f}}%<extra></extra>",
            ))
        # Optimal & current portfolios
        for res, label, sym, clr in [
            (max_sharpe, "Max Sharpe", "star", "#f1c40f"),
            (min_vol, "Min Vol", "diamond", "#2ecc71"),
        ]:
            fig_ef.add_trace(go.Scatter(
                x=[res["volatility"] * 100], y=[res["return"] * 100],
                mode="markers", name=label,
                marker=dict(color=clr, size=15, symbol=sym, line=dict(color="#000", width=1)),
                hovertemplate=f"<b>{label}</b><br>Vol: %{{x:.1f}}%<br>Return: %{{y:.1f}}%"
                              f"<br>Sharpe: {res['sharpe']:.2f}<extra></extra>",
            ))
        fig_ef.add_trace(go.Scatter(
            x=[port_vol * 100], y=[port_cagr * 100],
            mode="markers", name="Current",
            marker=dict(color=ACCENT, size=14, symbol="x", line=dict(width=1)),
            hovertemplate="<b>Current portfolio</b><br>Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
        ))
        fig_ef.update_layout(
            **CHART_LAYOUT, title="Efficient Frontier (annualized)",
            xaxis_title="Volatility (%)", yaxis_title="Return (%)",
        )

        def _weights_line(res):
            items = sorted(res["weights"].items(), key=lambda kv: -kv[1])
            return ", ".join(f"{t} {w * 100:.0f}%" for t, w in items if w > 0.005)

        opt_summary = dbc.Card(dbc.CardBody([
            html.H6("Optimal Portfolios", style={"color": TEXT}),
            html.P([html.Span("★ Max Sharpe  ", style={"color": "#f1c40f", "fontWeight": "bold"}),
                    f"Sharpe {max_sharpe['sharpe']:.2f} · Ret {fmt_pct(max_sharpe['return'])} · Vol {fmt_pct(max_sharpe['volatility'])}"],
                   className="mb-1", style={"color": TEXT, "fontSize": "0.85rem"}),
            html.Small(_weights_line(max_sharpe), className="text-muted d-block mb-2"),
            html.P([html.Span("◆ Min Vol  ", style={"color": "#2ecc71", "fontWeight": "bold"}),
                    f"Sharpe {min_vol['sharpe']:.2f} · Ret {fmt_pct(min_vol['return'])} · Vol {fmt_pct(min_vol['volatility'])}"],
                   className="mb-1", style={"color": TEXT, "fontSize": "0.85rem"}),
            html.Small(_weights_line(min_vol), className="text-muted d-block"),
            html.Hr(style={"borderColor": "#2a2a4a"}),
            html.Small([f"Risk-free rate: {rf * 100:.2f}%. ",
                        "Use the Max Sharpe / Min Vol buttons above to apply these weights."],
                       className="text-muted"),
        ]), style={"backgroundColor": CARD_BG, "border": "1px solid #2a2a4a", "height": "100%"})
    else:
        fig_ef.update_layout(**CHART_LAYOUT, title="Efficient Frontier (needs ≥ 2 assets)")

    # ── Stats table ───────────────────────────────────────────────────────────
    PCT_COLS = {"Ann. Return", "Ann. Volatility", "Max Drawdown", "VaR 95%", "CVaR 95%"}
    table_df = stats.copy()
    for col in table_df.columns:
        if col in PCT_COLS:
            table_df[col] = table_df[col].map(lambda v: fmt_pct(v))
        else:  # Sharpe, Sortino, Beta (vs <benchmark>)
            table_df[col] = table_df[col].map(lambda v: fmt_float(v))
    table_df = table_df.reset_index()

    stats_table = dash_table.DataTable(
        data=table_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in table_df.columns],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": PLOT_BG, "color": TEXT, "fontWeight": "bold", "border": "1px solid #2a2a4a"},
        style_cell={"backgroundColor": CARD_BG, "color": TEXT, "border": "1px solid #2a2a4a", "textAlign": "center", "padding": "8px"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#1a2a4a"},
            {"if": {"filter_query": '{Ticker} = "Portfolio"'},
             "backgroundColor": PLOT_BG, "fontWeight": "bold",
             "borderTop": f"2px solid {ACCENT}"},
        ],
    )

    # ── Individual stock section ───────────────────────────────────────────────
    default_stock = tickers[0]
    stock_selector = dbc.Row([
        dbc.Col(html.H5("Individual Stock", style={"color": TEXT}), width="auto"),
        dbc.Col(
            dbc.Select(
                id="stock-select",
                options=[{"label": t, "value": t} for t in tickers],
                value=default_stock,
                style={"backgroundColor": PLOT_BG, "color": TEXT, "borderColor": "#2a2a4a", "width": "120px"},
            ),
            width="auto",
        ),
    ], align="center", className="mb-2")

    # ── Assemble layout ───────────────────────────────────────────────────────
    return html.Div([
        cards_row,

        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_cum, config={"displayModeBar": False}), md=8),
            dbc.Col(dcc.Graph(figure=fig_alloc, config={"displayModeBar": False}), md=4),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_vol, config={"displayModeBar": False}), md=6),
            dbc.Col(dcc.Graph(figure=fig_rs, config={"displayModeBar": False}), md=6),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_corr, config={"displayModeBar": False}), md=5),
            dbc.Col(dcc.Graph(figure=fig_dd, config={"displayModeBar": False}), md=7),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_ef, config={"displayModeBar": False}), md=8),
            dbc.Col(opt_summary, md=4),
        ], className="mb-3"),

        dbc.Card(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(html.H5("Per-Asset Statistics", style={"color": TEXT}), width="auto"),
                    dbc.Col(dbc.Button("⬇ Export CSV", id="export-btn", color="secondary",
                                       outline=True, size="sm"), width="auto", className="ms-auto"),
                ], align="center", justify="between", className="mb-2"),
                stats_table,
            ]),
            style={"backgroundColor": CARD_BG, "border": "1px solid #2a2a4a", "marginBottom": "16px"},
        ),

        dbc.Card(
            dbc.CardBody([
                stock_selector,
                html.Div(build_fundamentals(default_stock), id="fundamentals-panel"),
                dcc.Graph(id="candle-chart", figure=build_candle_figure(default_stock, period),
                          config={"displayModeBar": "hover"}),
            ]),
            style={"backgroundColor": CARD_BG, "border": "1px solid #2a2a4a"},
        ),

        # Stash unformatted stats for CSV export
        dcc.Store(id="stats-store", data=stats.reset_index().to_dict("records")),
        dcc.Download(id="stats-download"),
    ])


# ── Callbacks ─────────────────────────────────────────────────────────────────
@callback(
    Output("portfolios-store", "data"),
    Output("portfolio-select", "value"),
    Output("portfolio-name", "value"),
    Output("portfolio-status", "children"),
    Input("save-portfolio-btn", "n_clicks"),
    Input("delete-portfolio-btn", "n_clicks"),
    State("portfolio-name", "value"),
    State("portfolio-select", "value"),
    State("ticker-input", "value"),
    State("period-select", "value"),
    State("benchmark-select", "value"),
    State("rf-input", "value"),
    State({"type": "weight-input", "index": ALL}, "value"),
    State({"type": "weight-input", "index": ALL}, "id"),
    State("portfolios-store", "data"),
    prevent_initial_call=True,
)
def save_or_delete_portfolio(n_save, n_delete, name, selected, ticker_input, period,
                             benchmark, rf_pct, weight_values, weight_ids, saved):
    """Write the current setup to localStorage under a name, or drop a saved one."""
    saved = dict(saved or {})

    if ctx.triggered_id == "delete-portfolio-btn":
        if not selected or selected not in saved:
            return no_update, no_update, no_update, "Pick a saved portfolio to delete."
        saved.pop(selected)
        return saved, None, no_update, f"Deleted “{selected}”."

    name = (name or "").strip()
    if not name:
        return no_update, no_update, no_update, "Give the portfolio a name to save it."
    tickers = parse_tickers(ticker_input)
    if not tickers:
        return no_update, no_update, no_update, "Add at least one ticker to save."

    existed = name in saved
    # Store the raw rf percentage, not parse_rf's decimal, so it round-trips
    # straight back into the input it came from.
    saved[name] = {
        "tickers": tickers,
        "weights": {
            wid["index"]: v
            for wid, v in zip(weight_ids, weight_values)
            if wid["index"] in tickers and v is not None and v >= 0
        },
        "period": period,
        "benchmark": benchmark,
        "rf": rf_pct,
    }
    verb = "Updated" if existed else "Saved"
    return saved, name, "", f"{verb} “{name}” · {len(tickers)} holdings."


@callback(
    Output("portfolio-select", "options"),
    Input("portfolios-store", "data"),
)
def list_saved_portfolios(saved):
    """Mirror localStorage into the recall dropdown, including on first paint."""
    saved = saved or {}
    return [
        {"label": f"{name} · {len(saved[name].get('tickers') or [])} holdings", "value": name}
        for name in sorted(saved, key=str.lower)
    ]


@callback(
    Output("ticker-input", "value"),
    Output("ticker-input", "options", allow_duplicate=True),
    Output("period-select", "value"),
    Output("benchmark-select", "value"),
    Output("rf-input", "value"),
    Input("portfolio-select", "value"),
    State("portfolios-store", "data"),
    prevent_initial_call=True,
)
def load_portfolio(name, saved):
    """Recall a saved portfolio into the controls. Weights ride along via
    update_weight_inputs, which reads the same record off the selection."""
    record = (saved or {}).get(name or "")
    if not record:
        raise PreventUpdate
    tickers = parse_tickers(record.get("tickers"))
    if not tickers:
        raise PreventUpdate
    # The search bar only knows the symbols it last searched, so re-seed options
    # or the recalled holdings would render as unlabelled chips.
    options = [{"label": t, "value": t, "search": t} for t in tickers]
    return (
        tickers,
        options,
        record.get("period") or DEFAULT_PERIOD,
        record.get("benchmark") or "SPY",
        record.get("rf", round(an.DEFAULT_RF * 100, 2)),
    )


@callback(
    Output("ticker-input", "options"),
    Input("ticker-input", "search_value"),
    State("ticker-input", "value"),
    prevent_initial_call=True,
)
def update_ticker_options(search_value, selected):
    """Feed live symbol lookups into the ticker search bar as the user types."""
    query = (search_value or "").strip()
    selected = parse_tickers(selected)

    # Selected symbols must stay in options or the dropdown renders the chosen
    # values with no label.
    options = [{"label": t, "value": t, "search": t} for t in selected]
    if len(query) < 2:
        return options

    try:
        matches = dt.search_tickers(query)
    except Exception as exc:
        log.warning("Ticker search failed for %r (%s)", query, exc)
        matches = []

    seen = set(selected)
    for m in matches:
        symbol = m["symbol"].upper()
        # Skip anything parse_tickers would drop later (futures like ES=F), so
        # the bar can't offer a symbol that silently vanishes on Analyze.
        if symbol in seen or not _TICKER_RE.match(symbol):
            continue
        seen.add(symbol)
        label = f"{symbol} · {m['name']}"
        if m["exchange"]:
            label = f"{label} ({m['exchange']})"
        # The dropdown re-filters options client-side over value/label/search.
        # Carrying the query in `search` keeps name hits whose label omits it
        # ("google" → GOOGL · Alphabet Inc.) from being filtered straight back out.
        options.append({
            "label": label,
            "value": symbol,
            "search": f"{symbol} {m['name']} {query}",
        })

    # Escape hatch: let a symbol through when lookup found nothing (or is down)
    # rather than making the bar a dead end.
    typed = query.upper()
    if len(options) == len(selected) and typed not in seen and _TICKER_RE.match(typed):
        options.append({"label": f'Add "{typed}"', "value": typed, "search": f"{typed} {query}"})

    return options


@callback(
    Output("weights-section", "children"),
    Input("ticker-input", "value"),
    State("portfolio-select", "value"),
    State("portfolios-store", "data"),
)
def update_weight_inputs(ticker_input, selected, saved):
    tickers = parse_tickers(ticker_input)
    if not tickers:
        return dbc.Alert("Search for one or more tickers to begin.", color="secondary",
                         className="mb-0 py-2")

    # A recalled portfolio brings its weights back with it. Reading them off the
    # selection here — rather than having load_portfolio push them into a store —
    # keeps this free of any ordering assumption: the selection is already
    # committed by the time the recalled tickers arrive on the Input above.
    record = (saved or {}).get(selected or "") or {}
    stored = record.get("weights") or {}
    if parse_tickers(record.get("tickers")) != tickers:
        stored = {}  # holdings edited since the save — that split no longer applies

    default_weight = round(100 / len(tickers), 1)
    cols = [
        dbc.Col([
            dbc.Label(ticker, style={"color": TEXT, "fontSize": "0.8rem", "fontWeight": "bold"}),
            dbc.Input(
                id={"type": "weight-input", "index": ticker},
                type="number",
                value=stored.get(ticker, default_weight),
                min=0,
                max=100,
                step=0.1,
                style={"backgroundColor": PLOT_BG, "color": TEXT, "borderColor": "#2a2a4a"},
            ),
        ], xs=6, sm=4, md=2)
        for ticker in tickers
    ]
    optimize_buttons = dbc.ButtonGroup([
        dbc.Button("Equal", id="opt-equal-btn", color="secondary", outline=True, size="sm"),
        dbc.Button("★ Max Sharpe", id="opt-sharpe-btn", color="warning", outline=True, size="sm"),
        dbc.Button("◆ Min Vol", id="opt-minvol-btn", color="success", outline=True, size="sm"),
    ], size="sm")
    return html.Div([
        dbc.Row([
            dbc.Col(dbc.Label("Portfolio Weights (%)", style={"color": TEXT, "marginBottom": 0}), width="auto"),
            dbc.Col(optimize_buttons, width="auto", className="ms-auto"),
        ], align="center", className="mb-2"),
        dbc.Row(cols),
        html.Small("Weights are normalized automatically. Optimize buttons fetch data and fill in weights — then click Analyze.",
                   className="text-muted mt-1", style={"display": "block"}),
    ])


@callback(
    Output({"type": "weight-input", "index": ALL}, "value"),
    Input("opt-equal-btn", "n_clicks"),
    Input("opt-sharpe-btn", "n_clicks"),
    Input("opt-minvol-btn", "n_clicks"),
    State("ticker-input", "value"),
    State("period-select", "value"),
    State("rf-input", "value"),
    State({"type": "weight-input", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def apply_optimization(n_equal, n_sharpe, n_minvol, ticker_input, period, rf_pct, weight_ids):
    if not weight_ids:
        raise PreventUpdate
    tickers = [wid["index"] for wid in weight_ids]

    if ctx.triggered_id == "opt-equal-btn":
        w = round(100 / len(tickers), 1)
        return [w for _ in weight_ids]

    try:
        prices = dt.fetch_prices(tickers, period=period).dropna(axis=1, how="all")
        rets = an.daily_returns(prices)
        objective = "sharpe" if ctx.triggered_id == "opt-sharpe-btn" else "min_vol"
        res = an.optimize_portfolio(rets, objective, rf=parse_rf(rf_pct))
    except Exception:
        return [no_update for _ in weight_ids]

    return [round(res["weights"].get(wid["index"], 0.0) * 100, 1) for wid in weight_ids]


@callback(
    Output("dashboard-content", "children"),
    Output("tickers-store", "data"),
    Input("analyze-btn", "n_clicks"),
    State("ticker-input", "value"),
    State("period-select", "value"),
    State("benchmark-select", "value"),
    State("rf-input", "value"),
    State({"type": "weight-input", "index": ALL}, "value"),
    State({"type": "weight-input", "index": ALL}, "id"),
    prevent_initial_call=False,
)
def update_dashboard(n_clicks, ticker_input, period, benchmark_sym, rf_pct, weight_values, weight_ids):
    tickers = parse_tickers(ticker_input)
    if not tickers:
        raise PreventUpdate
    rf = parse_rf(rf_pct)

    weights = (
        {wid["index"]: (v if v is not None and v >= 0 else 0.0) for wid, v in zip(weight_ids, weight_values)}
        if weight_ids else {t: 1.0 for t in tickers}
    )

    try:
        prices = dt.fetch_prices(tickers, period=period)
        # Drop tickers with all-NaN (failed fetch)
        prices = prices.dropna(axis=1, how="all")
        valid_tickers = prices.columns.tolist()
        if not valid_tickers:
            return dbc.Alert("Could not fetch data for any of the provided tickers.", color="danger"), []

        dropped = [t for t in tickers if t not in valid_tickers]

        benchmark = None
        if benchmark_sym != "none":
            bm_prices = dt.fetch_prices([benchmark_sym], period=period)
            if not bm_prices.empty:
                bm_col = bm_prices.iloc[:, 0]
                bm_col.name = benchmark_sym
                benchmark = bm_col.reindex(prices.index, method="ffill")

        dashboard = build_dashboard(prices, benchmark, valid_tickers, weights, period=period, rf=rf)
        if dropped:
            dashboard = html.Div([
                dbc.Alert(f"No data for: {', '.join(dropped)} — they were excluded.",
                          color="warning", dismissable=True, className="py-2"),
                dashboard,
            ])
        return dashboard, valid_tickers

    except Exception as exc:
        return dbc.Alert(f"Error fetching data: {exc}", color="danger"), []


@callback(
    Output("candle-chart", "figure"),
    Output("fundamentals-panel", "children"),
    Input("stock-select", "value"),
    State("period-select", "value"),
    prevent_initial_call=True,
)
def update_candle(ticker, period):
    if not ticker:
        raise PreventUpdate
    return build_candle_figure(ticker, period), build_fundamentals(ticker)


@callback(
    Output("stats-download", "data"),
    Input("export-btn", "n_clicks"),
    State("stats-store", "data"),
    prevent_initial_call=True,
)
def export_stats(n_clicks, records):
    if not records:
        raise PreventUpdate
    df = pd.DataFrame(records)
    return dcc.send_data_frame(df.to_csv, "portfolio_stats.csv", index=False)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    host = os.environ.get("HOST", "0.0.0.0")          # 0.0.0.0 = reachable from your LAN
    port = int(os.environ.get("PORT", "8050"))
    debug = os.environ.get("DEBUG", "0").lower() in ("1", "true", "yes")

    print(f"\n  Portfolio Analyzer serving on http://{host}:{port}")
    print("  From another device on your network, use this machine's LAN IP, e.g. http://192.168.1.149:8050\n")
    app.run(debug=debug, host=host, port=port)
