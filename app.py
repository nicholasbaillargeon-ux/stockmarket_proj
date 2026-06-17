"""Portfolio Analyzer — Dash dashboard with real market data and statistical analysis."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, callback, dcc, html, dash_table
from dash.exceptions import PreventUpdate

import analysis as an
import data as dt

# ── App init ──────────────────────────────────────────────────────────────────
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="Portfolio Analyzer",
    suppress_callback_exceptions=True,
)

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
DEFAULT_PERIOD = "1y"
PERIODS = {"6M": "6mo", "1Y": "1y", "2Y": "2y", "5Y": "5y"}

# ── Color palette ─────────────────────────────────────────────────────────────
COLORS = px.colors.qualitative.Plotly
BG = "#1a1a2e"
CARD_BG = "#16213e"
PLOT_BG = "#0f3460"
ACCENT = "#e94560"
TEXT = "#eaeaea"

CHART_LAYOUT = dict(
    paper_bgcolor=CARD_BG,
    plot_bgcolor=PLOT_BG,
    font=dict(color=TEXT, size=12),
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    xaxis=dict(gridcolor="#2a2a4a", showgrid=True),
    yaxis=dict(gridcolor="#2a2a4a", showgrid=True),
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


def metric_card(title, value, color="#eaeaea"):
    return dbc.Card(
        dbc.CardBody([
            html.P(title, className="text-muted mb-1", style={"fontSize": "0.75rem"}),
            html.H5(value, style={"color": color, "fontWeight": "bold", "marginBottom": 0}),
        ]),
        style={"backgroundColor": CARD_BG, "border": "1px solid #2a2a4a"},
    )


# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = dbc.Container(
    fluid=True,
    style={"backgroundColor": BG, "minHeight": "100vh", "padding": "20px"},
    children=[
        # Header
        dbc.Row([
            dbc.Col(html.H2("Portfolio Analyzer", style={"color": ACCENT, "fontWeight": "bold"}), width="auto"),
            dbc.Col(html.P("Real-time market data · Statistical risk analysis", className="text-muted mt-2"), width="auto"),
        ], align="center", className="mb-3"),

        # Controls
        dbc.Card(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Tickers (space or comma separated)", style={"color": TEXT}),
                        dbc.Input(
                            id="ticker-input",
                            value=" ".join(DEFAULT_TICKERS),
                            placeholder="e.g. AAPL MSFT NVDA",
                            style={"backgroundColor": PLOT_BG, "color": TEXT, "borderColor": "#2a2a4a"},
                        ),
                    ], md=6),
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
                        dbc.Label(" ", style={"display": "block"}),
                        dbc.Button("Analyze", id="analyze-btn", color="danger", n_clicks=0, className="w-100"),
                    ], md=1),
                ]),
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
    ],
)


# ── Dashboard builder ─────────────────────────────────────────────────────────
def build_dashboard(prices: pd.DataFrame, benchmark: pd.Series | None, tickers: list[str]):
    rets = an.daily_returns(prices)
    cum_rets = an.cumulative_returns(prices)
    stats = an.summary_stats(prices, benchmark)

    # Equal-weight portfolio
    weights = {t: 1.0 for t in tickers}
    port_ret = an.portfolio_returns(rets, weights)
    port_prices = (1 + port_ret).cumprod()

    # ── Metric cards ──────────────────────────────────────────────────────────
    port_sharpe = an.sharpe_ratio(port_ret)
    port_sortino = an.sortino_ratio(port_ret)
    port_vol = an.annualized_volatility(port_ret)
    port_dd = an.max_drawdown(port_prices)
    port_cagr = an.annualized_return(port_ret)
    total_ret = float(cum_rets.iloc[-1].mean())

    def color_for(v, good_positive=True):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return TEXT
        return "#2ecc71" if (v > 0) == good_positive else ACCENT

    cards_row = dbc.Row([
        dbc.Col(metric_card("Portfolio CAGR", fmt_pct(port_cagr), color_for(port_cagr)), md=2),
        dbc.Col(metric_card("Ann. Volatility", fmt_pct(port_vol, 1), TEXT), md=2),
        dbc.Col(metric_card("Sharpe Ratio", fmt_float(port_sharpe), color_for(port_sharpe)), md=2),
        dbc.Col(metric_card("Sortino Ratio", fmt_float(port_sortino), color_for(port_sortino)), md=2),
        dbc.Col(metric_card("Max Drawdown", fmt_pct(port_dd, 1), color_for(port_dd, good_positive=False)), md=2),
        dbc.Col(metric_card("Avg Total Return", fmt_pct(total_ret), color_for(total_ret)), md=2),
    ], className="mb-3")

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
    fig_alloc = go.Figure(go.Pie(
        labels=tickers,
        values=[1] * len(tickers),
        hole=0.55,
        marker=dict(colors=COLORS[:len(tickers)]),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Weight: %{percent}<extra></extra>",
    ))
    fig_alloc.update_layout(
        **CHART_LAYOUT,
        title="Portfolio Allocation (equal-weight)",
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
    roll_sharpe = an.rolling_sharpe(rets, window=63).dropna()
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

    # ── Stats table ───────────────────────────────────────────────────────────
    table_df = stats.copy()
    table_df["Ann. Return"] = table_df["Ann. Return"].map(lambda v: fmt_pct(v))
    table_df["Ann. Volatility"] = table_df["Ann. Volatility"].map(lambda v: fmt_pct(v))
    table_df["Sharpe"] = table_df["Sharpe"].map(lambda v: fmt_float(v))
    table_df["Sortino"] = table_df["Sortino"].map(lambda v: fmt_float(v))
    table_df["Max Drawdown"] = table_df["Max Drawdown"].map(lambda v: fmt_pct(v))
    table_df["Beta (vs SPY)"] = table_df["Beta (vs SPY)"].map(lambda v: fmt_float(v))
    table_df = table_df.reset_index()

    stats_table = dash_table.DataTable(
        data=table_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in table_df.columns],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": PLOT_BG, "color": TEXT, "fontWeight": "bold", "border": "1px solid #2a2a4a"},
        style_cell={"backgroundColor": CARD_BG, "color": TEXT, "border": "1px solid #2a2a4a", "textAlign": "center", "padding": "8px"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#1a2a4a"},
        ],
    )

    # ── Individual stock section ───────────────────────────────────────────────
    stock_selector = dbc.Row([
        dbc.Col(html.H5("Individual Stock", style={"color": TEXT}), width="auto"),
        dbc.Col(
            dbc.Select(
                id="stock-select",
                options=[{"label": t, "value": t} for t in tickers],
                value=tickers[0],
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

        dbc.Card(
            dbc.CardBody([
                html.H5("Per-Asset Statistics", style={"color": TEXT, "marginBottom": "12px"}),
                stats_table,
            ]),
            style={"backgroundColor": CARD_BG, "border": "1px solid #2a2a4a", "marginBottom": "16px"},
        ),

        dbc.Card(
            dbc.CardBody([
                stock_selector,
                dcc.Graph(id="candle-chart", config={"displayModeBar": "hover"}),
            ]),
            style={"backgroundColor": CARD_BG, "border": "1px solid #2a2a4a"},
        ),
    ])


# ── Callbacks ─────────────────────────────────────────────────────────────────
@callback(
    Output("dashboard-content", "children"),
    Output("tickers-store", "data"),
    Input("analyze-btn", "n_clicks"),
    State("ticker-input", "value"),
    State("period-select", "value"),
    State("benchmark-select", "value"),
    prevent_initial_call=False,
)
def update_dashboard(n_clicks, ticker_input, period, benchmark_sym):
    raw = ticker_input.replace(",", " ").upper().split()
    tickers = [t.strip() for t in raw if t.strip()]
    if not tickers:
        raise PreventUpdate

    try:
        prices = dt.fetch_prices(tickers, period=period)
        # Drop tickers with all-NaN (failed fetch)
        prices = prices.dropna(axis=1, how="all")
        valid_tickers = prices.columns.tolist()
        if not valid_tickers:
            return dbc.Alert("Could not fetch data for any of the provided tickers.", color="danger"), []

        benchmark = None
        if benchmark_sym != "none":
            bm_prices = dt.fetch_prices([benchmark_sym], period=period)
            if not bm_prices.empty:
                bm_col = bm_prices.iloc[:, 0]
                bm_col.name = benchmark_sym
                benchmark = bm_col.reindex(prices.index, method="ffill")

        return build_dashboard(prices, benchmark, valid_tickers), valid_tickers

    except Exception as exc:
        return dbc.Alert(f"Error fetching data: {exc}", color="danger"), []


@callback(
    Output("candle-chart", "figure"),
    Input("stock-select", "value"),
    State("period-select", "value"),
    prevent_initial_call=True,
)
def update_candle(ticker, period):
    if not ticker:
        raise PreventUpdate

    # Map analysis period to OHLCV period (cap at 2y for readability)
    period_map = {"6mo": "6mo", "1y": "1y", "2y": "2y", "5y": "2y"}
    ohlcv_period = period_map.get(period, "1y")

    try:
        df = dt.fetch_ohlcv(ticker, period=ohlcv_period)
    except Exception as exc:
        return go.Figure().update_layout(**CHART_LAYOUT, title=f"Error: {exc}")

    # Candlestick
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name=ticker,
        increasing=dict(line=dict(color="#2ecc71"), fillcolor="#2ecc71"),
        decreasing=dict(line=dict(color=ACCENT), fillcolor=ACCENT),
        hovertext=ticker,
    ))

    # 20-day and 50-day MA overlays
    for w, c in [(20, "#f39c12"), (50, "#9b59b6")]:
        if len(df) >= w:
            ma = df["Close"].rolling(w).mean()
            fig.add_trace(go.Scatter(
                x=ma.index, y=ma, name=f"{w}-day MA",
                line=dict(color=c, width=1.2, dash="dot"),
                hovertemplate=f"{w}-day MA: %{{y:.2f}}<extra></extra>",
            ))

    # Volume subplot
    colors = ["#2ecc71" if c >= o else ACCENT for o, c in zip(df["Open"], df["Close"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        name="Volume", marker_color=colors, opacity=0.4,
        yaxis="y2",
        hovertemplate="Vol: %{y:,.0f}<extra></extra>",
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        title=f"{ticker} — Price & Volume",
        xaxis_rangeslider_visible=False,
        yaxis=dict(domain=[0.25, 1.0], gridcolor="#2a2a4a"),
        yaxis2=dict(domain=[0.0, 0.2], showgrid=False, showticklabels=False),
    )
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
