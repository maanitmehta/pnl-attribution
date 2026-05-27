"""
PnL Attribution & Alpha Decay Framework — Dash App

Run with:
    /Users/maanitmehta/sentiment_signals/venv/bin/python app.py
    (or: python app.py if your venv is activated)

Serves on http://localhost:8050

Four panels:
  1  Strategy Overview     — equity curve vs buy-and-hold, summary stat cards
  2  PnL Attribution       — gross/slippage/drag waterfall + signal-score scatter
  3  Alpha Decay Curve     — IC vs horizon with 95% CI bands and significance annotation
  4  Rolling Signal Quality — 60-trade rolling IC with shaded degradation regions + hit-rate
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

from data_pipeline import (
    build_trade_log,
    compute_ic_table,
    compute_rolling_ic,
    compute_equity_curve,
    compute_pnl_attribution,
    compute_summary_stats,
    HORIZONS,
)

# ── Colour palette ──────────────────────────────────────────────────────────────
_GREEN  = "#2d9e6b"
_RED    = "#e74c3c"
_BLUE   = "#4a90d9"
_AMBER  = "#f39c12"
_GREY   = "#95a5a6"
_DARK   = "#1c1c2e"
_CARD   = "#16213e"
_BORDER = "#2a2a4a"

_SIG_COLORS = {"LONG": _GREEN, "SHORT": _RED, "HOLD": _GREY}

_LAYOUT_BASE = dict(
    paper_bgcolor=_CARD,
    plot_bgcolor=_DARK,
    font=dict(color="#e0e0e0", size=12),
    margin=dict(l=50, r=30, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=_BORDER),
)

# ── Ticker catalogue ────────────────────────────────────────────────────────────
_TICKERS = [
    # FinBERT DB already has signal data for these
    {"label": "AAPL — Apple",            "value": "AAPL"},
    {"label": "MSFT — Microsoft",        "value": "MSFT"},
    {"label": "GOOGL — Alphabet",        "value": "GOOGL"},
    {"label": "NFLX — Netflix",          "value": "NFLX"},
    {"label": "PLTR — Palantir",         "value": "PLTR"},
    {"label": "GS — Goldman Sachs",      "value": "GS"},
    {"label": "JPM — JPMorgan",          "value": "JPM"},
    {"label": "BA — Boeing",             "value": "BA"},
    {"label": "ABNB — Airbnb",           "value": "ABNB"},
    {"label": "CBRE — CBRE Group",       "value": "CBRE"},
    # Extra popular names (synthetic-only path; no DB signals)
    {"label": "AMZN — Amazon",           "value": "AMZN"},
    {"label": "NVDA — Nvidia",           "value": "NVDA"},
    {"label": "META — Meta",             "value": "META"},
    {"label": "TSLA — Tesla",            "value": "TSLA"},
    {"label": "SPY — S&P 500 ETF",       "value": "SPY"},
    {"label": "QQQ — Nasdaq 100 ETF",    "value": "QQQ"},
]

# ── Date range constants ────────────────────────────────────────────────────────
_DEFAULT_START = "2022-01-01"
_DEFAULT_END   = "2024-12-31"
_REAL_START    = "2025-01-01"
_REAL_END      = "2026-12-31"

# ── Per-ticker trade-log cache (keyed by "ticker|mode") ────────────────────────
# Populated lazily on first callback call for each combination.
_CACHE: dict[str, tuple[pd.DataFrame, str]] = {}


def _load(ticker: str, mode: str) -> tuple[pd.DataFrame, str]:
    """Return cached trade log or build it on first access."""
    key = f"{ticker}|{mode}"
    if key not in _CACHE:
        start = _REAL_START if mode == "real" else _DEFAULT_START
        end   = _REAL_END   if mode == "real" else _DEFAULT_END
        _CACHE[key] = build_trade_log(ticker, start, end)
        n = len(_CACHE[key][0])
        print(f"[cache] loaded {ticker}/{mode}: {n} rows, source={_CACHE[key][1]}")
    return _CACHE[key]


# Pre-warm default ticker so Panel 1 renders immediately
print("Pre-warming AAPL/synth …")
_load("AAPL", "synth")
print("Ready.\n")


# ── Chart builders ──────────────────────────────────────────────────────────────

def _axis(title: str = "", **kw) -> dict:
    return dict(
        title=title,
        gridcolor=_BORDER,
        zerolinecolor=_BORDER,
        showgrid=True,
        **kw,
    )


def build_equity_chart(trade_log: pd.DataFrame, ticker: str, horizon: int = 1) -> go.Figure:
    """Equity curve (strategy vs buy-and-hold) for Panel 1."""
    eq = compute_equity_curve(trade_log, horizon=horizon)
    if eq.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq["signal_date"], y=eq["portfolio_value"],
        name="Strategy", line=dict(color=_BLUE, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=eq["signal_date"], y=eq["bnh_value"],
        name="Buy & Hold", line=dict(color=_GREY, width=1, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=eq["signal_date"], y=eq["drawdown"] * 100,
        name="Drawdown %", fill="tozeroy",
        fillcolor="rgba(231,76,60,0.15)",
        line=dict(color="rgba(231,76,60,0.4)", width=1),
        yaxis="y2",
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        title=f"Equity Curve — {ticker} ({horizon}d hold)",
        xaxis=_axis("Date"),
        yaxis=_axis("Portfolio Value ($)"),
        yaxis2=dict(
            title="Drawdown (%)",
            overlaying="y",
            side="right",
            gridcolor=_BORDER,
            range=[-50, 5],
        ),
        hovermode="x unified",
    )
    return fig


def build_stats_cards(
    trade_log: pd.DataFrame,
    equity_df: pd.DataFrame,
    horizon: int = 1,
) -> list:
    """Summary stat cards for Panel 1."""
    s = compute_summary_stats(trade_log, equity_df, horizon)

    def card(label: str, value: str, sub: str = "", color: str = _BLUE) -> dbc.Col:
        return dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.P(label, className="text-muted mb-1", style={"fontSize": "0.75rem"}),
                    html.H5(value, style={"color": color, "fontWeight": "700", "margin": 0}),
                    html.Small(sub, className="text-muted"),
                ], style={"padding": "0.6rem 0.8rem"}),
            ], style={"backgroundColor": _CARD, "border": f"1px solid {_BORDER}"}),
            width=2,
        )

    sharpe_c = _GREEN if s["sharpe"] > 0  else _RED
    dd_c     = _RED   if s["max_drawdown"] < -10 else _AMBER
    pnl_c    = _GREEN if s["avg_pnl_bps"]  > 0  else _RED

    return [
        card("Sharpe Ratio",   f"{s['sharpe']:.2f}",          "annualised",    sharpe_c),
        card("Hit Rate",       f"{s['hit_rate']:.1f}%",        "active trades", _BLUE),
        card("Max Drawdown",   f"{s['max_drawdown']:.1f}%",    "peak-to-trough", dd_c),
        card("Avg PnL / Trade",f"{s['avg_pnl_bps']:.1f} bps", "before drag",   pnl_c),
        card("Active Trades",  str(s["n_active"]),
             f"L:{s['n_long']} S:{s['n_short']} H:{s['n_hold']}", _GREY),
    ]


def build_waterfall(trade_log: pd.DataFrame) -> go.Figure:
    """Gross/slippage/drag/net PnL waterfall for Panel 2."""
    attr = compute_pnl_attribution(trade_log)

    labels = ["Gross Signal\nAlpha", "Timing\nSlippage", "Execution\nDrag", "Net PnL"]
    values = [attr["gross_signal_pnl"], attr["timing_slippage"], attr["execution_drag"], 0]

    fig = go.Figure(go.Waterfall(
        name="PnL Attribution",
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=labels,
        y=values,
        textposition="outside",
        text=[f"{v:+.2f}%" if v != 0 else f"{attr['net_pnl']:+.2f}%" for v in values],
        connector=dict(line=dict(color=_BORDER, width=1)),
        increasing=dict(marker=dict(color=_GREEN)),
        decreasing=dict(marker=dict(color=_RED)),
        totals=dict(marker=dict(color=_BLUE)),
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        title="PnL Attribution (1-day hold)",
        yaxis=_axis("Return (%)"),
        showlegend=False,
    )
    return fig


def build_scatter(trade_log: pd.DataFrame, ticker: str, horizon: int = 1) -> go.Figure:
    """signal_score vs forward return scatter, coloured by signal type."""
    col = f"raw_ret_{horizon}d"
    df = trade_log.dropna(subset=["signal_score", col])

    fig = go.Figure()
    for sig_type in ["LONG", "SHORT", "HOLD"]:
        sub = df[df["signal_type"] == sig_type]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["signal_score"], y=sub[col] * 100,
            mode="markers", name=sig_type,
            marker=dict(color=_SIG_COLORS[sig_type], size=6, opacity=0.7,
                        line=dict(width=0.5, color=_DARK)),
            hovertemplate=(
                f"<b>{sig_type}</b><br>Score: %{{x:.3f}}<br>"
                f"Return ({horizon}d): %{{y:.2f}}%<extra></extra>"
            ),
        ))

    active = df[df["signal_type"] != "HOLD"]
    if len(active) > 2:
        m, b = np.polyfit(active["signal_score"], active[col] * 100, 1)
        xs = np.linspace(active["signal_score"].min(), active["signal_score"].max(), 50)
        fig.add_trace(go.Scatter(
            x=xs, y=m * xs + b, mode="lines", name="OLS fit",
            line=dict(color=_AMBER, width=1.5, dash="dash"),
        ))

    fig.add_hline(y=0, line_color=_GREY, line_width=0.8, line_dash="dot")
    fig.add_vline(x=0, line_color=_GREY, line_width=0.8, line_dash="dot")
    fig.update_layout(
        **_LAYOUT_BASE,
        title=f"{ticker} — Score vs {horizon}d Return",
        xaxis=_axis("Signal Score (FinBERT composite)"),
        yaxis=_axis("Forward Return (%)"),
        hovermode="closest",
    )
    return fig


def build_ic_decay(trade_log: pd.DataFrame, ticker: str) -> go.Figure:
    """IC vs horizon decay curve with 95% CI bands — Panel 3."""
    ic_tbl = compute_ic_table(trade_log)
    if ic_tbl.empty:
        return go.Figure()

    horizons  = ic_tbl.index.tolist()
    ic        = ic_tbl["ic"].tolist()
    ic_upper  = ic_tbl["ic_upper"].tolist()
    ic_lower  = ic_tbl["ic_lower"].tolist()
    sig_mask  = ic_tbl["significant"].tolist()

    first_insig = next(
        (h for h, s in zip(horizons, sig_mask)
         if not s and not np.isnan(ic_tbl.loc[h, "ic"])),
        None,
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 95% CI shaded band
    fig.add_trace(go.Scatter(
        x=horizons + horizons[::-1],
        y=ic_upper + ic_lower[::-1],
        fill="toself", fillcolor="rgba(74,144,217,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI", hoverinfo="skip",
    ), secondary_y=False)

    node_colors = [_GREEN if s else _RED for s in sig_mask]
    fig.add_trace(go.Scatter(
        x=horizons, y=ic,
        mode="lines+markers", name="IC (Spearman)",
        line=dict(color=_BLUE, width=2.5),
        marker=dict(size=9, color=node_colors, line=dict(color=_DARK, width=1.5)),
        customdata=np.column_stack([
            ic_tbl["t_stat"].tolist(),
            ic_tbl["p_value"].tolist(),
            ic_tbl["n_obs"].tolist(),
        ]),
        hovertemplate=(
            "<b>Horizon %{x}d</b><br>"
            "IC: %{y:.4f}<br>"
            "t-stat: %{customdata[0]:.2f}<br>"
            "p-value: %{customdata[1]:.4f}<br>"
            "N: %{customdata[2]}<extra></extra>"
        ),
    ), secondary_y=False)

    cum_ic = np.nancumsum(np.abs(ic))
    fig.add_trace(go.Scatter(
        x=horizons, y=cum_ic,
        mode="lines+markers", name="Cumulative |IC|",
        line=dict(color=_AMBER, width=1.5, dash="dot"),
        marker=dict(size=6, symbol="diamond"),
    ), secondary_y=True)

    fig.add_hline(y=0, line_color=_GREY, line_width=0.8, line_dash="dot")

    if first_insig is not None:
        fig.add_vline(
            x=first_insig,
            line_color=_RED, line_dash="dash", line_width=1.5,
            annotation_text=f"Edge lost >{first_insig}d",
            annotation_position="top right",
            annotation_font=dict(color=_RED, size=11),
        )

    _layout = {
        **_LAYOUT_BASE,
        "legend": dict(bgcolor="rgba(0,0,0,0)", bordercolor=_BORDER, x=0.65, y=0.95),
    }
    fig.update_layout(
        **_layout,
        title=f"{ticker} — Alpha Decay (IC vs Horizon)",
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Holding Horizon (trading days)", gridcolor=_BORDER)
    fig.update_yaxes(title_text="IC (Spearman)", gridcolor=_BORDER, secondary_y=False)
    fig.update_yaxes(title_text="Cumulative |IC|", gridcolor=_BORDER, secondary_y=True)

    return fig


def build_rolling_ic(trade_log: pd.DataFrame, ticker: str, window: int = 60) -> go.Figure:
    """Rolling IC with shaded degradation regions + hit-rate overlay — Panel 4."""
    roll = compute_rolling_ic(trade_log, window=window)
    if roll.empty:
        return go.Figure()

    dates = roll["signal_date"]
    ic    = roll["rolling_ic"]
    hr    = roll["rolling_hit_rate"]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    ic_pos = ic.clip(lower=0)
    ic_neg = ic.clip(upper=0)

    fig.add_trace(go.Scatter(
        x=dates, y=ic_pos,
        fill="tozeroy", fillcolor="rgba(45,158,107,0.20)",
        line=dict(color="rgba(0,0,0,0)"),
        name="IC > 0", hoverinfo="skip",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=dates, y=ic_neg,
        fill="tozeroy", fillcolor="rgba(231,76,60,0.20)",
        line=dict(color="rgba(0,0,0,0)"),
        name="IC < 0 (degradation)", hoverinfo="skip",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=dates, y=ic,
        mode="lines", name=f"Rolling IC ({window}-trade)",
        line=dict(color=_BLUE, width=2),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=dates, y=hr * 100,
        mode="lines", name="Rolling Hit Rate %",
        line=dict(color=_AMBER, width=1.5, dash="dash"),
    ), secondary_y=True)

    fig.add_hline(y=0,  line_color=_GREY, line_width=0.8, line_dash="dot")
    fig.add_hline(y=50, line_color=_GREY, line_width=0.8, line_dash="dot", secondary_y=True)

    fig.update_layout(
        **_LAYOUT_BASE,
        title=f"{ticker} — Rolling Signal Quality ({window}-trade window)",
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Signal Date", gridcolor=_BORDER)
    fig.update_yaxes(title_text="Rolling IC", gridcolor=_BORDER, secondary_y=False)
    fig.update_yaxes(title_text="Hit Rate (%)", gridcolor=_BORDER, secondary_y=True)

    return fig


def _build_ic_table_html(trade_log: pd.DataFrame) -> dbc.Table:
    """Small IC summary table shown below the decay chart."""
    ic_tbl = compute_ic_table(trade_log)

    header = html.Thead(html.Tr([
        html.Th("Horizon", className="text-muted"),
        html.Th("IC",      className="text-muted"),
        html.Th("t-stat",  className="text-muted"),
        html.Th("p-value", className="text-muted"),
        html.Th("N",       className="text-muted"),
        html.Th("Sig?",    className="text-muted"),
    ]))

    rows = []
    for h, row in ic_tbl.iterrows():
        ic_val = row["ic"]
        sig    = bool(row["significant"])
        color  = _GREEN if (sig and ic_val > 0) else (_RED if ic_val < 0 else _GREY)
        rows.append(html.Tr([
            html.Td(f"{h}d"),
            html.Td(f"{ic_val:.4f}" if not np.isnan(ic_val) else "—",
                    style={"color": color, "fontWeight": "600"}),
            html.Td(f"{row['t_stat']:.2f}" if not np.isnan(row["t_stat"]) else "—"),
            html.Td(f"{row['p_value']:.4f}"),
            html.Td(str(row["n_obs"])),
            html.Td(
                html.Span("✓", style={"color": _GREEN}) if sig
                else html.Span("✗", style={"color": _RED})
            ),
        ]))

    return dbc.Table(
        [header, html.Tbody(rows)],
        bordered=False, hover=True, size="sm",
        style={"fontSize": "0.8rem", "backgroundColor": _CARD},
    )


def _info_badge(source: str, n: int, start: str, end: str) -> dbc.Badge:
    label = "Real FinBERT" if source == "real" else "Synthetic"
    color = "success" if source == "real" else "warning"
    return dbc.Badge(
        f"{label}  |  {n} active trades  |  {start} – {end}",
        color=color, className="ms-2", style={"fontSize": "0.75rem"},
    )


# ── App Layout ──────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="PnL Attribution | Alpha Decay",
)
server = app.server

app.layout = dbc.Container([

    # ── Header ──────────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            html.H3(
                "PnL Attribution & Alpha Decay Framework",
                className="mb-0",
                style={"fontWeight": "700", "letterSpacing": "0.02em"},
            ),
            html.Small(id="subtitle", className="text-muted"),
        ], width=5),

        dbc.Col([
            dbc.Row([
                dbc.Col(
                    html.Label("Ticker", className="text-muted col-form-label",
                               style={"fontSize": "0.8rem", "whiteSpace": "nowrap"}),
                    width="auto",
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="ticker-dd",
                        options=_TICKERS,
                        value="AAPL",
                        clearable=False,
                        style={"minWidth": "220px", "fontSize": "0.85rem"},
                    ),
                    width="auto",
                ),
            ], className="g-2 align-items-center justify-content-end mb-1"),
            dbc.Row([
                dbc.Col(
                    html.Label("Dataset", className="text-muted",
                               style={"fontSize": "0.8rem"}),
                    width="auto",
                ),
                dbc.Col(
                    dcc.RadioItems(
                        id="data-mode",
                        options=[
                            {"label": " Synthetic 2022–24 ", "value": "synth"},
                            {"label": " Real FinBERT 2025–26", "value": "real"},
                        ],
                        value="synth",
                        inline=True,
                        inputStyle={"marginRight": "4px"},
                        labelStyle={"marginRight": "12px", "fontSize": "0.85rem"},
                    ),
                    width="auto",
                ),
                dbc.Col(html.Div(id="data-badge"), width="auto"),
            ], className="g-2 align-items-center justify-content-end"),
        ], width=7),

    ], className="py-3 border-bottom border-secondary mb-3"),

    # ── Panels (wrapped in Loading for feedback on ticker change) ───────────
    dcc.Loading(
        id="global-loading",
        type="circle",
        color=_BLUE,
        children=[

            # Panel 1 + Panel 2
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.B("Panel 1 — Strategy Overview"), className="py-2"),
                        dbc.CardBody([
                            dbc.Row(id="stats-cards", className="g-2 mb-2"),
                            dcc.Graph(id="equity-chart", config={"displayModeBar": False}),
                        ]),
                    ], style={"backgroundColor": _CARD, "border": f"1px solid {_BORDER}"}),
                ], width=6),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.B("Panel 2 — PnL Attribution"),
                            dbc.Row([
                                dbc.Col(html.Label("Scatter horizon:", className="text-muted col-form-label",
                                                   style={"fontSize": "0.8rem"}), width="auto"),
                                dbc.Col(dcc.Dropdown(
                                    id="scatter-horizon",
                                    options=[{"label": f"{h}d", "value": h} for h in HORIZONS],
                                    value=1, clearable=False,
                                    style={"width": "80px", "fontSize": "0.85rem"},
                                ), width="auto"),
                            ], className="g-1 align-items-center mt-1"),
                        ], className="py-2"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col(dcc.Graph(id="waterfall-chart",
                                                  config={"displayModeBar": False}), width=5),
                                dbc.Col(dcc.Graph(id="scatter-chart",
                                                  config={"displayModeBar": False}), width=7),
                            ]),
                        ]),
                    ], style={"backgroundColor": _CARD, "border": f"1px solid {_BORDER}"}),
                ], width=6),
            ], className="mb-3"),

            # Panel 3 + Panel 4
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.B("Panel 3 — Alpha Decay Curve"), className="py-2"),
                        dbc.CardBody([
                            dcc.Graph(id="ic-decay-chart", config={"displayModeBar": False}),
                            html.Div(id="ic-table", className="mt-2"),
                        ]),
                    ], style={"backgroundColor": _CARD, "border": f"1px solid {_BORDER}"}),
                ], width=6),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.B("Panel 4 — Rolling Signal Quality"),
                            dbc.Row([
                                dbc.Col(html.Label("Window (trades):", className="text-muted col-form-label",
                                                   style={"fontSize": "0.8rem"}), width="auto"),
                                dbc.Col(dcc.Slider(
                                    id="window-slider",
                                    min=20, max=100, step=10, value=60,
                                    marks={v: str(v) for v in [20, 40, 60, 80, 100]},
                                ), width=8),
                            ], className="g-1 align-items-center mt-1"),
                        ], className="py-2"),
                        dbc.CardBody([
                            dcc.Graph(id="rolling-ic-chart", config={"displayModeBar": False}),
                        ]),
                    ], style={"backgroundColor": _CARD, "border": f"1px solid {_BORDER}"}),
                ], width=6),
            ], className="mb-3"),

        ],
    ),

    # ── Footer ───────────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col(html.Small(
            "IC = Spearman rank-correlation(signal_score, forward_return)  |  "
            "Execution drag = 7 bps/trade (5 bps spread + 2 bps market impact)  |  "
            "95% CI bands use asymptotic SE = √((1−IC²)/(N−2))",
            className="text-muted",
        )),
    ], className="py-2 border-top border-secondary"),

], fluid=True, style={"backgroundColor": _DARK, "minHeight": "100vh", "padding": "1rem"})


# ── Callback ──────────────────────────────────────────────────────────────────

@app.callback(
    Output("subtitle",        "children"),
    Output("data-badge",      "children"),
    Output("stats-cards",     "children"),
    Output("equity-chart",    "figure"),
    Output("waterfall-chart", "figure"),
    Output("scatter-chart",   "figure"),
    Output("ic-decay-chart",  "figure"),
    Output("ic-table",        "children"),
    Output("rolling-ic-chart","figure"),
    Input("ticker-dd",       "value"),
    Input("data-mode",       "value"),
    Input("scatter-horizon", "value"),
    Input("window-slider",   "value"),
)
def update_all(ticker: str, mode: str, scatter_h: int, window: int):
    ticker = (ticker or "AAPL").upper()
    tl, source = _load(ticker, mode)
    start = _REAL_START if mode == "real" else _DEFAULT_START
    end   = _REAL_END   if mode == "real" else _DEFAULT_END

    eq = compute_equity_curve(tl, horizon=1)

    n_active = int((tl["signal_type"] != "HOLD").sum())
    subtitle = f"FinBERT sentiment signal on {ticker}  |  Keyrock strategy forensics demo"
    badge    = _info_badge(source, n_active, start, end)
    stats    = build_stats_cards(tl, eq, horizon=1)
    equity   = build_equity_chart(tl, ticker, horizon=1)
    wfall    = build_waterfall(tl)
    scat     = build_scatter(tl, ticker, horizon=int(scatter_h))
    decay    = build_ic_decay(tl, ticker)
    ic_tbl   = _build_ic_table_html(tl)
    roll     = build_rolling_ic(tl, ticker, window=int(window))

    return subtitle, badge, stats, equity, wfall, scat, decay, ic_tbl, roll


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("PnL Attribution & Alpha Decay Framework")
    print("Serving on http://localhost:8050\n")
    app.run(debug=False, host="0.0.0.0", port=8050)
