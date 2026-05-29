#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  FIN ANALYTICS  ·  Dashboard Hub  ·  Nicoletti 2026                 ║
║  Tutti i dashboard su un unico URL con navigazione a tab            ║
║                                                                      ║
║  Tab:  💧 Fiscal Flow  ·  ⚡ Volatility  ·  📈 Rates  ·  🌊 GLI    ║
║        📊 Market Internals  ·  🎯 AGE  ·  📐 Timmer               ║
║  Port: 8050                                                          ║
╚══════════════════════════════════════════════════════════════════════╝

Avvio:
    python hub.py  →  http://localhost:8050
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

# ── Import funzioni pure da ciascun modulo (senza avviare le loro app) ───────
from fiscal_flow_monitor import (
    load_data        as ff_load_data,
    chart_net_liq, chart_components, chart_weekly_delta,
    chart_waterfall, chart_zscore,
    build_kpi_row    as ff_build_kpi_row,
    build_app_layout as ff_build_layout,
)
from volatility_radar import (
    load_data        as vol_load_data,
    chart_vix_history, chart_term_structure, chart_vvix,
    chart_move, chart_percentile_gauge, chart_zscore_combined,
    build_kpi_row    as vol_build_kpi_row,
    build_app_layout as vol_build_layout,
)
from rates_treasury import (
    load_data        as rt_load_data,
    chart_yield_curve_snapshot, chart_yield_history, chart_spread,
    chart_yield_heatmap, chart_breakeven, chart_real_yield,
    build_kpi_row    as rt_build_kpi_row,
    build_app_layout as rt_build_layout,
)
from global_liquidity_index import (
    load_data        as gli_load_data,
    chart_gli_score, chart_cb_levels, chart_cb_yoy,
    chart_gli_vs_market, chart_phase_donut, chart_rolling_correlation,
    build_kpi_row    as gli_build_kpi_row,
    build_app_layout as gli_build_layout,
)
from market_internals import (
    load_data        as mi_load_data,
    chart_breadth_ma, chart_ad_line, chart_nh_nl,
    chart_put_call, chart_buffett, chart_hindenburg_detail, chart_spx_price,
    build_kpi_row    as mi_build_kpi_row,
    build_app_layout as mi_build_layout,
)
from age_indicators import (
    load_data            as age_load_data,
    chart_trin, chart_seasonality_monthly, chart_seasonality_annual,
    chart_cot, chart_fear_greed_gauge, chart_fear_greed_components,
    build_kpi_row        as age_build_kpi_row,
    build_app_layout     as age_build_layout,
)
from timmer_framework import (
    load_data            as tmr_load_data,
    _tmr_charts,
    build_kpi_row        as tmr_build_kpi_row,
    build_app_layout     as tmr_build_layout,
)

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

PORT = int(os.environ.get("PORT", 8050))

C = {
    "bg":          "#080c18",
    "surface":     "#0d1117",
    "border":      "#1a2236",
    "text":        "#e2e8f0",
    "muted":       "#556080",
    "accent":      "#00d4ff",
}

# ─────────────────────────────────────────────────────────────────────────────
#  CARICAMENTO DATI (una sola volta, condiviso tra tutti i tab)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 60)
print("  FIN ANALYTICS  ·  Hub  ·  Avvio...")
print("─" * 60)

print("\n💧 Fiscal Flow Monitor...")
_ff   = ff_load_data()

print("\n⚡ Volatility Radar...")
_vol  = vol_load_data()

print("\n📈 Rates & Treasury...")
_rt   = rt_load_data()

print("\n🌊 Global Liquidity Index...")
_gli  = gli_load_data()

print("\n📊 Market Internals...")
_mi   = mi_load_data()

print("\n🎯 AGE Indicators...")
_age  = age_load_data()

print("\n📐 Timmer Framework...")
_tmr  = tmr_load_data()

print("\n✅ Tutti i dati caricati\n" + "─" * 60 + "\n")

# ─────────────────────────────────────────────────────────────────────────────
#  APP
# ─────────────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="FIN ANALYTICS · Dashboard Suite",
    suppress_callback_exceptions=True,
)
server = app.server   # per deploy WSGI se necessario

# ─────────────────────────────────────────────────────────────────────────────
#  STILE TAB  (replica stile FIN ANALYTICS)
# ─────────────────────────────────────────────────────────────────────────────

_TAB = {
    "backgroundColor":  C["bg"],
    "color":            C["muted"],
    "border":           f"1px solid {C['border']}",
    "borderBottom":     "none",
    "fontFamily":       "monospace",
    "fontSize":         "12px",
    "fontWeight":       "600",
    "letterSpacing":    "0.05em",
    "padding":          "10px 20px",
}
_TAB_SEL = {
    **_TAB,
    "color":            C["accent"],
    "backgroundColor":  C["surface"],
    "borderTop":        f"2px solid {C['accent']}",
    "borderLeft":       f"1px solid {C['border']}",
    "borderRight":      f"1px solid {C['border']}",
    "borderBottom":     f"1px solid {C['surface']}",
}

# ─────────────────────────────────────────────────────────────────────────────
#  LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

app.layout = html.Div([

    # ── Intestazione globale ──────────────────────────────────────────────────
    html.Div([
        html.Span("◆ ", style={"color": C["accent"], "fontSize": "16px"}),
        html.Span("FIN ANALYTICS", style={
            "fontFamily": "monospace", "fontSize": "16px",
            "fontWeight": "700", "color": C["accent"],
            "letterSpacing": "0.08em",
        }),
        html.Span("  ·  Dashboard Suite  ·  Nicoletti 2026", style={
            "fontFamily": "monospace", "fontSize": "11px",
            "color": C["muted"], "marginLeft": "8px",
        }),
        html.Span("  ·  Local", style={
            "fontFamily": "monospace", "fontSize": "10px",
            "color": C["border"], "marginLeft": "4px",
        }),
    ], style={
        "backgroundColor": C["surface"],
        "borderBottom":    f"1px solid {C['border']}",
        "padding":         "10px 28px",
        "display":         "flex",
        "alignItems":      "center",
    }),

    # ── Tabs principali ───────────────────────────────────────────────────────
    dcc.Tabs(
        id="main-tabs",
        value="fiscal-flow",
        style={
            "backgroundColor": C["bg"],
            "borderBottom":    f"1px solid {C['border']}",
            "paddingLeft":     "20px",
        },
        children=[

            # ── TAB 1: Fiscal Flow Monitor ────────────────────────────────────
            dcc.Tab(
                label="💧 Fiscal Flow",
                value="fiscal-flow",
                style=_TAB, selected_style=_TAB_SEL,
                children=[ff_build_layout(_ff)],
            ),

            # ── TAB 2: Volatility Radar ───────────────────────────────────────
            dcc.Tab(
                label="⚡ Volatility",
                value="volatility",
                style=_TAB, selected_style=_TAB_SEL,
                children=[vol_build_layout(_vol)],
            ),

            # ── TAB 3: Rates & Treasury ───────────────────────────────────────
            dcc.Tab(
                label="📈 Rates & Treasury",
                value="rates",
                style=_TAB, selected_style=_TAB_SEL,
                children=[rt_build_layout(_rt)],
            ),

            # ── TAB 4: Global Liquidity Index ────────────────────────────────
            dcc.Tab(
                label="🌊 GLI",
                value="gli",
                style=_TAB, selected_style=_TAB_SEL,
                children=[gli_build_layout(_gli)],
            ),

            # ── TAB 5: Market Internals ───────────────────────────────────────
            dcc.Tab(
                label="📊 Market Internals",
                value="market-internals",
                style=_TAB, selected_style=_TAB_SEL,
                children=[mi_build_layout(_mi)],
            ),

            # ── TAB 6: AGE Indicators ─────────────────────────────────────────
            dcc.Tab(
                label="🎯 AGE Indicators",
                value="age-indicators",
                style=_TAB, selected_style=_TAB_SEL,
                children=[age_build_layout(_age)],
            ),

            # ── TAB 7: Timmer Framework ───────────────────────────────────────
            dcc.Tab(
                label="📐 Timmer",
                value="timmer",
                style=_TAB, selected_style=_TAB_SEL,
                children=[tmr_build_layout(_tmr)],
            ),
        ],
    ),

], style={"backgroundColor": C["bg"], "minHeight": "100vh"})


# ─────────────────────────────────────────────────────────────────────────────
#  CALLBACKS  —  Fiscal Flow Monitor
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("g-net-liq",    "figure"),
    Output("g-components", "figure"),
    Output("g-weekly",     "figure"),
    Output("g-waterfall",  "figure"),
    Output("g-zscore",     "figure"),
    Input("lookback",      "value"),
    Input("weeks-bar",     "value"),
)
def ff_update(years, weeks):
    return (
        chart_net_liq(_ff, years),
        chart_components(_ff, years),
        chart_weekly_delta(_ff, weeks),
        chart_waterfall(_ff),
        chart_zscore(_ff, years),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CALLBACKS  —  Volatility Radar
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("g-vix-hist",    "figure"),
    Output("g-term-struct", "figure"),
    Output("g-vvix",        "figure"),
    Output("g-move",        "figure"),
    Output("g-pct-gauge",   "figure"),
    Output("g-zscore-vol",  "figure"),
    Input("vol-lookback",   "value"),
)
def vol_update(years):
    return (
        chart_vix_history(_vol, years),
        chart_term_structure(_vol, years),
        chart_vvix(_vol, years),
        chart_move(_vol, years),
        chart_percentile_gauge(_vol),
        chart_zscore_combined(_vol, years),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CALLBACKS  —  Rates & Treasury
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("g-yc-snap",    "figure"),
    Output("g-yc-hist",    "figure"),
    Output("g-spread",     "figure"),
    Output("g-heatmap",    "figure"),
    Output("g-breakeven",  "figure"),
    Output("g-real-yield", "figure"),
    Input("rt-lookback",   "value"),
    Input("rt-heatmap-sid","value"),
)
def rt_update(years, heatmap_sid):
    return (
        chart_yield_curve_snapshot(_rt),
        chart_yield_history(_rt, years),
        chart_spread(_rt, years),
        chart_yield_heatmap(_rt, heatmap_sid),
        chart_breakeven(_rt, years),
        chart_real_yield(_rt, years),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CALLBACKS  —  Global Liquidity Index
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("g-gli-score",    "figure"),
    Output("g-cb-levels",    "figure"),
    Output("g-cb-yoy",       "figure"),
    Output("g-gli-mkt",      "figure"),
    Output("g-phase-donut",  "figure"),
    Output("g-rolling-corr", "figure"),
    Input("gli-lookback",    "value"),
    Input("gli-market",      "value"),
)
def gli_update(years, market_key):
    return (
        chart_gli_score(_gli, years),
        chart_cb_levels(_gli, years),
        chart_cb_yoy(_gli, years),
        chart_gli_vs_market(_gli, market_key, years),
        chart_phase_donut(_gli),
        chart_rolling_correlation(_gli, years),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CALLBACKS  —  Market Internals
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("g-mi-breadth-ma",  "figure"),
    Output("g-mi-ad-line",     "figure"),
    Output("g-mi-nh-nl",       "figure"),
    Output("g-mi-put-call",    "figure"),
    Output("g-mi-buffett",     "figure"),
    Output("g-mi-hindenburg",  "figure"),
    Output("g-mi-spx",         "figure"),
    Input("mi-lookback",       "value"),
)
def mi_update(years):
    return (
        chart_breadth_ma(_mi, years),
        chart_ad_line(_mi, years),
        chart_nh_nl(_mi, years),
        chart_put_call(_mi, years),
        chart_buffett(_mi, years),
        chart_hindenburg_detail(_mi),
        chart_spx_price(_mi, years),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CALLBACKS  —  AGE Indicators
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("g-age-trin",            "figure"),
    Output("g-age-season-monthly",  "figure"),
    Output("g-age-season-annual",   "figure"),
    Output("g-age-cot",             "figure"),
    Output("g-age-fg-gauge",        "figure"),
    Output("g-age-fg-components",   "figure"),
    Input("age-lookback",           "value"),
)
def age_update(years):
    return (
        chart_trin(_age, years),
        chart_seasonality_monthly(_age),
        chart_seasonality_annual(_age),
        chart_cot(_age, years),
        chart_fear_greed_gauge(_age),
        chart_fear_greed_components(_age),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
#  CALLBACKS  —  Timmer Framework
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("g-tmr-leaderboard",  "figure"),
    Output("g-tmr-corr-heatmap", "figure"),
    Output("g-tmr-cap-equal",    "figure"),
    Output("g-tmr-sb-corr",      "figure"),
    Output("g-tmr-breadth",      "figure"),
    Output("g-tmr-erp",          "figure"),
    Output("g-tmr-cape",         "figure"),
    Output("g-tmr-margins",      "figure"),
    Output("tmr-alert-log",      "children"),
    Input("tmr-lookback",        "value"),
    Input("tmr-corr-window",     "value"),
)
def tmr_update(years, window):
    return _tmr_charts(_tmr, years, window)


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"🚀  FIN ANALYTICS Hub  →  http://localhost:{PORT}\n")
    app.run(debug=False, port=PORT, host="0.0.0.0")
