#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  FIN ANALYTICS  ·  Dashboard Hub  ·  APA Quant                 ║
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
import threading
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────────────────────────────────────
#  TIMEOUT GLOBALE — copre yfinance, urllib, requests e qualsiasi socket
# ─────────────────────────────────────────────────────────────────────────────
socket.setdefaulttimeout(12)

# ─────────────────────────────────────────────────────────────────────────────
#  PATCH requests.get — aggiunge User-Agent prima di qualsiasi import
#  (FRED e altri servizi bloccano le richieste senza un browser User-Agent)
# ─────────────────────────────────────────────────────────────────────────────

import requests as _requests_module

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
_orig_requests_get = _requests_module.get
_orig_session_get  = _requests_module.Session.get

def _get_with_ua(url, **kwargs):
    h = kwargs.get("headers") or {}
    h.setdefault("User-Agent", _UA)
    kwargs["headers"] = h
    kwargs.setdefault("timeout", 10)
    return _orig_requests_get(url, **kwargs)

def _session_get_with_ua(self, url, **kwargs):
    h = kwargs.get("headers") or {}
    h.setdefault("User-Agent", _UA)
    kwargs["headers"] = h
    kwargs.setdefault("timeout", 10)
    return _orig_session_get(self, url, **kwargs)

_requests_module.get        = _get_with_ua        # monkey-patch globale
_requests_module.Session.get = _session_get_with_ua  # copre yfinance

# ─────────────────────────────────────────────────────────────────────────────
#  Import moduli dashboard  (solo funzioni pure, nessun avvio di app)
# ─────────────────────────────────────────────────────────────────────────────

import dash
from dash import dcc, html, Input, Output, clientside_callback
import dash_bootstrap_components as dbc

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
    "bg":      "#080c18",
    "surface": "#0d1117",
    "border":  "#1a2236",
    "text":    "#e2e8f0",
    "muted":   "#556080",
    "accent":  "#00d4ff",
}

# ─────────────────────────────────────────────────────────────────────────────
#  CARICAMENTO DATI IN BACKGROUND
# ─────────────────────────────────────────────────────────────────────────────

_all_data: dict = {}
_loading_error: list = []


def _load_all_data():
    print("\n" + "─" * 60)
    print("  FIN ANALYTICS  ·  Hub  ·  Caricamento parallelo in background...")
    print("─" * 60)

    loaders = {
        "ff":  ("💧 Fiscal Flow Monitor",   ff_load_data),
        "vol": ("⚡ Volatility Radar",       vol_load_data),
        "rt":  ("📈 Rates & Treasury",       rt_load_data),
        "gli": ("🌊 Global Liquidity Index", gli_load_data),
        "mi":  ("📊 Market Internals",       mi_load_data),
        "age": ("🎯 AGE Indicators",         age_load_data),
        "tmr": ("📐 Timmer Framework",       tmr_load_data),
    }

    try:
        with ThreadPoolExecutor(max_workers=7) as executor:
            futures = {executor.submit(fn): (key, label)
                       for key, (label, fn) in loaders.items()}
            for future in as_completed(futures, timeout=90):
                key, label = futures[future]
                try:
                    _all_data[key] = future.result(timeout=30)
                    print(f"  ✓ {label}")
                except Exception as e:
                    print(f"  ✗ {label}: {e}")
                    _loading_error.append(f"{key}: {e}")
                    _all_data[key] = {}
    except Exception as e:
        print(f"\n❌ Errore executor: {e}")
        _loading_error.append(str(e))
    finally:
        # Garantisci chiavi + sblocca layout — SEMPRE eseguito
        for k in loaders:
            _all_data.setdefault(k, {})
        print("\n✅ Caricamento completato\n" + "─" * 60 + "\n")


# Avvia il caricamento in background — il server HTTP parte subito
_loader_thread = threading.Thread(target=_load_all_data, daemon=True)
_loader_thread.start()


# ─────────────────────────────────────────────────────────────────────────────
#  APP
# ─────────────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="FIN ANALYTICS · APA Quant",
    suppress_callback_exceptions=True,
)
server = app.server   # per deploy WSGI

# ─────────────────────────────────────────────────────────────────────────────
#  STILE TAB
# ─────────────────────────────────────────────────────────────────────────────

_TAB = {
    "backgroundColor":  C["bg"],
    "color":            C["muted"],
    "border":           f"1px solid {C['border']}",
    "borderBottom":     "none",
    "fontFamily":       "monospace",
    "fontSize":         "11px",
    "fontWeight":       "600",
    "letterSpacing":    "0.03em",
    "padding":          "9px 14px",
    "whiteSpace":       "nowrap",
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
#  HEADER COMUNE
# ─────────────────────────────────────────────────────────────────────────────

_HEADER = html.Div([
    html.Span("◆ ", style={"color": C["accent"], "fontSize": "16px"}),
    html.Span("FIN ANALYTICS", style={
        "fontFamily": "monospace", "fontSize": "16px",
        "fontWeight": "700", "color": C["accent"],
        "letterSpacing": "0.08em",
    }),
    html.Span("  ·  Dashboard Suite  ·  APA Quant", style={
        "fontFamily": "monospace", "fontSize": "11px",
        "color": C["muted"], "marginLeft": "8px",
    }),
], style={
    "backgroundColor": C["surface"],
    "borderBottom":    f"1px solid {C['border']}",
    "padding":         "10px 28px",
    "display":         "flex",
    "alignItems":      "center",
})



# ─────────────────────────────────────────────────────────────────────────────
#  LAYOUT PRINCIPALE  — sempre visibile, i grafici si aggiornano via callback
# ─────────────────────────────────────────────────────────────────────────────

app.layout = html.Div([
    _HEADER,
    # Interval che rilancia tutti i callback ogni 10s per 3 minuti
    # (permette ai grafici di aggiornarsi mentre i dati vengono caricati)
    dcc.Interval(id="hub-refresh", interval=10_000, max_intervals=18),
    dcc.Tabs(
        id="main-tabs",
        value="fiscal-flow",
        style={
            "backgroundColor": C["bg"],
            "borderBottom":    f"1px solid {C['border']}",
            "paddingLeft":     "12px",
            "overflowX":       "auto",
            "overflowY":       "hidden",
            "display":         "flex",
            "flexWrap":        "nowrap",
        },
        children=[
            dcc.Tab(label="💧 Fiscal Flow",      value="fiscal-flow",
                    style=_TAB, selected_style=_TAB_SEL,
                    children=[ff_build_layout({})]),

            dcc.Tab(label="⚡ Volatility",       value="volatility",
                    style=_TAB, selected_style=_TAB_SEL,
                    children=[vol_build_layout({})]),

            dcc.Tab(label="📈 Rates & Treasury", value="rates",
                    style=_TAB, selected_style=_TAB_SEL,
                    children=[rt_build_layout({})]),

            dcc.Tab(label="🌊 GLI",              value="gli",
                    style=_TAB, selected_style=_TAB_SEL,
                    children=[gli_build_layout({})]),

            dcc.Tab(label="📊 Market Internals", value="market-internals",
                    style=_TAB, selected_style=_TAB_SEL,
                    children=[mi_build_layout({})]),

            dcc.Tab(label="🎯 AGE Indicators",  value="age-indicators",
                    style=_TAB, selected_style=_TAB_SEL,
                    children=[age_build_layout({})]),

            dcc.Tab(label="📐 Timmer",           value="timmer",
                    style=_TAB, selected_style=_TAB_SEL,
                    children=[tmr_build_layout({})]),
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
    Input("hub-refresh",   "n_intervals"),
)
def ff_update(years, weeks, _n):
    ff = _all_data.get("ff", {})
    return (
        chart_net_liq(ff, years),
        chart_components(ff, years),
        chart_weekly_delta(ff, weeks),
        chart_waterfall(ff),
        chart_zscore(ff, years),
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
    Input("hub-refresh",    "n_intervals"),
)
def vol_update(years, _n):
    vol = _all_data.get("vol", {})
    return (
        chart_vix_history(vol, years),
        chart_term_structure(vol, years),
        chart_vvix(vol, years),
        chart_move(vol, years),
        chart_percentile_gauge(vol),
        chart_zscore_combined(vol, years),
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
    Input("hub-refresh",   "n_intervals"),
)
def rt_update(years, heatmap_sid, _n):
    rt = _all_data.get("rt", {})
    return (
        chart_yield_curve_snapshot(rt),
        chart_yield_history(rt, years),
        chart_spread(rt, years),
        chart_yield_heatmap(rt, heatmap_sid),
        chart_breakeven(rt, years),
        chart_real_yield(rt, years),
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
    Input("hub-refresh",     "n_intervals"),
)
def gli_update(years, market_key, _n):
    gli = _all_data.get("gli", {})
    return (
        chart_gli_score(gli, years),
        chart_cb_levels(gli, years),
        chart_cb_yoy(gli, years),
        chart_gli_vs_market(gli, market_key, years),
        chart_phase_donut(gli),
        chart_rolling_correlation(gli, years),
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
    Input("hub-refresh",       "n_intervals"),
)
def mi_update(years, _n):
    mi = _all_data.get("mi", {})
    return (
        chart_breadth_ma(mi, years),
        chart_ad_line(mi, years),
        chart_nh_nl(mi, years),
        chart_put_call(mi, years),
        chart_buffett(mi, years),
        chart_hindenburg_detail(mi),
        chart_spx_price(mi, years),
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
    Input("hub-refresh",            "n_intervals"),
)
def age_update(years, _n):
    age = _all_data.get("age", {})
    return (
        chart_trin(age, years),
        chart_seasonality_monthly(age),
        chart_seasonality_annual(age),
        chart_cot(age, years),
        chart_fear_greed_gauge(age),
        chart_fear_greed_components(age),
    )


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
    Input("hub-refresh",         "n_intervals"),
)
def tmr_update(years, window, _n):
    tmr = _all_data.get("tmr", {})
    return _tmr_charts(tmr, years, window)


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"🚀  FIN ANALYTICS Hub  →  http://localhost:{PORT}\n")
    app.run(debug=False, port=PORT, host="0.0.0.0")
