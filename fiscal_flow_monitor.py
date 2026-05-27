#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  FISCAL FLOW MONITOR  ·  FIN ANALYTICS Suite  ·  Nicoletti 2026     ║
║  Net Liquidity = Fed Balance Sheet − TGA − Reverse Repo              ║
║  Data source: FRED API (public CSV, no key required)                 ║
║  Cache: DuckDB local  ·  Port: 8052                                  ║
╚══════════════════════════════════════════════════════════════════════╝

Installazione (una tantum):
    pip install dash dash-bootstrap-components plotly pandas duckdb requests

Avvio:
    python fiscal_flow_monitor.py
    → http://localhost:8052
"""

import os
import sys
import requests
import pandas as pd
import numpy as np
import duckdb
import plotly.graph_objects as go
from datetime import datetime, timedelta
from io import StringIO

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

PORT          = 8052
CACHE_TTL_H   = 6          # ore prima di ri-scaricare da FRED
_LOCAL_CACHE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
DB_PATH       = os.path.join(
    _LOCAL_CACHE if os.access(os.path.dirname(os.path.abspath(__file__)), os.W_OK)
    else "/tmp",
    "fiscal_flow.duckdb",
)

# ─── Palette (dark theme coerente con FIN ANALYTICS) ───────────────────────
C = {
    "bg":          "#080c18",
    "surface":     "#0d1117",
    "border":      "#1a2236",
    "border_lit":  "#2a3550",
    "text":        "#e2e8f0",
    "muted":       "#556080",
    "accent":      "#00d4ff",   # cyan  — Net Liquidity
    "fed":         "#38bdf8",   # light blue — Fed Assets
    "tga":         "#fb923c",   # orange      — TGA (drain)
    "rrp":         "#f43f5e",   # rose        — RRP (drain)
    "green":       "#22c55e",
    "red":         "#ef4444",
    "yellow":      "#eab308",
    "grid":        "#131929",
}

# ─── Serie FRED ─────────────────────────────────────────────────────────────
# WALCL    : Fed Total Assets               (weekly Wed, milioni $)
# WTREGEN  : Treasury General Account       (weekly Wed, milioni $)
# RRPONTSYD: Overnight Reverse Repo         (giornaliero, miliardi $)
FRED_SERIES = ["WALCL", "WTREGEN", "RRPONTSYD"]

# ─────────────────────────────────────────────────────────────────────────────
#  DATA LAYER
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_fred(series_id: str) -> pd.DataFrame:
    """Scarica la serie FRED come CSV (endpoint pubblico, senza API key)."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["value"] != "."].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().sort_values("date").reset_index(drop=True)
    print(f"  [FRED]   {series_id}: {len(df)} obs  (ultimo: {df['date'].max().date()})")
    return df


def _load_series(series_id: str, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Restituisce la serie dal cache DuckDB, ri-scaricando se stale."""
    table = f"fred_{series_id.lower()}"

    # Assicura tabella meta
    con.execute("""
        CREATE TABLE IF NOT EXISTS _cache_meta (
            series_id   VARCHAR PRIMARY KEY,
            updated_at  TIMESTAMP
        )
    """)

    # Verifica staleness
    stale = True
    try:
        row = con.execute(
            "SELECT updated_at FROM _cache_meta WHERE series_id = ?", [series_id]
        ).fetchone()
        if row:
            age_h = (datetime.now() - row[0]).total_seconds() / 3600
            stale = age_h > CACHE_TTL_H
    except Exception:
        stale = True

    if not stale:
        try:
            df = con.execute(f"SELECT date, value FROM {table} ORDER BY date").df()
            df["date"] = pd.to_datetime(df["date"])
            print(f"  [cache]  {series_id}: {len(df)} obs")
            return df
        except Exception:
            stale = True

    # Fetch da FRED
    try:
        df = _fetch_fred(series_id)
        con.execute(f"DROP TABLE IF EXISTS {table}")
        con.execute(f"CREATE TABLE {table} AS SELECT * FROM df")
        con.execute(
            "INSERT OR REPLACE INTO _cache_meta VALUES (?, ?)",
            [series_id, datetime.now()]
        )
        return df
    except Exception as e:
        print(f"  [warn]   {series_id} fetch fallito ({e}). Provo cache stale...")
        try:
            df = con.execute(f"SELECT date, value FROM {table} ORDER BY date").df()
            df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception:
            return pd.DataFrame(columns=["date", "value"])


def load_data() -> pd.DataFrame:
    """
    Carica e combina le tre serie FRED, calcola Net Liquidity e derivati.
    Restituisce un DataFrame settimanale (frequenza Wed).
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = duckdb.connect(DB_PATH)

    print("\n📡 Caricamento dati FRED...")
    raw = {sid: _load_series(sid, con) for sid in FRED_SERIES}
    con.close()

    # Converti unità → miliardi $
    fed = raw["WALCL"].set_index("date")["value"].rename("fed") / 1000   # M$ → B$
    tga = raw["WTREGEN"].set_index("date")["value"].rename("tga") / 1000  # M$ → B$
    rrp = raw["RRPONTSYD"].set_index("date")["value"].rename("rrp")        # già B$

    # Assicura DatetimeIndex su tutti e tre (necessario per resample in pandas 2.x)
    fed.index = pd.DatetimeIndex(fed.index)
    tga.index = pd.DatetimeIndex(tga.index)
    rrp.index = pd.DatetimeIndex(rrp.index)

    # Normalizza a frequenza settimanale (mercoledì) — allinea RRP giornaliero
    rrp_w = rrp.resample("W-WED").last().ffill()

    df = pd.concat([fed, tga, rrp_w], axis=1).dropna().copy()
    df.index.name = "date"
    df = df.reset_index()

    # ── Calcoli principali ──────────────────────────────────────────────────
    df["net"] = df["fed"] - df["tga"] - df["rrp"]

    # Variazioni settimanali
    df["d_fed"] = df["fed"].diff()
    df["d_tga"] = df["tga"].diff()
    df["d_rrp"] = df["rrp"].diff()
    df["d_net"] = df["net"].diff()

    # Medie mobili per regime
    df["ma26"] = df["net"].rolling(26).mean()
    df["ma52"] = df["net"].rolling(52).mean()

    # Regime: Easing / Tightening / Neutral
    df["regime"] = np.where(
        df["net"] > df["ma52"] * 1.02, "Easing",
        np.where(df["net"] < df["ma52"] * 0.98, "Tightening", "Neutral")
    )

    # Momentum 4w (somma rolling variazioni settimanali)
    df["mom4w"] = df["d_net"].rolling(4).sum()

    # Z-score 52w di Net Liquidity
    roll = df["net"].rolling(52)
    df["zscore"] = (df["net"] - roll.mean()) / roll.std()

    print(f"✅ Dati pronti: {len(df)} osservazioni settimanali "
          f"({df['date'].min().date()} → {df['date'].max().date()})\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _base_layout(title: str = "", yaxis_title: str = "$B",
                  height: int = 320, show_legend: bool = True) -> dict:
    return dict(
        title=dict(
            text=title,
            font=dict(color=C["muted"], size=10, family="monospace"),
            x=0, y=1, pad=dict(b=8),
        ),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C["text"], family="monospace", size=11),
        xaxis=dict(
            gridcolor=C["grid"], showgrid=True, zeroline=False,
            tickfont=dict(size=10), rangeslider=dict(visible=False),
            showspikes=True, spikecolor=C["muted"], spikemode="across",
            spikethickness=1, spikedash="dot",
        ),
        yaxis=dict(
            gridcolor=C["grid"], showgrid=True, zeroline=False,
            tickfont=dict(size=10),
            title=dict(text=yaxis_title, font=dict(size=10, color=C["muted"])),
        ),
        margin=dict(l=52, r=18, t=32, b=36),
        showlegend=show_legend,
        legend=dict(
            bgcolor="rgba(0,0,0,0)", font=dict(size=10),
            orientation="h", y=-0.18, x=0,
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=C["surface"], font=dict(family="monospace", size=11),
            bordercolor=C["border_lit"],
        ),
    )


def _cutoff(df: pd.DataFrame, years: int) -> pd.DataFrame:
    return df[df["date"] >= df["date"].max() - pd.DateOffset(years=years)].copy()


# ── Grafico 1: Net Liquidity + MA52 + regime shading ─────────────────────
def chart_net_liq(df: pd.DataFrame, years: int = 3) -> go.Figure:
    d = _cutoff(df, years)
    fig = go.Figure()

    # Regime background shading
    regime_color = {
        "Easing":     "rgba(34,197,94,0.07)",
        "Tightening": "rgba(239,68,68,0.07)",
        "Neutral":    "rgba(0,0,0,0)",
    }
    prev = None
    seg_start = None
    for _, row in d.iterrows():
        if row["regime"] != prev:
            if prev is not None and prev != "Neutral":
                fig.add_vrect(
                    x0=seg_start, x1=row["date"],
                    fillcolor=regime_color[prev], layer="below", line_width=0,
                )
            seg_start = row["date"]
            prev = row["regime"]
    if prev and prev != "Neutral" and seg_start:
        fig.add_vrect(
            x0=seg_start, x1=d["date"].max(),
            fillcolor=regime_color[prev], layer="below", line_width=0,
        )

    # MA 52w
    fig.add_trace(go.Scatter(
        x=d["date"], y=d["ma52"], name="MA 52w",
        line=dict(color=C["muted"], width=1, dash="dot"),
        hovertemplate="%{y:,.0f}B<extra>MA 52w</extra>",
    ))
    # MA 26w
    fig.add_trace(go.Scatter(
        x=d["date"], y=d["ma26"], name="MA 26w",
        line=dict(color=C["border_lit"], width=1, dash="dash"),
        hovertemplate="%{y:,.0f}B<extra>MA 26w</extra>",
    ))
    # Net Liquidity
    fig.add_trace(go.Scatter(
        x=d["date"], y=d["net"], name="Net Liquidity",
        line=dict(color=C["accent"], width=2.5),
        fill="tonexty", fillcolor="rgba(0,212,255,0.04)",
        hovertemplate="%{y:,.0f}B<extra>Net Liquidity</extra>",
    ))

    layout = _base_layout("NET LIQUIDITY  ·  Fed − TGA − RRP", "$B", 400)
    fig.update_layout(**layout)
    return fig


# ── Grafico 2: Componenti ──────────────────────────────────────────────────
def chart_components(df: pd.DataFrame, years: int = 2) -> go.Figure:
    d = _cutoff(df, years)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=d["date"], y=d["fed"], name="Fed Assets",
        line=dict(color=C["fed"], width=1.8),
        hovertemplate="%{y:,.0f}B<extra>Fed Assets</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=d["date"], y=d["tga"], name="TGA",
        line=dict(color=C["tga"], width=1.5),
        hovertemplate="%{y:,.0f}B<extra>TGA (drain)</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=d["date"], y=d["rrp"], name="Reverse Repo",
        line=dict(color=C["rrp"], width=1.5),
        hovertemplate="%{y:,.0f}B<extra>RRP (drain)</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=d["date"], y=d["net"], name="Net Liquidity",
        line=dict(color=C["accent"], width=2, dash="dot"),
        hovertemplate="%{y:,.0f}B<extra>Net Liquidity</extra>",
    ))

    fig.update_layout(**_base_layout(
        "COMPONENTI  ·  Fed Assets  /  TGA  /  Reverse Repo", "$B", 310
    ))
    return fig


# ── Grafico 3: Variazioni settimanali ─────────────────────────────────────
def chart_weekly_delta(df: pd.DataFrame, weeks: int = 52) -> go.Figure:
    d = df.dropna(subset=["d_net"]).tail(weeks).copy()

    bar_colors = [C["green"] if v >= 0 else C["red"] for v in d["d_net"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=d["date"], y=d["d_net"], name="Δ Net (1w)",
        marker_color=bar_colors,
        hovertemplate="%{y:+,.0f}B<extra>Δ 1w</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=d["date"], y=d["mom4w"], name="Σ 4w",
        line=dict(color=C["yellow"], width=1.8),
        hovertemplate="%{y:+,.0f}B<extra>Σ 4w</extra>",
    ))
    fig.add_hline(y=0, line_color=C["muted"], line_width=0.8)
    fig.update_layout(**_base_layout(
        "VARIAZIONE SETTIMANALE  ·  Iniezione (+) / Drenaggio (−)", "$B", 280
    ))
    return fig


# ── Grafico 4: Waterfall ultima settimana ────────────────────────────────
def chart_waterfall(df: pd.DataFrame) -> go.Figure:
    last = df.dropna(subset=["d_net"]).iloc[-1]
    date_str = last["date"].strftime("%d %b %Y")

    # Fed Δ (+/−) ; TGA drain = -(Δ TGA) ; RRP drain = -(Δ RRP)
    labels  = ["Fed Assets Δ", "TGA Δ (drain−)", "RRP Δ (drain−)", "Net Liquidity Δ"]
    values  = [last["d_fed"], -last["d_tga"], -last["d_rrp"], last["d_net"]]
    measure = ["relative",   "relative",       "relative",       "total"]

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=measure,
        x=labels, y=values,
        connector=dict(line=dict(color=C["border_lit"], width=1)),
        increasing=dict(marker_color=C["green"]),
        decreasing=dict(marker_color=C["red"]),
        totals=dict(marker_color=C["accent"]),
        texttemplate="%{y:+,.0f}B",
        textposition="outside",
        textfont=dict(color=C["text"], size=10, family="monospace"),
    ))
    fig.add_hline(y=0, line_color=C["muted"], line_width=0.8)

    layout = _base_layout(f"ULTIMA SETTIMANA  ·  {date_str}", "$B", 280, show_legend=False)
    fig.update_layout(**layout)
    return fig


# ── Grafico 5: Z-score 52w ───────────────────────────────────────────────
def chart_zscore(df: pd.DataFrame, years: int = 3) -> go.Figure:
    d = _cutoff(df, years).dropna(subset=["zscore"])
    colors = [C["green"] if v > 0 else C["red"] for v in d["zscore"]]

    fig = go.Figure()
    fig.add_hrect(y0=1, y1=3,  fillcolor="rgba(34,197,94,0.06)",  layer="below", line_width=0)
    fig.add_hrect(y0=-3, y1=-1, fillcolor="rgba(239,68,68,0.06)", layer="below", line_width=0)
    fig.add_hline(y=1,  line_color=C["green"], line_width=0.6, line_dash="dot")
    fig.add_hline(y=-1, line_color=C["red"],   line_width=0.6, line_dash="dot")
    fig.add_hline(y=0,  line_color=C["muted"], line_width=0.8)

    fig.add_trace(go.Bar(
        x=d["date"], y=d["zscore"], name="Z-score 52w",
        marker_color=colors,
        hovertemplate="%{y:+.2f}σ<extra>Z-score</extra>",
    ))

    layout = _base_layout("Z-SCORE 52w  ·  Livello di liquidità normalizzato", "σ", 240, show_legend=False)
    fig.update_layout(**layout)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  KPI COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def _kpi(title: str, value: str, delta: str, up_is_good: bool,
         delta_positive: bool, accent_color: str) -> html.Div:
    good  = delta_positive if up_is_good else not delta_positive
    d_col = C["green"] if good else C["red"]
    arrow = "▲" if delta_positive else "▼"
    return html.Div([
        html.Div(title, className="kpi-title"),
        html.Div(value, className="kpi-value", style={"color": accent_color}),
        html.Span(f"{arrow} {delta} ", style={"color": d_col, "fontSize": "11px", "fontFamily": "monospace"}),
        html.Span("1w", style={"color": C["muted"], "fontSize": "10px", "fontFamily": "monospace"}),
    ], className="kpi-card", style={"borderLeftColor": accent_color})


def _regime_card(row: pd.Series) -> html.Div:
    r = row["regime"]
    col = {"Easing": C["green"], "Tightening": C["red"], "Neutral": C["yellow"]}[r]
    icon = {"Easing": "▲", "Tightening": "▼", "Neutral": "◆"}[r]
    pct  = (row["net"] - row["ma52"]) / row["ma52"] * 100 if row["ma52"] else 0
    zs   = row["zscore"] if not np.isnan(row["zscore"]) else 0
    return html.Div([
        html.Div("REGIME LIQUIDITÀ", className="kpi-title"),
        html.Div(f"{icon} {r.upper()}", className="kpi-value", style={"color": col}),
        html.Div(f"{'+' if pct>0 else ''}{pct:.1f}% vs MA52  ·  Z {zs:+.2f}σ",
                 style={"color": col, "fontSize": "11px", "fontFamily": "monospace"}),
    ], className="kpi-card", style={"borderLeftColor": col})


def build_kpi_row(df: pd.DataFrame) -> html.Div:
    last = df.dropna(subset=["d_net", "zscore"]).iloc[-1]
    return html.Div([
        _kpi("NET LIQUIDITY",
             f"${last['net']:,.0f}B",
             f"${abs(last['d_net']):,.0f}B",
             True, last["d_net"] >= 0, C["accent"]),
        _kpi("FED ASSETS",
             f"${last['fed']:,.0f}B",
             f"${abs(last['d_fed']):,.0f}B",
             True, last["d_fed"] >= 0, C["fed"]),
        _kpi("TGA (drain)",
             f"${last['tga']:,.0f}B",
             f"${abs(last['d_tga']):,.0f}B",
             False, last["d_tga"] >= 0, C["tga"]),
        _kpi("REVERSE REPO (drain)",
             f"${last['rrp']:,.0f}B",
             f"${abs(last['d_rrp']):,.0f}B",
             False, last["d_rrp"] >= 0, C["rrp"]),
        _regime_card(last),
    ], className="kpi-row")


# ─────────────────────────────────────────────────────────────────────────────
#  LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

def build_app_layout(df: pd.DataFrame) -> html.Div:
    last      = df.dropna(subset=["d_net"]).iloc[-1]
    last_date = last["date"].strftime("%d %b %Y")

    return html.Div([
        html.Div([
            # ── Header ────────────────────────────────────────────────────
            html.Div([
                html.Div([
                    html.Span("💧 "),
                    html.Span("FISCAL FLOW MONITOR", className="header-title"),
                ]),
                html.Div(
                    f"Net Liquidity = Fed Balance Sheet − TGA − Reverse Repo  ·  "
                    f"Dati: FRED  ·  Ultimo aggiornamento: {last_date}",
                    className="header-sub"
                ),
            ], className="header-wrap"),

            # ── KPI ──────────────────────────────────────────────────────
            build_kpi_row(df),

            # ── Controlli orizzonte ───────────────────────────────────────
            html.Div([
                html.Span("Orizzonte:", className="ctrl-label"),
                dcc.RadioItems(
                    id="lookback",
                    options=[
                        {"label": " 1A", "value": 1},
                        {"label": " 2A", "value": 2},
                        {"label": " 3A", "value": 3},
                        {"label": " 5A", "value": 5},
                        {"label": "10A", "value": 10},
                    ],
                    value=3,
                    inline=True,
                    className="radio-group",
                    inputStyle={"marginRight": "4px", "accentColor": C["accent"]},
                ),
                html.Span("  |  Variaz. settimanali:", className="ctrl-label"),
                dcc.RadioItems(
                    id="weeks-bar",
                    options=[
                        {"label": "26s", "value": 26},
                        {"label": "52s", "value": 52},
                        {"label": "2A",  "value": 104},
                    ],
                    value=52,
                    inline=True,
                    className="radio-group",
                    inputStyle={"marginRight": "4px", "accentColor": C["accent"]},
                ),
            ], className="controls-wrap"),

            # ── Net Liquidity (principale) ────────────────────────────────
            html.Div(
                dcc.Graph(id="g-net-liq", config={"displayModeBar": False}),
                className="chart-card"
            ),

            # ── Componenti ───────────────────────────────────────────────
            html.Div(
                dcc.Graph(id="g-components", config={"displayModeBar": False}),
                className="chart-card"
            ),

            # ── Waterfall + Var. settimanali ──────────────────────────────
            html.Div([
                html.Div(
                    dcc.Graph(id="g-waterfall", config={"displayModeBar": False}),
                    className="chart-card", style={"flex": "0 0 340px"}
                ),
                html.Div(
                    dcc.Graph(id="g-weekly", config={"displayModeBar": False}),
                    className="chart-card", style={"flex": "1"}
                ),
            ], className="row-2"),

            # ── Z-score ───────────────────────────────────────────────────
            html.Div(
                dcc.Graph(id="g-zscore", config={"displayModeBar": False}),
                className="chart-card"
            ),

            # ── Footer ───────────────────────────────────────────────────
            html.Div(
                "⚠️  Fiscal Flow Monitor  ·  Net Liquidity = Fed Balance Sheet − TGA − RRP  ·  "
                "Non costituisce consulenza finanziaria  ·  FIN ANALYTICS  ·  Nicoletti 2026",
                className="footer"
            ),
        ], className="page-wrap"),
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  APP & CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
#  STANDALONE (python fiscal_flow_monitor.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.CYBORG],
        title="Fiscal Flow Monitor · FIN ANALYTICS",
        suppress_callback_exceptions=True,
    )
    _df_global = load_data()
    app.layout = build_app_layout(_df_global)

    @app.callback(
        Output("g-net-liq",    "figure"),
        Output("g-components", "figure"),
        Output("g-weekly",     "figure"),
        Output("g-waterfall",  "figure"),
        Output("g-zscore",     "figure"),
        Input("lookback",      "value"),
        Input("weeks-bar",     "value"),
    )
    def update_charts(years: int, weeks: int):
        return (
            chart_net_liq(_df_global, years),
            chart_components(_df_global, years),
            chart_weekly_delta(_df_global, weeks),
            chart_waterfall(_df_global),
            chart_zscore(_df_global, years),
        )

    print(f"\n🚀  Fiscal Flow Monitor  →  http://localhost:{PORT}\n")
    app.run(debug=False, port=PORT, host="0.0.0.0")
