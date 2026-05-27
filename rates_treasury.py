#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  RATES & TREASURY  ·  FIN ANALYTICS Suite  ·  Nicoletti 2026        ║
║  Curva dei rendimenti USA · Spread · Regime tassi · Fed Funds        ║
║  Dati: FRED API (pubblico, senza API key)                            ║
║  Port: 8054                                                          ║
╚══════════════════════════════════════════════════════════════════════╝

Installazione:
    pip install dash dash-bootstrap-components plotly pandas duckdb requests

Avvio:
    python rates_treasury.py  →  http://localhost:8054
"""

import os
import warnings
warnings.filterwarnings("ignore")

import requests
import pandas as pd
import numpy as np
import duckdb
import plotly.graph_objects as go
from datetime import datetime
from io import StringIO

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

PORT        = 8054
CACHE_TTL_H = 6
_LOCAL_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
DB_PATH      = os.path.join(
    _LOCAL_CACHE if os.access(os.path.dirname(os.path.abspath(__file__)), os.W_OK)
    else "/tmp",
    "rates.duckdb",
)

C = {
    "bg":         "#080c18",
    "surface":    "#0d1117",
    "border":     "#1a2236",
    "border_lit": "#2a3550",
    "text":       "#e2e8f0",
    "muted":      "#556080",
    "accent":     "#00d4ff",
    "green":      "#22c55e",
    "red":        "#ef4444",
    "orange":     "#fb923c",
    "yellow":     "#eab308",
    "grid":       "#131929",
}

# Serie FRED — Treasury yields
YIELD_SERIES = {
    "DFF":    {"label": "Fed Funds",  "tenor": 0,    "color": "#64748b"},
    "DGS1MO": {"label": "1M",         "tenor": 1/12, "color": "#94a3b8"},
    "DGS3MO": {"label": "3M",         "tenor": 3/12, "color": "#cbd5e1"},
    "DGS6MO": {"label": "6M",         "tenor": 6/12, "color": "#fbbf24"},
    "DGS1":   {"label": "1Y",         "tenor": 1,    "color": "#fb923c"},
    "DGS2":   {"label": "2Y",         "tenor": 2,    "color": "#f97316"},
    "DGS3":   {"label": "3Y",         "tenor": 3,    "color": "#ef4444"},
    "DGS5":   {"label": "5Y",         "tenor": 5,    "color": "#dc2626"},
    "DGS7":   {"label": "7Y",         "tenor": 7,    "color": "#a855f7"},
    "DGS10":  {"label": "10Y",        "tenor": 10,   "color": "#00d4ff"},
    "DGS20":  {"label": "20Y",        "tenor": 20,   "color": "#38bdf8"},
    "DGS30":  {"label": "30Y",        "tenor": 30,   "color": "#22c55e"},
}

# Spread chiave
SPREAD_SERIES = {
    "T10Y2Y":  {"label": "10Y−2Y",   "color": "#00d4ff"},
    "T10Y3M":  {"label": "10Y−3M",   "color": "#fb923c"},
    "T5YIFR":  {"label": "5Y5Y Fwd", "color": "#a855f7"},
}

# Breakeven inflazione
BREAKEVEN_SERIES = {
    "T5YIE":  {"label": "Breakeven 5Y",  "color": "#eab308"},
    "T10YIE": {"label": "Breakeven 10Y", "color": "#fb923c"},
}

# ─────────────────────────────────────────────────────────────────────────────
#  DATA LAYER
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = duckdb.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS _cache_meta (
            series_id  VARCHAR PRIMARY KEY,
            updated_at TIMESTAMP
        )
    """)
    return con


def _is_stale(con, sid):
    try:
        row = con.execute("SELECT updated_at FROM _cache_meta WHERE series_id=?", [sid]).fetchone()
        if row:
            return (datetime.now() - row[0]).total_seconds() / 3600 > CACHE_TTL_H
    except Exception:
        pass
    return True


def _fetch_fred(sid: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        df.columns = ["date", "value"]
        df["date"]  = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["value"] != "."].copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna().sort_values("date").reset_index(drop=True)
        print(f"  [FRED]   {sid}: {len(df)} obs  (ultimo: {df['date'].max().date()})")
        return df
    except Exception as e:
        print(f"  [warn]   FRED {sid}: {e}")
        return pd.DataFrame(columns=["date", "value"])


def _get_series(con, sid: str) -> pd.DataFrame:
    table = f"rates_{sid.lower()}"
    if not _is_stale(con, sid):
        try:
            df = con.execute(f"SELECT date, value FROM {table} ORDER BY date").df()
            df["date"] = pd.to_datetime(df["date"])
            print(f"  [cache]  {sid}: {len(df)} obs")
            return df
        except Exception:
            pass
    df = _fetch_fred(sid)
    if not df.empty:
        con.execute(f"DROP TABLE IF EXISTS {table}")
        con.execute(f"CREATE TABLE {table} AS SELECT * FROM df")
        con.execute("INSERT OR REPLACE INTO _cache_meta VALUES (?,?)", [sid, datetime.now()])
    else:
        try:
            df = con.execute(f"SELECT date, value FROM {table} ORDER BY date").df()
            df["date"] = pd.to_datetime(df["date"])
        except Exception:
            pass
    return df


def load_data() -> dict:
    con = _ensure_db()
    print("\n📡 Caricamento dati Rates & Treasury...")

    all_series = list(YIELD_SERIES.keys()) + list(SPREAD_SERIES.keys()) + list(BREAKEVEN_SERIES.keys())
    raw = {sid: _get_series(con, sid) for sid in all_series}
    con.close()

    # ── Costruisce DataFrame wide (colonna per scadenza) ─────────────────────
    yield_dfs = []
    for sid, meta in YIELD_SERIES.items():
        df = raw.get(sid, pd.DataFrame())
        if not df.empty:
            s = df.set_index("date")["value"].rename(sid)
            yield_dfs.append(s)

    yields_wide = pd.concat(yield_dfs, axis=1).sort_index() if yield_dfs else pd.DataFrame()
    yields_wide.index = pd.DatetimeIndex(yields_wide.index)

    # ── Spread ───────────────────────────────────────────────────────────────
    spread_dfs = {}
    for sid in SPREAD_SERIES:
        df = raw.get(sid, pd.DataFrame())
        if not df.empty:
            spread_dfs[sid] = df.set_index("date")["value"].rename(sid)
            spread_dfs[sid].index = pd.DatetimeIndex(spread_dfs[sid].index)

    # ── Breakeven ────────────────────────────────────────────────────────────
    be_dfs = {}
    for sid in BREAKEVEN_SERIES:
        df = raw.get(sid, pd.DataFrame())
        if not df.empty:
            be_dfs[sid] = df.set_index("date")["value"].rename(sid)
            be_dfs[sid].index = pd.DatetimeIndex(be_dfs[sid].index)

    print(f"✅ Dati rates pronti\n")
    return {
        "raw":         raw,
        "yields_wide": yields_wide.reset_index(),
        "spreads":     spread_dfs,
        "breakevens":  be_dfs,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def _base(title="", yt="", h=300, legend=True):
    return dict(
        title=dict(text=title, font=dict(color=C["muted"], size=10, family="monospace"), x=0),
        height=h,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C["text"], family="monospace", size=11),
        xaxis=dict(gridcolor=C["grid"], showgrid=True, zeroline=False,
                   tickfont=dict(size=10), rangeslider=dict(visible=False),
                   showspikes=True, spikecolor=C["muted"], spikethickness=1),
        yaxis=dict(gridcolor=C["grid"], showgrid=True, zeroline=False,
                   tickfont=dict(size=10),
                   title=dict(text=yt, font=dict(size=10, color=C["muted"]))),
        margin=dict(l=52, r=18, t=32, b=36),
        showlegend=legend,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10),
                    orientation="h", y=-0.2, x=0),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=C["surface"], font=dict(family="monospace", size=11),
                        bordercolor=C["border_lit"]),
    )


def _cutoff(df, years):
    if isinstance(df, pd.DataFrame) and df.empty:
        return df
    if isinstance(df, pd.Series) and df.empty:
        return df
    date_col = df.index if isinstance(df, pd.Series) else df["date"]
    mask = date_col >= date_col.max() - pd.DateOffset(years=years)
    return df[mask].copy()


# ── 1. Curva dei rendimenti (snapshot istantanea) ─────────────────────────
def chart_yield_curve_snapshot(data: dict, dates_to_plot: list = None) -> go.Figure:
    """Curva dei rendimenti per N date selezionate."""
    yields_wide = data["yields_wide"]
    if yields_wide.empty:
        fig = go.Figure()
        fig.add_annotation(text="Dati non disponibili", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=C["muted"]))
        fig.update_layout(**_base("CURVA DEI RENDIMENTI USA", "%", 340))
        return fig

    yields_wide = yields_wide.copy()
    yields_wide["date"] = pd.to_datetime(yields_wide["date"])

    # Scadenze da plottare (escludiamo DFF dal grafico curva)
    curve_cols = [sid for sid in YIELD_SERIES if sid != "DFF" and sid in yields_wide.columns]
    tenors     = [YIELD_SERIES[sid]["tenor"] for sid in curve_cols]
    labels_x   = [YIELD_SERIES[sid]["label"] for sid in curve_cols]

    # Date di riferimento: oggi, 1M fa, 3M fa, 1A fa
    latest = yields_wide["date"].max()
    ref_dates = [
        ("Oggi",  latest),
        ("3M fa", latest - pd.DateOffset(months=3)),
        ("6M fa", latest - pd.DateOffset(months=6)),
        ("1A fa", latest - pd.DateOffset(years=1)),
    ]
    date_colors = [C["accent"], C["yellow"], C["orange"], C["muted"]]

    fig = go.Figure()
    for (label, dt), col in zip(ref_dates, date_colors):
        # trova riga più vicina
        idx = (yields_wide["date"] - dt).abs().idxmin()
        row = yields_wide.iloc[idx]
        y_vals = [row[sid] if sid in row and not pd.isna(row[sid]) else None for sid in curve_cols]
        fig.add_trace(go.Scatter(
            x=labels_x, y=y_vals, name=f"{label} ({row['date'].strftime('%d/%m/%y')})",
            line=dict(color=col, width=2 if label == "Oggi" else 1.2,
                      dash="solid" if label == "Oggi" else "dot"),
            hovertemplate="%{y:.2f}%<extra>" + label + "</extra>",
        ))

    fig.update_layout(**_base(
        "YIELD CURVE USA  ·  Snapshot multi-data", "%", 360
    ))
    fig.update_xaxes(showgrid=True)
    return fig


# ── 2. Storico yield selezionati ──────────────────────────────────────────
def chart_yield_history(data: dict, years: int = 3) -> go.Figure:
    """Serie storiche 2Y, 5Y, 10Y, 30Y + Fed Funds."""
    yields_wide = data["yields_wide"].copy()
    if yields_wide.empty:
        fig = go.Figure()
        fig.update_layout(**_base("STORICO RENDIMENTI", "%", 340))
        return fig

    yields_wide["date"] = pd.to_datetime(yields_wide["date"])
    cutoff = yields_wide["date"].max() - pd.DateOffset(years=years)
    d = yields_wide[yields_wide["date"] >= cutoff]

    key_series = [
        ("DFF",  "Fed Funds", "#64748b", "dot"),
        ("DGS2", "2Y",        "#fb923c", "solid"),
        ("DGS5", "5Y",        "#ef4444", "solid"),
        ("DGS10","10Y",       C["accent"], "solid"),
        ("DGS30","30Y",       C["green"], "solid"),
    ]

    fig = go.Figure()
    for sid, label, col, dash in key_series:
        if sid in d.columns:
            fig.add_trace(go.Scatter(
                x=d["date"], y=d[sid], name=label,
                line=dict(color=col, width=1.8 if sid != "DFF" else 1.2, dash=dash),
                hovertemplate=f"%{{y:.2f}}%<extra>{label}</extra>",
            ))

    fig.update_layout(**_base("RENDIMENTI USA  ·  Fed Funds · 2Y · 5Y · 10Y · 30Y", "%", 360))
    return fig


# ── 3. Spread 10Y−2Y ──────────────────────────────────────────────────────
def chart_spread(data: dict, years: int = 3) -> go.Figure:
    spreads = data["spreads"]
    fig = go.Figure()
    plotted = False

    for sid, meta in SPREAD_SERIES.items():
        if sid in spreads:
            s = spreads[sid]
            cutoff = s.index.max() - pd.DateOffset(years=years)
            s = s[s.index >= cutoff]
            if not s.empty:
                bar_colors = [C["green"] if v >= 0 else C["red"] for v in s.values]
                fig.add_trace(go.Scatter(
                    x=s.index, y=s.values, name=meta["label"],
                    line=dict(color=meta["color"], width=1.8),
                    hovertemplate=f"%{{y:+.2f}}bps<extra>{meta['label']}</extra>",
                ))
                plotted = True

    fig.add_hline(y=0, line_color=C["muted"], line_width=1)
    fig.add_hrect(y0=-3, y1=0, fillcolor="rgba(239,68,68,0.05)", layer="below", line_width=0)

    fig.update_layout(**_base(
        "SPREAD  ·  10Y−2Y (inversione = recessione)  ·  10Y−3M  ·  5Y5Y Fwd",
        "%", 300
    ))
    return fig


# ── 4. Heatmap rendimenti mensili ────────────────────────────────────────
def chart_yield_heatmap(data: dict, sid: str = "DGS10") -> go.Figure:
    """Heatmap anno × mese del rendimento selezionato."""
    raw = data["raw"].get(sid, pd.DataFrame())
    if raw.empty:
        fig = go.Figure()
        fig.update_layout(**_base(f"HEATMAP {sid}", "", 300))
        return fig

    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= df["date"].max() - pd.DateOffset(years=10)]
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month

    pivot = df.groupby(["year", "month"])["value"].last().unstack(level=1)
    pivot.columns = ["Gen","Feb","Mar","Apr","Mag","Giu","Lug","Ago","Set","Ott","Nov","Dic"][: len(pivot.columns)]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=list(pivot.columns),
        y=[str(y) for y in pivot.index],
        colorscale=[
            [0.0,  "#1e3a5f"],
            [0.35, "#1d4ed8"],
            [0.5,  "#0d1117"],
            [0.65, "#dc2626"],
            [1.0,  "#7f1d1d"],
        ],
        hovertemplate="Anno %{y}  %{x}: %{z:.2f}%<extra></extra>",
        showscale=True,
        colorbar=dict(
            tickfont=dict(color=C["muted"], size=9, family="monospace"),
            outlinewidth=0, thickness=10,
        ),
    ))
    fig.update_layout(
        **_base(f"HEATMAP RENDIMENTI  ·  {YIELD_SERIES.get(sid, {}).get('label', sid)}  (ultimi 10 anni)", "", 320, False)
    )
    return fig


# ── 5. Breakeven inflazione ───────────────────────────────────────────────
def chart_breakeven(data: dict, years: int = 3) -> go.Figure:
    be = data["breakevens"]
    fig = go.Figure()

    for sid, meta in BREAKEVEN_SERIES.items():
        if sid in be:
            s = be[sid]
            cutoff = s.index.max() - pd.DateOffset(years=years)
            s = s[s.index >= cutoff]
            if not s.empty:
                fig.add_trace(go.Scatter(
                    x=s.index, y=s.values, name=meta["label"],
                    line=dict(color=meta["color"], width=1.8),
                    hovertemplate=f"%{{y:.2f}}%<extra>{meta['label']}</extra>",
                ))

    fig.add_hline(y=2.0, line_color=C["muted"], line_width=0.8, line_dash="dot",
                  annotation_text="Fed target 2%", annotation_position="right",
                  annotation_font=dict(color=C["muted"], size=9))

    fig.update_layout(**_base("BREAKEVEN INFLAZIONE  ·  5Y · 10Y", "%", 260))
    return fig


# ── 6. Real yield (10Y nominale − Breakeven 10Y) ─────────────────────────
def chart_real_yield(data: dict, years: int = 3) -> go.Figure:
    yields_wide = data["yields_wide"].copy()
    be = data["breakevens"]
    fig = go.Figure()

    if "DGS10" not in yields_wide.columns or "T10YIE" not in be:
        fig.add_annotation(text="Dati real yield non disponibili", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=C["muted"]))
        fig.update_layout(**_base("REAL YIELD 10Y", "%", 260))
        return fig

    yields_wide["date"] = pd.to_datetime(yields_wide["date"])
    nominal = yields_wide.set_index("date")["DGS10"]
    be10y   = be["T10YIE"]
    # Allinea
    merged  = pd.concat([nominal.rename("nom"), be10y.rename("be")], axis=1).dropna()
    merged["real"] = merged["nom"] - merged["be"]

    cutoff = merged.index.max() - pd.DateOffset(years=years)
    m = merged[merged.index >= cutoff]

    bar_colors = [C["green"] if v >= 0 else C["red"] for v in m["real"]]
    fig.add_trace(go.Bar(x=m.index, y=m["real"], name="Real yield 10Y",
                         marker_color=bar_colors, opacity=0.8,
                         hovertemplate="%{y:+.2f}%<extra>Real yield</extra>"))
    fig.add_hline(y=0, line_color=C["muted"], line_width=0.8)
    fig.update_layout(**_base("REAL YIELD 10Y  ·  Nominale − Breakeven 10Y", "%", 260))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  KPI
# ─────────────────────────────────────────────────────────────────────────────

def _yield_kpi(label: str, sid: str, data: dict, color: str) -> html.Div:
    yields_wide = data["yields_wide"]
    val_str, chg_str, chg_pos = "N/D", "—", True
    if not yields_wide.empty and sid in yields_wide.columns:
        yields_wide = yields_wide.copy()
        yields_wide["date"] = pd.to_datetime(yields_wide["date"])
        col_data = yields_wide.dropna(subset=[sid])
        if not col_data.empty:
            last = col_data.iloc[-1]
            prev = col_data.iloc[-6] if len(col_data) >= 6 else col_data.iloc[0]
            val_str = f"{last[sid]:.2f}%"
            chg = last[sid] - prev[sid]
            chg_str = f"{chg:+.2f}%"
            chg_pos = chg >= 0

    # Per tassi: salita non è necessariamente positiva (dipende dal contesto)
    d_col = C["red"] if chg_pos else C["green"]  # salita tassi = pressione
    arrow = "▲" if chg_pos else "▼"

    return html.Div([
        html.Div(label, className="kpi-title"),
        html.Div(val_str, className="kpi-value", style={"color": color}),
        html.Span(f"{arrow} {chg_str}", style={"color": d_col, "fontSize": "11px", "fontFamily": "monospace"}),
        html.Span(" 1W", style={"color": C["muted"], "fontSize": "10px", "fontFamily": "monospace"}),
    ], className="kpi-card", style={"borderLeftColor": color})


def _curve_kpi(data: dict) -> html.Div:
    spreads = data["spreads"]
    val_str = "N/D"
    col     = C["muted"]
    label   = "—"

    if "T10Y2Y" in spreads:
        s = spreads["T10Y2Y"]
        if not s.empty:
            val = s.iloc[-1]
            val_str = f"{val:+.2f}%"
            if val < -0.5:
                col, label = C["red"],    "INVERTITA 🔴"
            elif val < 0:
                col, label = C["orange"], "PIATTA / NEG 🟠"
            elif val < 0.5:
                col, label = C["yellow"], "PIATTA 🟡"
            else:
                col, label = C["green"],  "NORMALE 🟢"

    return html.Div([
        html.Div("CURVA 10Y−2Y", className="kpi-title"),
        html.Div(val_str, className="kpi-value", style={"color": col}),
        html.Div(label, style={"color": col, "fontSize": "12px", "fontFamily": "monospace"}),
    ], className="kpi-card", style={"borderLeftColor": col})


def build_kpi_row(data: dict) -> html.Div:
    colors = {
        "DFF":   "#64748b",
        "DGS2":  C["orange"],
        "DGS5":  C["red"],
        "DGS10": C["accent"],
        "DGS30": C["green"],
    }
    labels = {"DFF": "FED FUNDS", "DGS2": "2Y", "DGS5": "5Y", "DGS10": "10Y", "DGS30": "30Y"}

    return html.Div([
        _yield_kpi(labels[sid], sid, data, col)
        for sid, col in colors.items()
    ] + [_curve_kpi(data)], className="kpi-row")


# ─────────────────────────────────────────────────────────────────────────────
#  LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

def build_app_layout(data: dict) -> html.Div:
    yw = data["yields_wide"]
    last_date = pd.to_datetime(yw["date"]).max().strftime("%d %b %Y") if not yw.empty else "N/D"

    return html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.Span("📈 "),
                    html.Span("RATES & TREASURY", className="header-title"),
                ]),
                html.Div(
                    f"Yield curve USA · Spread · Breakeven · Real yield  ·  "
                    f"Dati: FRED  ·  Ultimo: {last_date}",
                    className="header-sub"
                ),
            ], className="header-wrap"),

            build_kpi_row(data),

            html.Div([
                html.Span("Orizzonte:", className="ctrl-label"),
                dcc.RadioItems(
                    id="rt-lookback",
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
                html.Span("  |  Heatmap:", className="ctrl-label"),
                dcc.RadioItems(
                    id="rt-heatmap-sid",
                    options=[
                        {"label": " 2Y",  "value": "DGS2"},
                        {"label": " 5Y",  "value": "DGS5"},
                        {"label": "10Y",  "value": "DGS10"},
                        {"label": "30Y",  "value": "DGS30"},
                    ],
                    value="DGS10",
                    inline=True,
                    className="radio-group",
                    inputStyle={"marginRight": "4px", "accentColor": C["accent"]},
                ),
            ], className="controls-wrap"),

            # Riga 1: Yield curve snapshot + Storico
            html.Div([
                html.Div(dcc.Graph(id="g-yc-snap",  config={"displayModeBar": False}),
                         className="chart-card", style={"flex": "1"}),
                html.Div(dcc.Graph(id="g-yc-hist",  config={"displayModeBar": False}),
                         className="chart-card", style={"flex": "2"}),
            ], className="row-2"),

            # Riga 2: Spread + Heatmap
            html.Div([
                html.Div(dcc.Graph(id="g-spread",   config={"displayModeBar": False}),
                         className="chart-card", style={"flex": "2"}),
                html.Div(dcc.Graph(id="g-heatmap",  config={"displayModeBar": False}),
                         className="chart-card", style={"flex": "1"}),
            ], className="row-2"),

            # Riga 3: Breakeven + Real yield
            html.Div([
                html.Div(dcc.Graph(id="g-breakeven",  config={"displayModeBar": False}),
                         className="chart-card", style={"flex": "1"}),
                html.Div(dcc.Graph(id="g-real-yield", config={"displayModeBar": False}),
                         className="chart-card", style={"flex": "1"}),
            ], className="row-2"),

            html.Div(
                "⚠️  Rates & Treasury  ·  Dati: FRED (US Treasury, Federal Reserve)  ·  "
                "Non costituisce consulenza finanziaria  ·  FIN ANALYTICS  ·  Nicoletti 2026",
                className="footer"
            ),
        ], className="page-wrap"),
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  APP & CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.CYBORG],
        title="Rates & Treasury · FIN ANALYTICS",
        suppress_callback_exceptions=True,
    )
    print("\n📈 Caricamento Rates & Treasury...")
    _data_global = load_data()
    app.layout = build_app_layout(_data_global)

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
    def update(years, heatmap_sid):
        return (
            chart_yield_curve_snapshot(_data_global),
            chart_yield_history(_data_global, years),
            chart_spread(_data_global, years),
            chart_yield_heatmap(_data_global, heatmap_sid),
            chart_breakeven(_data_global, years),
            chart_real_yield(_data_global, years),
        )

    print(f"\n🚀  Rates & Treasury  →  http://localhost:{PORT}\n")
    app.run(debug=False, port=PORT, host="0.0.0.0")
