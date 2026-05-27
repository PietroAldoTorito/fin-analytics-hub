#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  VOLATILITY RADAR  ·  FIN ANALYTICS Suite  ·  Nicoletti 2026        ║
║  VIX · VIX 3M (VXMT) · VVIX · MOVE Index                           ║
║  Regime di rischio · Term Structure · Percentile storico             ║
║  Data: yfinance (VIX/VXMT/VVIX) + FRED (MOVE via ICE proxy)        ║
║  Port: 8053                                                          ║
╚══════════════════════════════════════════════════════════════════════╝

Installazione (una tantum):
    pip install yfinance dash dash-bootstrap-components plotly pandas duckdb

Avvio:
    python volatility_radar.py  →  http://localhost:8053
"""

import os
import warnings
warnings.filterwarnings("ignore")

import requests
import pandas as pd
import numpy as np
import duckdb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from io import StringIO

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False
    print("⚠️  yfinance non installato. Esegui: pip install yfinance")

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

PORT        = 8053
CACHE_TTL_H = 6
_LOCAL_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
DB_PATH      = os.path.join(
    _LOCAL_CACHE if os.access(os.path.dirname(os.path.abspath(__file__)), os.W_OK)
    else "/tmp",
    "volatility.duckdb",
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
    "purple":     "#a855f7",
    "grid":       "#131929",
    # per strumento
    "vix":        "#ef4444",
    "vxmt":       "#fb923c",
    "vvix":       "#a855f7",
    "move":       "#38bdf8",
}

# Soglie regime VIX
VIX_LOW    = 15
VIX_MEDIUM = 20
VIX_HIGH   = 30

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


def _is_stale(con, series_id):
    try:
        row = con.execute(
            "SELECT updated_at FROM _cache_meta WHERE series_id = ?", [series_id]
        ).fetchone()
        if row:
            return (datetime.now() - row[0]).total_seconds() / 3600 > CACHE_TTL_H
    except Exception:
        pass
    return True


def _save_cache(con, series_id, df):
    table = f"vol_{series_id.lower().replace('^','').replace('-','_')}"
    con.execute(f"DROP TABLE IF EXISTS {table}")
    con.execute(f"CREATE TABLE {table} AS SELECT * FROM df")
    con.execute(
        "INSERT OR REPLACE INTO _cache_meta VALUES (?, ?)",
        [series_id, datetime.now()]
    )
    return table


def _load_cache(con, series_id):
    table = f"vol_{series_id.lower().replace('^','').replace('-','_')}"
    try:
        df = con.execute(f"SELECT date, value FROM {table} ORDER BY date").df()
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return None


def fetch_yfinance(ticker: str, start: str = "2005-01-01") -> pd.DataFrame:
    """Scarica serie storica da Yahoo Finance."""
    if not HAS_YF:
        return pd.DataFrame(columns=["date", "value"])
    try:
        raw = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        if raw.empty:
            return pd.DataFrame(columns=["date", "value"])
        # yfinance può restituire MultiIndex
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        col = "Close" if "Close" in raw.columns else raw.columns[0]
        df = raw[[col]].copy()
        df.columns = ["value"]
        df.index.name = "date"
        df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.dropna().sort_values("date").reset_index(drop=True)
        print(f"  [yf]     {ticker}: {len(df)} obs  (ultimo: {df['date'].max().date()})")
        return df
    except Exception as e:
        print(f"  [warn]   {ticker} yfinance fallito: {e}")
        return pd.DataFrame(columns=["date", "value"])


def fetch_fred_csv(series_id: str) -> pd.DataFrame:
    """Scarica serie da FRED (endpoint CSV pubblico)."""
    try:
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
    except Exception as e:
        print(f"  [warn]   FRED {series_id} fallito: {e}")
        return pd.DataFrame(columns=["date", "value"])


def get_series(con, series_id: str, source: str = "yf") -> pd.DataFrame:
    """Carica dal cache o ri-scarica."""
    if not _is_stale(con, series_id):
        cached = _load_cache(con, series_id)
        if cached is not None and not cached.empty:
            print(f"  [cache]  {series_id}: {len(cached)} obs")
            return cached

    if source == "yf":
        df = fetch_yfinance(series_id)
    else:
        df = fetch_fred_csv(series_id)

    if not df.empty:
        _save_cache(con, series_id, df)
    else:
        # prova cache stale
        cached = _load_cache(con, series_id)
        if cached is not None:
            return cached

    return df


def load_data() -> dict:
    """
    Ritorna dict con chiavi: vix, vxmt, vvix, move
    Ogni valore è un DataFrame con colonne [date, value] + metriche derivate.
    """
    con = _ensure_db()
    print("\n📡 Caricamento dati volatilità...")

    # VIX — CBOE Volatility Index
    vix  = get_series(con, "^VIX",  source="yf")
    # VXMT — VIX 3-Month (Cboe Mid-Term)
    # VIX 3-Month: prova ^VIX3M su Yahoo, poi VXMTCLS su FRED come fallback
    vxmt = get_series(con, "^VIX3M", source="yf")
    if vxmt.empty or len(vxmt) < 100:
        vxmt = get_series(con, "VXMTCLS", source="fred")
    if vxmt.empty or len(vxmt) < 100:
        vxmt = pd.DataFrame(columns=["date", "value"])   # nessuna fonte disponibile
    # VVIX — VIX of VIX
    vvix = get_series(con, "^VVIX", source="yf")
    # MOVE — ICE BofA MOVE Index (tasso obbligazionario)
    # FRED series ICEMAAB = non esiste liberamente; usiamo proxy FRED MOVE tramite
    # serie BAMLH0A0HYM2EY come proxy vol obbligazionaria, oppure FRED MOVE se disponibile
    move = get_series(con, "MOVE",  source="fred")  # alcuni broker lo espongono su FRED
    if move.empty:
        # Fallback: usa FRED VIXCLS come cross-check
        move = get_series(con, "VIXCLS", source="fred")
        move.columns = move.columns  # rinomina internamente solo per display

    con.close()

    # Allinea su date comuni (join esterno, poi gestisci NaN per display)
    dfs = {"vix": vix, "vxmt": vxmt, "vvix": vvix, "move": move}

    # Aggiungi metriche derivate per ciascuna serie
    result = {}
    for key, df in dfs.items():
        if df.empty:
            result[key] = df
            continue
        df = df.copy().sort_values("date").reset_index(drop=True)
        df["pct10"]  = df["value"].rolling(252).quantile(0.10)
        df["pct25"]  = df["value"].rolling(252).quantile(0.25)
        df["pct75"]  = df["value"].rolling(252).quantile(0.75)
        df["pct90"]  = df["value"].rolling(252).quantile(0.90)
        df["ma20"]   = df["value"].rolling(20).mean()
        df["ma60"]   = df["value"].rolling(60).mean()
        # percentile rank 1Y rolling
        df["rank1y"] = df["value"].rolling(252).rank(pct=True) * 100
        # percentile rank full history
        df["rank_all"] = df["value"].expanding().rank(pct=True) * 100
        # 1M change
        df["chg1m"]  = df["value"].pct_change(21) * 100
        result[key] = df

    # DataFrame combinato per confronti
    combined = pd.DataFrame()
    for key, df in result.items():
        if not df.empty:
            s = df.set_index("date")["value"].rename(key)
            combined = pd.concat([combined, s], axis=1)

    # Normalizza a z-score per confronto (su finestra 1Y)
    for col in combined.columns:
        roll = combined[col].rolling(252)
        combined[f"{col}_z"] = (combined[col] - roll.mean()) / roll.std()

    combined.index.name = "date"
    result["combined"] = combined.reset_index()

    print(f"✅ Dati volatilità pronti\n")
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  REGIME
# ─────────────────────────────────────────────────────────────────────────────

def get_vix_regime(vix_val: float) -> tuple:
    """Restituisce (label, color, emoji) basato sul livello VIX."""
    if vix_val < VIX_LOW:
        return "COMPLACENCY", C["green"], "🟢"
    elif vix_val < VIX_MEDIUM:
        return "LOW FEAR", C["yellow"], "🟡"
    elif vix_val < VIX_HIGH:
        return "ELEVATED", C["orange"], "🟠"
    else:
        return "STRESS / PANIC", C["red"], "🔴"


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
    if isinstance(df, pd.DataFrame):
        if df.empty:
            return df
        if "date" not in df.columns:
            # date potrebbe essere l'indice
            df = df.reset_index().rename(columns={"index": "date"})
        df["date"] = pd.to_datetime(df["date"])
        return df[df["date"] >= df["date"].max() - pd.DateOffset(years=years)].copy()
    elif isinstance(df, pd.Series):
        if df.empty:
            return df
        return df[df.index >= df.index.max() - pd.DateOffset(years=years)].copy()
    return df


# ── 1. VIX storico con regime shading ──────────────────────────────────────
def chart_vix_history(data: dict, years: int = 3) -> go.Figure:
    df  = _cutoff(data["vix"], years) if not data["vix"].empty else pd.DataFrame()
    fig = go.Figure()

    if df.empty:
        fig.add_annotation(text="Dati VIX non disponibili", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=C["muted"]))
        fig.update_layout(**_base("VIX  ·  CBOE Volatility Index", "", 360))
        return fig

    # Regime bands
    fig.add_hrect(y0=0,       y1=VIX_LOW,    fillcolor="rgba(34,197,94,0.06)",   layer="below", line_width=0)
    fig.add_hrect(y0=VIX_LOW, y1=VIX_MEDIUM, fillcolor="rgba(234,179,8,0.05)",   layer="below", line_width=0)
    fig.add_hrect(y0=VIX_MEDIUM, y1=VIX_HIGH,fillcolor="rgba(251,146,60,0.05)",  layer="below", line_width=0)
    fig.add_hrect(y0=VIX_HIGH,y1=100,         fillcolor="rgba(239,68,68,0.06)",   layer="below", line_width=0)

    # Soglie
    for lvl, lbl in [(VIX_LOW, "15"), (VIX_MEDIUM, "20"), (VIX_HIGH, "30")]:
        fig.add_hline(y=lvl, line_color=C["muted"], line_width=0.6, line_dash="dot",
                      annotation_text=lbl, annotation_position="right",
                      annotation_font=dict(color=C["muted"], size=9))

    # MA 20
    fig.add_trace(go.Scatter(x=df["date"], y=df["ma20"], name="MA 20",
                             line=dict(color=C["muted"], width=1, dash="dot"),
                             hovertemplate="%{y:.1f}<extra>MA20</extra>"))
    # VIX
    fig.add_trace(go.Scatter(x=df["date"], y=df["value"], name="VIX",
                             line=dict(color=C["vix"], width=2),
                             fill="tozeroy", fillcolor="rgba(239,68,68,0.06)",
                             hovertemplate="%{y:.2f}<extra>VIX</extra>"))

    fig.update_layout(**_base("VIX  ·  CBOE Volatility Index", "punti", 360))
    return fig


# ── 2. Term structure VIX (VIX vs VXMT) ────────────────────────────────────
def chart_term_structure(data: dict, years: int = 2) -> go.Figure:
    vix  = _cutoff(data["vix"],  years)
    vxmt = _cutoff(data["vxmt"], years)
    fig  = go.Figure()

    if not vix.empty:
        fig.add_trace(go.Scatter(x=vix["date"], y=vix["value"], name="VIX (1M)",
                                 line=dict(color=C["vix"], width=2),
                                 hovertemplate="%{y:.2f}<extra>VIX</extra>"))
    if not vxmt.empty:
        fig.add_trace(go.Scatter(x=vxmt["date"], y=vxmt["value"], name="VXV (3M)",
                                 line=dict(color=C["vxmt"], width=1.8),
                                 hovertemplate="%{y:.2f}<extra>VXV 3M</extra>"))

    # Spread (contango / backwardation)
    if not vix.empty and not vxmt.empty:
        merged = pd.merge(vix[["date","value"]], vxmt[["date","value"]],
                          on="date", suffixes=("_vix","_vxmt")).dropna()
        if not merged.empty:
            spread = merged["value_vxmt"] - merged["value_vix"]
            fig.add_trace(go.Bar(x=merged["date"], y=spread, name="Spread 3M−1M",
                                 marker_color=[C["green"] if v >= 0 else C["red"] for v in spread],
                                 opacity=0.4, yaxis="y2",
                                 hovertemplate="%{y:+.2f}<extra>Spread</extra>"))

    fig.update_layout(
        **_base("TERM STRUCTURE VIX  ·  1M vs 3M  (contango = normalità, backwardation = stress)", "punti", 300),
        yaxis2=dict(overlaying="y", side="right", showgrid=False,
                    tickfont=dict(size=9), zeroline=True, zerolinecolor=C["muted"],
                    title=dict(text="spread", font=dict(size=9, color=C["muted"]))),
    )
    return fig


# ── 3. VVIX (vol della vol) ─────────────────────────────────────────────────
def chart_vvix(data: dict, years: int = 2) -> go.Figure:
    df  = _cutoff(data["vvix"], years)
    fig = go.Figure()

    if df.empty:
        fig.add_annotation(text="Dati VVIX non disponibili", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=C["muted"]))
        fig.update_layout(**_base("VVIX  ·  VIX of VIX", "", 280))
        return fig

    fig.add_hline(y=100, line_color=C["orange"], line_width=0.8, line_dash="dot",
                  annotation_text="100", annotation_position="right",
                  annotation_font=dict(color=C["orange"], size=9))
    fig.add_hline(y=120, line_color=C["red"], line_width=0.6, line_dash="dot",
                  annotation_text="120", annotation_position="right",
                  annotation_font=dict(color=C["red"], size=9))

    fig.add_trace(go.Scatter(x=df["date"], y=df["ma20"], name="MA 20",
                             line=dict(color=C["muted"], width=1, dash="dot"),
                             hovertemplate="%{y:.1f}<extra>MA20</extra>"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["value"], name="VVIX",
                             line=dict(color=C["vvix"], width=2),
                             fill="tozeroy", fillcolor="rgba(168,85,247,0.06)",
                             hovertemplate="%{y:.2f}<extra>VVIX</extra>"))

    fig.update_layout(**_base("VVIX  ·  Volatilità della volatilità  (>100 = tail risk elevato)", "punti", 280))
    return fig


# ── 4. MOVE index ────────────────────────────────────────────────────────────
def chart_move(data: dict, years: int = 2) -> go.Figure:
    df  = _cutoff(data["move"], years)
    fig = go.Figure()

    if df.empty:
        fig.add_annotation(text="Dati MOVE non disponibili", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=C["muted"]))
        fig.update_layout(**_base("MOVE  ·  ICE BofA Bond Volatility", "", 280))
        return fig

    fig.add_hline(y=100, line_color=C["orange"], line_width=0.8, line_dash="dot",
                  annotation_text="100", annotation_position="right",
                  annotation_font=dict(color=C["orange"], size=9))

    fig.add_trace(go.Scatter(x=df["date"], y=df["ma20"], name="MA 20",
                             line=dict(color=C["muted"], width=1, dash="dot"),
                             hovertemplate="%{y:.1f}<extra>MA20</extra>"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["value"], name="MOVE",
                             line=dict(color=C["move"], width=2),
                             fill="tozeroy", fillcolor="rgba(56,189,248,0.06)",
                             hovertemplate="%{y:.2f}<extra>MOVE</extra>"))

    fig.update_layout(**_base("MOVE  ·  Volatilità obbligazionaria USA  (proxy ICE BofA)", "bps", 280))
    return fig


# ── 5. Percentile rank radar ─────────────────────────────────────────────────
def chart_percentile_gauge(data: dict) -> go.Figure:
    """Barre orizzontali del percentile rank 1Y per ogni indice."""
    items = [
        ("VIX",  data["vix"],  C["vix"]),
        ("VXV",  data["vxmt"], C["vxmt"]),
        ("VVIX", data["vvix"], C["vvix"]),
        ("MOVE", data["move"], C["move"]),
    ]

    labels, values, colors = [], [], []
    for name, df, col in items:
        if not df.empty and "rank1y" in df.columns:
            last = df.dropna(subset=["rank1y"]).iloc[-1]
            labels.append(f"{name}  {last['value']:.1f}")
            values.append(last["rank1y"])
            colors.append(col)

    fig = go.Figure()
    if not labels:
        fig.add_annotation(text="Dati insufficienti", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=C["muted"]))
    else:
        fig.add_trace(go.Bar(
            x=values, y=labels, orientation="h",
            marker_color=colors,
            text=[f"{v:.0f}°pct" for v in values],
            textposition="outside",
            textfont=dict(color=C["text"], size=11, family="monospace"),
            hovertemplate="%{x:.0f}° percentile 1Y<extra>%{y}</extra>",
        ))
        # Zona di attenzione
        fig.add_vline(x=75, line_color=C["orange"], line_dash="dot", line_width=1)
        fig.add_vline(x=90, line_color=C["red"],    line_dash="dot", line_width=1)

    layout = _base("PERCENTILE RANK 1Y  ·  Livello attuale vs ultimi 252 gg", "percentile", 240, False)
    layout["xaxis"]["range"] = [0, 110]
    layout["xaxis"]["ticksuffix"] = "%"
    layout["margin"]["r"] = 80
    fig.update_layout(**layout)
    return fig


# ── 6. Z-score normalizzato (confronto cross) ───────────────────────────────
def chart_zscore_combined(data: dict, years: int = 2) -> go.Figure:
    combined = _cutoff(data.get("combined", pd.DataFrame()), years)
    fig = go.Figure()

    mapping = [
        ("vix_z",  "VIX z-score",  C["vix"]),
        ("vxmt_z", "VXV z-score",  C["vxmt"]),
        ("vvix_z", "VVIX z-score", C["vvix"]),
        ("move_z", "MOVE z-score", C["move"]),
    ]

    for col, label, color in mapping:
        if col in combined.columns:
            fig.add_trace(go.Scatter(x=combined["date"], y=combined[col], name=label,
                                     line=dict(color=color, width=1.5),
                                     hovertemplate="%{y:+.2f}σ<extra>" + label + "</extra>"))

    fig.add_hline(y=0,   line_color=C["muted"],  line_width=0.8)
    fig.add_hline(y=1.5, line_color=C["orange"], line_width=0.6, line_dash="dot")
    fig.add_hline(y=2,   line_color=C["red"],    line_width=0.6, line_dash="dot")
    fig.add_hrect(y0=1.5, y1=5, fillcolor="rgba(239,68,68,0.04)", layer="below", line_width=0)

    fig.update_layout(**_base("Z-SCORE 1Y  ·  Confronto cross-asset volatilità", "σ", 280))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  KPI COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def _vol_kpi(name: str, df: pd.DataFrame, color: str) -> html.Div:
    ranked = df.dropna(subset=["rank1y"]) if (not df.empty and "rank1y" in df.columns) else pd.DataFrame()
    if df.empty or "rank1y" not in df.columns or ranked.empty:
        val_str = f"{df['value'].iloc[-1]:.1f}" if not df.empty and "value" in df.columns else "N/D"
        chg_str = "—"
        pct_str = "N/D pct"
        chg_pos = True
    else:
        last = ranked.iloc[-1]
        val_str = f"{last['value']:.1f}"
        chg1m   = last.get("chg1m", np.nan)
        chg_str = f"{chg1m:+.1f}%" if not np.isnan(chg1m) else "—"
        pct_str = f"{last['rank1y']:.0f}°pct"
        chg_pos = chg1m >= 0 if not np.isnan(chg1m) else True

    # Per volatilità: salita = brutta notizia → colore inverso
    bad_up  = chg_pos
    d_col   = C["red"] if bad_up else C["green"]
    arrow   = "▲" if chg_pos else "▼"

    return html.Div([
        html.Div(name, className="kpi-title"),
        html.Div(val_str, className="kpi-value", style={"color": color}),
        html.Span(f"{arrow} {chg_str} ", style={"color": d_col, "fontSize": "11px", "fontFamily": "monospace"}),
        html.Span("1M  ", style={"color": C["muted"], "fontSize": "10px", "fontFamily": "monospace"}),
        html.Span(pct_str, style={"color": color, "fontSize": "10px", "fontFamily": "monospace"}),
    ], className="kpi-card", style={"borderLeftColor": color})


def _regime_kpi(data: dict) -> html.Div:
    vix_df = data.get("vix", pd.DataFrame())
    if vix_df.empty:
        label, col, icon = "N/D", C["muted"], "⚪"
        vix_val = 0.0
    else:
        vix_val = vix_df["value"].iloc[-1]
        label, col, icon = get_vix_regime(vix_val)

    return html.Div([
        html.Div("REGIME DI RISCHIO", className="kpi-title"),
        html.Div(f"{icon}  {label}", className="kpi-value",
                 style={"color": col, "fontSize": "16px"}),
        html.Div(f"VIX = {vix_val:.1f}  ·  soglie: <{VIX_LOW} calm  <{VIX_MEDIUM} low  <{VIX_HIGH} high",
                 style={"color": C["muted"], "fontSize": "10px", "fontFamily": "monospace",
                        "marginTop": "4px"}),
    ], className="kpi-card", style={"borderLeftColor": col})


def build_kpi_row(data: dict) -> html.Div:
    return html.Div([
        _vol_kpi("VIX",        data.get("vix",  pd.DataFrame()), C["vix"]),
        _vol_kpi("VXV (3M)",   data.get("vxmt", pd.DataFrame()), C["vxmt"]),
        _vol_kpi("VVIX",       data.get("vvix", pd.DataFrame()), C["vvix"]),
        _vol_kpi("MOVE",       data.get("move", pd.DataFrame()), C["move"]),
        _regime_kpi(data),
    ], className="kpi-row")


# ─────────────────────────────────────────────────────────────────────────────
#  LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

def build_app_layout(data: dict) -> html.Div:
    vix_df = data.get("vix", pd.DataFrame())
    last_date = vix_df["date"].max().strftime("%d %b %Y") if not vix_df.empty else "N/D"

    return html.Div([
        html.Div([
            # Header
            html.Div([
                html.Div([
                    html.Span("⚡ "),
                    html.Span("VOLATILITY RADAR", className="header-title"),
                ]),
                html.Div(
                    f"VIX · VXV 3M · VVIX · MOVE  ·  Dati: Yahoo Finance + FRED  ·  "
                    f"Ultimo aggiornamento: {last_date}",
                    className="header-sub"
                ),
            ], className="header-wrap"),

            # KPI
            build_kpi_row(data),

            # Controlli
            html.Div([
                html.Span("Orizzonte:", className="ctrl-label"),
                dcc.RadioItems(
                    id="vol-lookback",
                    options=[
                        {"label": " 6M", "value": 0.5},
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
            ], className="controls-wrap"),

            # Riga 1: VIX + Term Structure
            html.Div([
                html.Div(dcc.Graph(id="g-vix-hist",  config={"displayModeBar": False}),
                         className="chart-card", style={"flex": "3"}),
                html.Div(dcc.Graph(id="g-pct-gauge", config={"displayModeBar": False}),
                         className="chart-card", style={"flex": "1", "minWidth": "220px"}),
            ], className="row-2"),

            # Riga 2: Term Structure + Z-score
            html.Div([
                html.Div(dcc.Graph(id="g-term-struct", config={"displayModeBar": False}),
                         className="chart-card", style={"flex": "1"}),
                html.Div(dcc.Graph(id="g-zscore-vol",  config={"displayModeBar": False}),
                         className="chart-card", style={"flex": "1"}),
            ], className="row-2"),

            # Riga 3: VVIX + MOVE
            html.Div([
                html.Div(dcc.Graph(id="g-vvix", config={"displayModeBar": False}),
                         className="chart-card", style={"flex": "1"}),
                html.Div(dcc.Graph(id="g-move", config={"displayModeBar": False}),
                         className="chart-card", style={"flex": "1"}),
            ], className="row-2"),

            # Footer
            html.Div(
                "⚠️  Volatility Radar  ·  VIX/VVIX/VXV: CBOE via Yahoo Finance  ·  "
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
        title="Volatility Radar · FIN ANALYTICS",
        suppress_callback_exceptions=True,
    )
    print("\n⚡ Caricamento Volatility Radar...")
    _data_global = load_data()
    app.layout = build_app_layout(_data_global)

    @app.callback(
        Output("g-vix-hist",    "figure"),
        Output("g-term-struct", "figure"),
        Output("g-vvix",        "figure"),
        Output("g-move",        "figure"),
        Output("g-pct-gauge",   "figure"),
        Output("g-zscore-vol",  "figure"),
        Input("vol-lookback",   "value"),
    )
    def update(years):
        return (
            chart_vix_history(_data_global, years),
            chart_term_structure(_data_global, years),
            chart_vvix(_data_global, years),
            chart_move(_data_global, years),
            chart_percentile_gauge(_data_global),
            chart_zscore_combined(_data_global, years),
        )

    print(f"\n🚀  Volatility Radar  →  http://localhost:{PORT}\n")
    app.run(debug=False, port=PORT, host="0.0.0.0")
