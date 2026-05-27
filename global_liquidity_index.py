#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  GLOBAL LIQUIDITY INDEX  ·  FIN ANALYTICS Suite  ·  Nicoletti 2026  ║
║  Fed · ECB · BOJ · BOE · PBOC  ≈ 85% del GLI globale               ║
║  Score 0-100 · Fasi ciclo · Lead/Lag vs azionario/bond/crypto       ║
║  Dati: FRED API (pubblico)                                           ║
║  Port: 8055                                                          ║
╚══════════════════════════════════════════════════════════════════════╝

Installazione:
    pip install dash dash-bootstrap-components plotly pandas duckdb requests

Avvio:
    python global_liquidity_index.py  →  http://localhost:8055

Note sui dati FRED:
    WALCL   = Fed Total Assets (Weekly, M$)
    ECBASSETSW = ECB Assets (Weekly, M€)  — oppure proxy WECBEUROASS
    JPNASSETS  = BOJ Assets (proxy, Monthly)
    UKASSETSW  = BOE Assets (proxy)
    Per PBOC usiamo M2 cinese come proxy (MYAGM2CNM189N)
    Per il confronto asset: WILL5000IND (Wilshire 5000), BAMLH0A0HYM2EY (HY spread)
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

PORT        = 8055
CACHE_TTL_H = 12   # GLI mensile, refresh meno frequente
_LOCAL_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
DB_PATH      = os.path.join(
    _LOCAL_CACHE if os.access(os.path.dirname(os.path.abspath(__file__)), os.W_OK)
    else "/tmp",
    "gli.duckdb",
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
    # colori per banca centrale
    "fed":   "#00d4ff",
    "ecb":   "#3b82f6",
    "boj":   "#f97316",
    "boe":   "#a855f7",
    "pboc":  "#22c55e",
}

# ── FRED series per bilanci banche centrali ──────────────────────────────────
# Nota: non tutte le banche centrali hanno serie FRED complete.
# Usiamo i migliori proxy disponibili pubblicamente.
CB_SERIES = {
    "fed":  {
        "sid":   "WALCL",
        "label": "Federal Reserve",
        "unit":  "M$",
        "scale": 1/1000,   # M$ → B$
        "freq":  "W",
        "color": C["fed"],
    },
    "ecb":  {
        # ECB total assets in EUR billion
        "sid":   "ECBASSETSW",
        "label": "ECB",
        "unit":  "M€",
        "scale": 1/1000,
        "freq":  "W",
        "color": C["ecb"],
    },
    "boj":  {
        # BOJ balance sheet (proxy: Japan M2 + reserves)
        # FRED: JPNASSETS non esiste direttamente; usiamo RBJAJABTPSDN (BOJ reserves, monthly)
        "sid":   "RBJAJABTPSDN",
        "label": "Bank of Japan",
        "unit":  "100M JPY",
        "scale": 1/100000,  # → T JPY approssimativo
        "freq":  "M",
        "color": C["boj"],
    },
    "boe":  {
        # BOE: RBOEASBIS (BIS proxy) o MABMM301GBM189S (M1 UK)
        "sid":   "MABMM301GBM189S",
        "label": "Bank of England",
        "unit":  "M£",
        "scale": 1/1000,
        "freq":  "M",
        "color": C["boe"],
    },
    "pboc": {
        # PBOC: M2 China come proxy della liquidità (mensile)
        "sid":   "MYAGM2CNM189N",
        "label": "PBOC (M2 proxy)",
        "unit":  "100M CNY",
        "scale": 1/100000,
        "freq":  "M",
        "color": C["pboc"],
    },
}

# ── Serie mercato per lead/lag ───────────────────────────────────────────────
MARKET_SERIES = {
    "sp500":   {"sid": "SP500",          "label": "S&P 500",       "color": C["accent"]},
    "wilsh":   {"sid": "WILL5000IND",    "label": "Wilshire 5000", "color": "#38bdf8"},
    "hy":      {"sid": "BAMLH0A0HYM2EY", "label": "HY Yield",      "color": C["orange"]},
    "dgs10":   {"sid": "DGS10",          "label": "10Y Treasury",  "color": C["yellow"]},
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


def _is_stale(con, sid, ttl=CACHE_TTL_H):
    try:
        row = con.execute("SELECT updated_at FROM _cache_meta WHERE series_id=?", [sid]).fetchone()
        if row:
            return (datetime.now() - row[0]).total_seconds() / 3600 > ttl
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
        if not df.empty:
            print(f"  [FRED]   {sid}: {len(df)} obs  (ultimo: {df['date'].max().date()})")
        return df
    except Exception as e:
        print(f"  [warn]   FRED {sid}: {e}")
        return pd.DataFrame(columns=["date", "value"])


def _get_series(con, sid: str, ttl: int = CACHE_TTL_H) -> pd.DataFrame:
    table = f"gli_{sid.lower().replace('-','_')}"
    if not _is_stale(con, sid, ttl):
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


def _normalize_to_monthly(df: pd.DataFrame, scale: float = 1.0) -> pd.Series:
    """Resample a frequenza mensile, applica scala."""
    if df.empty:
        return pd.Series(dtype=float)
    s = df.set_index("date")["value"] * scale
    s.index = pd.DatetimeIndex(s.index)
    return s.resample("MS").last().ffill()


def _zscore_norm(s: pd.Series, window: int = 36) -> pd.Series:
    """Z-score rolling su finestra `window` mesi."""
    roll = s.rolling(window)
    return (s - roll.mean()) / roll.std()


def _minmax_norm(s: pd.Series, window: int = 60) -> pd.Series:
    """Min-max scaling rolling 0-100."""
    roll_min = s.rolling(window).min()
    roll_max = s.rolling(window).max()
    scaled = (s - roll_min) / (roll_max - roll_min + 1e-9) * 100
    return scaled.clip(0, 100)


def compute_gli(components: dict) -> pd.DataFrame:
    """
    Calcola il Global Liquidity Index:
    1. Normalizza ogni bilancio a YoY% change
    2. Applica pesi approssimativi per PIL
    3. Calcola score 0-100 con min-max rolling
    4. Classifica fase del ciclo
    """
    # Pesi approssimati per PIL (Fed/ECB/BOJ/BOE/PBOC ≈ 85% GLI)
    WEIGHTS = {"fed": 0.30, "ecb": 0.22, "boj": 0.15, "boe": 0.08, "pboc": 0.25}

    monthly = {}
    for key, meta in CB_SERIES.items():
        df = components.get(key, pd.DataFrame())
        s  = _normalize_to_monthly(df, meta["scale"])
        if not s.empty:
            monthly[key] = s

    if not monthly:
        return pd.DataFrame()

    # Allinea su griglia mensile comune
    combined = pd.concat(monthly, axis=1).sort_index()

    # YoY% change per ogni banca centrale
    yoy = combined.pct_change(12) * 100

    # Weighted average YoY (pesi normalizzati sulle CB disponibili)
    avail_weights = {k: WEIGHTS[k] for k in yoy.columns if k in WEIGHTS}
    total_w = sum(avail_weights.values())
    weighted = sum(yoy[k] * (v / total_w) for k, v in avail_weights.items() if k in yoy.columns)
    weighted.name = "gli_raw"

    # Livello aggregato (somma B$ / T-unit)
    level_cols = [c for c in combined.columns if c in avail_weights]
    level = combined[level_cols].sum(axis=1)
    level.name = "gli_level"

    # Score 0-100
    score = _minmax_norm(weighted.dropna(), window=60)
    score.name = "gli_score"

    df_out = pd.concat([
        combined,            # livelli singole CB
        yoy.add_suffix("_yoy"),
        weighted,
        level,
        score,
    ], axis=1).reset_index()
    df_out.rename(columns={"index": "date"}, inplace=True)
    if "date" not in df_out.columns:
        df_out = df_out.rename(columns={df_out.columns[0]: "date"})

    # Fase del ciclo (basata su score e momentum)
    score_series = df_out["gli_score"].copy()
    momentum = score_series.diff(3)   # cambio a 3 mesi

    conditions = [
        (score_series >= 60) & (momentum >= 0),   # Easing Forte
        (score_series >= 60) & (momentum < 0),    # Easing in rallentamento
        (score_series < 40)  & (momentum <= 0),   # Tightening Forte
        (score_series < 40)  & (momentum > 0),    # Tightening in attenuazione
    ]
    choices = ["Easing Forte", "Easing Attenuato", "Tightening Forte", "Tightening Attenuato"]
    df_out["phase"] = np.select(conditions, choices, default="Neutro")

    return df_out


def load_data() -> dict:
    con = _ensure_db()
    print("\n📡 Caricamento dati Global Liquidity Index...")

    # Banche centrali
    cb_raw = {}
    for key, meta in CB_SERIES.items():
        cb_raw[key] = _get_series(con, meta["sid"])

    # Mercato (lead/lag)
    mkt_raw = {}
    for key, meta in MARKET_SERIES.items():
        mkt_raw[key] = _get_series(con, meta["sid"])

    con.close()

    # GLI aggregato
    gli_df = compute_gli(cb_raw)

    # Normalizza anche mercato a mensile per correlazioni
    mkt_monthly = {}
    for key, meta in MARKET_SERIES.items():
        df = mkt_raw.get(key, pd.DataFrame())
        s  = _normalize_to_monthly(df, 1.0)
        if not s.empty:
            mkt_monthly[key] = s

    print(f"✅ GLI calcolato: {len(gli_df)} osservazioni mensili\n")
    return {
        "gli":       gli_df,
        "cb_raw":    cb_raw,
        "mkt_raw":   mkt_raw,
        "mkt_monthly": mkt_monthly,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def _base(title="", yt="", h=320, legend=True):
    return dict(
        title=dict(text=title, font=dict(color=C["muted"], size=10, family="monospace"), x=0),
        height=h,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C["text"], family="monospace", size=11),
        xaxis=dict(gridcolor=C["grid"], showgrid=True, zeroline=False,
                   tickfont=dict(size=10), showspikes=True,
                   spikecolor=C["muted"], spikethickness=1),
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
        if df.empty: return df
        col = "date" if "date" in df.columns else df.columns[0]
        mask = pd.to_datetime(df[col]) >= pd.to_datetime(df[col]).max() - pd.DateOffset(years=years)
        return df[mask].copy()
    elif isinstance(df, pd.Series):
        if df.empty: return df
        return df[df.index >= df.index.max() - pd.DateOffset(years=years)].copy()
    return df


# ── 1. GLI Score (principale) ────────────────────────────────────────────
def chart_gli_score(data: dict, years: int = 5) -> go.Figure:
    gli = _cutoff(data["gli"], years)
    fig = go.Figure()

    if gli.empty or "gli_score" not in gli.columns:
        fig.add_annotation(text="Dati GLI non disponibili — verifica connessione FRED",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(color=C["muted"], size=12))
        fig.update_layout(**_base("GLOBAL LIQUIDITY INDEX  ·  Score 0-100", "", 400))
        return fig

    gli["date"] = pd.to_datetime(gli["date"])

    # Phase shading
    phase_colors = {
        "Easing Forte":         "rgba(34,197,94,0.10)",
        "Easing Attenuato":     "rgba(34,197,94,0.04)",
        "Tightening Forte":     "rgba(239,68,68,0.10)",
        "Tightening Attenuato": "rgba(239,68,68,0.04)",
        "Neutro":               "rgba(0,0,0,0)",
    }
    prev_phase, seg_start = None, None
    for _, row in gli.iterrows():
        ph = row.get("phase", "Neutro")
        if ph != prev_phase:
            if prev_phase is not None and ph != "Neutro":
                fig.add_vrect(x0=seg_start, x1=row["date"],
                              fillcolor=phase_colors.get(prev_phase,"rgba(0,0,0,0)"),
                              layer="below", line_width=0)
            seg_start = row["date"]
            prev_phase = ph
    if prev_phase and seg_start:
        fig.add_vrect(x0=seg_start, x1=gli["date"].max(),
                      fillcolor=phase_colors.get(prev_phase,"rgba(0,0,0,0)"),
                      layer="below", line_width=0)

    # Zone
    fig.add_hrect(y0=60, y1=100, fillcolor="rgba(34,197,94,0.04)", layer="below", line_width=0)
    fig.add_hrect(y0=0,  y1=40,  fillcolor="rgba(239,68,68,0.04)", layer="below", line_width=0)
    fig.add_hline(y=50, line_color=C["muted"], line_width=0.8, line_dash="dot")
    fig.add_hline(y=60, line_color=C["green"], line_width=0.5, line_dash="dot",
                  annotation_text="Easing", annotation_position="right",
                  annotation_font=dict(color=C["green"], size=9))
    fig.add_hline(y=40, line_color=C["red"], line_width=0.5, line_dash="dot",
                  annotation_text="Tightening", annotation_position="right",
                  annotation_font=dict(color=C["red"], size=9))

    # Score
    score_valid = gli.dropna(subset=["gli_score"])
    fig.add_trace(go.Scatter(
        x=score_valid["date"], y=score_valid["gli_score"],
        name="GLI Score",
        line=dict(color=C["accent"], width=3),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.04)",
        hovertemplate="%{y:.1f}<extra>GLI Score</extra>",
    ))

    layout = _base("GLOBAL LIQUIDITY INDEX  ·  Score 0-100  (min-max rolling 5Y)", "score", 400)
    layout["yaxis"]["range"] = [0, 100]
    fig.update_layout(**layout)
    return fig


# ── 2. Bilanci banche centrali (livello) ─────────────────────────────────
def chart_cb_levels(data: dict, years: int = 5) -> go.Figure:
    gli = _cutoff(data["gli"], years)
    fig = go.Figure()

    if gli.empty:
        fig.update_layout(**_base("BILANCI BANCHE CENTRALI  ·  Livello normalizzato", "", 320))
        return fig

    gli["date"] = pd.to_datetime(gli["date"])

    cb_cols = [k for k in CB_SERIES.keys() if k in gli.columns]
    for key in cb_cols:
        meta = CB_SERIES[key]
        col_data = gli.dropna(subset=[key])
        if not col_data.empty:
            # Normalizza a 100 al punto iniziale per confronto
            base = col_data[key].iloc[0]
            if base != 0:
                y_vals = col_data[key] / base * 100
            else:
                y_vals = col_data[key]
            fig.add_trace(go.Scatter(
                x=col_data["date"], y=y_vals,
                name=meta["label"],
                line=dict(color=meta["color"], width=1.8),
                hovertemplate=f"%{{y:.0f}} (base=100)<extra>{meta['label']}</extra>",
            ))

    fig.update_layout(**_base("BILANCI CB  ·  Indice (base=100 al dato iniziale)", "indice", 340))
    return fig


# ── 3. YoY% per banca centrale ────────────────────────────────────────────
def chart_cb_yoy(data: dict, years: int = 5) -> go.Figure:
    gli = _cutoff(data["gli"], years)
    fig = go.Figure()

    if gli.empty:
        fig.update_layout(**_base("VARIAZIONE YoY% BANCHE CENTRALI", "%", 300))
        return fig

    gli["date"] = pd.to_datetime(gli["date"])

    for key in CB_SERIES.keys():
        col = f"{key}_yoy"
        if col in gli.columns:
            meta = CB_SERIES[key]
            col_data = gli.dropna(subset=[col])
            if not col_data.empty:
                fig.add_trace(go.Scatter(
                    x=col_data["date"], y=col_data[col],
                    name=meta["label"],
                    line=dict(color=meta["color"], width=1.5),
                    hovertemplate=f"%{{y:+.1f}}%<extra>{meta['label']}</extra>",
                ))

    # GLI weighted
    if "gli_raw" in gli.columns:
        raw_data = gli.dropna(subset=["gli_raw"])
        if not raw_data.empty:
            fig.add_trace(go.Scatter(
                x=raw_data["date"], y=raw_data["gli_raw"],
                name="GLI ponderato",
                line=dict(color=C["accent"], width=2.5),
                hovertemplate="%{y:+.1f}%<extra>GLI</extra>",
            ))

    fig.add_hline(y=0, line_color=C["muted"], line_width=0.8)
    fig.update_layout(**_base("ESPANSIONE BILANCI CB  ·  Variazione YoY%", "%", 320))
    return fig


# ── 4. Lead/Lag: GLI vs mercati ───────────────────────────────────────────
def chart_gli_vs_market(data: dict, market_key: str = "sp500", years: int = 5) -> go.Figure:
    gli = _cutoff(data["gli"], years)
    mkt = data["mkt_monthly"].get(market_key, pd.Series())

    fig = go.Figure()

    if gli.empty or "gli_score" not in gli.columns:
        fig.update_layout(**_base("GLI vs MERCATO", "", 340))
        return fig

    gli["date"] = pd.to_datetime(gli["date"])
    score = gli.dropna(subset=["gli_score"])

    # Normalizza mercato su scala secondaria
    if not mkt.empty:
        mkt_cut = mkt[mkt.index >= gli["date"].min()]
        if not mkt_cut.empty:
            mkt_norm = (mkt_cut - mkt_cut.min()) / (mkt_cut.max() - mkt_cut.min()) * 100
            fig.add_trace(go.Scatter(
                x=mkt_cut.index, y=mkt_norm.values,
                name=MARKET_SERIES[market_key]["label"],
                line=dict(color=MARKET_SERIES[market_key]["color"], width=1.5),
                yaxis="y2",
                hovertemplate="%{y:.0f} (norm)<extra>" + MARKET_SERIES[market_key]["label"] + "</extra>",
            ))

    fig.add_trace(go.Scatter(
        x=score["date"], y=score["gli_score"],
        name="GLI Score",
        line=dict(color=C["accent"], width=2.5),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.04)",
        hovertemplate="%{y:.1f}<extra>GLI Score</extra>",
    ))

    fig.add_hline(y=50, line_color=C["muted"], line_width=0.6, line_dash="dot")

    layout = _base(
        f"GLI  vs  {MARKET_SERIES.get(market_key,{}).get('label','Mercato')}  ·  Lead/Lag",
        "score", 360
    )
    layout["yaxis2"] = dict(
        overlaying="y", side="right", showgrid=False,
        tickfont=dict(size=9), zeroline=False,
        title=dict(text="indice (norm 0-100)", font=dict(size=9, color=C["muted"])),
    )
    layout["yaxis"]["range"] = [0, 100]
    fig.update_layout(**layout)
    return fig


# ── 5. Correlazione rolling GLI-Mercati ──────────────────────────────────
def chart_rolling_correlation(data: dict, years: int = 5) -> go.Figure:
    gli = data["gli"]
    fig = go.Figure()

    if gli.empty or "gli_score" not in gli.columns:
        fig.update_layout(**_base("CORRELAZIONE ROLLING  ·  GLI vs Mercati", "", 280))
        return fig

    gli = gli.copy()
    gli["date"] = pd.to_datetime(gli["date"])
    score = gli.set_index("date")["gli_score"].dropna()

    for key, meta in MARKET_SERIES.items():
        mkt = data["mkt_monthly"].get(key, pd.Series())
        if mkt.empty:
            continue
        merged = pd.concat([score.rename("gli"), mkt.rename("mkt")], axis=1).dropna()
        if len(merged) < 24:
            continue
        rolling_corr = merged["gli"].rolling(24).corr(merged["mkt"])
        cutoff = rolling_corr.index.max() - pd.DateOffset(years=years)
        rc = rolling_corr[rolling_corr.index >= cutoff]
        if not rc.empty:
            fig.add_trace(go.Scatter(
                x=rc.index, y=rc.values,
                name=meta["label"],
                line=dict(color=meta["color"], width=1.5),
                hovertemplate=f"%{{y:+.2f}}<extra>Corr GLI-{meta['label']}</extra>",
            ))

    fig.add_hline(y=0,    line_color=C["muted"],  line_width=0.8)
    fig.add_hline(y=0.5,  line_color=C["green"],  line_width=0.5, line_dash="dot")
    fig.add_hline(y=-0.5, line_color=C["red"],    line_width=0.5, line_dash="dot")

    fig.update_layout(**_base(
        "CORRELAZIONE ROLLING 24M  ·  GLI Score vs Azionario / Bond / HY",
        "coefficiente", 300
    ))
    return fig


# ── 6. Phase distribution (donut) ───────────────────────────────────────
def chart_phase_donut(data: dict) -> go.Figure:
    gli = data["gli"]
    fig = go.Figure()

    if gli.empty or "phase" not in gli.columns:
        fig.update_layout(**_base("DISTRIBUZIONE FASI", "", 260, False))
        return fig

    last_3y = gli[pd.to_datetime(gli["date"]) >= pd.to_datetime(gli["date"]).max() - pd.DateOffset(years=3)]
    counts = last_3y["phase"].value_counts()

    phase_colors_map = {
        "Easing Forte":         C["green"],
        "Easing Attenuato":     "#86efac",
        "Tightening Forte":     C["red"],
        "Tightening Attenuato": "#fca5a5",
        "Neutro":               C["muted"],
    }
    labels = counts.index.tolist()
    values = counts.values.tolist()
    colors = [phase_colors_map.get(l, C["muted"]) for l in labels]

    fig.add_trace(go.Pie(
        labels=labels, values=values,
        hole=0.55,
        marker_colors=colors,
        textfont=dict(family="monospace", size=10, color=C["text"]),
        hovertemplate="%{label}: %{percent}<extra></extra>",
    ))

    # Fase corrente al centro
    current_phase = gli.dropna(subset=["phase"]).iloc[-1]["phase"] if not gli.empty else "N/D"
    current_score = gli.dropna(subset=["gli_score"]).iloc[-1]["gli_score"] if "gli_score" in gli.columns else 0
    fig.add_annotation(
        text=f"<b>{current_score:.0f}</b><br><span style='font-size:9px'>{current_phase}</span>",
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False,
        font=dict(family="monospace", size=14, color=C["accent"]),
    )

    layout = _base("FASI CICLO LIQUIDITÀ  ·  ultimi 3 anni", "", 280, True)
    layout["showlegend"] = True
    layout["legend"]["orientation"] = "v"
    layout["legend"]["x"] = 1.0
    layout["legend"]["y"] = 0.5
    fig.update_layout(**layout)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  KPI
# ─────────────────────────────────────────────────────────────────────────────

def _cb_kpi(key: str, data: dict) -> html.Div:
    meta = CB_SERIES[key]
    gli  = data["gli"]
    val_str, chg_str, chg_pos = "N/D", "—", True

    yoy_col = f"{key}_yoy"
    if not gli.empty and yoy_col in gli.columns:
        col_data = gli.dropna(subset=[yoy_col])
        if not col_data.empty:
            last = col_data.iloc[-1]
            val_str = f"{last[yoy_col]:+.1f}%"
            chg_pos = last[yoy_col] >= 0

    d_col = C["green"] if chg_pos else C["red"]
    arrow = "▲" if chg_pos else "▼"

    return html.Div([
        html.Div(meta["label"].upper(), className="kpi-title"),
        html.Div(val_str, className="kpi-value", style={"color": meta["color"], "fontSize": "18px"}),
        html.Span(f"{arrow} YoY", style={"color": d_col, "fontSize": "11px", "fontFamily": "monospace"}),
    ], className="kpi-card", style={"borderLeftColor": meta["color"]})


def _gli_kpi(data: dict) -> html.Div:
    gli = data["gli"]
    score_val, phase, col = 50, "N/D", C["muted"]

    if not gli.empty and "gli_score" in gli.columns and "phase" in gli.columns:
        last = gli.dropna(subset=["gli_score", "phase"]).iloc[-1]
        score_val = last["gli_score"]
        phase     = last["phase"]
        if "Easing Forte"     == phase: col = C["green"]
        elif "Easing"         in phase: col = "#86efac"
        elif "Tightening Forte" == phase: col = C["red"]
        elif "Tightening"     in phase: col = "#fca5a5"
        else: col = C["yellow"]

    return html.Div([
        html.Div("GLI SCORE", className="kpi-title"),
        html.Div(f"{score_val:.0f} / 100", className="kpi-value", style={"color": col}),
        html.Div(phase, style={"color": col, "fontSize": "12px", "fontFamily": "monospace", "marginTop": "2px"}),
    ], className="kpi-card", style={"borderLeftColor": col, "flex": "2"})


def build_kpi_row(data: dict) -> html.Div:
    return html.Div([
        _gli_kpi(data),
        *[_cb_kpi(k, data) for k in CB_SERIES.keys()],
    ], className="kpi-row")


# ─────────────────────────────────────────────────────────────────────────────
#  LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

def build_app_layout(data: dict) -> html.Div:
    gli = data["gli"]
    last_date = pd.to_datetime(gli["date"]).max().strftime("%b %Y") if not gli.empty else "N/D"

    return html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.Span("🌊 "),
                    html.Span("GLOBAL LIQUIDITY INDEX", className="header-title"),
                ]),
                html.Div(
                    f"Fed · ECB · BOJ · BOE · PBOC  ≈ 85% GLI  ·  "
                    f"Dati: FRED  ·  Ultimo: {last_date}",
                    className="header-sub"
                ),
            ], className="header-wrap"),

            build_kpi_row(data),

            html.Div([
                html.Span("Orizzonte:", className="ctrl-label"),
                dcc.RadioItems(
                    id="gli-lookback",
                    options=[
                        {"label": " 2A", "value": 2},
                        {"label": " 3A", "value": 3},
                        {"label": " 5A", "value": 5},
                        {"label": "10A", "value": 10},
                        {"label": "15A", "value": 15},
                    ],
                    value=5,
                    inline=True,
                    className="radio-group",
                    inputStyle={"marginRight": "4px", "accentColor": C["accent"]},
                ),
                html.Span("  |  Lead/Lag vs:", className="ctrl-label"),
                dcc.RadioItems(
                    id="gli-market",
                    options=[
                        {"label": " S&P 500", "value": "sp500"},
                        {"label": " Wilshire", "value": "wilsh"},
                        {"label": " HY Yield","value": "hy"},
                        {"label": " 10Y",     "value": "dgs10"},
                    ],
                    value="sp500",
                    inline=True,
                    className="radio-group",
                    inputStyle={"marginRight": "4px", "accentColor": C["accent"]},
                ),
            ], className="controls-wrap"),

            # Riga 1: GLI Score (principale, largo)
            html.Div(
                dcc.Graph(id="g-gli-score", config={"displayModeBar": False}),
                className="chart-card"
            ),

            # Riga 2: Bilanci CB + YoY
            html.Div([
                html.Div(dcc.Graph(id="g-cb-levels", config={"displayModeBar": False}),
                         className="chart-card", style={"flex": "1"}),
                html.Div(dcc.Graph(id="g-cb-yoy",    config={"displayModeBar": False}),
                         className="chart-card", style={"flex": "1"}),
            ], className="row-2"),

            # Riga 3: Lead/Lag + Donut fasi
            html.Div([
                html.Div(dcc.Graph(id="g-gli-mkt",   config={"displayModeBar": False}),
                         className="chart-card", style={"flex": "3"}),
                html.Div(dcc.Graph(id="g-phase-donut", config={"displayModeBar": False}),
                         className="chart-card", style={"flex": "1", "minWidth": "240px"}),
            ], className="row-2"),

            # Riga 4: Correlazione rolling
            html.Div(
                dcc.Graph(id="g-rolling-corr", config={"displayModeBar": False}),
                className="chart-card"
            ),

            html.Div(
                "⚠️  Global Liquidity Index  ·  Stima costruita su proxy FRED  ·  "
                "BOJ/BOE/PBOC = serie approssimate  ·  Non costituisce consulenza finanziaria  ·  "
                "FIN ANALYTICS  ·  Nicoletti 2026",
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
        title="Global Liquidity Index · FIN ANALYTICS",
        suppress_callback_exceptions=True,
    )
    print("\n🌊 Caricamento Global Liquidity Index...")
    _data_global = load_data()
    app.layout = build_app_layout(_data_global)

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
    def update(years, market_key):
        return (
            chart_gli_score(_data_global, years),
            chart_cb_levels(_data_global, years),
            chart_cb_yoy(_data_global, years),
            chart_gli_vs_market(_data_global, market_key, years),
            chart_phase_donut(_data_global),
            chart_rolling_correlation(_data_global, years),
        )

    print(f"\n🚀  Global Liquidity Index  →  http://localhost:{PORT}\n")
    app.run(debug=False, port=PORT, host="0.0.0.0")
