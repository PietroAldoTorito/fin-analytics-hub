#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  FIN ANALYTICS  ·  Market Internals  ·  Nicoletti 2026              ║
║  Breadth · Sentiment · Valuation · S&P 500 & Nasdaq                 ║
║                                                                      ║
║  Indicatori:  %>200MA · %>50MA · A/D Line · McClellan               ║
║               NH/NL Ratio · Hindenburg Omen · Put/Call              ║
║               Buffett Indicator · SPX 12M Momentum                  ║
║  Alert system: semaforo con log storico dei cambi di colore         ║
║  Port: 8056                                                          ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os, sys, warnings
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import duckdb
import requests
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

PORT        = 8056
CACHE_TTL_H = 6

_LOCAL_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
DB_PATH      = os.path.join(
    _LOCAL_CACHE if os.access(os.path.dirname(os.path.abspath(__file__)), os.W_OK)
    else "/tmp",
    "market_internals.duckdb",
)

C = {
    "bg":      "#080c18",
    "surface": "#0d1117",
    "border":  "#1a2236",
    "text":    "#e2e8f0",
    "muted":   "#556080",
    "accent":  "#00d4ff",
    "green":   "#00e676",
    "yellow":  "#ffd600",
    "red":     "#ff1744",
    "gray":    "#37474f",
}

_LAYOUT_BASE = dict(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(0,0,0,0)",
    font          = dict(color=C["text"], family="monospace", size=11),
    margin        = dict(l=48, r=16, t=36, b=36),
    xaxis         = dict(gridcolor=C["border"], zerolinecolor=C["border"]),
    yaxis         = dict(gridcolor=C["border"], zerolinecolor=C["border"]),
)

NYSE_ISSUES = 3200   # approx total NYSE listed issues (for Hindenburg % calc)

# ─────────────────────────────────────────────────────────────────────────────
#  TRAFFIC LIGHT THRESHOLDS
#  Each entry: green/yellow lambdas. If neither → RED.
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLDS = {
    "pct_above_200ma": {
        "label":      "% S&P >200MA",
        "unit":       "%",
        "green":      lambda v: v >= 60,
        "yellow":     lambda v: 40 <= v < 60,
        "desc_green": "Breadth sana",
        "desc_yellow":"Breadth in deterioramento",
        "desc_red":   "Breadth rotta",
        "desc_gray":  "N/D",
    },
    "pct_above_50ma": {
        "label":      "% S&P >50MA",
        "unit":       "%",
        "green":      lambda v: v >= 65,
        "yellow":     lambda v: 45 <= v < 65,
        "desc_green": "Momentum positivo",
        "desc_yellow":"Momentum in calo",
        "desc_red":   "Momentum negativo",
        "desc_gray":  "N/D",
    },
    "mcclellan": {
        "label":      "McClellan Osc",
        "unit":       "",
        "green":      lambda v: v >= 20,
        "yellow":     lambda v: -20 <= v < 20,
        "desc_green": "A/D in espansione",
        "desc_yellow":"A/D neutro",
        "desc_red":   "A/D in contrazione",
        "desc_gray":  "N/D",
    },
    "nh_nl_ratio": {
        "label":      "NH / NL Ratio",
        "unit":       "x",
        "green":      lambda v: v >= 1.5,
        "yellow":     lambda v: 0.7 <= v < 1.5,
        "desc_green": "Highs dominanti",
        "desc_yellow":"Equilibrio NH/NL",
        "desc_red":   "Lows dominanti",
        "desc_gray":  "N/D",
    },
    "hindenburg": {
        "label":      "Hindenburg Omen",
        "unit":       "/4",
        "green":      lambda v: v <= 1,
        "yellow":     lambda v: v == 2,
        "desc_green": "Nessun segnale",
        "desc_yellow":"Segnale parziale",
        "desc_red":   "Omen attivo ⚠",
        "desc_gray":  "N/D",
    },
    "put_call": {
        "label":      "Put/Call 10d MA",
        "unit":       "",
        "green":      lambda v: v <= 0.80,
        "yellow":     lambda v: 0.80 < v <= 1.00,
        "desc_green": "Sentiment equilibrato",
        "desc_yellow":"Cautela crescente",
        "desc_red":   "Fear elevata",
        "desc_gray":  "N/D",
    },
    "buffett": {
        "label":      "Buffett Indicator",
        "unit":       "%",
        "green":      lambda v: v <= 110,
        "yellow":     lambda v: 110 < v <= 150,
        "desc_green": "Valuation ragionevole",
        "desc_yellow":"Mercato sopravvalutato",
        "desc_red":   "Valuation estrema",
        "desc_gray":  "N/D",
    },
    "spx_momentum": {
        "label":      "SPX 12M Return",
        "unit":       "%",
        "green":      lambda v: 5 <= v <= 25,
        "yellow":     lambda v: 0 <= v < 5 or 25 < v <= 35,
        "desc_green": "Trend sostenibile",
        "desc_yellow":"Momentum ai limiti",
        "desc_red":   "Negativo o parabólico",
        "desc_gray":  "N/D",
    },
}


def _color(name, value):
    """Returns 'green', 'yellow', 'red', or 'gray' (missing data)."""
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "gray"
    t = THRESHOLDS.get(name, {})
    if t.get("green",  lambda v: False)(value):
        return "green"
    if t.get("yellow", lambda v: False)(value):
        return "yellow"
    return "red"


def _color_hex(col):
    return {"green": C["green"], "yellow": C["yellow"], "red": C["red"]}.get(col, C["gray"])


# ─────────────────────────────────────────────────────────────────────────────
#  DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_yf(ticker, period="5y"):
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Close"]].rename(columns={"Close": "value"})
    except Exception as e:
        print(f"  [warn] yfinance {ticker}: {e}")
        return pd.DataFrame()


def _fetch_fred(series):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    try:
        df = pd.read_csv(url, names=["date", "value"], skiprows=1)
        df["date"]  = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date")
        df.index = pd.DatetimeIndex(df.index)
        return df
    except Exception as e:
        print(f"  [warn] FRED {series}: {e}")
        return pd.DataFrame()


def _fetch_cboe_pc():
    """CBOE Total Put/Call Ratio — daily."""
    url = "https://www.cboe.com/publish/scheduledtask/mktdata/datahouse/totalpc.csv"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
        # Skip non-data header lines; keep only rows where col-0 parses as date
        data_lines = []
        for line in lines:
            parts = line.split(",")
            if len(parts) < 4:
                continue
            try:
                pd.to_datetime(parts[0].strip())
                float(parts[3].strip())   # total ratio column
                data_lines.append(line)
            except Exception:
                pass
        if not data_lines:
            return pd.DataFrame()
        from io import StringIO
        df = pd.read_csv(StringIO("\n".join(data_lines)),
                         header=None, names=["date", "call", "put", "total"])
        df["date"]  = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["total"], errors="coerce")
        df = df.dropna(subset=["date", "value"]).set_index("date")
        df.index = pd.DatetimeIndex(df.index)
        return df[["value"]]
    except Exception as e:
        print(f"  [warn] CBOE P/C: {e}")
        return pd.DataFrame()


def _last_val(df, col="value"):
    if df is None or df.empty:
        return float("nan")
    try:
        return float(df[col].dropna().iloc[-1])
    except Exception:
        return float("nan")


def _compute_mcclellan(nyad_df):
    """
    McClellan Oscillator = EMA19 - EMA39 of daily A/D values.
    Summation Index     = cumulative McClellan.
    """
    if nyad_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    s = nyad_df["value"].dropna()
    # If series looks cumulative (large values), diff it first
    if s.abs().mean() > 5000:
        s = s.diff().dropna()
    ema19 = s.ewm(span=19, adjust=False).mean()
    ema39 = s.ewm(span=39, adjust=False).mean()
    osc       = (ema19 - ema39).rename("value").to_frame()
    summation = osc["value"].cumsum().rename("value").to_frame()
    return osc, summation


def _hindenburg_score(d):
    """
    Returns (score: int, conditions: list of (name, met: bool, detail: str)).
    """
    nyhgh = d.get("nyhgh", pd.DataFrame())
    nylow = d.get("nylow", pd.DataFrame())
    sp500 = d.get("sp500", pd.DataFrame())
    mc    = d.get("mcclellan", pd.DataFrame())

    nh = _last_val(nyhgh)
    nl = _last_val(nylow)

    # Condition 1 — NH and NL both > 2.1% of NYSE issues on same day
    cond1 = (not np.isnan(nh)) and (not np.isnan(nl)) \
            and (nh / NYSE_ISSUES > 0.021) and (nl / NYSE_ISSUES > 0.021)
    det1 = (f"NH={nh:.0f} ({nh/NYSE_ISSUES*100:.2f}%) / "
            f"NL={nl:.0f} ({nl/NYSE_ISSUES*100:.2f}%)"
            if not np.isnan(nh) else "N/D")

    # Condition 2 — SPX above its 50-day MA
    cond2, det2 = False, "N/D"
    if not sp500.empty:
        spx = sp500["value"].dropna()
        if len(spx) >= 50:
            ma50    = spx.rolling(50).mean().iloc[-1]
            last_px = spx.iloc[-1]
            cond2   = last_px > ma50
            det2    = f"SPX {last_px:,.0f} vs MA50 {ma50:,.0f}"

    # Condition 3 — McClellan Oscillator negative
    cond3, det3 = False, "N/D"
    if not mc.empty:
        mc_val = _last_val(mc)
        if not np.isnan(mc_val):
            cond3 = mc_val < 0
            det3  = f"McClellan = {mc_val:.1f}"

    # Condition 4 — NH count >= NL count
    cond4 = (not np.isnan(nh)) and (not np.isnan(nl)) and (nh >= nl)
    det4  = f"NH={nh:.0f} / NL={nl:.0f}" if not np.isnan(nh) else "N/D"

    conditions = [
        ("NH & NL entrambi > 2.1% issues", cond1, det1),
        ("SPX sopra MA50",                 cond2, det2),
        ("McClellan Oscillator < 0",       cond3, det3),
        ("New Highs ≥ New Lows",           cond4, det4),
    ]
    score = sum(1 for _, met, _ in conditions if met)
    return score, conditions


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    """Fetch, cache, derive all indicators. Returns dict."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = duckdb.connect(DB_PATH)

    # ── Init tables ──────────────────────────────────────────────────────────
    con.execute("""
        CREATE TABLE IF NOT EXISTS cache_meta (
            key        VARCHAR PRIMARY KEY,
            updated_at TIMESTAMP
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS indicator_states (
            indicator  VARCHAR PRIMARY KEY,
            color      VARCHAR,
            value      DOUBLE,
            updated_at TIMESTAMP
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS alert_history (
            ts        TIMESTAMP,
            indicator VARCHAR,
            label     VARCHAR,
            old_color VARCHAR,
            new_color VARCHAR,
            value     DOUBLE
        )
    """)

    def _is_fresh(key):
        row = con.execute(
            "SELECT updated_at FROM cache_meta WHERE key=?", [key]
        ).fetchone()
        if not row:
            return False
        return (datetime.utcnow() - row[0]).total_seconds() / 3600 < CACHE_TTL_H

    def _save_meta(key):
        con.execute("INSERT OR REPLACE INTO cache_meta VALUES (?,?)",
                    [key, datetime.utcnow()])

    def _cache_df(key, df):
        tbl = f"ds_{key}"
        con.execute(f"DROP TABLE IF EXISTS {tbl}")
        if not df.empty:
            con.execute(f"CREATE TABLE {tbl} AS SELECT * FROM df")

    def _load_cached(key):
        tbl = f"ds_{key}"
        try:
            df = con.execute(f"SELECT * FROM {tbl}").df()
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
                df.index = pd.DatetimeIndex(df.index)
            return df
        except Exception:
            return pd.DataFrame()

    def _get(key, fetch_fn):
        if _is_fresh(key):
            df = _load_cached(key)
            if not df.empty:
                return df
        print(f"  → fetching {key} …")
        df = fetch_fn()
        df_save = df.reset_index() if (not df.empty and df.index.name) else df
        _cache_df(key, df_save)
        _save_meta(key)
        return df

    # ── Raw fetches ───────────────────────────────────────────────────────────
    result = {}
    result["spxa200r"]   = _get("spxa200r",  lambda: _fetch_yf("^SPXA200R"))
    result["spxa50r"]    = _get("spxa50r",   lambda: _fetch_yf("^SPXA50R"))
    result["ndxa200r"]   = _get("ndxa200r",  lambda: _fetch_yf("^NDXA200R"))
    result["nyad"]       = _get("nyad",      lambda: _fetch_yf("^NYAD"))
    result["nyhgh"]      = _get("nyhgh",     lambda: _fetch_yf("^NYHGH"))
    result["nylow"]      = _get("nylow",     lambda: _fetch_yf("^NYLOW"))
    result["sp500"]      = _get("sp500_mi",  lambda: _fetch_yf("^GSPC"))
    result["nasdaq"]     = _get("nasdaq_mi", lambda: _fetch_yf("^IXIC"))
    result["pcr"]        = _get("pcr",       lambda: _fetch_cboe_pc())
    result["dgs10"]      = _get("mi_dgs10",  lambda: _fetch_fred("DGS10"))
    result["will5k"]     = _get("mi_will5k", lambda: _fetch_fred("WILL5000IND"))
    result["gdp"]        = _get("mi_gdp",    lambda: _fetch_fred("GDP"))

    # ── Derived: McClellan ────────────────────────────────────────────────────
    result["mcclellan"], result["summation"] = _compute_mcclellan(result["nyad"])

    # ── Derived: Buffett Indicator ────────────────────────────────────────────
    will, gdp = result["will5k"], result["gdp"]
    if not will.empty and not gdp.empty:
        gdp_m  = gdp.resample("ME").last().ffill()
        will_m = will.resample("ME").last()
        al = pd.DataFrame({"will": will_m["value"], "gdp": gdp_m["value"]}).dropna()
        al["buffett"] = al["will"] / al["gdp"] * 100
        result["buffett_df"] = al[["buffett"]].rename(columns={"buffett": "value"})
    else:
        result["buffett_df"] = pd.DataFrame()

    # ── Hindenburg Omen ───────────────────────────────────────────────────────
    hind_score, hind_conditions = _hindenburg_score(result)
    result["hind_score"]      = hind_score
    result["hind_conditions"] = hind_conditions

    # ── Current scalar values ─────────────────────────────────────────────────
    pcr_val = float("nan")
    if not result["pcr"].empty and len(result["pcr"]) >= 5:
        pcr_val = float(result["pcr"]["value"].rolling(10, min_periods=5).mean().dropna().iloc[-1])

    nh_nl_ratio = float("nan")
    nh, nl = _last_val(result["nyhgh"]), _last_val(result["nylow"])
    if not np.isnan(nh) and not np.isnan(nl):
        nh_nl_ratio = nh / nl if nl > 0 else (10.0 if nh > 0 else float("nan"))

    spx_mom = float("nan")
    if not result["sp500"].empty:
        spx = result["sp500"]["value"].dropna()
        if len(spx) >= 252:
            spx_mom = (spx.iloc[-1] / spx.iloc[-252] - 1) * 100
        elif len(spx) >= 2:
            spx_mom = (spx.iloc[-1] / spx.iloc[0] - 1) * 100

    current_values = {
        "pct_above_200ma": _last_val(result["spxa200r"]),
        "pct_above_50ma":  _last_val(result["spxa50r"]),
        "mcclellan":       _last_val(result["mcclellan"]),
        "nh_nl_ratio":     nh_nl_ratio,
        "hindenburg":      float(hind_score),
        "put_call":        pcr_val,
        "buffett":         _last_val(result["buffett_df"]),
        "spx_momentum":    spx_mom,
    }
    result["current_values"] = current_values

    # ── Traffic light colors ──────────────────────────────────────────────────
    colors = {k: _color(k, v) for k, v in current_values.items()}
    result["colors"] = colors

    # ── Alert system: detect color changes, persist to DB ────────────────────
    new_alerts = []
    now = datetime.utcnow()
    for ind, new_col in colors.items():
        if new_col == "gray":
            continue
        row = con.execute(
            "SELECT color FROM indicator_states WHERE indicator=?", [ind]
        ).fetchone()
        old_col = row[0] if row else None
        # Update current state
        con.execute(
            "INSERT OR REPLACE INTO indicator_states VALUES (?,?,?,?)",
            [ind, new_col, current_values.get(ind), now]
        )
        # Record change
        if old_col is not None and old_col != new_col:
            lbl = THRESHOLDS.get(ind, {}).get("label", ind)
            con.execute(
                "INSERT INTO alert_history VALUES (?,?,?,?,?,?)",
                [now, ind, lbl, old_col, new_col, current_values.get(ind)]
            )
            new_alerts.append({
                "ts": now, "indicator": ind, "label": lbl,
                "old_color": old_col, "new_color": new_col,
                "value": current_values.get(ind),
            })
            print(f"  🚨 ALERT: {lbl}  {old_col.upper()} → {new_col.upper()}")

    # ── Load alert history ────────────────────────────────────────────────────
    try:
        alerts_df = con.execute(
            "SELECT * FROM alert_history ORDER BY ts DESC LIMIT 100"
        ).df()
    except Exception:
        alerts_df = pd.DataFrame()

    result["alerts"]     = alerts_df
    result["new_alerts"] = new_alerts
    con.close()
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  CHART FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _empty_fig(msg="Dati non disponibili"):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(color=C["muted"], size=13))
    fig.update_layout(**_LAYOUT_BASE)
    return fig


def _cutoff(df, years):
    if df is None or df.empty:
        return df
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
    try:
        if "date" in df.columns:
            return df[df["date"] >= cutoff]
        return df[df.index >= cutoff]
    except Exception:
        return df


# ── Chart 1: % Stocks Above Moving Average ───────────────────────────────────

def chart_breadth_ma(d, years=2):
    sp200 = _cutoff(d.get("spxa200r", pd.DataFrame()), years)
    sp50  = _cutoff(d.get("spxa50r",  pd.DataFrame()), years)
    nq200 = _cutoff(d.get("ndxa200r", pd.DataFrame()), years)

    if sp200.empty and sp50.empty:
        return _empty_fig("% Stocks Above MA — N/D")

    fig = go.Figure()
    if not sp200.empty:
        fig.add_trace(go.Scatter(x=sp200.index, y=sp200["value"],
            name="S&P >200MA", line=dict(color=C["green"], width=2),
            fill="tozeroy", fillcolor="rgba(0,230,118,0.05)"))
    if not sp50.empty:
        fig.add_trace(go.Scatter(x=sp50.index, y=sp50["value"],
            name="S&P >50MA", line=dict(color=C["accent"], width=2)))
    if not nq200.empty:
        fig.add_trace(go.Scatter(x=nq200.index, y=nq200["value"],
            name="NDX >200MA", line=dict(color=C["yellow"], width=1.5, dash="dash")))

    for y_val, col, lbl in [(80, C["red"], "80%"), (60, C["yellow"], "60%"),
                             (40, C["yellow"], "40%"), (20, C["green"], "20%")]:
        fig.add_hline(y=y_val, line=dict(color=col, dash="dot", width=1),
                      annotation_text=lbl, annotation_font=dict(color=col, size=9))

    fig.update_layout(**_LAYOUT_BASE,
        title=dict(text="% Stocks Above Moving Average", font=dict(size=12, color=C["text"]), x=0),
        yaxis=dict(**_LAYOUT_BASE["yaxis"], range=[0, 100]),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )
    return fig


# ── Chart 2: NYSE A/D Line + McClellan Oscillator ────────────────────────────

def chart_ad_line(d, years=2):
    nyad = _cutoff(d.get("nyad", pd.DataFrame()), years)
    mc   = _cutoff(d.get("mcclellan", pd.DataFrame()), years)

    if nyad.empty:
        return _empty_fig("A/D Line — N/D")

    cumad = nyad["value"].cumsum()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.62, 0.38], vertical_spacing=0.04)

    fig.add_trace(go.Scatter(x=cumad.index, y=cumad.values,
        name="A/D Cumulativa", line=dict(color=C["accent"], width=2)), row=1, col=1)

    if not mc.empty:
        bar_colors = [C["green"] if v >= 0 else C["red"] for v in mc["value"]]
        fig.add_trace(go.Bar(x=mc.index, y=mc["value"],
            name="McClellan", marker_color=bar_colors, opacity=0.85), row=2, col=1)
        fig.add_hline(y=0, line=dict(color=C["muted"], width=1), row=2, col=1)

    fig.update_layout(**_LAYOUT_BASE,
        title=dict(text="NYSE A/D Line + McClellan Oscillator",
                   font=dict(size=12, color=C["text"]), x=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )
    fig.update_xaxes(gridcolor=C["border"])
    fig.update_yaxes(gridcolor=C["border"])
    return fig


# ── Chart 3: NYSE New Highs vs New Lows ──────────────────────────────────────

def chart_nh_nl(d, years=2):
    nh = _cutoff(d.get("nyhgh", pd.DataFrame()), years)
    nl = _cutoff(d.get("nylow", pd.DataFrame()), years)

    if nh.empty and nl.empty:
        return _empty_fig("NH/NL — N/D")

    fig = go.Figure()
    if not nh.empty:
        ma20_nh = nh["value"].rolling(20, min_periods=1).mean()
        fig.add_trace(go.Bar(x=nh.index, y=nh["value"],
            name="New Highs", marker_color=C["green"], opacity=0.45))
        fig.add_trace(go.Scatter(x=ma20_nh.index, y=ma20_nh.values,
            name="MA20 NH", line=dict(color=C["green"], width=2)))
    if not nl.empty:
        ma20_nl = nl["value"].rolling(20, min_periods=1).mean()
        fig.add_trace(go.Bar(x=nl.index, y=-nl["value"],
            name="New Lows", marker_color=C["red"], opacity=0.45))
        fig.add_trace(go.Scatter(x=ma20_nl.index, y=-ma20_nl.values,
            name="MA20 NL", line=dict(color=C["red"], width=2)))

    fig.add_hline(y=0, line=dict(color=C["muted"], width=1))
    fig.update_layout(**_LAYOUT_BASE,
        title=dict(text="NYSE New Highs vs New Lows",
                   font=dict(size=12, color=C["text"]), x=0),
        barmode="overlay",
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )
    return fig


# ── Chart 4: CBOE Put/Call Ratio ─────────────────────────────────────────────

def chart_put_call(d, years=2):
    pcr = _cutoff(d.get("pcr", pd.DataFrame()), years)

    if pcr.empty:
        return _empty_fig("Put/Call Ratio — N/D")

    ma10 = pcr["value"].rolling(10, min_periods=1).mean()
    ma21 = pcr["value"].rolling(21, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pcr.index, y=pcr["value"],
        name="P/C Daily", line=dict(color=C["muted"], width=1), opacity=0.35))
    fig.add_trace(go.Scatter(x=ma10.index, y=ma10.values,
        name="MA10", line=dict(color=C["accent"], width=2)))
    fig.add_trace(go.Scatter(x=ma21.index, y=ma21.values,
        name="MA21", line=dict(color=C["yellow"], width=1.5, dash="dash")))

    for y_val, col, lbl in [
        (1.2, C["green"],  "1.20 — Fear extrema (contrarian BUY)"),
        (1.0, C["yellow"], "1.00"),
        (0.8, C["yellow"], "0.80"),
        (0.6, C["red"],    "0.60 — Complacency (contrarian SELL)"),
    ]:
        fig.add_hline(y=y_val, line=dict(color=col, dash="dot", width=1),
                      annotation_text=lbl, annotation_font=dict(color=col, size=8),
                      annotation_position="right")

    fig.update_layout(**_LAYOUT_BASE,
        title=dict(text="CBOE Total Put/Call Ratio",
                   font=dict(size=12, color=C["text"]), x=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )
    return fig


# ── Chart 5: Buffett Indicator ───────────────────────────────────────────────

def chart_buffett(d, years=10):
    buf = _cutoff(d.get("buffett_df", pd.DataFrame()), years)
    sp  = _cutoff(d.get("sp500",     pd.DataFrame()), years)

    if buf.empty:
        return _empty_fig("Buffett Indicator — N/D")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=buf.index, y=buf["value"],
        name="Mkt Cap / GDP %", line=dict(color=C["accent"], width=2),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.06)"),
        secondary_y=False)
    if not sp.empty:
        fig.add_trace(go.Scatter(x=sp.index, y=sp["value"],
            name="S&P 500", line=dict(color=C["muted"], width=1, dash="dot"), opacity=0.5),
            secondary_y=True)

    for y_val, col, lbl in [
        (200, C["red"],    "200% Bolla"),
        (150, C["red"],    "150% Sopravvalutato"),
        (110, C["yellow"], "110% Fair value"),
        (80,  C["green"],  "80% Sottovalutato"),
    ]:
        fig.add_hline(y=y_val, line=dict(color=col, dash="dot", width=1),
                      annotation_text=lbl, annotation_font=dict(color=col, size=9),
                      secondary_y=False)

    fig.update_layout(**_LAYOUT_BASE,
        title=dict(text="Buffett Indicator — Wilshire 5000 / GDP",
                   font=dict(size=12, color=C["text"]), x=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        yaxis =dict(**_LAYOUT_BASE["yaxis"], title="Mkt Cap / GDP (%)"),
        yaxis2=dict(showgrid=False, title="S&P 500"),
    )
    return fig


# ── Chart 6: Hindenburg Omen Scorecard ───────────────────────────────────────

def chart_hindenburg_detail(d):
    score      = d.get("hind_score", 0)
    conditions = d.get("hind_conditions", [])

    score_col = (C["green"] if score <= 1 else
                 C["yellow"] if score == 2 else C["red"])

    fig = go.Figure()

    # Gauge
    fig.add_trace(go.Indicator(
        mode  = "gauge+number",
        value = score,
        number= dict(suffix=f"/4", font=dict(color=score_col, size=36, family="monospace")),
        gauge = dict(
            axis    = dict(range=[0, 4], tickwidth=1, tickcolor=C["muted"],
                           tickfont=dict(color=C["muted"])),
            bar     = dict(color=score_col),
            bgcolor = C["surface"],
            bordercolor = C["border"],
            steps   = [
                dict(range=[0, 1.5], color="rgba(0,230,118,0.12)"),
                dict(range=[1.5,2.5], color="rgba(255,214,0,0.12)"),
                dict(range=[2.5,4],   color="rgba(255,23,68,0.12)"),
            ],
            threshold=dict(line=dict(color=C["red"], width=3), thickness=0.75, value=3),
        ),
        title  = dict(text="Hindenburg Omen<br>Condizioni Attive",
                      font=dict(color=C["text"], size=11)),
        domain = dict(x=[0, 0.38], y=[0.05, 1]),
    ))

    # Conditions checklist
    for i, (cond_name, met, detail) in enumerate(conditions):
        col    = C["red"] if met else C["green"]
        symbol = "✗" if met else "✓"
        fig.add_annotation(
            x=0.44, y=0.88 - i * 0.23, xref="paper", yref="paper",
            xanchor="left", showarrow=False,
            text=f'<span style="color:{col};font-size:14px">{symbol}</span>'
                 f'<span style="font-size:11px"> {cond_name}</span>',
            font=dict(family="monospace", color=C["text"]),
        )
        fig.add_annotation(
            x=0.44, y=0.80 - i * 0.23, xref="paper", yref="paper",
            xanchor="left", showarrow=False,
            text=f'<span style="font-size:9px;color:{C["muted"]}">{detail}</span>',
            font=dict(family="monospace", color=C["muted"]),
        )

    fig.update_layout(**_LAYOUT_BASE, height=260)
    return fig


# ── Chart 7: SPX & NDX Price + 200MA ─────────────────────────────────────────

def chart_spx_price(d, years=2):
    sp  = _cutoff(d.get("sp500",  pd.DataFrame()), years)
    nq  = _cutoff(d.get("nasdaq", pd.DataFrame()), years)

    if sp.empty:
        return _empty_fig("SPX / NDX — N/D")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if not sp.empty:
        ma200_sp = sp["value"].rolling(200, min_periods=30).mean()
        fig.add_trace(go.Scatter(x=sp.index, y=sp["value"],
            name="S&P 500", line=dict(color=C["accent"], width=2)), secondary_y=False)
        fig.add_trace(go.Scatter(x=ma200_sp.index, y=ma200_sp.values,
            name="S&P 200MA", line=dict(color=C["accent"], width=1, dash="dot"),
            opacity=0.6), secondary_y=False)

    if not nq.empty:
        ma200_nq = nq["value"].rolling(200, min_periods=30).mean()
        fig.add_trace(go.Scatter(x=nq.index, y=nq["value"],
            name="Nasdaq", line=dict(color=C["yellow"], width=2)), secondary_y=True)
        fig.add_trace(go.Scatter(x=ma200_nq.index, y=ma200_nq.values,
            name="NDX 200MA", line=dict(color=C["yellow"], width=1, dash="dot"),
            opacity=0.6), secondary_y=True)

    fig.update_layout(**_LAYOUT_BASE,
        title=dict(text="S&P 500 & Nasdaq — Prezzo vs 200MA",
                   font=dict(size=12, color=C["text"]), x=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        yaxis =dict(**_LAYOUT_BASE["yaxis"], title="S&P 500"),
        yaxis2=dict(showgrid=False, title="Nasdaq"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  KPI ROW  (traffic lights)
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_val(name, val):
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return "N/D"
    unit = THRESHOLDS.get(name, {}).get("unit", "")
    if name == "hindenburg":
        return f"{int(val)}{unit}"
    if name in ("pct_above_200ma", "pct_above_50ma", "buffett", "spx_momentum"):
        return f"{val:.1f}{unit}"
    if name == "put_call":
        return f"{val:.2f}"
    if name == "nh_nl_ratio":
        return f"{val:.2f}{unit}"
    return f"{val:.1f}{unit}"


def build_kpi_row(d):
    cv     = d.get("current_values", {})
    colors = d.get("colors", {})

    cards = []
    for name, info in THRESHOLDS.items():
        val  = cv.get(name, float("nan"))
        col  = colors.get(name, "gray")
        dot  = _color_hex(col)
        fval = _fmt_val(name, val)
        desc = info.get(f"desc_{col}", "")

        cards.append(html.Div([
            html.Div(info["label"], className="kpi-title"),
            html.Div([
                html.Span("● ", style={"color": dot, "fontSize": "15px", "lineHeight": "1"}),
                html.Span(fval, style={
                    "color": C["text"], "fontSize": "20px",
                    "fontFamily": "monospace", "fontWeight": "700",
                }),
            ], style={"display": "flex", "alignItems": "center", "gap": "4px",
                      "marginTop": "4px"}),
            html.Div(desc, style={
                "fontSize": "9px", "color": C["muted"],
                "fontFamily": "monospace", "marginTop": "3px",
                "letterSpacing": "0.03em",
            }),
        ], className="kpi-card"))

    return html.Div(cards, className="kpi-row")


# ─────────────────────────────────────────────────────────────────────────────
#  ALERT PANEL
# ─────────────────────────────────────────────────────────────────────────────

_ARROW = {
    "green→red":    ("▼ DOWNGRADE", C["red"]),
    "green→yellow": ("▼ cautela",   C["yellow"]),
    "yellow→red":   ("▼ DOWNGRADE", C["red"]),
    "yellow→green": ("▲ upgrade",   C["green"]),
    "red→yellow":   ("▲ migliora",  C["yellow"]),
    "red→green":    ("▲ UPGRADE",   C["green"]),
}


def _build_alert_panel(d):
    alerts = d.get("alerts", pd.DataFrame())

    if alerts.empty:
        return html.Div("📭  Nessun alert registrato", style={
            "color": C["muted"], "fontFamily": "monospace", "fontSize": "11px",
            "padding": "12px 20px", "backgroundColor": C["surface"],
            "border": f"1px solid {C['border']}", "borderRadius": "4px",
            "marginBottom": "16px",
        })

    rows = []
    for _, row in alerts.head(30).iterrows():
        old_c = str(row.get("old_color", ""))
        new_c = str(row.get("new_color", ""))
        key   = f"{old_c}→{new_c}"
        arrow_txt, arrow_col = _ARROW.get(key, ("↔ cambio", C["muted"]))
        try:
            ts_str = pd.Timestamp(row["ts"]).strftime("%Y-%m-%d  %H:%M")
        except Exception:
            ts_str = "—"
        val_str = f"  ({row['value']:.2f})" if pd.notna(row.get("value")) else ""

        rows.append(html.Div([
            html.Span(ts_str + "  ", style={
                "color": C["muted"], "fontSize": "9px",
                "fontFamily": "monospace", "minWidth": "130px", "display": "inline-block",
            }),
            html.Span(arrow_txt + "  ", style={
                "color": arrow_col, "fontSize": "11px", "fontFamily": "monospace",
                "fontWeight": "700", "minWidth": "120px", "display": "inline-block",
            }),
            html.Span(str(row.get("label", row.get("indicator", "?"))), style={
                "color": C["text"], "fontSize": "11px", "fontFamily": "monospace",
            }),
            html.Span(f"  ● → ●  ", style={"fontSize": "10px"}),
            html.Span(val_str, style={"color": C["muted"], "fontSize": "9px", "fontFamily": "monospace"}),
        ], style={
            "padding":      "6px 14px",
            "borderBottom": f"1px solid {C['border']}",
            "display":      "flex",
            "alignItems":   "center",
        }))

    new_count = len(d.get("new_alerts", []))
    badge_txt = f"  {new_count} nuovi ↑" if new_count else ""

    return html.Div([
        html.Div([
            html.Span("🚨  ALERT LOG", style={
                "fontFamily": "monospace", "fontSize": "11px",
                "fontWeight": "700", "color": C["accent"], "letterSpacing": "0.08em",
            }),
            html.Span(badge_txt, style={
                "fontFamily": "monospace", "fontSize": "10px", "color": C["red"],
                "marginLeft": "10px",
            }),
        ], style={"padding": "8px 14px", "borderBottom": f"1px solid {C['border']}"}),
        html.Div(rows, style={"maxHeight": "180px", "overflowY": "auto"}),
    ], style={
        "backgroundColor": C["surface"],
        "border":          f"1px solid {C['border']}",
        "borderRadius":    "4px",
        "marginBottom":    "18px",
    })


# ─────────────────────────────────────────────────────────────────────────────
#  FULL LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

def build_app_layout(d):
    new_alerts = d.get("new_alerts", [])
    badge = f"  ·  {len(new_alerts)} nuovi alert" if new_alerts else ""

    return html.Div([

        # ── Sub-header ────────────────────────────────────────────────────────
        html.Div([
            html.Span("◆ ", style={"color": C["accent"]}),
            html.Span("MARKET INTERNALS", style={
                "fontFamily": "monospace", "fontSize": "13px",
                "fontWeight": "700", "color": C["accent"], "letterSpacing": "0.08em",
            }),
            html.Span("  ·  Breadth · Sentiment · Valuation", style={
                "fontFamily": "monospace", "fontSize": "10px",
                "color": C["muted"], "marginLeft": "8px",
            }),
            html.Span(badge, style={
                "fontFamily": "monospace", "fontSize": "10px",
                "color": C["red"], "marginLeft": "6px",
            }),
        ], style={
            "borderBottom": f"1px solid {C['border']}",
            "padding":      "8px 20px",
        }),

        html.Div([

            # KPI row
            build_kpi_row(d),

            # Controls
            html.Div([
                html.Span("Lookback:", style={
                    "fontFamily": "monospace", "fontSize": "11px", "color": C["muted"],
                }),
                dcc.RadioItems(
                    id="mi-lookback",
                    options=[{"label": k, "value": v}
                             for k, v in [("1A", 1), ("2A", 2), ("3A", 3), ("5A", 5)]],
                    value=2, inline=True,
                    style={"display": "inline-flex", "gap": "14px", "marginLeft": "12px"},
                    labelStyle={"fontFamily": "monospace", "fontSize": "11px",
                                "color": C["muted"], "cursor": "pointer"},
                    inputStyle={"accentColor": C["accent"]},
                ),
            ], className="controls-wrap"),

            # Alert panel
            _build_alert_panel(d),

            # Row 1: % above MA  +  A/D Line + McClellan
            html.Div([
                html.Div([dcc.Graph(id="g-mi-breadth-ma",
                          config={"displayModeBar": False})],
                          className="chart-card", style={"flex": 1}),
                html.Div([dcc.Graph(id="g-mi-ad-line",
                          config={"displayModeBar": False})],
                          className="chart-card", style={"flex": 1}),
            ], style={"display": "flex", "gap": "12px", "marginBottom": "12px"}),

            # Row 2: NH/NL  +  Put/Call
            html.Div([
                html.Div([dcc.Graph(id="g-mi-nh-nl",
                          config={"displayModeBar": False})],
                          className="chart-card", style={"flex": 1}),
                html.Div([dcc.Graph(id="g-mi-put-call",
                          config={"displayModeBar": False})],
                          className="chart-card", style={"flex": 1}),
            ], style={"display": "flex", "gap": "12px", "marginBottom": "12px"}),

            # Row 3: Buffett Indicator  +  Hindenburg Omen
            html.Div([
                html.Div([dcc.Graph(id="g-mi-buffett",
                          config={"displayModeBar": False})],
                          className="chart-card", style={"flex": 1}),
                html.Div([
                    html.Div("HINDENBURG OMEN — SCORECARD", style={
                        "fontFamily": "monospace", "fontSize": "10px",
                        "fontWeight": "700", "color": C["accent"],
                        "letterSpacing": "0.08em", "padding": "8px 12px 0",
                    }),
                    dcc.Graph(id="g-mi-hindenburg",
                              config={"displayModeBar": False}),
                ], className="chart-card", style={"flex": 1}),
            ], style={"display": "flex", "gap": "12px", "marginBottom": "12px"}),

            # Row 4: SPX + NDX full width
            html.Div([
                html.Div([dcc.Graph(id="g-mi-spx",
                          config={"displayModeBar": False})],
                          className="chart-card", style={"flex": 1}),
            ], style={"display": "flex", "gap": "12px", "marginBottom": "12px"}),

        ], className="page-wrap"),
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  STANDALONE APP
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", PORT))
    print(f"\n{'─'*60}")
    print("  FIN ANALYTICS  ·  Market Internals  ·  Avvio...")
    print(f"{'─'*60}\n")

    _d = load_data()

    _app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.CYBORG],
        title="Market Internals",
        suppress_callback_exceptions=True,
    )
    _app.layout = html.Div(
        [build_app_layout(_d)],
        style={"backgroundColor": C["bg"], "minHeight": "100vh"},
    )

    @_app.callback(
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
            chart_breadth_ma(_d, years),
            chart_ad_line(_d, years),
            chart_nh_nl(_d, years),
            chart_put_call(_d, years),
            chart_buffett(_d, years),
            chart_hindenburg_detail(_d),
            chart_spx_price(_d, years),
        )

    print(f"🚀  Market Internals  →  http://localhost:{PORT}\n")
    _app.run(debug=False, port=PORT, host="0.0.0.0")
