#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  FIN ANALYTICS  ·  Timmer Framework  ·  APA Quant              ║
║  Ispirato all'analisi settimanale di Jurrien Timmer (Fidelity)       ║
║                                                                      ║
║  Indicatori:                                                         ║
║    - Asset Class Leaderboard (YTD returns)                           ║
║    - Correlation Heatmap rolling 60d (SPX/Bonds/Gold/Commod/BTC)    ║
║    - Cap vs Equal-Weight ratio (SPX concentrazione)                  ║
║    - Forward P/E + Equity Risk Premium                               ║
║    - Earnings Growth trajectory                                      ║
║    - CAPE ratio + 10Y forward return model                           ║
║    - Capex% + Operating Margins trend                                ║
║    - Stock-Bond correlation regime (60/40 vs 60/20/20)              ║
║  Port: 8058 (standalone) / tab hub                                   ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import warnings
import datetime as dt

import numpy as np
import pandas as pd
import duckdb
import requests
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────

_LOCAL_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
DB_PATH = os.path.join(
    _LOCAL_CACHE if os.access(os.path.dirname(os.path.abspath(__file__)), os.W_OK)
    else "/tmp",
    "timmer_framework.duckdb",
)

PORT = int(os.environ.get("PORT", 8058))
TTL_HOURS = 6

C = {
    "bg":      "#080c18",
    "surface": "#0d1117",
    "border":  "#1a2236",
    "text":    "#e2e8f0",
    "muted":   "#556080",
    "accent":  "#00d4ff",
    "green":   "#00e676",
    "yellow":  "#ffea00",
    "red":     "#ff1744",
    "orange":  "#ff9100",
}

LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="monospace", size=11, color=C["text"]),
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
)
AXIS_BASE = dict(
    gridcolor=C["border"],
    zerolinecolor=C["border"],
    showgrid=True,
)

# ─────────────────────────────────────────────────────────────────────────────
#  THRESHOLDS  (traffic light)
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLDS = {
    "erp": {
        "label": "Equity Risk Premium",
        "unit":  "%",
        "green":  lambda v: v >= 3.0,
        "yellow": lambda v: 1.5 <= v < 3.0,
    },
    "fwd_pe": {
        "label": "Forward P/E",
        "unit":  "x",
        "green":  lambda v: v <= 20,
        "yellow": lambda v: 20 < v <= 25,
    },
    "eps_growth": {
        "label": "EPS Growth (YoY)",
        "unit":  "%",
        "green":  lambda v: v >= 10,
        "yellow": lambda v: 0 <= v < 10,
    },
    "sb_corr": {
        "label": "Stock-Bond 60d Corr",
        "unit":  "",
        "green":  lambda v: v < 0,
        "yellow": lambda v: 0 <= v < 0.3,
    },
    "breadth_200": {
        "label": "% S&P>200MA",
        "unit":  "%",
        "green":  lambda v: v >= 60,
        "yellow": lambda v: 40 <= v < 60,
    },
    "concentration": {
        "label": "Concentration (SPX/RSP trend)",
        "unit":  "",
        "green":  lambda v: v <= -0.02,      # ratio falling = broadening
        "yellow": lambda v: -0.02 < v < 0.02,
    },
}


def _color(key, value):
    """Returns 'green'/'yellow'/'red'/'gray'."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "gray"
    t = THRESHOLDS.get(key, {})
    if t.get("green", lambda v: False)(value):
        return "green"
    if t.get("yellow", lambda v: False)(value):
        return "yellow"
    return "red"


# ─────────────────────────────────────────────────────────────────────────────
#  DB HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_con():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = duckdb.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS cache_meta (
            key       VARCHAR PRIMARY KEY,
            fetched_at TIMESTAMP
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS indicator_states (
            key       VARCHAR PRIMARY KEY,
            color     VARCHAR,
            value     DOUBLE,
            updated_at TIMESTAMP
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS alert_history (
            id         INTEGER,
            ts         TIMESTAMP,
            key        VARCHAR,
            label      VARCHAR,
            old_color  VARCHAR,
            new_color  VARCHAR,
            new_value  DOUBLE
        )
    """)
    return con


def _is_fresh(con, key):
    row = con.execute(
        "SELECT fetched_at FROM cache_meta WHERE key=?", [key]
    ).fetchone()
    if not row:
        return False
    return (dt.datetime.utcnow() - row[0]).total_seconds() < TTL_HOURS * 3600


def _touch(con, key):
    con.execute(
        "INSERT OR REPLACE INTO cache_meta VALUES (?,?)",
        [key, dt.datetime.utcnow()],
    )


def _check_alert(con, key, new_color, new_value):
    label = THRESHOLDS.get(key, {}).get("label", key)
    row = con.execute(
        "SELECT color FROM indicator_states WHERE key=?", [key]
    ).fetchone()
    old_color = row[0] if row else None
    if old_color != new_color:
        next_id = (con.execute("SELECT COUNT(*) FROM alert_history").fetchone()[0] or 0) + 1
        con.execute(
            "INSERT INTO alert_history VALUES (?,?,?,?,?,?,?)",
            [next_id, dt.datetime.utcnow(), key, label, old_color, new_color, new_value],
        )
    con.execute(
        "INSERT OR REPLACE INTO indicator_states VALUES (?,?,?,?)",
        [key, new_color, new_value, dt.datetime.utcnow()],
    )


# ─────────────────────────────────────────────────────────────────────────────
#  DATA FETCHERS
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_yf(tickers, period="2y", interval="1d"):
    """Download adjusted close prices for a list of tickers."""
    raw = yf.download(tickers, period=period, interval=interval,
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        df = raw["Close"]
    else:
        df = raw[["Close"]] if "Close" in raw.columns else raw
        df.columns = tickers[:1]
    df = df.dropna(how="all")
    return df


def _fetch_fred(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(
        __import__("io").StringIO(r.text),
        parse_dates=["DATE"], index_col="DATE"
    )
    df = df.replace(".", np.nan).astype(float)
    df = df.dropna()
    return df


def _fetch_spy_pe():
    """Try to get SPY/SPX forward P/E and EPS growth from yfinance (timeout 20s)."""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
    def _get_info():
        return yf.Ticker("SPY").info
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_get_info)
            try:
                info = future.result(timeout=20)
            except (FuturesTimeout, Exception):
                print("  [warn] SPY.info timeout — skip P/E")
                return {}
        fwd_pe = info.get("forwardPE") or info.get("trailingPE")
        trailing_pe = info.get("trailingPE")
        eps_growth = info.get("earningsGrowth")  # annual forward
        if eps_growth is not None:
            eps_growth *= 100  # convert to %
        fwd_eps = info.get("forwardEps")
        trailing_eps = info.get("trailingEps")
        return {
            "fwd_pe": float(fwd_pe) if fwd_pe else None,
            "trailing_pe": float(trailing_pe) if trailing_pe else None,
            "eps_growth": float(eps_growth) if eps_growth else None,
            "fwd_eps": float(fwd_eps) if fwd_eps else None,
            "trailing_eps": float(trailing_eps) if trailing_eps else None,
        }
    except Exception:
        return {"fwd_pe": None, "trailing_pe": None, "eps_growth": None,
                "fwd_eps": None, "trailing_eps": None}


def _fetch_cape():
    """Download Shiller CAPE from online CSV (online.wsj, multpl, or Yale)."""
    # Try multpl.com table data via requests
    urls = [
        "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/shiller_ie.csv",
        "https://datahub.io/core/s-and-p-500/r/shiller_ie.csv",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                df = pd.read_csv(__import__("io").StringIO(r.text))
                # find date and CAPE columns
                date_col = next((c for c in df.columns if "date" in c.lower()), None)
                cape_col = next((c for c in df.columns
                                 if any(k in c.lower() for k in ["cape", "pe10", "shiller"])), None)
                if date_col and cape_col:
                    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                    df = df.dropna(subset=[date_col, cape_col])
                    df = df.set_index(date_col).sort_index()
                    return df[[cape_col]].rename(columns={cape_col: "CAPE"}).astype(float)
        except Exception:
            continue
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD DATA  (entry point called by hub.py and __main__)
# ─────────────────────────────────────────────────────────────────────────────

def _df_from_duckdb(con, table):
    """Legge un DataFrame da DuckDB ripristinando l'indice temporale."""
    df = con.execute(f"SELECT * FROM {table}").df()
    date_col = next((c for c in df.columns if c.lower() in ("index", "date", "datetime")), None)
    if date_col:
        df.index = pd.to_datetime(df.pop(date_col))
    else:
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    return df.sort_index()


def load_data():
    con = _get_con()
    d = {}

    # ── 1. Asset Class Prices (2 year history) ───────────────────────────────
    # ^SPXA200R e ^SPXA50R sono delisted — rimossi per non rallentare il download
    tickers_ac = ["SPY", "RSP", "GLD", "TLT", "DJP", "BTC-USD", "EFA", "EEM", "^VIX"]
    key_ac = "asset_class_prices"
    if not _is_fresh(con, key_ac):
        try:
            prices = _fetch_yf(tickers_ac, period="2y")
            con.execute("DROP TABLE IF EXISTS asset_class_prices")
            con.execute(
                "CREATE TABLE asset_class_prices AS SELECT * FROM prices"
            )
            _touch(con, key_ac)
            print(f"  ✓ Asset class prices fetched ({len(prices)} rows)")
        except Exception as e:
            print(f"  ✗ Asset class prices: {e}")

    try:
        prices_df = con.execute("SELECT * FROM asset_class_prices").df()
        # DuckDB può salvare l'indice come "index", "Date" o "Datetime"
        _date_col = next((c for c in prices_df.columns
                          if c.lower() in ("index", "date", "datetime")), None)
        if _date_col:
            prices_df.index = pd.to_datetime(prices_df.pop(_date_col))
        else:
            prices_df.index = pd.to_datetime(prices_df.index)
        prices_df = prices_df.sort_index()
        d["prices"] = prices_df
    except Exception:
        d["prices"] = pd.DataFrame()

    # ── 2. 10Y Treasury yield (FRED) ────────────────────────────────────────
    key_dgs10 = "fred_dgs10"
    if not _is_fresh(con, key_dgs10):
        try:
            dgs10 = _fetch_fred("DGS10")
            con.execute("DROP TABLE IF EXISTS fred_dgs10")
            con.execute("CREATE TABLE fred_dgs10 AS SELECT * FROM dgs10")
            _touch(con, key_dgs10)
            print("  ✓ DGS10 fetched")
        except Exception as e:
            print(f"  ✗ DGS10: {e}")

    try:
        dgs10_df = con.execute("SELECT * FROM fred_dgs10").df()
        dgs10_df.index = pd.to_datetime(dgs10_df["DATE"] if "DATE" in dgs10_df.columns
                                        else dgs10_df.index)
        if "DATE" in dgs10_df.columns:
            dgs10_df = dgs10_df.drop(columns=["DATE"])
        dgs10_df = dgs10_df.sort_index()
        d["dgs10"] = dgs10_df
    except Exception:
        d["dgs10"] = pd.DataFrame()

    # ── 3. IG & HY Credit Spreads (FRED) ────────────────────────────────────
    for sid, name in [("BAMLC0A0CM", "ig_spread"), ("BAMLH0A0HYM2EY", "hy_spread")]:
        key = f"fred_{name}"
        if not _is_fresh(con, key):
            try:
                df = _fetch_fred(sid)
                con.execute(f"DROP TABLE IF EXISTS {key}")
                con.execute(f"CREATE TABLE {key} AS SELECT * FROM df")
                _touch(con, key)
                print(f"  ✓ {sid} fetched")
            except Exception as e:
                print(f"  ✗ {sid}: {e}")
        try:
            df = con.execute(f"SELECT * FROM {key}").df()
            df.index = pd.to_datetime(df["DATE"] if "DATE" in df.columns else df.index)
            if "DATE" in df.columns:
                df = df.drop(columns=["DATE"])
            d[name] = df.sort_index()
        except Exception:
            d[name] = pd.DataFrame()

    # ── 4. Corporate Profits & Margins (FRED) ───────────────────────────────
    for sid, name in [("CP", "corp_profits"), ("NFCPATAX", "corp_profits_after_tax")]:
        key = f"fred_{name}"
        if not _is_fresh(con, key):
            try:
                df = _fetch_fred(sid)
                con.execute(f"DROP TABLE IF EXISTS {key}")
                con.execute(f"CREATE TABLE {key} AS SELECT * FROM df")
                _touch(con, key)
                print(f"  ✓ {sid} fetched")
            except Exception as e:
                print(f"  ✗ {sid}: {e}")
        try:
            df = con.execute(f"SELECT * FROM {key}").df()
            df.index = pd.to_datetime(df["DATE"] if "DATE" in df.columns else df.index)
            if "DATE" in df.columns:
                df = df.drop(columns=["DATE"])
            d[name] = df.sort_index()
        except Exception:
            d[name] = pd.DataFrame()

    # ── 5. S&P 500 Operating Margins (FRED: NFCPITAXFLD for proxy) ──────────
    # Use FRED series A463RC1Q027SBEA (corporate profits as % of GDP) as margins proxy
    key_marg = "fred_margins"
    if not _is_fresh(con, key_marg):
        try:
            df = _fetch_fred("A463RC1Q027SBEA")
            con.execute("DROP TABLE IF EXISTS fred_margins")
            con.execute("CREATE TABLE fred_margins AS SELECT * FROM df")
            _touch(con, key_marg)
            print("  ✓ Margins proxy fetched")
        except Exception as e:
            print(f"  ✗ Margins: {e}")
    try:
        df = con.execute("SELECT * FROM fred_margins").df()
        df.index = pd.to_datetime(df["DATE"] if "DATE" in df.columns else df.index)
        if "DATE" in df.columns:
            df = df.drop(columns=["DATE"])
        d["margins"] = df.sort_index()
    except Exception:
        d["margins"] = pd.DataFrame()

    # ── 6. SPY P/E, EPS Growth (live from yfinance) ─────────────────────────
    key_pe = "spy_pe"
    if not _is_fresh(con, key_pe):
        pe_data = _fetch_spy_pe()
        for k, v in pe_data.items():
            con.execute(f"DROP TABLE IF EXISTS pe_{k}")
            if v is not None:
                df_tmp = pd.DataFrame({"value": [v]})
                con.execute(f"CREATE TABLE pe_{k} AS SELECT * FROM df_tmp")
        _touch(con, key_pe)
        print(f"  ✓ SPY P/E data fetched: fwd_pe={pe_data.get('fwd_pe')}")

    pe_vals = {}
    for k in ["fwd_pe", "trailing_pe", "eps_growth", "fwd_eps", "trailing_eps"]:
        try:
            row = con.execute(f"SELECT value FROM pe_{k}").fetchone()
            pe_vals[k] = float(row[0]) if row else None
        except Exception:
            pe_vals[k] = None
    d["pe"] = pe_vals

    # ── 7. CAPE (Shiller PE) ─────────────────────────────────────────────────
    key_cape = "shiller_cape"
    if not _is_fresh(con, key_cape):
        try:
            cape_df = _fetch_cape()
            if not cape_df.empty:
                con.execute("DROP TABLE IF EXISTS shiller_cape")
                con.execute("CREATE TABLE shiller_cape AS SELECT * FROM cape_df")
                _touch(con, key_cape)
                print(f"  ✓ CAPE fetched ({len(cape_df)} rows)")
            else:
                print("  ✗ CAPE: empty data")
        except Exception as e:
            print(f"  ✗ CAPE: {e}")
    try:
        df = con.execute("SELECT * FROM shiller_cape").df()
        df.index = pd.to_datetime(df.get("Date", df.index))
        if "Date" in df.columns:
            df = df.drop(columns=["Date"])
        d["cape"] = df.sort_index()
    except Exception:
        d["cape"] = pd.DataFrame()

    # ── 8. Derived indicators ────────────────────────────────────────────────
    prices = d.get("prices", pd.DataFrame())
    dgs10 = d.get("dgs10", pd.DataFrame())

    # ERP = 1/fwd_pe - 10Y_yield/100
    fwd_pe = d["pe"].get("fwd_pe")
    ten_y = float(dgs10.iloc[-1, 0]) if not dgs10.empty else None
    if fwd_pe and fwd_pe > 0 and ten_y:
        erp = (1 / fwd_pe * 100) - ten_y
    else:
        erp = None
    d["erp"] = erp

    # Stock-Bond correlation (rolling 60d)
    if not prices.empty and "SPY" in prices.columns and "TLT" in prices.columns:
        rets = prices[["SPY", "TLT"]].pct_change().dropna()
        sb_corr = rets["SPY"].rolling(60).corr(rets["TLT"])
        d["sb_corr_series"] = sb_corr.dropna()
        d["sb_corr_current"] = float(sb_corr.dropna().iloc[-1]) if not sb_corr.dropna().empty else None
    else:
        d["sb_corr_series"] = pd.Series(dtype=float)
        d["sb_corr_current"] = None

    # SPX/RSP concentration ratio
    if not prices.empty and "SPY" in prices.columns and "RSP" in prices.columns:
        ratio = (prices["SPY"] / prices["RSP"]).dropna()
        if ratio.empty:
            ratio_norm = ratio
        else:
            ratio_norm = ratio / ratio.iloc[0]  # normalize to 1.0
        d["concentration_series"] = ratio_norm
        # trend: 20d change in normalized ratio
        if len(ratio_norm) >= 20:
            conc_trend = float(ratio_norm.iloc[-1]) - float(ratio_norm.iloc[-20])
        else:
            conc_trend = 0.0
        d["concentration_trend"] = conc_trend
    else:
        d["concentration_series"] = pd.Series(dtype=float)
        d["concentration_trend"] = None

    # Breadth
    breadth_200 = None
    if not prices.empty and "^SPXA200R" in prices.columns:
        s = prices["^SPXA200R"].dropna()
        breadth_200 = float(s.iloc[-1]) if not s.empty else None
    d["breadth_200"] = breadth_200

    breadth_50 = None
    if not prices.empty and "^SPXA50R" in prices.columns:
        s = prices["^SPXA50R"].dropna()
        breadth_50 = float(s.iloc[-1]) if not s.empty else None
    d["breadth_50"] = breadth_50

    # ── 9. Traffic lights & alerts ────────────────────────────────────────────
    checks = {
        "erp":          d.get("erp"),
        "fwd_pe":       d["pe"].get("fwd_pe"),
        "eps_growth":   d["pe"].get("eps_growth"),
        "sb_corr":      d.get("sb_corr_current"),
        "breadth_200":  d.get("breadth_200"),
        "concentration": d.get("concentration_trend"),
    }
    d["colors"] = {}
    for k, v in checks.items():
        c = _color(k, v)
        d["colors"][k] = c
        _check_alert(con, k, c, v if v is not None else float("nan"))

    # Alert history
    try:
        d["alerts"] = con.execute(
            "SELECT * FROM alert_history ORDER BY ts DESC LIMIT 30"
        ).df()
    except Exception:
        d["alerts"] = pd.DataFrame()

    con.close()
    return d


# ─────────────────────────────────────────────────────────────────────────────
#  CHART FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _fig_empty(title="No data"):
    fig = go.Figure()
    fig.update_layout(**LAYOUT_BASE, title=dict(text=title, font=dict(color=C["muted"])))
    return fig


def chart_leaderboard(d, years=1):
    """YTD / rolling returns for main asset classes."""
    prices = d.get("prices", pd.DataFrame())
    if prices.empty:
        return _fig_empty("Asset Class Leaderboard — no data")

    tickers = {
        "SPX cap-w (SPY)":     "SPY",
        "SPX equal-w (RSP)":   "RSP",
        "Intl Equities (EFA)": "EFA",
        "EM Equities (EEM)":   "EEM",
        "Gold (GLD)":          "GLD",
        "Bonds (TLT)":         "TLT",
        "Commodities (DJP)":   "DJP",
        "Bitcoin":             "BTC-USD",
    }
    # YTD: from Jan 1 of current year
    today = pd.Timestamp.today()
    ytd_start = pd.Timestamp(today.year, 1, 1)

    rows = []
    for label, tkr in tickers.items():
        if tkr not in prices.columns:
            continue
        s = prices[tkr].dropna()
        # YTD
        s_ytd = s[s.index >= ytd_start]
        if len(s_ytd) >= 2:
            ytd_ret = (s_ytd.iloc[-1] / s_ytd.iloc[0] - 1) * 100
        else:
            ytd_ret = None
        # Rolling N years
        cutoff = today - pd.DateOffset(years=years)
        s_roll = s[s.index >= cutoff]
        roll_ret = (s_roll.iloc[-1] / s_roll.iloc[0] - 1) * 100 if len(s_roll) >= 2 else None
        rows.append({"Asset": label, "YTD %": ytd_ret, f"{years}Y %": roll_ret})

    df = pd.DataFrame(rows).dropna(subset=["YTD %"])
    df = df.sort_values("YTD %")

    colors = [C["green"] if v >= 0 else C["red"] for v in df["YTD %"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["Asset"], x=df["YTD %"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in df["YTD %"]],
        textposition="outside",
        name="YTD",
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title="Asset Class Leaderboard (YTD Returns)",
        xaxis={**AXIS_BASE, "title": "Return %", "zeroline": True,
               "zerolinecolor": C["muted"], "zerolinewidth": 1},
        yaxis={**AXIS_BASE, "showgrid": False},
        height=340,
        showlegend=False,
    )
    return fig


def chart_correlation_heatmap(d, window=60):
    """Rolling correlation matrix: SPY, TLT, GLD, DJP, BTC-USD."""
    prices = d.get("prices", pd.DataFrame())
    tickers = ["SPY", "TLT", "GLD", "DJP", "BTC-USD"]
    labels  = ["S&P500", "Bonds", "Gold", "Commodities", "Bitcoin"]

    avail = [t for t in tickers if t in prices.columns]
    if len(avail) < 2:
        return _fig_empty("Correlation Matrix — insufficient data")

    rets = prices[avail].pct_change().dropna()
    # Use last `window` observations for the correlation
    corr = rets.tail(window).corr()

    label_map = dict(zip(tickers, labels))
    corr.index   = [label_map.get(t, t) for t in corr.index]
    corr.columns = [label_map.get(t, t) for t in corr.columns]

    # Round for display
    z = corr.values.round(2)
    text = [[f"{v:.2f}" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=list(corr.columns),
        y=list(corr.index),
        text=text,
        texttemplate="%{text}",
        colorscale=[
            [0.0,  "#ff1744"],
            [0.35, "#0d1117"],
            [0.5,  "#1a2236"],
            [0.65, "#0d1117"],
            [1.0,  "#00e676"],
        ],
        zmin=-1, zmax=1,
        showscale=True,
        colorbar=dict(thickness=12, len=0.8,
                      tickfont=dict(size=9, color=C["text"])),
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=f"Cross-Asset Correlation Matrix (rolling {window}d)",
        height=340,
        xaxis=dict(side="bottom"),
    )
    return fig


def chart_cap_vs_equalweight(d, years=2):
    """SPY vs RSP normalized price + ratio trend."""
    prices = d.get("prices", pd.DataFrame())
    if prices.empty or "SPY" not in prices.columns or "RSP" not in prices.columns:
        return _fig_empty("Cap vs Equal-Weight — no data")

    cutoff = pd.Timestamp.today() - pd.DateOffset(years=years)
    spy = prices["SPY"].dropna()
    rsp = prices["RSP"].dropna()
    spy = spy[spy.index >= cutoff]
    rsp = rsp[rsp.index >= cutoff]

    if spy.empty or rsp.empty:
        raise ValueError("Cap vs Equal-Weight — no SPY/RSP data")
    # Normalize both to 100
    spy_n = spy / spy.iloc[0] * 100
    rsp_n = rsp / rsp.iloc[0] * 100

    ratio = spy_n / rsp_n  # rising = more concentrated in mega-caps

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.04)

    fig.add_trace(go.Scatter(x=spy_n.index, y=spy_n.values, name="SPX cap-weighted",
                             line=dict(color=C["accent"], width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=rsp_n.index, y=rsp_n.values, name="SPX equal-weighted",
                             line=dict(color=C["orange"], width=1.5)), row=1, col=1)

    # Ratio: rising = concentrated market, falling = broadening
    ratio_color = [C["red"] if r > 1.02 else C["green"] if r < 0.98 else C["yellow"]
                   for r in ratio.values]
    fig.add_trace(go.Scatter(x=ratio.index, y=ratio.values, name="Ratio (cap/equal)",
                             line=dict(color=C["muted"], width=1),
                             fill="tozeroy",
                             fillcolor="rgba(0,212,255,0.05)"), row=2, col=1)
    fig.add_hline(y=1.0, line_dash="dot", line_color=C["muted"], row=2, col=1)

    fig.update_layout(
        **LAYOUT_BASE,
        title="Market Concentration: Cap-Weighted vs Equal-Weighted",
        height=380,
        legend=dict(orientation="h", y=1.05),
    )
    for row in [1, 2]:
        fig.update_xaxes(**AXIS_BASE, row=row, col=1)
        fig.update_yaxes(**AXIS_BASE, row=row, col=1)

    return fig


def chart_erp_and_pe(d, years=2):
    """Equity Risk Premium + Forward P/E over time (using trailing P/E from prices + DGS10)."""
    prices = d.get("prices", pd.DataFrame())
    dgs10  = d.get("dgs10", pd.DataFrame())
    pe_data = d.get("pe", {})

    # Compute ERP timeline: SPY earnings yield (rough: 1/fwd_pe_proxy) - 10Y yield
    # We approximate rolling P/E by: use current P/E anchored to current price,
    # and reconstruct historical P/E as price_ratio × current_pe
    if prices.empty or dgs10.empty or not pe_data.get("trailing_pe"):
        return _fig_empty("Equity Risk Premium — insufficient data")

    cutoff = pd.Timestamp.today() - pd.DateOffset(years=years)

    # Use SPY price relative to current as P/E proxy
    spy = prices["SPY"].dropna() if "SPY" in prices.columns else pd.Series()
    spy = spy[spy.index >= cutoff]

    current_pe = pe_data.get("trailing_pe") or pe_data.get("fwd_pe")
    if not current_pe:
        return _fig_empty("Equity Risk Premium — no P/E data")
    if spy.empty:
        return _fig_empty("Equity Risk Premium — no SPY data")

    # Historical P/E ≈ current_pe × (current_price / historical_price)
    pe_series = current_pe * (spy.iloc[-1] / spy)
    earnings_yield = 100 / pe_series  # percent

    # Align DGS10 with spy dates
    ten_y = dgs10.copy()
    ten_y.index = pd.to_datetime(ten_y.index)
    ten_y = ten_y[ten_y.index >= cutoff].iloc[:, 0]
    ten_y = ten_y.reindex(spy.index, method="ffill").dropna()

    common = spy.index.intersection(ten_y.index)
    erp = earnings_yield.reindex(common) - ten_y.reindex(common)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.04)

    # P/E
    fig.add_trace(go.Scatter(x=pe_series.index, y=pe_series.values, name="Trailing P/E",
                             line=dict(color=C["accent"], width=1.5)), row=1, col=1)
    fig.add_hline(y=20, line_dash="dot", line_color=C["yellow"],
                  annotation_text="20x", annotation_font_color=C["yellow"], row=1, col=1)
    fig.add_hline(y=25, line_dash="dot", line_color=C["red"],
                  annotation_text="25x", annotation_font_color=C["red"], row=1, col=1)

    # ERP with color fill
    erp_colors = [C["green"] if v >= 3 else C["yellow"] if v >= 1.5 else C["red"]
                  for v in erp.values]
    current_erp = float(erp.iloc[-1]) if not erp.empty else None
    erp_fill = C["green"] if (current_erp or 0) >= 3 else C["yellow"] if (current_erp or 0) >= 1.5 else C["red"]

    fig.add_trace(go.Scatter(x=erp.index, y=erp.values, name="ERP",
                             line=dict(color=erp_fill, width=2),
                             fill="tozeroy",
                             fillcolor=f"rgba{tuple(int(erp_fill.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.15,)}"),
                  row=2, col=1)
    fig.add_hline(y=3, line_dash="dot", line_color=C["green"],
                  annotation_text="3% (healthy)", annotation_font_color=C["green"], row=2, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color=C["red"], row=2, col=1)

    erp_str = f"{current_erp:.1f}%" if current_erp else "N/A"
    fig.update_layout(
        **LAYOUT_BASE,
        title=f"Valuation: P/E & Equity Risk Premium  (ERP={erp_str})",
        height=380,
    )
    for row in [1, 2]:
        fig.update_xaxes(**AXIS_BASE, row=row, col=1)
        fig.update_yaxes(**AXIS_BASE, row=row, col=1)

    return fig


def chart_breadth_and_concentration(d, years=2):
    """% above 50/200 MA + rolling breadth context."""
    prices = d.get("prices", pd.DataFrame())
    if prices.empty:
        return _fig_empty("Breadth — no data")

    cutoff = pd.Timestamp.today() - pd.DateOffset(years=years)

    fig = go.Figure()

    for col, label, color in [("^SPXA200R", "% above 200MA", C["accent"]),
                               ("^SPXA50R",  "% above 50MA",  C["orange"])]:
        if col in prices.columns:
            s = prices[col].dropna()
            s = s[s.index >= cutoff]
            fig.add_trace(go.Scatter(x=s.index, y=s.values, name=label,
                                     line=dict(color=color, width=1.5)))

    fig.add_hline(y=60, line_dash="dot", line_color=C["green"],
                  annotation_text="Healthy (60%)", annotation_font_color=C["green"])
    fig.add_hline(y=40, line_dash="dot", line_color=C["red"],
                  annotation_text="Oversold (40%)", annotation_font_color=C["red"])

    b200 = d.get("breadth_200")
    b50  = d.get("breadth_50")
    subtitle = ""
    if b200:
        subtitle += f"  200MA: {b200:.0f}%"
    if b50:
        subtitle += f"  |  50MA: {b50:.0f}%"

    fig.update_layout(
        **LAYOUT_BASE,
        title=f"Market Breadth (% stocks above MA){subtitle}",
        xaxis=dict(**AXIS_BASE),
        yaxis=dict(**AXIS_BASE, title="%", range=[0, 100]),
        height=300,
    )
    return fig


def chart_stock_bond_corr(d, years=2):
    """Rolling 60d stock-bond correlation — regime indicator for 60/40 vs 60/20/20."""
    sb_series = d.get("sb_corr_series", pd.Series())
    if sb_series.empty:
        return _fig_empty("Stock-Bond Correlation — no data")

    cutoff = pd.Timestamp.today() - pd.DateOffset(years=years)
    sb = sb_series[sb_series.index >= cutoff]

    current = float(sb.iloc[-1]) if not sb.empty else None

    # Regime background
    regime = "60/40 WORKS" if (current or 0) < 0 else "60/40 BROKEN → ADD ALTS"
    regime_color = C["green"] if (current or 0) < 0 else C["red"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sb.index, y=sb.values, name="Stock-Bond Corr (60d)",
                             line=dict(color=C["accent"], width=1.5),
                             fill="tozeroy",
                             fillcolor="rgba(0,212,255,0.08)"))
    fig.add_hline(y=0, line_dash="solid", line_color=C["muted"])
    fig.add_hline(y=0.3, line_dash="dot", line_color=C["red"],
                  annotation_text="Danger zone", annotation_font_color=C["red"])

    corr_str = f"{current:.2f}" if current else "N/A"
    fig.update_layout(
        **LAYOUT_BASE,
        title=f"Stock-Bond Correlation (60d rolling)  [{corr_str}]  →  {regime}",
        xaxis=dict(**AXIS_BASE),
        yaxis=dict(**AXIS_BASE, range=[-1, 1]),
        height=280,
    )
    return fig


def chart_cape(d):
    """CAPE ratio historical + 10-year forward return estimate."""
    cape_df = d.get("cape", pd.DataFrame())
    if cape_df.empty:
        # Fallback: show a message with current known CAPE if we have P/E
        return _fig_empty("CAPE (Shiller P/E) — data unavailable")

    cape = cape_df["CAPE"].dropna()
    # Historical forward 10Y return regression: ~r = 1/CAPE approx
    # Timmer uses the model: high CAPE → lower 10Y CAGR
    implied_ret = (1 / cape * 100).clip(0, 20)  # rough

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.04)

    fig.add_trace(go.Scatter(x=cape.index, y=cape.values, name="CAPE",
                             line=dict(color=C["accent"], width=1.5)), row=1, col=1)
    # Long-term average
    avg = cape.mean()
    fig.add_hline(y=avg, line_dash="dot", line_color=C["muted"],
                  annotation_text=f"LT avg {avg:.1f}x",
                  annotation_font_color=C["muted"], row=1, col=1)

    fig.add_trace(go.Scatter(x=implied_ret.index, y=implied_ret.values,
                             name="Implied 10Y Return (1/CAPE)",
                             line=dict(color=C["green"], width=1.5)), row=2, col=1)

    current_cape = float(cape.iloc[-1]) if not cape.empty else None
    cape_str = f"{current_cape:.1f}x" if current_cape else "N/A"

    fig.update_layout(
        **LAYOUT_BASE,
        title=f"CAPE Shiller P/E  (current: {cape_str})  —  10Y Forward Return Model",
        height=380,
    )
    for row in [1, 2]:
        fig.update_xaxes(**AXIS_BASE, row=row, col=1)
        fig.update_yaxes(**AXIS_BASE, row=row, col=1)

    return fig


def chart_margins_and_spreads(d, years=5):
    """Corporate profit margins (FRED proxy) + IG/HY credit spreads."""
    margins = d.get("margins", pd.DataFrame())
    ig = d.get("ig_spread", pd.DataFrame())
    hy = d.get("hy_spread", pd.DataFrame())

    cutoff = pd.Timestamp.today() - pd.DateOffset(years=years)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=False,
                        subplot_titles=["Corporate Profit Margins (% of GDP)",
                                        "Credit Spreads (IG & HY)"],
                        vertical_spacing=0.12)

    if not margins.empty:
        m = margins.iloc[:, 0].dropna()
        m = m[m.index >= cutoff]
        fig.add_trace(go.Scatter(x=m.index, y=m.values, name="Corp Margins (FRED)",
                                 line=dict(color=C["accent"], width=1.5),
                                 fill="tozeroy",
                                 fillcolor="rgba(0,212,255,0.06)"), row=1, col=1)

    if not ig.empty:
        s = ig.iloc[:, 0].dropna()
        s = s[s.index >= cutoff]
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name="IG Spread",
                                 line=dict(color=C["green"], width=1.5)), row=2, col=1)
    if not hy.empty:
        s = hy.iloc[:, 0].dropna()
        s = s[s.index >= cutoff]
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name="HY Spread",
                                 line=dict(color=C["orange"], width=1.5)), row=2, col=1)

    fig.update_layout(**LAYOUT_BASE, height=380)
    for row in [1, 2]:
        fig.update_xaxes(**AXIS_BASE, row=row, col=1)
        fig.update_yaxes(**AXIS_BASE, row=row, col=1)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  KPI ROW
# ─────────────────────────────────────────────────────────────────────────────

def build_kpi_row(d):
    colors = d.get("colors", {})
    pe = d.get("pe", {})
    erp = d.get("erp")
    sb  = d.get("sb_corr_current")
    b200 = d.get("breadth_200")

    def _dot(key):
        c = colors.get(key, "gray")
        return html.Span("● ", style={"color": C.get(c, C["muted"]), "fontSize": "14px"})

    def _kpi(key, label, value_str):
        return html.Div([
            _dot(key),
            html.Span(label + "  ", style={"color": C["muted"], "fontSize": "10px",
                                           "fontFamily": "monospace"}),
            html.Span(value_str, style={"color": C["text"], "fontSize": "13px",
                                        "fontFamily": "monospace", "fontWeight": "700"}),
        ], style={"display": "inline-block", "marginRight": "24px"})

    erp_str   = f"{erp:.1f}%" if erp else "N/A"
    pe_str    = f"{pe.get('fwd_pe') or pe.get('trailing_pe') or 'N/A':.1f}x" \
                if (pe.get("fwd_pe") or pe.get("trailing_pe")) else "N/A"
    eps_str   = f"{pe.get('eps_growth'):.0f}%" if pe.get("eps_growth") else "N/A"
    sb_str    = f"{sb:.2f}" if sb is not None else "N/A"
    b200_str  = f"{b200:.0f}%" if b200 else "N/A"

    return html.Div([
        _kpi("erp",         "ERP",           erp_str),
        _kpi("fwd_pe",      "Fwd P/E",       pe_str),
        _kpi("eps_growth",  "EPS Growth",    eps_str),
        _kpi("sb_corr",     "Stock/Bond ρ",  sb_str),
        _kpi("breadth_200", "% > 200MA",     b200_str),
    ], style={"padding": "8px 24px", "backgroundColor": C["surface"],
              "borderBottom": f"1px solid {C['border']}"})


# ─────────────────────────────────────────────────────────────────────────────
#  LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

def _graph(gid, height=None):
    style = {"height": f"{height}px"} if height else {}
    return dcc.Graph(id=gid, config={"displayModeBar": False}, style=style)


def _section(title, children):
    return html.Div([
        html.Div(title, style={
            "color": C["muted"], "fontFamily": "monospace",
            "fontSize": "10px", "letterSpacing": "0.1em",
            "padding": "12px 24px 4px",
            "borderBottom": f"1px solid {C['border']}",
        }),
        *children,
    ])


def build_app_layout(d):
    return html.Div([

        build_kpi_row(d),

        # ── Controls ────────────────────────────────────────────────────────
        html.Div([
            html.Span("LOOKBACK  ", style={"color": C["muted"], "fontSize": "10px",
                                           "fontFamily": "monospace"}),
            dcc.RadioItems(
                id="tmr-lookback",
                options=[{"label": f" {y}Y ", "value": y} for y in [1, 2, 5, 10]],
                value=2,
                inline=True,
                style={"color": C["text"], "fontFamily": "monospace", "fontSize": "11px"},
                inputStyle={"marginRight": "4px", "accentColor": C["accent"]},
                labelStyle={"marginRight": "16px"},
            ),
            html.Span("  CORR WINDOW  ", style={"color": C["muted"], "fontSize": "10px",
                                                "fontFamily": "monospace"}),
            dcc.RadioItems(
                id="tmr-corr-window",
                options=[{"label": f" {w}d ", "value": w} for w in [20, 60, 126, 252]],
                value=60,
                inline=True,
                style={"color": C["text"], "fontFamily": "monospace", "fontSize": "11px"},
                inputStyle={"marginRight": "4px", "accentColor": C["accent"]},
                labelStyle={"marginRight": "16px"},
            ),
        ], style={"padding": "8px 24px", "backgroundColor": C["bg"],
                  "borderBottom": f"1px solid {C['border']}"}),

        # ── Row 1: Leaderboard + Correlation Heatmap ─────────────────────
        _section("CROSS-ASSET OVERVIEW", [
            html.Div([
                html.Div(_graph("g-tmr-leaderboard"), style={"flex": "1", "minWidth": "340px"}),
                html.Div(_graph("g-tmr-corr-heatmap"), style={"flex": "1", "minWidth": "340px"}),
            ], style={"display": "flex", "gap": "2px", "padding": "8px 0"}),
        ]),

        # ── Row 2: Cap vs Equal-Weight + Stock-Bond Correlation ──────────
        _section("MARKET STRUCTURE & REGIME", [
            html.Div([
                html.Div(_graph("g-tmr-cap-equal"), style={"flex": "2", "minWidth": "400px"}),
                html.Div(_graph("g-tmr-sb-corr"),  style={"flex": "1", "minWidth": "300px"}),
            ], style={"display": "flex", "gap": "2px", "padding": "8px 0"}),
        ]),

        # ── Row 3: Breadth ──────────────────────────────────────────────
        _section("BREADTH", [
            html.Div(_graph("g-tmr-breadth"), style={"padding": "8px 0"}),
        ]),

        # ── Row 4: Valuation — ERP & P/E ────────────────────────────────
        _section("VALUATION — EQUITY RISK PREMIUM", [
            html.Div([
                html.Div(_graph("g-tmr-erp"), style={"flex": "1", "minWidth": "340px"}),
                html.Div(_graph("g-tmr-cape"), style={"flex": "1", "minWidth": "340px"}),
            ], style={"display": "flex", "gap": "2px", "padding": "8px 0"}),
        ]),

        # ── Row 5: Margins & Spreads ─────────────────────────────────────
        _section("CORPORATE HEALTH & CREDIT", [
            html.Div(_graph("g-tmr-margins"), style={"padding": "8px 0"}),
        ]),

        # ── Alert Log ───────────────────────────────────────────────────
        _section("ALERT LOG", [
            html.Div(id="tmr-alert-log",
                     style={"padding": "8px 24px", "fontFamily": "monospace",
                            "fontSize": "10px", "color": C["muted"]}),
        ]),

    ], style={"backgroundColor": C["bg"]})


# ─────────────────────────────────────────────────────────────────────────────
#  STANDALONE APP (only when run directly)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n📐 Timmer Framework  ·  Avvio...")
    _d = load_data()

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.CYBORG],
        title="Timmer Framework",
        suppress_callback_exceptions=True,
    )

    app.layout = html.Div([
        html.Div([
            html.Span("◆ ", style={"color": C["accent"]}),
            html.Span("TIMMER FRAMEWORK", style={
                "fontFamily": "monospace", "fontWeight": "700",
                "color": C["accent"], "fontSize": "14px",
            }),
        ], style={"backgroundColor": C["surface"],
                  "borderBottom": f"1px solid {C['border']}",
                  "padding": "10px 24px"}),
        build_app_layout(_d),
    ], style={"backgroundColor": C["bg"], "minHeight": "100vh"})

    @app.callback(
        [
            dash.Output("g-tmr-leaderboard",  "figure"),
            dash.Output("g-tmr-corr-heatmap", "figure"),
            dash.Output("g-tmr-cap-equal",    "figure"),
            dash.Output("g-tmr-sb-corr",      "figure"),
            dash.Output("g-tmr-breadth",      "figure"),
            dash.Output("g-tmr-erp",          "figure"),
            dash.Output("g-tmr-cape",         "figure"),
            dash.Output("g-tmr-margins",      "figure"),
            dash.Output("tmr-alert-log",      "children"),
        ],
        [
            dash.Input("tmr-lookback",     "value"),
            dash.Input("tmr-corr-window",  "value"),
        ],
    )
    def _update(years, window):
        return _tmr_charts(_d, years, window)

    print(f"🚀  http://localhost:{PORT}\n")
    app.run(debug=False, port=PORT, host="0.0.0.0")


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED CALLBACK LOGIC (used by hub.py)
# ─────────────────────────────────────────────────────────────────────────────

def _build_alert_log(d):
    alerts = d.get("alerts", pd.DataFrame())
    if alerts.empty:
        return html.Span("Nessun alert registrato.", style={"color": C["muted"]})
    rows = []
    for _, row in alerts.head(10).iterrows():
        old_c = str(row.get("old_color", "—"))
        new_c = str(row.get("new_color", ""))
        ts    = str(row.get("ts", ""))[:16]
        label = str(row.get("label", ""))
        val   = row.get("new_value", "")
        val_s = f"{val:.2f}" if isinstance(val, float) else str(val)
        dot   = html.Span("● ", style={"color": C.get(new_c, C["muted"])})
        rows.append(html.Div([
            dot,
            html.Span(f"{ts}  {label}  {old_c} → {new_c}  ({val_s})",
                      style={"color": C["text"]}),
        ]))
    return html.Div(rows)


def _safe_chart(fn, *args):
    try:
        return fn(*args)
    except Exception as e:
        print(f"  [chart error] {fn.__name__}: {e}")
        return _fig_empty(f"{fn.__name__} — {type(e).__name__}")


def _tmr_charts(d, years, window):
    return (
        _safe_chart(chart_leaderboard, d, years),
        _safe_chart(chart_correlation_heatmap, d, window),
        _safe_chart(chart_cap_vs_equalweight, d, years),
        _safe_chart(chart_stock_bond_corr, d, years),
        _safe_chart(chart_breadth_and_concentration, d, years),
        _safe_chart(chart_erp_and_pe, d, years),
        _safe_chart(chart_cape, d),
        _safe_chart(chart_margins_and_spreads, d, years),
        _build_alert_log(d),
    )
