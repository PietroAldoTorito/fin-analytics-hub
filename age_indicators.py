#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  FIN ANALYTICS  ·  AGE Indicators  ·  APA Quant                ║
║  Ispirato a AGE Italia / Gaetano Evangelista                        ║
║                                                                      ║
║  TRIN (Arms Index)   ·  Stagionalità S&P 500                        ║
║  CoT Index (CFTC)    ·  Fear & Greed Composito                      ║
║  Alert system con semaforo per ogni indicatore                      ║
║  Port: 8057                                                          ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os, sys, warnings, zipfile
from datetime import datetime
from io import BytesIO, StringIO

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

PORT        = 8057
CACHE_TTL_H = 6

_LOCAL_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
DB_PATH      = os.path.join(
    _LOCAL_CACHE if os.access(os.path.dirname(os.path.abspath(__file__)), os.W_OK)
    else "/tmp",
    "age_indicators.duckdb",
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
    "orange":  "#ff6d00",
    "gray":    "#37474f",
    "purple":  "#e040fb",
}

_LAYOUT_BASE = dict(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(0,0,0,0)",
    font          = dict(color=C["text"], family="monospace", size=11),
    margin        = dict(l=52, r=16, t=36, b=36),
    xaxis         = dict(gridcolor=C["border"], zerolinecolor=C["border"]),
    yaxis         = dict(gridcolor=C["border"], zerolinecolor=C["border"]),
)

# ─────────────────────────────────────────────────────────────────────────────
#  TRAFFIC LIGHT THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLDS = {
    "trin": {
        "label":      "TRIN MA10",
        "unit":       "",
        "green":      lambda v: 0.70 <= v <= 1.10,
        "yellow":     lambda v: (0.50 <= v < 0.70) or (1.10 < v <= 1.50),
        "desc_green": "Mercato equilibrato",
        "desc_yellow":"Pressione in corso",
        "desc_red":   "Estremo — segnale contrarian",
        "desc_gray":  "N/D",
    },
    "seasonality": {
        "label":      "Stagionalità 20d",
        "unit":       "%",
        "green":      lambda v: v >= 1.5,
        "yellow":     lambda v: 0.0 <= v < 1.5,
        "desc_green": "Finestra stagionale favorevole",
        "desc_yellow":"Stagionalità neutra",
        "desc_red":   "Finestra stagionale sfavorevole",
        "desc_gray":  "N/D",
    },
    "cot": {
        "label":      "CoT Index",
        "unit":       "/100",
        "green":      lambda v: v <= 35,
        "yellow":     lambda v: 35 < v <= 65,
        "desc_green": "Specs corti — segnale contrarian rialzista",
        "desc_yellow":"Positioning neutro",
        "desc_red":   "Specs molto lunghi — trade affollato",
        "desc_gray":  "N/D",
    },
    "fear_greed": {
        "label":      "Fear & Greed",
        "unit":       "/100",
        "green":      lambda v: 30 <= v <= 70,
        "yellow":     lambda v: (15 <= v < 30) or (70 < v <= 85),
        "desc_green": "Sentiment bilanciato",
        "desc_yellow":"Avvicinarsi agli estremi",
        "desc_red":   "Estremo — rischio di inversione",
        "desc_gray":  "N/D",
    },
}


def _color(name, value):
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
    url = "https://www.cboe.com/publish/scheduledtask/mktdata/datahouse/totalpc.csv"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data_lines = []
        for line in resp.text.strip().split("\n"):
            parts = line.split(",")
            if len(parts) < 4:
                continue
            try:
                pd.to_datetime(parts[0].strip())
                float(parts[3].strip())
                data_lines.append(line)
            except Exception:
                pass
        if not data_lines:
            return pd.DataFrame()
        df = pd.read_csv(StringIO("\n".join(data_lines)),
                         header=None, names=["date", "call", "put", "total"])
        df["date"]  = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["total"], errors="coerce")
        return df.dropna(subset=["date", "value"]).set_index("date")[["value"]]
    except Exception as e:
        print(f"  [warn] CBOE P/C: {e}")
        return pd.DataFrame()


def _fetch_cot_sp500():
    """
    Download CFTC COT legacy-format financial futures data.
    Returns DataFrame with columns: nc_net (large specs), comm_net, oi
    indexed by date (weekly, Tuesdays).
    """
    current_year = datetime.now().year
    dfs = []

    for year in range(current_year - 5, current_year + 1):
        url = f"https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code != 200:
                continue
            with zipfile.ZipFile(BytesIO(resp.content)) as z:
                txt_files = [f for f in z.namelist() if f.lower().endswith(".txt")]
                if not txt_files:
                    continue
                with z.open(txt_files[0]) as f:
                    dfs.append(pd.read_csv(f, low_memory=False))
            print(f"    CoT {year}: {len(dfs[-1])} rows")
        except Exception as e:
            print(f"    [warn] CoT {year}: {e}")

    # Append current-week file
    try:
        resp = requests.get(
            "https://www.cftc.gov/dea/newcot/FinFutWk.txt", timeout=30
        )
        if resp.status_code == 200:
            dfs.append(pd.read_csv(StringIO(resp.text), low_memory=False))
    except Exception as e:
        print(f"    [warn] CoT current week: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    # ── flexible column lookup ────────────────────────────────────────────────
    col_map = {c.strip().lower(): c.strip() for c in combined.columns}

    def _find(patterns):
        for p in patterns:
            if p.lower() in col_map:
                return col_map[p.lower()]
        # partial match
        for p in patterns:
            for k, v in col_map.items():
                if p.lower().replace("_", "") in k.replace("_", "").replace(" ", ""):
                    return v
        return None

    date_col = _find(["As_of_Date_in_Form_YYYY-MM-DD",
                       "Report_Date_as_YYYY-MM-DD", "report date"])
    mkt_col  = _find(["Market_and_Exchange_Names",
                       "Market and Exchange Names"])
    nc_long  = _find(["Noncomm_Positions_Long_All",  "NonComm_Positions_Long_All"])
    nc_short = _find(["Noncomm_Positions_Short_All", "NonComm_Positions_Short_All"])
    c_long   = _find(["Comm_Positions_Long_All"])
    c_short  = _find(["Comm_Positions_Short_All"])
    oi_col   = _find(["Open_Interest_All"])

    if date_col is None or mkt_col is None:
        print("  [warn] CoT: colonne data/mercato non trovate")
        return pd.DataFrame()

    # Filter S&P 500 futures
    mask = combined[mkt_col].astype(str).str.contains(
        r"E-MINI S&P 500|S&P 500 STOCK INDEX", case=False, na=False
    )
    sp = combined[mask].copy()
    if sp.empty:
        print("  [warn] CoT: nessun record S&P 500")
        return pd.DataFrame()

    sp["date"] = pd.to_datetime(sp[date_col], errors="coerce")
    sp = (sp.dropna(subset=["date"])
            .sort_values("date")
            .drop_duplicates(subset=["date"])
            .set_index("date"))
    sp.index = pd.DatetimeIndex(sp.index)

    result = pd.DataFrame(index=sp.index)
    if nc_long and nc_short:
        result["nc_net"] = (pd.to_numeric(sp[nc_long],  errors="coerce") -
                            pd.to_numeric(sp[nc_short], errors="coerce"))
    if c_long and c_short:
        result["comm_net"] = (pd.to_numeric(sp[c_long],  errors="coerce") -
                              pd.to_numeric(sp[c_short], errors="coerce"))
    if oi_col:
        result["oi"] = pd.to_numeric(sp[oi_col], errors="coerce")

    return result.dropna(how="all")


# ─────────────────────────────────────────────────────────────────────────────
#  DERIVED COMPUTATIONS
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_pctrank(series, window):
    """Percentile rank of the latest value within a rolling window (0–100)."""
    return series.rolling(window, min_periods=max(10, window // 4)).apply(
        lambda x: float((x[-1] >= x[:-1]).sum()) / max(len(x) - 1, 1) * 100,
        raw=True,
    )


def _compute_cot_index(cot_df, lookback=52):
    """
    Stochastic CoT Index for non-commercial (large speculator) net.
    Score 0–100: 0 = specs most short ever, 100 = specs most long ever.
    Contrarian: low score (specs short) → bullish, high (specs long) → bearish.
    """
    if cot_df.empty or "nc_net" not in cot_df.columns:
        return pd.DataFrame()
    net = cot_df["nc_net"].dropna()

    def _stoch(x):
        mn, mx = x.min(), x.max()
        return 50.0 if mx == mn else float((x[-1] - mn) / (mx - mn) * 100)

    idx = net.rolling(lookback, min_periods=lookback // 2).apply(_stoch, raw=True)
    return idx.rename("value").to_frame()


def _compute_seasonality(sp500_df):
    """
    Compute seasonal statistics from full S&P 500 price history.
    Returns dict with monthly, day-of-week, cumulative path, forecast.
    """
    if sp500_df.empty:
        return {}
    sp = sp500_df["value"].dropna()
    if len(sp) < 252:
        return {}

    ret = sp.pct_change().dropna() * 100
    df  = ret.to_frame("ret")
    df.index = pd.DatetimeIndex(df.index)
    df["month"] = df.index.month
    df["dow"]   = df.index.dayofweek   # 0=Mon
    df["year"]  = df.index.year

    # Monthly stats
    monthly = df.groupby("month")["ret"].agg(
        avg_ret="mean",
        pct_pos=lambda x: (x > 0).mean() * 100,
    ).round(3)

    # Day-of-week stats
    dow_stats = df.groupby("dow")["ret"].agg(
        avg_ret="mean",
        pct_pos=lambda x: (x > 0).mean() * 100,
    ).round(3)

    # Trading-day-of-year seasonal path (average "year" profile)
    df["tdy"] = df.groupby("year").cumcount() + 1
    daily_stats = df.groupby("tdy")["ret"].agg(
        avg_ret="mean",
        pct_pos=lambda x: (x > 0).mean() * 100,
    ).round(4)
    cumulative = daily_stats["avg_ret"].cumsum()

    # Current position in the year
    today_year = pd.Timestamp.now().year
    current_td = len(df[df["year"] == today_year]) + 1
    current_td = min(current_td, 252)

    # Seasonal forecast: sum of expected daily returns for next 20 trading days
    fwd_days  = list(range(current_td, min(current_td + 21, 253)))
    forecast  = float(daily_stats.loc[daily_stats.index.isin(fwd_days), "avg_ret"].sum())

    # Average monthly returns (full period)
    monthly_df = (df.groupby(["year", "month"])["ret"]
                    .sum().reset_index()
                    .rename(columns={"ret": "monthly_ret"}))

    return {
        "monthly":     monthly,
        "dow_stats":   dow_stats,
        "daily_stats": daily_stats,
        "cumulative":  cumulative,
        "current_td":  current_td,
        "forecast_20d": forecast,
        "monthly_ts":  monthly_df,
    }


def _compute_fear_greed(vix_df, pcr_df, spxa200_df, sp500_df, trin_df, hy_df, lookback=252):
    """
    Fear & Greed composite (0 = Extreme Fear, 100 = Extreme Greed).
    Returns (score, components_dict, history_series).
    """
    MONTH_NAMES = {1:"Gen",2:"Feb",3:"Mar",4:"Apr",5:"Mag",6:"Giu",
                   7:"Lug",8:"Ago",9:"Set",10:"Ott",11:"Nov",12:"Dic"}

    components = {}
    hist_parts = {}

    # ── VIX (inverse rank — high VIX = fear) ─────────────────────────────────
    if not vix_df.empty:
        vix = vix_df["value"].dropna()
        rank = _rolling_pctrank(vix, lookback)
        score = 100 - float(rank.dropna().iloc[-1]) if not rank.dropna().empty else np.nan
        components["VIX"] = {"label": "Volatilità VIX", "score": score, "raw": float(vix.iloc[-1])}
        hist_parts["vix"] = (100 - rank).rename("vix")

    # ── Put/Call MA5 (inverse rank) ───────────────────────────────────────────
    if not pcr_df.empty:
        pcr5 = pcr_df["value"].rolling(5, min_periods=3).mean().dropna()
        rank  = _rolling_pctrank(pcr5, lookback)
        score = 100 - float(rank.dropna().iloc[-1]) if not rank.dropna().empty else np.nan
        raw   = float(pcr5.iloc[-1]) if not pcr5.empty else np.nan
        components["P/C Ratio"] = {"label": "Put/Call Ratio", "score": score, "raw": raw}
        hist_parts["pcr"] = (100 - rank).rename("pcr")

    # ── % S&P above 200MA (direct) ────────────────────────────────────────────
    if not spxa200_df.empty:
        sp200 = spxa200_df["value"].dropna()
        score = float(sp200.iloc[-1])
        components["%>200MA"] = {"label": "% S&P >200MA", "score": score, "raw": score}
        hist_parts["sp200"] = sp200.rename("sp200")

    # ── SPX vs MA125 (normalized deviation) ──────────────────────────────────
    if not sp500_df.empty:
        spx = sp500_df["value"].dropna()
        if len(spx) >= 125:
            ma125  = spx.rolling(125, min_periods=60).mean()
            dev    = (spx / ma125 - 1) * 100
            # normalize: -15% → 0, 0 → 50, +15% → 100
            norm   = ((dev + 15) / 30 * 100).clip(0, 100)
            score  = float(norm.dropna().iloc[-1])
            components["Momentum"] = {"label": "SPX vs MA125", "score": score, "raw": float(dev.dropna().iloc[-1])}
            hist_parts["mom"] = norm.rename("mom")

    # ── TRIN MA5 (inverse rank — high TRIN = fear) ───────────────────────────
    if not trin_df.empty:
        trin5 = trin_df["value"].rolling(5, min_periods=3).mean().dropna()
        if len(trin5) >= 20:
            rank  = _rolling_pctrank(trin5, min(lookback, len(trin5) - 1))
            score = 100 - float(rank.dropna().iloc[-1]) if not rank.dropna().empty else np.nan
            raw   = float(trin5.iloc[-1])
            components["TRIN"] = {"label": "TRIN Arms Index", "score": score, "raw": raw}
            hist_parts["trin"] = (100 - rank).rename("trin")

    # ── HY Spread (inverse rank — wide spread = fear) ────────────────────────
    if not hy_df.empty:
        hy   = hy_df["value"].dropna()
        rank = _rolling_pctrank(hy, lookback)
        score = 100 - float(rank.dropna().iloc[-1]) if not rank.dropna().empty else np.nan
        components["HY Spread"] = {"label": "HY Spread (OAS)", "score": score, "raw": float(hy.iloc[-1])}
        hist_parts["hy"] = (100 - rank).rename("hy")

    # ── Composite ─────────────────────────────────────────────────────────────
    valid_scores = [v["score"] for v in components.values()
                    if not np.isnan(v.get("score", np.nan))]
    composite = float(np.mean(valid_scores)) if valid_scores else np.nan

    # ── Historical composite (daily) ──────────────────────────────────────────
    history = pd.DataFrame()
    if hist_parts:
        hist_df  = pd.concat(hist_parts.values(), axis=1).dropna(how="all")
        history  = hist_df.mean(axis=1).rename("value").to_frame()

    return composite, components, history


def _last_val(df, col="value"):
    if df is None or df.empty:
        return np.nan
    try:
        return float(df[col].dropna().iloc[-1])
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = duckdb.connect(DB_PATH)

    con.execute("""
        CREATE TABLE IF NOT EXISTS cache_meta (key VARCHAR PRIMARY KEY, updated_at TIMESTAMP)
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS indicator_states (
            indicator VARCHAR PRIMARY KEY, color VARCHAR, value DOUBLE, updated_at TIMESTAMP)
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS alert_history (
            ts TIMESTAMP, indicator VARCHAR, label VARCHAR,
            old_color VARCHAR, new_color VARCHAR, value DOUBLE)
    """)

    def _is_fresh(key):
        row = con.execute("SELECT updated_at FROM cache_meta WHERE key=?", [key]).fetchone()
        if not row:
            return False
        return (datetime.utcnow() - row[0]).total_seconds() / 3600 < CACHE_TTL_H

    def _save_meta(key):
        con.execute("INSERT OR REPLACE INTO cache_meta VALUES (?,?)", [key, datetime.utcnow()])

    def _cache_df(key, df):
        tbl = f"ds_{key}"
        con.execute(f"DROP TABLE IF EXISTS {tbl}")
        if not df.empty:
            con.execute(f"CREATE TABLE {tbl} AS SELECT * FROM df")

    def _load_cached(key):
        try:
            df = con.execute(f"SELECT * FROM ds_{key}").df()
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
        print(f"  → {key} …")
        df = fetch_fn()
        df_save = df.reset_index() if (not df.empty and df.index.name) else df
        _cache_df(key, df_save)
        _save_meta(key)
        return df

    result = {}

    # ── Raw fetches ───────────────────────────────────────────────────────────
    # TRIN — NYSE Arms Index
    result["trin"]    = _get("age_trin",    lambda: _fetch_yf("^TRIN"))
    result["trinq"]   = _get("age_trinq",   lambda: _fetch_yf("^TRINQ"))

    # S&P 500 full history for seasonality (max)
    result["sp500_full"] = _get("age_sp500_full", lambda: _fetch_yf("^GSPC", period="max"))

    # Components for Fear & Greed
    result["vix"]     = _get("age_vix",     lambda: _fetch_yf("^VIX"))
    result["pcr"]     = _get("age_pcr",     lambda: _fetch_cboe_pc())
    result["spxa200"] = _get("age_spxa200", lambda: _fetch_yf("^SPXA200R"))
    result["sp500"]   = _get("age_sp500",   lambda: _fetch_yf("^GSPC"))
    result["hy"]      = _get("age_hy",      lambda: _fetch_fred("BAMLH0A0HYM2EY"))

    # CoT — CFTC
    result["cot_raw"] = _get("age_cot_raw", _fetch_cot_sp500)

    # ── Derived ───────────────────────────────────────────────────────────────
    result["cot_index"] = _compute_cot_index(result["cot_raw"])

    seasonal = _compute_seasonality(result["sp500_full"])
    result["seasonal"] = seasonal

    fg_score, fg_components, fg_history = _compute_fear_greed(
        result["vix"], result["pcr"], result["spxa200"],
        result["sp500"], result["trin"], result["hy"],
    )
    result["fg_score"]      = fg_score
    result["fg_components"] = fg_components
    result["fg_history"]    = fg_history

    # ── TRIN MA10 current value ───────────────────────────────────────────────
    trin_ma10 = np.nan
    if not result["trin"].empty:
        s = result["trin"]["value"].rolling(10, min_periods=5).mean().dropna()
        if not s.empty:
            trin_ma10 = float(s.iloc[-1])

    current_values = {
        "trin":        trin_ma10,
        "seasonality": seasonal.get("forecast_20d", np.nan),
        "cot":         _last_val(result["cot_index"]),
        "fear_greed":  fg_score,
    }
    result["current_values"] = current_values

    colors = {k: _color(k, v) for k, v in current_values.items()}
    result["colors"] = colors

    # ── Alert system ──────────────────────────────────────────────────────────
    new_alerts = []
    now = datetime.utcnow()
    for ind, new_col in colors.items():
        if new_col == "gray":
            continue
        row     = con.execute("SELECT color FROM indicator_states WHERE indicator=?", [ind]).fetchone()
        old_col = row[0] if row else None
        con.execute("INSERT OR REPLACE INTO indicator_states VALUES (?,?,?,?)",
                    [ind, new_col, current_values.get(ind), now])
        if old_col is not None and old_col != new_col:
            lbl = THRESHOLDS.get(ind, {}).get("label", ind)
            con.execute("INSERT INTO alert_history VALUES (?,?,?,?,?,?)",
                        [now, ind, lbl, old_col, new_col, current_values.get(ind)])
            new_alerts.append({"ts": now, "indicator": ind, "label": lbl,
                                "old_color": old_col, "new_color": new_col,
                                "value": current_values.get(ind)})
            print(f"  🚨 ALERT: {lbl}  {old_col.upper()} → {new_col.upper()}")

    try:
        alerts_df = con.execute("SELECT * FROM alert_history ORDER BY ts DESC LIMIT 100").df()
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


# ── Chart 1: TRIN / Arms Index ───────────────────────────────────────────────

def chart_trin(d, years=2):
    trin  = _cutoff(d.get("trin",  pd.DataFrame()), years)
    trinq = _cutoff(d.get("trinq", pd.DataFrame()), years)

    if trin.empty:
        return _empty_fig("TRIN — N/D (ticker ^TRIN non disponibile)")

    ma5  = trin["value"].rolling(5,  min_periods=3).mean()
    ma10 = trin["value"].rolling(10, min_periods=5).mean()

    # Color bars: green below 1, red above 1
    bar_colors = [C["green"] if v <= 1.0 else C["red"]
                  for v in trin["value"].fillna(1.0)]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.70, 0.30], vertical_spacing=0.04)

    fig.add_trace(go.Bar(
        x=trin.index, y=trin["value"].values,
        name="TRIN (NYSE)", marker_color=bar_colors, opacity=0.55,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=ma5.index, y=ma5.values, name="MA5",
        line=dict(color=C["yellow"], width=1.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=ma10.index, y=ma10.values, name="MA10",
        line=dict(color=C["accent"], width=2),
    ), row=1, col=1)

    if not trinq.empty:
        fig.add_trace(go.Scatter(
            x=trinq.index, y=trinq["value"].values,
            name="TRINQ (Nasdaq)", line=dict(color=C["purple"], width=1.5, dash="dash"),
            opacity=0.8,
        ), row=2, col=1)
        trinq_ma10 = trinq["value"].rolling(10, min_periods=5).mean()
        fig.add_trace(go.Scatter(
            x=trinq_ma10.index, y=trinq_ma10.values,
            name="TRINQ MA10", line=dict(color=C["purple"], width=1),
        ), row=2, col=1)
        fig.add_hline(y=1.0, line=dict(color=C["muted"], width=1), row=2, col=1)

    for y_val, col, lbl in [
        (2.0, C["green"],  "2.0 — Panic sell (contrarian BUY)"),
        (1.5, C["red"],    "1.5 — Forte pressione"),
        (1.0, C["muted"],  "1.0 — Neutro"),
        (0.5, C["orange"], "0.5 — Complacency"),
    ]:
        fig.add_hline(y=y_val, line=dict(color=col, dash="dot", width=1),
                      annotation_text=lbl,
                      annotation_font=dict(color=col, size=9),
                      row=1, col=1)

    fig.update_layout(**_LAYOUT_BASE,
        title=dict(text="TRIN / Arms Index — NYSE & Nasdaq",
                   font=dict(size=12, color=C["text"]), x=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )
    fig.update_xaxes(gridcolor=C["border"])
    fig.update_yaxes(gridcolor=C["border"])
    return fig


# ── Chart 2 + 3: Stagionalità S&P 500 ───────────────────────────────────────

def chart_seasonality_monthly(d):
    seasonal = d.get("seasonal", {})
    monthly  = seasonal.get("monthly", pd.DataFrame())

    if monthly.empty:
        return _empty_fig("Stagionalità mensile — N/D")

    MONTHS = ["Gen","Feb","Mar","Apr","Mag","Giu","Lug","Ago","Set","Ott","Nov","Dic"]
    months_idx = monthly.index.tolist()
    labels  = [MONTHS[m - 1] for m in months_idx]
    avg_ret = monthly["avg_ret"].tolist()
    pct_pos = monthly["pct_pos"].tolist()

    bar_colors = [C["green"] if v >= 0 else C["red"] for v in avg_ret]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=labels, y=avg_ret,
        name="Rendimento medio %", marker_color=bar_colors, opacity=0.8,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=labels, y=pct_pos,
        name="% anni positivi", mode="lines+markers",
        line=dict(color=C["accent"], width=2),
        marker=dict(size=6, color=C["accent"]),
    ), secondary_y=True)

    fig.add_hline(y=0, line=dict(color=C["muted"], width=1), secondary_y=False)
    fig.add_hline(y=50, line=dict(color=C["muted"], dash="dot", width=1), secondary_y=True)

    # Highlight current month — usa indice numerico (stringa non supportata su asse categorico)
    current_month = pd.Timestamp.now().month
    cm_idx = current_month - 1
    if cm_idx < len(labels):
        try:
            fig.add_vrect(x0=cm_idx - 0.4, x1=cm_idx + 0.4,
                          fillcolor=C["yellow"], opacity=0.08,
                          line_width=0, annotation_text="Oggi",
                          annotation_font=dict(color=C["yellow"], size=9),
                          annotation_position="top left")
        except Exception:
            pass

    fig.update_layout(**_LAYOUT_BASE,
        title=dict(text="Stagionalità Mensile — S&P 500 (storico completo)",
                   font=dict(size=12, color=C["text"]), x=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        yaxis =dict(**_LAYOUT_BASE["yaxis"], title="Rend. medio (%)"),
        yaxis2=dict(showgrid=False, title="% anni positivi", range=[0, 100]),
        bargap=0.25,
    )
    return fig


def chart_seasonality_annual(d):
    """Annual seasonal profile (average year cumulative path) + next 20-day window."""
    seasonal   = d.get("seasonal", {})
    cumulative = seasonal.get("cumulative", pd.Series())
    current_td = seasonal.get("current_td", 0)

    if cumulative.empty:
        return _empty_fig("Profilo stagionale annuale — N/D")

    fig = go.Figure()

    # Full cumulative seasonal path
    fig.add_trace(go.Scatter(
        x=cumulative.index, y=cumulative.values,
        name="Percorso stagionale medio",
        line=dict(color=C["accent"], width=2),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.05)",
    ))

    # Highlight current position
    if 0 < current_td <= len(cumulative):
        fig.add_vline(
            x=int(current_td),
            line=dict(color=C["yellow"], width=2, dash="dash"),
            annotation_text=f"Oggi (GG {current_td})",
            annotation_font=dict(color=C["yellow"], size=10),
        )
        # Shade next 20 days
        x_start = current_td
        x_end   = min(current_td + 20, len(cumulative))
        fig.add_vrect(
            x0=x_start, x1=x_end,
            fillcolor="rgba(255,214,0,0.08)",
            line_width=0,
        )

    # Seasonal forecast annotation
    forecast = seasonal.get("forecast_20d", np.nan)
    if not np.isnan(forecast):
        col = C["green"] if forecast > 0 else C["red"]
        fig.add_annotation(
            x=0.98, y=0.95, xref="paper", yref="paper", xanchor="right",
            text=f"Forecast 20 sessioni: <b>{forecast:+.2f}%</b>",
            font=dict(color=col, size=12, family="monospace"),
            showarrow=False,
            bgcolor=C["surface"],
            bordercolor=col,
            borderwidth=1,
        )

    fig.add_hline(y=0, line=dict(color=C["muted"], width=1))
    fig.update_layout(**_LAYOUT_BASE,
        title=dict(text="Profilo Stagionale Annuale — S&P 500 (anno medio cumulato)",
                   font=dict(size=12, color=C["text"]), x=0),
        xaxis=dict(**_LAYOUT_BASE["xaxis"], title="Giorno di trading dell'anno"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )
    return fig


# ── Chart 4: CoT Index ───────────────────────────────────────────────────────

def chart_cot(d, years=4):
    cot_raw   = _cutoff(d.get("cot_raw",   pd.DataFrame()), years)
    cot_index = _cutoff(d.get("cot_index", pd.DataFrame()), years)

    if cot_raw.empty and cot_index.empty:
        return _empty_fig("CoT Index — N/D (dati CFTC non disponibili)")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.05)

    # Row 1: raw net positions
    if not cot_raw.empty:
        if "nc_net" in cot_raw.columns:
            nc = cot_raw["nc_net"].dropna()
            nc_colors = [C["red"] if v > 0 else C["green"] for v in nc]
            fig.add_trace(go.Bar(
                x=nc.index, y=nc.values, name="Large Specs Net",
                marker_color=nc_colors, opacity=0.75,
            ), row=1, col=1)
        if "comm_net" in cot_raw.columns:
            cm = cot_raw["comm_net"].dropna()
            fig.add_trace(go.Scatter(
                x=cm.index, y=cm.values, name="Commercials Net",
                line=dict(color=C["accent"], width=2),
            ), row=1, col=1)
        fig.add_hline(y=0, line=dict(color=C["muted"], width=1), row=1, col=1)

    # Row 2: CoT Index (0-100 stochastic)
    if not cot_index.empty:
        ci = cot_index["value"].dropna()
        ci_colors = [
            C["green"] if v <= 35 else (C["red"] if v >= 65 else C["yellow"])
            for v in ci
        ]
        fig.add_trace(go.Scatter(
            x=ci.index, y=ci.values, name="CoT Index",
            line=dict(color=C["accent"], width=2),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.06)",
        ), row=2, col=1)
        for y_val, col, lbl in [
            (80, C["red"],   "80 — Specs molto lunghi (bearish)"),
            (65, C["red"],   "65"),
            (35, C["green"], "35"),
            (20, C["green"], "20 — Specs molto corti (bullish)"),
        ]:
            fig.add_hline(y=y_val, line=dict(color=col, dash="dot", width=1),
                          annotation_text=lbl, annotation_font=dict(color=col, size=9),
                          row=2, col=1)

    fig.update_layout(**_LAYOUT_BASE,
        title=dict(text="CoT Index — E-Mini S&P 500 (CFTC)",
                   font=dict(size=12, color=C["text"]), x=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )
    fig.update_xaxes(gridcolor=C["border"])
    fig.update_yaxes(gridcolor=C["border"])
    return fig


# ── Chart 5: Fear & Greed Gauge ──────────────────────────────────────────────

def chart_fear_greed_gauge(d):
    score      = d.get("fg_score", np.nan)
    components = d.get("fg_components", {})

    if np.isnan(score):
        return _empty_fig("Fear & Greed — N/D")

    if score <= 20:
        label = "PAURA ESTREMA"
        col   = C["red"]
    elif score <= 40:
        label = "Paura"
        col   = C["orange"]
    elif score <= 60:
        label = "Neutro"
        col   = C["yellow"]
    elif score <= 80:
        label = "Greed"
        col   = C["accent"]
    else:
        label = "GREED ESTREMO"
        col   = C["red"]

    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode  = "gauge+number+delta",
        value = round(score, 1),
        number= dict(font=dict(color=col, size=42, family="monospace")),
        delta = dict(reference=50, increasing=dict(color=C["red"]),
                     decreasing=dict(color=C["green"])),
        gauge = dict(
            axis    = dict(range=[0, 100], tickwidth=1, tickcolor=C["muted"],
                           tickfont=dict(color=C["muted"], size=9),
                           tickvals=[0,20,40,60,80,100],
                           ticktext=["0","20","40","60","80","100"]),
            bar     = dict(color=col, thickness=0.25),
            bgcolor = C["surface"],
            bordercolor = C["border"],
            steps   = [
                dict(range=[0,  20], color="rgba(255,23,68,0.18)"),
                dict(range=[20, 40], color="rgba(255,109,0,0.12)"),
                dict(range=[40, 60], color="rgba(255,214,0,0.10)"),
                dict(range=[60, 80], color="rgba(0,212,255,0.10)"),
                dict(range=[80,100], color="rgba(255,23,68,0.18)"),
            ],
            threshold=dict(
                line=dict(color=C["yellow"], width=3),
                thickness=0.75, value=score,
            ),
        ),
        title  = dict(
            text=f"FEAR & GREED COMPOSITO<br><span style='color:{col};font-size:16px'>{label}</span>",
            font=dict(color=C["text"], size=11),
        ),
        domain = dict(x=[0, 1], y=[0, 1]),
    ))

    fig.update_layout(**_LAYOUT_BASE, height=320,
        annotations=[dict(
            x=0.5, y=-0.05, xref="paper", yref="paper",
            text="0 = Paura Estrema  ·  50 = Neutro  ·  100 = Greed Estremo",
            font=dict(color=C["muted"], size=9, family="monospace"),
            showarrow=False,
        )],
    )
    return fig


def chart_fear_greed_components(d):
    components = d.get("fg_components", {})
    history    = d.get("fg_history", pd.DataFrame())

    if not components and history.empty:
        return _empty_fig("Componenti Fear & Greed — N/D")

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Componenti attuali", "Storico composito"],
                        column_widths=[0.40, 0.60])

    # Left: current components bar
    if components:
        names  = [v["label"] for v in components.values()]
        scores = [v["score"] for v in components.values()]
        raw    = [v.get("raw", np.nan) for v in components.values()]
        bar_cols = []
        for s in scores:
            if np.isnan(s):
                bar_cols.append(C["gray"])
            elif s <= 30:
                bar_cols.append(C["red"])
            elif s <= 50:
                bar_cols.append(C["orange"])
            elif s <= 70:
                bar_cols.append(C["yellow"])
            else:
                bar_cols.append(C["accent"])

        hover_texts = [f"{n}<br>Score: {s:.1f}<br>Raw: {r:.2f}"
                       if not np.isnan(s) and not np.isnan(r) else n
                       for n, s, r in zip(names, scores, raw)]

        fig.add_trace(go.Bar(
            x=[s if not np.isnan(s) else 0 for s in scores],
            y=names, orientation="h",
            marker_color=bar_cols, opacity=0.85,
            hovertext=hover_texts, hoverinfo="text",
            name="",
        ), row=1, col=1)
        fig.add_vline(x=50, line=dict(color=C["muted"], dash="dot", width=1),
                      row=1, col=1)
        fig.update_xaxes(range=[0, 100], row=1, col=1)

    # Right: historical line
    if not history.empty:
        hist2y = history[history.index >= pd.Timestamp.now() - pd.DateOffset(years=2)]
        if not hist2y.empty:
            hist_col = hist2y["value"].clip(0, 100)
            fig.add_trace(go.Scatter(
                x=hist2y.index, y=hist_col.values,
                name="F&G storico", line=dict(color=C["accent"], width=2),
            ), row=1, col=2)
            for y_val, col in [(80, C["red"]), (60, C["yellow"]), (40, C["yellow"]), (20, C["green"])]:
                fig.add_hline(y=y_val, line=dict(color=col, dash="dot", width=1), row=1, col=2)

    fig.update_layout(**_LAYOUT_BASE,
        title=dict(text="Fear & Greed — Composito e Componenti",
                   font=dict(size=12, color=C["text"]), x=0),
        showlegend=False,
        height=320,
    )
    fig.update_xaxes(gridcolor=C["border"])
    fig.update_yaxes(gridcolor=C["border"])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  KPI ROW
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_val(name, val):
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return "N/D"
    u = THRESHOLDS.get(name, {}).get("unit", "")
    if name == "trin":
        return f"{val:.2f}"
    if name == "seasonality":
        return f"{val:+.2f}%"
    if name in ("cot", "fear_greed"):
        return f"{val:.0f}{u}"
    return f"{val:.2f}"


def build_kpi_row(d):
    cv     = d.get("current_values", {})
    colors = d.get("colors", {})
    cards  = []
    for name, info in THRESHOLDS.items():
        val  = cv.get(name, np.nan)
        col  = colors.get(name, "gray")
        dot  = _color_hex(col)
        fval = _fmt_val(name, val)
        desc = info.get(f"desc_{col}", "")
        cards.append(html.Div([
            html.Div(info["label"], className="kpi-title"),
            html.Div([
                html.Span("● ", style={"color": dot, "fontSize": "15px", "lineHeight": "1"}),
                html.Span(fval, style={"color": C["text"], "fontSize": "20px",
                                       "fontFamily": "monospace", "fontWeight": "700"}),
            ], style={"display": "flex", "alignItems": "center", "gap": "4px", "marginTop": "4px"}),
            html.Div(desc, style={"fontSize": "9px", "color": C["muted"],
                                  "fontFamily": "monospace", "marginTop": "3px",
                                  "letterSpacing": "0.03em"}),
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
            html.Span(ts_str + "  ", style={"color": C["muted"], "fontSize": "9px",
                                             "fontFamily": "monospace", "minWidth": "130px",
                                             "display": "inline-block"}),
            html.Span(arrow_txt + "  ", style={"color": arrow_col, "fontSize": "11px",
                                                "fontFamily": "monospace", "fontWeight": "700",
                                                "minWidth": "120px", "display": "inline-block"}),
            html.Span(str(row.get("label", row.get("indicator", "?"))),
                      style={"color": C["text"], "fontSize": "11px", "fontFamily": "monospace"}),
            html.Span(val_str, style={"color": C["muted"], "fontSize": "9px",
                                      "fontFamily": "monospace"}),
        ], style={"padding": "6px 14px", "borderBottom": f"1px solid {C['border']}",
                  "display": "flex", "alignItems": "center"}))

    new_count = len(d.get("new_alerts", []))
    badge = f"  {new_count} nuovi ↑" if new_count else ""

    return html.Div([
        html.Div([
            html.Span("🚨  ALERT LOG", style={"fontFamily": "monospace", "fontSize": "11px",
                                              "fontWeight": "700", "color": C["accent"],
                                              "letterSpacing": "0.08em"}),
            html.Span(badge, style={"fontFamily": "monospace", "fontSize": "10px",
                                    "color": C["red"], "marginLeft": "10px"}),
        ], style={"padding": "8px 14px", "borderBottom": f"1px solid {C['border']}"}),
        html.Div(rows, style={"maxHeight": "180px", "overflowY": "auto"}),
    ], style={"backgroundColor": C["surface"], "border": f"1px solid {C['border']}",
              "borderRadius": "4px", "marginBottom": "18px"})


# ─────────────────────────────────────────────────────────────────────────────
#  FULL LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

def build_app_layout(d):
    new_alerts = d.get("new_alerts", [])
    badge = f"  ·  {len(new_alerts)} nuovi alert" if new_alerts else ""

    # Seasonality context string
    seasonal   = d.get("seasonal", {})
    current_td = seasonal.get("current_td", 0)
    forecast   = seasonal.get("forecast_20d", np.nan)
    season_ctx = ""
    if current_td > 0 and not np.isnan(forecast):
        season_ctx = (f"  ·  GG {current_td}/252  ·  "
                      f"Forecast 20 sessioni: {forecast:+.2f}%")

    return html.Div([
        # ── Sub-header ────────────────────────────────────────────────────────
        html.Div([
            html.Span("◆ ", style={"color": C["accent"]}),
            html.Span("SENTIMENT", style={"fontFamily": "monospace", "fontSize": "13px",
                                               "fontWeight": "700", "color": C["accent"],
                                               "letterSpacing": "0.08em"}),
            html.Span("  ·  TRIN · Stagionalità · CoT · Fear & Greed", style={
                "fontFamily": "monospace", "fontSize": "10px",
                "color": C["muted"], "marginLeft": "8px"}),
            html.Span(season_ctx, style={"fontFamily": "monospace", "fontSize": "10px",
                                         "color": C["accent"], "marginLeft": "8px"}),
            html.Span(badge, style={"fontFamily": "monospace", "fontSize": "10px",
                                    "color": C["red"], "marginLeft": "8px"}),
        ], style={"borderBottom": f"1px solid {C['border']}", "padding": "8px 20px"}),

        html.Div([
            # KPI row
            build_kpi_row(d),

            # Controls
            html.Div([
                html.Span("Lookback:", style={"fontFamily": "monospace", "fontSize": "11px",
                                              "color": C["muted"]}),
                dcc.RadioItems(
                    id="age-lookback",
                    options=[{"label": k, "value": v}
                             for k, v in [("1A",1),("2A",2),("3A",3),("5A",5)]],
                    value=2, inline=True,
                    style={"display": "inline-flex", "gap": "14px", "marginLeft": "12px"},
                    labelStyle={"fontFamily": "monospace", "fontSize": "11px",
                                "color": C["muted"], "cursor": "pointer"},
                    inputStyle={"accentColor": C["accent"]},
                ),
            ], className="controls-wrap"),

            # Alert panel
            _build_alert_panel(d),

            # Row 1: TRIN (full width)
            html.Div([
                html.Div([dcc.Graph(id="g-age-trin", config={"displayModeBar": False})],
                         className="chart-card"),
            ], style={"marginBottom": "12px"}),

            # Row 2: Stagionalità (2 charts)
            html.Div([
                html.Div([dcc.Graph(id="g-age-season-monthly",
                                    config={"displayModeBar": False})],
                         className="chart-card", style={"flex": 1}),
                html.Div([dcc.Graph(id="g-age-season-annual",
                                    config={"displayModeBar": False})],
                         className="chart-card", style={"flex": 1}),
            ], style={"display": "flex", "gap": "12px", "marginBottom": "12px"}),

            # Row 3: CoT (full width)
            html.Div([
                html.Div([dcc.Graph(id="g-age-cot", config={"displayModeBar": False})],
                         className="chart-card"),
            ], style={"marginBottom": "12px"}),

            # Row 4: Fear & Greed gauge + components + history
            html.Div([
                html.Div([dcc.Graph(id="g-age-fg-gauge",
                                    config={"displayModeBar": False})],
                         className="chart-card", style={"flex": "0 0 300px"}),
                html.Div([dcc.Graph(id="g-age-fg-components",
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
    print("  FIN ANALYTICS  ·  AGE Indicators  ·  Avvio...")
    print(f"{'─'*60}\n")

    _d = load_data()

    _app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.CYBORG],
        title="AGE Indicators",
        suppress_callback_exceptions=True,
    )
    _app.layout = html.Div(
        [build_app_layout(_d)],
        style={"backgroundColor": C["bg"], "minHeight": "100vh"},
    )

    @_app.callback(
        Output("g-age-trin",           "figure"),
        Output("g-age-season-monthly", "figure"),
        Output("g-age-season-annual",  "figure"),
        Output("g-age-cot",            "figure"),
        Output("g-age-fg-gauge",       "figure"),
        Output("g-age-fg-components",  "figure"),
        Input("age-lookback",          "value"),
    )
    def age_update(years):
        return (
            chart_trin(_d, years),
            chart_seasonality_monthly(_d),
            chart_seasonality_annual(_d),
            chart_cot(_d, years),
            chart_fear_greed_gauge(_d),
            chart_fear_greed_components(_d),
        )

    print(f"🚀  AGE Indicators  →  http://localhost:{PORT}\n")
    _app.run(debug=False, port=PORT, host="0.0.0.0")
