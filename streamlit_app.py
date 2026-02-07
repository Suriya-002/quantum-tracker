import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone

import streamlit as st
import numpy as np
import pandas as pd
import feedparser
import yfinance as yf
import requests

from anthropic import Anthropic
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import plotly.express as px
import plotly.graph_objects as go


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Quantum Tracker - ML Portfolio Analysis",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.12);
        border-radius: 10px;
        padding: 12px 22px;
        color: white;
    }
    .stTabs [aria-selected="true"] { background-color: rgba(255, 255, 255, 0.28); }
    h1, h2, h3 { color: white !important; }
    .stMarkdown { color: white; }
</style>
""", unsafe_allow_html=True)


# =========================
# Helpers
# =========================
def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def utc_now_iso() -> str:
    return utc_now().isoformat(timespec="seconds")

def get_secret(key: str, default: str = "") -> str:
    try:
        v = st.secrets.get(key, default)
        return v if v is not None else default
    except Exception:
        return os.getenv(key, default) or default


# =========================
# Universe & RSS
# =========================
UNIVERSE = {
    "IONQ": "IonQ Inc",
    "RGTI": "Rigetti Computing",
    "QBTS": "D-Wave Quantum",
    "IBM":  "IBM",
    "GOOGL":"Alphabet Inc",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "INTC": "Intel",
    "HON":  "Honeywell",
}
PURE_PLAY = {"IONQ", "RGTI", "QBTS"}

RSS_FEEDS = [
    ("https://thequantuminsider.com/feed/", "The Quantum Insider"),
    ("https://quantumcomputingreport.com/feed/", "Quantum Computing Report"),
]

QUANTUM_KEYWORDS = [
    "quantum", "qubit", "ionq", "rigetti", "ibm quantum", "google quantum",
    "d-wave", "dwave", "error correction", "fault tolerant"
]

DEFAULT_BENCH = "QQQ"


# =========================
# Keys
# =========================
ANTHROPIC_API_KEY = get_secret("ANTHROPIC_API_KEY", "")


# =========================
# DB (Cloud-safe path)
# =========================
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "quantum_tracker.db")

@st.cache_resource
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS news_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            url TEXT UNIQUE,
            source TEXT,
            published_date TEXT,
            description TEXT,
            fetched_date TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS stock_daily (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            company_name TEXT,
            d TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            market_cap REAL,
            UNIQUE(ticker, d)
        )
    """)

    c.execute("CREATE INDEX IF NOT EXISTS idx_stock_daily_ticker_d ON stock_daily(ticker, d)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_news_published_date ON news_articles(published_date)")

    conn.commit()
    return conn

conn = init_db()


def meta_get(key: str, default: str = "") -> str:
    c = conn.cursor()
    row = c.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default

def meta_set(key: str, value: str) -> None:
    c = conn.cursor()
    c.execute(
        "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value)
    )
    conn.commit()


# =========================
# News
# =========================
def normalize_date_to_ymd(s: str) -> str:
    try:
        dt = pd.to_datetime(s, errors="coerce", utc=True)
        if pd.isna(dt):
            return utc_now().date().isoformat()
        return dt.date().isoformat()
    except Exception:
        return utc_now().date().isoformat()

def extract_tickers(text: str) -> list[str]:
    if not text:
        return []
    t = text.upper()
    found = []
    for tk in UNIVERSE.keys():
        if re.search(rf"\b{re.escape(tk)}\b", t):
            found.append(tk)
    name_map = {
        "IONQ": ["IONQ", "IONQ INC"],
        "RGTI": ["RIGETTI"],
        "QBTS": ["D-WAVE", "DWAVE", "D WAVE"],
        "GOOGL": ["ALPHABET", "GOOGLE"],
        "MSFT": ["MICROSOFT"],
        "AMZN": ["AMAZON"],
        "INTC": ["INTEL"],
        "HON": ["HONEYWELL"],
        "IBM": ["IBM"],
    }
    for tk, keys in name_map.items():
        if tk in found:
            continue
        for k in keys:
            if k in t:
                found.append(tk)
                break
    return sorted(set(found))

@st.cache_data(ttl=3600)
def fetch_news(max_per_feed: int = 20) -> int:
    c = conn.cursor()
    added = 0
    for feed_url, source in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:max_per_feed]:
                title = (entry.get("title") or "").strip()
                link = (entry.get("link") or "").strip()
                summary = (entry.get("summary") or "").strip()
                hay = (title + " " + summary).lower()

                if not any(kw in hay for kw in QUANTUM_KEYWORDS):
                    continue

                published_raw = entry.get("published") or entry.get("updated") or utc_now_iso()
                published_ymd = normalize_date_to_ymd(published_raw)

                c.execute("""
                    INSERT OR IGNORE INTO news_articles
                    (title, url, source, published_date, description, fetched_date)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (title, link, source, published_ymd, summary[:900], utc_now_iso()))
                added += 1
        except Exception as e:
            st.sidebar.error(f"News fetch error ({source}): {e}")

    conn.commit()
    return added

@st.cache_data(ttl=300)
def get_news(limit: int = 60) -> pd.DataFrame:
    df = pd.read_sql_query("""
        SELECT * FROM news_articles
        ORDER BY published_date DESC, id DESC
        LIMIT ?
    """, conn, params=(int(limit),))
    if not df.empty:
        df["tickers"] = df.apply(lambda r: extract_tickers(f"{r['title']} {r.get('description','')}"), axis=1)
    return df


# =========================
# Stocks
# =========================
def safe_market_cap(ticker: str) -> float:
    try:
        info = yf.Ticker(ticker).get_info()
        return float(info.get("marketCap", 0) or 0)
    except Exception:
        return 0.0

@st.cache_data(ttl=1800)
def fetch_stocks_daily(days: int = 365, universe: list[str] | None = None) -> bool:
    tickers = universe or list(UNIVERSE.keys())
    end = utc_now().date()
    start = end - timedelta(days=int(days))

    try:
        df = yf.download(
            tickers=" ".join(tickers),
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True
        )
    except Exception as e:
        st.sidebar.error(f"yfinance download failed: {e}")
        return False

    c = conn.cursor()
    mcaps = {tk: safe_market_cap(tk) for tk in tickers}

    multi = isinstance(df.columns, pd.MultiIndex)

    def insert_row(tk: str, d: str, row: pd.Series):
        open_ = float(row.get("Open", np.nan))
        close_= float(row.get("Close", np.nan))
        high_ = float(row.get("High", np.nan))
        low_  = float(row.get("Low", np.nan))
        vol_  = int(row.get("Volume", 0) or 0)
        if not np.isfinite(open_) or not np.isfinite(close_):
            return
        c.execute("""
            INSERT OR IGNORE INTO stock_daily
            (ticker, company_name, d, open, high, low, close, volume, market_cap)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (tk, UNIVERSE.get(tk, tk), d, open_, high_, low_, close_, vol_, float(mcaps.get(tk, 0.0))))

    if multi:
        for tk in tickers:
            if tk not in df.columns.get_level_values(0):
                continue
            sub = df[tk].dropna(how="all")
            for idx, row in sub.iterrows():
                d = pd.to_datetime(idx).date().isoformat()
                insert_row(tk, d, row)
    else:
        tk = tickers[0] if tickers else "UNKNOWN"
        sub = df.dropna(how="all")
        for idx, row in sub.iterrows():
            d = pd.to_datetime(idx).date().isoformat()
            insert_row(tk, d, row)

    conn.commit()
    return True

@st.cache_data(ttl=300)
def get_latest_stocks(universe: list[str]) -> pd.DataFrame:
    if not universe:
        return pd.DataFrame()
    placeholders = ",".join(["?"] * len(universe))
    query = f"""
        SELECT s.ticker, s.company_name, s.d, s.open, s.close, s.volume, s.market_cap
        FROM stock_daily s
        JOIN (
            SELECT ticker, MAX(d) AS max_d
            FROM stock_daily
            WHERE ticker IN ({placeholders})
            GROUP BY ticker
        ) t
        ON s.ticker = t.ticker AND s.d = t.max_d
        ORDER BY s.ticker
    """
    df = pd.read_sql_query(query, conn, params=tuple(universe))
    if df.empty:
        return df
    df["price"] = df["close"]
    df["change"] = df["close"] - df["open"]
    df["change_percent"] = np.where(df["open"] != 0, (df["change"] / df["open"]) * 100.0, 0.0)
    return df

@st.cache_data(ttl=300)
def get_stock_history(ticker: str, days: int = 365) -> pd.DataFrame:
    start = (utc_now().date() - timedelta(days=int(days))).isoformat()
    df = pd.read_sql_query("""
        SELECT * FROM stock_daily
        WHERE ticker = ?
          AND d >= ?
        ORDER BY d
    """, conn, params=(ticker, start))
    return df


# =========================
# REFRESH policy
# =========================
def should_refresh(ttl_minutes: int) -> bool:
    last = meta_get("last_refresh_utc", "")
    if not last:
        return True
    try:
        last_dt = pd.to_datetime(last, utc=True)
        return (utc_now() - last_dt.to_pydatetime().replace(tzinfo=timezone.utc)) > timedelta(minutes=int(ttl_minutes))
    except Exception:
        return True

def refresh_all(force: bool, ttl_minutes: int, universe: list[str], days: int) -> dict:
    do = force or should_refresh(ttl_minutes)
    status = {"ran": False, "news_added": 0, "stocks_ok": False}
    if not do:
        return status
    status["ran"] = True
    status["news_added"] = fetch_news(max_per_feed=20)
    status["stocks_ok"] = fetch_stocks_daily(days=days, universe=universe)
    meta_set("last_refresh_utc", utc_now_iso())
    return status


# =========================
# FIXED ML signal (NO KeyError 'd')
# =========================
def make_features(dates: pd.Series, price: pd.Series, volume: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({
        "d": pd.to_datetime(dates),
        "price": price.astype(float),
        "volume": volume.astype(float),
    }).copy()

    df["ret"] = df["price"].pct_change()
    df["ret_1d"] = df["price"].pct_change(1)
    df["ret_5d"] = df["price"].pct_change(5)
    df["vol_5d"] = df["ret"].rolling(5).std()
    df["vol_20d"] = df["ret"].rolling(20).std()
    df["vol_chg"] = df["volume"].pct_change()
    df["ma5"] = df["price"].rolling(5).mean()
    df["ma20"] = df["price"].rolling(20).mean()
    df["p_to_ma5"] = df["price"] / df["ma5"]
    df["p_to_ma20"] = df["price"] / df["ma20"]
    return df

def walk_forward_ml_signal(hist: pd.DataFrame, min_train: int = 60) -> pd.DataFrame:
    """
    Walk-forward expanding window, 1-step ahead prediction.
    Returns index=d with columns pred, conf.
    """
    df = hist.copy()
    if "d" not in df.columns:
        raise ValueError("Expected 'd' column in stock history.")
    df["d"] = pd.to_datetime(df["d"])
    df = df.sort_values("d")

    feat = make_features(df["d"], df["close"], df["volume"])
    feat["target"] = feat["ret"].shift(-1)
    feat = feat.dropna().copy()

    feature_cols = ["ret_1d", "ret_5d", "vol_5d", "vol_20d", "vol_chg", "p_to_ma5", "p_to_ma20"]

    X_all = feat[feature_cols].values
    y_all = feat["target"].values
    d_all = feat["d"].values  # ✅ exists now

    scaler = StandardScaler()
    ridge = Ridge(alpha=1.0)
    rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)

    preds, confs, out_dates = [], [], []

    for i in range(len(X_all)):
        if i < min_train:
            continue

        X_train, y_train = X_all[:i], y_all[:i]
        X_test = X_all[i:i+1]

        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        ridge.fit(X_train_s, y_train)
        rf.fit(X_train_s, y_train)

        r_pred = float(ridge.predict(X_test_s)[0])
        f_pred = float(rf.predict(X_test_s)[0])
        pred = 0.5 * (r_pred + f_pred)

        recent = y_train[-20:] if len(y_train) >= 20 else y_train
        recent_vol = float(np.std(recent)) if len(recent) > 1 else 1e-9
        recent_vol = recent_vol if recent_vol > 1e-9 else 1e-9

        conf = abs(pred) / recent_vol

        preds.append(pred)
        confs.append(conf)
        out_dates.append(pd.to_datetime(d_all[i]))

    out = pd.DataFrame({"d": out_dates, "pred": preds, "conf": confs})
    out = out.set_index("d").sort_index()
    return out


# =========================
# Portfolio helpers
# =========================
def get_returns_matrix(tickers: list[str], days: int = 365) -> pd.DataFrame:
    mats = []
    for tk in tickers:
        h = get_stock_history(tk, days=days)
        if h.empty:
            continue
        s = pd.Series(h["close"].values, index=pd.to_datetime(h["d"]), name=tk).sort_index()
        mats.append(s)
    if not mats:
        return pd.DataFrame()
    prices = pd.concat(mats, axis=1).dropna(how="all").sort_index()
    rets = prices.pct_change().dropna(how="all")
    return rets


# =========================
# UI
# =========================
st.title("⚛️ Quantum Tracker")
st.markdown("### Real-time ML-Powered Quantum Computing Portfolio Analysis")

with st.sidebar:
    st.markdown("### ⚙️ Controls")

    universe_mode = st.selectbox("Universe", ["Full (Quantum + Big Tech)", "Pure-play (IONQ/RGTI/QBTS)"], index=0)
    universe_list = sorted(list(UNIVERSE.keys())) if "Full" in universe_mode else sorted(list(PURE_PLAY))

    refresh_ttl = st.slider("Auto-refresh TTL (minutes)", 5, 180, 30, 5)
    hist_days = st.slider("History window (days)", 90, 720, 365, 30)
    force_refresh = st.button("🔄 Force refresh now", type="primary")

    st.markdown("---")
    st.caption(f"Last refresh (UTC): {meta_get('last_refresh_utc','Never')}")

with st.spinner("Checking refresh policy..."):
    r = refresh_all(force_refresh, refresh_ttl, universe_list, hist_days)

stocks_df = get_latest_stocks(universe_list)
news_df = get_news(60)

tab1, tab2, tab3 = st.tabs(["📊 Overview", "🧠 ML Test", "📰 News"])

with tab1:
    st.markdown("### 📈 Market Snapshot")
    if stocks_df.empty:
        st.info("No stock data yet. Force refresh and try again.")
    else:
        disp = stocks_df[["ticker","company_name","price","change_percent","volume","market_cap","d"]].copy()
        disp["price"] = disp["price"].apply(lambda x: f"${float(x):.2f}")
        disp["change_percent"] = disp["change_percent"].apply(lambda x: f"{float(x):.2f}%")
        disp["volume"] = disp["volume"].apply(lambda x: f"{int(x)/1e6:.2f}M")
        disp["market_cap"] = disp["market_cap"].apply(lambda x: f"${float(x)/1e9:.2f}B" if float(x) > 0 else "N/A")
        st.dataframe(disp, width="stretch", hide_index=True)

with tab2:
    st.markdown("### 🧠 ML Walk-forward Sanity Test")
    st.markdown("This tab is specifically to ensure the ML pipeline runs without crashing.")

    pick = st.selectbox("Pick a ticker", universe_list, index=0)
    bt_days = st.slider("History days for ML", 120, 720, 365, 30)
    conf_th = st.slider("Confidence threshold", 0.0, 5.0, 1.0, 0.1)

    h = get_stock_history(pick, days=bt_days + 120)
    if h.empty or len(h) < 100:
        st.warning("Not enough data yet. Force refresh.")
    else:
        out = walk_forward_ml_signal(h, min_train=60)
        out = out.tail(120)
        st.write(out.tail(10))

        sig = ((out["pred"] > 0) & (out["conf"] >= conf_th)).astype(float)
        st.metric("Signals (last 120 days)", int(sig.sum()))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=out.index, y=out["pred"], name="Predicted next-day return"))
        fig.add_trace(go.Scatter(x=out.index, y=out["conf"], name="Confidence"))
        fig.update_layout(height=420)
        st.plotly_chart(fig, width="stretch")

with tab3:
    st.markdown("### 📰 Latest News (Ticker-mapped)")
    if news_df.empty:
        st.info("No news yet.")
    else:
        for _, a in news_df.head(30).iterrows():
            tks = a.get("tickers", [])
            tks_s = ", ".join(tks) if isinstance(tks, list) and tks else "N/A"
            with st.expander(f"{a['published_date']} | {a['title']}"):
                st.markdown(f"**Source:** {a['source']}")
                st.markdown(f"**Tickers:** {tks_s}")
                st.markdown(f"**Link:** {a['url']}")
                if a.get("description"):
                    st.markdown(a["description"][:800])

st.caption("Quantum Tracker • Streamlit Cloud-ready • UTC time")
