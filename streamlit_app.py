import os
import re
import sqlite3
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta, date

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

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Quantum Tracker - ML Portfolio Analysis",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# CSS
# ----------------------------
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
    div[data-testid="stMetricValue"] { font-size: 28px; color: #667eea; }
    .card {
        background: rgba(255,255,255,0.92);
        border-radius: 14px;
        padding: 16px;
        margin: 10px 0;
        box-shadow: 0 6px 14px rgba(0,0,0,0.12);
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Constants / Universe
# ----------------------------
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
PURE_PLAY = {"IONQ", "RGTI", "QBTS"}  # you can expand later

DEFAULT_BENCH = "QQQ"  # or SPY

RSS_FEEDS = [
    ("https://thequantuminsider.com/feed/", "The Quantum Insider"),
    ("https://quantumcomputingreport.com/feed/", "Quantum Computing Report"),
]

QUANTUM_KEYWORDS = [
    "quantum", "qubit", "ionq", "rigetti", "ibm quantum", "google quantum",
    "d-wave", "dwave", "quantinuum", "psi quantum", "neutral atom", "superconducting",
    "error correction", "fault tolerant", "photon", "spin qubit"
]

# ----------------------------
# Secrets helper
# ----------------------------
def get_secret(key: str, default: str = "") -> str:
    try:
        v = st.secrets.get(key, default)
        return v if v is not None else default
    except Exception:
        return os.getenv(key, default) or default

ANTHROPIC_API_KEY = get_secret("ANTHROPIC_API_KEY", "")
SLACK_WEBHOOK_URL = get_secret("SLACK_WEBHOOK_URL", "")
SMTP_HOST = get_secret("SMTP_HOST", "")
SMTP_PORT = int(get_secret("SMTP_PORT", "587") or "587")
SMTP_USER = get_secret("SMTP_USER", "")
SMTP_PASS = get_secret("SMTP_PASS", "")
ALERT_TO_EMAIL = get_secret("ALERT_TO_EMAIL", "")

# ----------------------------
# DB setup (Cloud-safe path)
# ----------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "quantum_tracker.db")

@st.cache_resource
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()

    # Meta key/value table (for refresh timestamps, etc.)
    c.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    # News
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

    # Stocks (daily granularity)
    c.execute("""
        CREATE TABLE IF NOT EXISTS stock_daily (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            company_name TEXT,
            d TEXT,              -- YYYY-MM-DD
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            market_cap REAL,
            UNIQUE(ticker, d)
        )
    """)

    # Indexes for speed
    c.execute("CREATE INDEX IF NOT EXISTS idx_stock_daily_ticker_d ON stock_daily(ticker, d)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_news_published_date ON news_articles(published_date)")

    conn.commit()
    return conn

conn = init_db()

# ----------------------------
# Meta helpers (refresh control)
# ----------------------------
def meta_get(key: str, default: str = "") -> str:
    c = conn.cursor()
    row = c.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default

def meta_set(key: str, value: str) -> None:
    c = conn.cursor()
    c.execute("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
    conn.commit()

def utc_now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")

# ----------------------------
# Fetch: RSS news
# ----------------------------
def normalize_date_to_ymd(s: str) -> str:
    # Many feeds provide RFC822; feedparser provides parsed struct_time sometimes.
    try:
        # If it's already YYYY-MM-DD
        if re.match(r"^\d{4}-\d{2}-\d{2}$", str(s).strip()):
            return str(s).strip()
        # Try parse via pandas
        return pd.to_datetime(s, errors="coerce", utc=True).date().isoformat()
    except Exception:
        return datetime.utcnow().date().isoformat()

def extract_tickers(text: str) -> list[str]:
    if not text:
        return []
    t = text.upper()
    found = []
    for tk in UNIVERSE.keys():
        # match as whole word
        if re.search(rf"\b{re.escape(tk)}\b", t):
            found.append(tk)
    # also map company names lightly
    name_map = {
        "IONQ": ["IONQ", "IONQ INC"],
        "RGTI": ["RIGETTI"],
        "QBTS": ["D-WAVE", "DWAVE", "D WAVE"],
        "IBM": ["IBM"],
        "GOOGL": ["ALPHABET", "GOOGLE"],
        "MSFT": ["MICROSOFT"],
        "AMZN": ["AMAZON"],
        "INTC": ["INTEL"],
        "HON": ["HONEYWELL"],
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

# ----------------------------
# Fetch: Stocks (batch via yf.download)
# ----------------------------
def safe_market_cap(ticker: str) -> float:
    try:
        info = yf.Ticker(ticker).get_info()
        mc = info.get("marketCap", 0) or 0
        return float(mc)
    except Exception:
        return 0.0

@st.cache_data(ttl=1800)
def fetch_stocks_daily(days: int = 180, universe: list[str] | None = None) -> bool:
    tickers = universe or list(UNIVERSE.keys())
    end = datetime.utcnow().date()
    start = end - timedelta(days=int(days))

    # batch download (fast)
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
        st.sidebar.error(f"yfinance batch download failed: {e}")
        return False

    c = conn.cursor()

    # If single ticker, yfinance returns columns not multiindex
    multi = isinstance(df.columns, pd.MultiIndex)

    # market caps (cached-ish)
    mcaps = {}
    for tk in tickers:
        mcaps[tk] = safe_market_cap(tk)

    def insert_row(tk: str, d: str, row: pd.Series):
        open_ = float(row.get("Open", np.nan))
        high_ = float(row.get("High", np.nan))
        low_  = float(row.get("Low", np.nan))
        close_= float(row.get("Close", np.nan))
        vol_  = int(row.get("Volume", 0) or 0)

        if not np.isfinite(open_) or not np.isfinite(close_):
            return

        c.execute("""
            INSERT OR IGNORE INTO stock_daily
            (ticker, company_name, d, open, high, low, close, volume, market_cap)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (tk, UNIVERSE.get(tk, tk), d, open_, high_, low_, close_, vol_, float(mcaps.get(tk, 0.0))))

    if multi:
        # df has columns: (ticker, field)
        for tk in tickers:
            if tk not in df.columns.get_level_values(0):
                continue
            sub = df[tk].dropna(how="all")
            for idx, row in sub.iterrows():
                d = pd.to_datetime(idx).date().isoformat()
                insert_row(tk, d, row)
    else:
        # single ticker case
        tk = tickers[0] if tickers else "UNKNOWN"
        sub = df.dropna(how="all")
        for idx, row in sub.iterrows():
            d = pd.to_datetime(idx).date().isoformat()
            insert_row(tk, d, row)

    conn.commit()
    return True

# ----------------------------
# Queries
# ----------------------------
@st.cache_data(ttl=300)
def get_latest_stocks(universe: list[str]) -> pd.DataFrame:
    if not universe:
        return pd.DataFrame()

    placeholders = ",".join(["?"] * len(universe))
    query = f"""
        SELECT s.ticker, s.company_name, s.close, s.open, s.volume, s.market_cap, s.d
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
    df["change"] = df["close"] - df["open"]
    df["change_percent"] = np.where(df["open"] != 0, (df["change"] / df["open"]) * 100.0, 0.0)
    df.rename(columns={"close": "price"}, inplace=True)
    return df

@st.cache_data(ttl=300)
def get_stock_history(ticker: str, days: int = 180) -> pd.DataFrame:
    start = (datetime.utcnow().date() - timedelta(days=int(days))).isoformat()
    df = pd.read_sql_query("""
        SELECT * FROM stock_daily
        WHERE ticker = ?
          AND d >= ?
        ORDER BY d
    """, conn, params=(ticker, start))
    return df

@st.cache_data(ttl=300)
def get_news(limit: int = 50) -> pd.DataFrame:
    df = pd.read_sql_query("""
        SELECT * FROM news_articles
        ORDER BY published_date DESC, id DESC
        LIMIT ?
    """, conn, params=(int(limit),))
    # add tickers column dynamically
    if not df.empty:
        df["tickers"] = df.apply(lambda r: extract_tickers(f"{r['title']} {r.get('description','')}"), axis=1)
    return df

# ----------------------------
# Metrics & backtest helpers
# ----------------------------
def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())

def annualize_return(daily_ret: pd.Series, periods: int = 252) -> float:
    if daily_ret.empty:
        return 0.0
    equity = (1.0 + daily_ret.fillna(0)).cumprod()
    n = len(equity)
    if n < 2:
        return 0.0
    years = n / periods
    return float(equity.iloc[-1] ** (1 / years) - 1)

def annualize_vol(daily_ret: pd.Series, periods: int = 252) -> float:
    if daily_ret.empty:
        return 0.0
    return float(daily_ret.std(ddof=0) * np.sqrt(periods))

def sharpe(daily_ret: pd.Series, rf: float = 0.0, periods: int = 252) -> float:
    if daily_ret.empty:
        return 0.0
    excess = daily_ret - (rf / periods)
    vol = excess.std(ddof=0)
    if vol == 0:
        return 0.0
    return float(excess.mean() / vol * np.sqrt(periods))

def sortino(daily_ret: pd.Series, rf: float = 0.0, periods: int = 252) -> float:
    if daily_ret.empty:
        return 0.0
    excess = daily_ret - (rf / periods)
    downside = excess[excess < 0].std(ddof=0)
    if downside == 0:
        return 0.0
    return float(excess.mean() / downside * np.sqrt(periods))

def calmar(daily_ret: pd.Series, periods: int = 252) -> float:
    if daily_ret.empty:
        return 0.0
    eq = (1.0 + daily_ret.fillna(0)).cumprod()
    mdd = abs(max_drawdown(eq))
    cagr = annualize_return(daily_ret, periods=periods)
    return float(cagr / mdd) if mdd > 0 else 0.0

def turnover_from_weights(w: pd.DataFrame) -> pd.Series:
    # sum of absolute daily changes in weights
    return w.diff().abs().sum(axis=1).fillna(0.0)

def compute_portfolio_returns(returns: pd.DataFrame, weights: pd.DataFrame, tc_bps: float = 5.0) -> pd.Series:
    # Align
    returns = returns.loc[weights.index, weights.columns].fillna(0.0)
    w = weights.fillna(0.0)

    gross = (w.shift(1).fillna(0.0) * returns).sum(axis=1)
    t = turnover_from_weights(w)
    tc = (tc_bps / 10000.0) * t
    net = gross - tc
    return net

# ----------------------------
# Strategy signals
# ----------------------------
def make_features(price: pd.Series, volume: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"price": price, "volume": volume}).copy()
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
    Returns DataFrame indexed by date with columns:
      pred (next-day return prediction), conf (confidence proxy)
    Walk-forward expanding window, 1-step ahead.
    """
    df = hist.copy()
    df["d"] = pd.to_datetime(df["d"])
    df = df.sort_values("d")

    feat = make_features(df["close"], df["volume"])
    feat["target"] = feat["ret"].shift(-1)
    feat = feat.dropna().copy()

    feature_cols = ["ret_1d", "ret_5d", "vol_5d", "vol_20d", "vol_chg", "p_to_ma5", "p_to_ma20"]
    X_all = feat[feature_cols].values
    y_all = feat["target"].values
    dates = feat["d"].values

    preds = []
    confs = []
    out_dates = []

    scaler = StandardScaler()
    ridge = Ridge(alpha=1.0)
    rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)

    for i in range(len(X_all)):
        if i < min_train:
            continue
        X_train = X_all[:i]
        y_train = y_all[:i]
        X_test = X_all[i:i+1]

        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        ridge.fit(X_train_s, y_train)
        rf.fit(X_train_s, y_train)

        r_pred = float(ridge.predict(X_test_s)[0])
        f_pred = float(rf.predict(X_test_s)[0])
        pred = 0.5 * (r_pred + f_pred)

        # Confidence proxy: magnitude relative to recent volatility
        recent_vol = float(np.std(y_train[-20:])) if len(y_train) >= 20 else float(np.std(y_train))
        recent_vol = recent_vol if recent_vol > 1e-9 else 1e-9
        conf = abs(pred) / recent_vol

        preds.append(pred)
        confs.append(conf)
        out_dates.append(pd.to_datetime(dates[i]).date().isoformat())

    out = pd.DataFrame({"d": out_dates, "pred": preds, "conf": confs})
    out["d"] = pd.to_datetime(out["d"])
    out = out.set_index("d").sort_index()
    return out

def momentum_signal(price: pd.Series, lookback: int = 20) -> pd.Series:
    # simple momentum: +1 if lookback return > 0 else 0
    mom = price.pct_change(lookback)
    sig = (mom > 0).astype(float)
    return sig

# ----------------------------
# Build universe returns
# ----------------------------
@st.cache_data(ttl=600)
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

# ----------------------------
# News event impact (1d/3d)
# ----------------------------
def event_impact_for_ticker(ticker: str, event_date_ymd: str) -> dict:
    # Use close-to-close returns from stored daily data
    h = get_stock_history(ticker, days=365)
    if h.empty:
        return {"r1d": np.nan, "r3d": np.nan}
    h = h.copy()
    h["d"] = pd.to_datetime(h["d"])
    h = h.set_index("d").sort_index()
    event_dt = pd.to_datetime(event_date_ymd)
    if event_dt not in h.index:
        # pick next trading day
        idx = h.index.searchsorted(event_dt)
        if idx >= len(h.index):
            return {"r1d": np.nan, "r3d": np.nan}
        event_dt = h.index[idx]

    close0 = float(h.loc[event_dt, "close"])
    # next 1d
    idx0 = h.index.get_loc(event_dt)
    r1d = np.nan
    r3d = np.nan
    if idx0 + 1 < len(h.index):
        close1 = float(h.iloc[idx0 + 1]["close"])
        r1d = (close1 / close0) - 1.0
    if idx0 + 3 < len(h.index):
        close3 = float(h.iloc[idx0 + 3]["close"])
        r3d = (close3 / close0) - 1.0
    return {"r1d": r1d, "r3d": r3d}

# ----------------------------
# AI summary
# ----------------------------
@st.cache_data(ttl=3600)
def generate_ai_summary(news_limit: int = 12) -> str:
    if not ANTHROPIC_API_KEY:
        return "⚠️ Anthropic API key not configured. Add ANTHROPIC_API_KEY in Streamlit secrets or environment."

    news_df = get_news(news_limit)
    if news_df.empty:
        return "No news available for summary."

    # keep compact
    lines = []
    for _, r in news_df.iterrows():
        tks = ", ".join(r.get("tickers", [])) if isinstance(r.get("tickers", []), list) else ""
        lines.append(f"- {r['title']} ({r['source']}) [Tickers: {tks}]")

    prompt = (
        "You are an investor-focused analyst.\n"
        "Summarize the quantum computing sector using the headlines below.\n\n"
        + "\n".join(lines) +
        "\n\nReturn a structured summary:\n"
        "1) Market overview (2-3 sentences)\n"
        "2) Key company developments (bullets)\n"
        "3) Tech progress (bullets)\n"
        "4) Funding/partnerships (bullets)\n"
        "5) What to watch next week (bullets)\n"
        "Be concise and actionable."
    )

    try:
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=700,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text
    except Exception as e:
        return f"Error generating summary: {e}"

# ----------------------------
# Refresh policy
# ----------------------------
def should_refresh(ttl_minutes: int) -> bool:
    last = meta_get("last_refresh_utc", "")
    if not last:
        return True
    try:
        last_dt = pd.to_datetime(last, utc=True)
        return (datetime.utcnow() - last_dt.to_pydatetime()) > timedelta(minutes=int(ttl_minutes))
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

# ----------------------------
# Alerts
# ----------------------------
def send_slack(msg: str) -> bool:
    if not SLACK_WEBHOOK_URL:
        return False
    try:
        r = requests.post(SLACK_WEBHOOK_URL, json={"text": msg}, timeout=10)
        return r.status_code >= 200 and r.status_code < 300
    except Exception:
        return False

def send_email(subject: str, body: str) -> bool:
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and ALERT_TO_EMAIL):
        return False
    try:
        mime = MIMEText(body, "plain", "utf-8")
        mime["Subject"] = subject
        mime["From"] = SMTP_USER
        mime["To"] = ALERT_TO_EMAIL

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.sendmail(SMTP_USER, [ALERT_TO_EMAIL], mime.as_string())
        return True
    except Exception:
        return False

def scan_alerts(stocks_df: pd.DataFrame, news_df: pd.DataFrame, move_threshold: float = 5.0) -> list[str]:
    alerts = []

    # price move alerts
    if not stocks_df.empty:
        movers = stocks_df[stocks_df["change_percent"].abs() >= move_threshold]
        for _, r in movers.iterrows():
            alerts.append(f"⚡ {r['ticker']} moved {r['change_percent']:.2f}% today (price: {r['price']:.2f}).")

    # "funding/partnership" news alerts
    if not news_df.empty:
        recent = news_df.head(15)
        for _, r in recent.iterrows():
            text = f"{r['title']} {(r.get('description','') or '')}".lower()
            if any(k in text for k in ["funding", "raised", "series", "investment", "partnership", "collaboration"]):
                tks = r.get("tickers", [])
                tks_s = ", ".join(tks) if isinstance(tks, list) and tks else "N/A"
                alerts.append(f"📰 Funding/Partnership headline: {r['title']} (Tickers: {tks_s})")

    return alerts

# ----------------------------
# UI: Sidebar controls
# ----------------------------
st.title("⚛️ Quantum Tracker")
st.markdown("### Real-time ML-Powered Quantum Computing Portfolio Analysis")

colA, colB, colC = st.columns([6, 1, 1])
with colC:
    st.markdown('<span style="color: #00ff00;">● LIVE</span>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Controls")

    universe_mode = st.selectbox("Universe", ["Full (Quantum + Big Tech)", "Pure-play (IONQ/RGTI/QBTS)"], index=0)
    universe_list = sorted(list(UNIVERSE.keys())) if "Full" in universe_mode else sorted(list(PURE_PLAY))

    refresh_ttl = st.slider("Auto-refresh TTL (minutes)", 5, 180, 30, 5)
    hist_days = st.slider("History window (days)", 90, 720, 365, 30)

    force_refresh = st.button("🔄 Force refresh now", type="primary")
    st.caption("Tip: Auto-refresh only runs when TTL expires (or forced).")

    st.markdown("---")
    st.markdown("### 🔔 Alerts")
    move_th = st.slider("Alert threshold: abs daily move (%)", 2.0, 15.0, 5.0, 0.5)
    do_alert_scan = st.button("Run alert scan")

    st.markdown("---")
    last_r = meta_get("last_refresh_utc", "Never")
    st.metric("Last refresh (UTC)", last_r if last_r != "Never" else "Never")

# ----------------------------
# Refresh policy execution (first thing)
# ----------------------------
with st.spinner("Checking refresh policy..."):
    r = refresh_all(force_refresh, refresh_ttl, universe_list, hist_days)

if r["ran"]:
    st.sidebar.success(f"✅ Refreshed. News added: {r['news_added']}. Stocks OK: {r['stocks_ok']}")
else:
    st.sidebar.info("Using cached DB (refresh TTL not reached).")

# ----------------------------
# Load current data
# ----------------------------
stocks_df = get_latest_stocks(universe_list)
news_df = get_news(80)

# Alert scan trigger
if do_alert_scan:
    alerts = scan_alerts(stocks_df, news_df, move_threshold=move_th)
    if not alerts:
        st.sidebar.success("No alerts triggered.")
    else:
        msg = "Quantum Tracker Alerts:\n" + "\n".join([f"- {a}" for a in alerts[:12]])
        slack_ok = send_slack(msg)
        email_ok = send_email("Quantum Tracker Alerts", msg)
        st.sidebar.warning(f"Triggered {len(alerts)} alerts.")
        st.sidebar.caption(f"Slack sent: {slack_ok} | Email sent: {email_ok}")

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "💼 Portfolio Builder",
    "🧠 ML & Backtests",
    "📰 News Intelligence",
    "⚛️ Sector Index"
])

# ============================================================
# TAB 1: Overview
# ============================================================
with tab1:
    c1, c2 = st.columns([2, 1])

    with c1:
        st.markdown("### 🤖 AI Market Summary")
        with st.spinner("Generating AI insights..."):
            summary = generate_ai_summary(news_limit=12)
        st.markdown(summary)

    with c2:
        st.markdown("### 📈 Market Snapshot")
        if stocks_df.empty:
            st.info("No stock data yet (refresh and try again).")
        else:
            avg_change = float(stocks_df["change_percent"].mean())
            gainers = int((stocks_df["change_percent"] > 0).sum())
            total_vol = int(stocks_df["volume"].sum())

            st.metric("Average Change", f"{avg_change:.2f}%", delta=f"{avg_change:.2f}%")
            st.metric("Gainers / Total", f"{gainers} / {len(stocks_df)}")
            st.metric("Total Volume", f"{total_vol/1e6:.1f}M")

            st.markdown("#### 🔥 Top Movers")
            top = stocks_df.reindex(stocks_df["change_percent"].abs().sort_values(ascending=False).index).head(3)
            for _, r2 in top.iterrows():
                emoji = "🟢" if r2["change_percent"] >= 0 else "🔴"
                st.markdown(f"{emoji} **{r2['ticker']}**: {r2['change_percent']:.2f}%")

    st.markdown("---")
    st.markdown("### 📋 Live Table")
    if not stocks_df.empty:
        disp = stocks_df[["ticker", "company_name", "price", "change_percent", "volume", "market_cap", "d"]].copy()
        disp["price"] = disp["price"].apply(lambda x: f"${float(x):.2f}")
        disp["change_percent"] = disp["change_percent"].apply(lambda x: f"{float(x):.2f}%")
        disp["volume"] = disp["volume"].apply(lambda x: f"{int(x)/1e6:.2f}M")
        disp["market_cap"] = disp["market_cap"].apply(lambda x: f"${float(x)/1e9:.2f}B" if float(x) > 0 else "N/A")
        st.dataframe(disp, width="stretch", hide_index=True)
    else:
        st.info("Loading...")

# ============================================================
# TAB 2: Portfolio Builder (user-defined) + backtest
# ============================================================
with tab2:
    st.markdown("### 💼 Portfolio Builder")
    st.markdown("Build a portfolio, run a walk-forward backtest, and compare against a benchmark.")

    colL, colR = st.columns([1, 1])

    with colL:
        pick = st.multiselect("Select tickers", options=universe_list, default=universe_list[:5])
        benchmark = st.selectbox("Benchmark", ["QQQ", "SPY"], index=0 if DEFAULT_BENCH == "QQQ" else 1)
        strat = st.selectbox("Strategy", ["Equal Weight (Buy & Hold)", "Momentum (20D)", "ML Ensemble (Walk-forward)"], index=2)

    with colR:
        bt_days = st.slider("Backtest length (days)", 90, 720, 365, 30)
        tc_bps = st.slider("Transaction cost (bps per 1.0 turnover)", 0.0, 50.0, 5.0, 1.0)
        conf_th = st.slider("ML confidence threshold (higher = fewer trades)", 0.0, 5.0, 1.0, 0.1)

    if not pick:
        st.info("Select at least 1 ticker.")
    else:
        with st.spinner("Preparing backtest..."):
            rets = get_returns_matrix(pick, days=bt_days)
            bench_rets = None
            try:
                bdf = yf.download(benchmark, period=f"{bt_days+30}d", progress=False, auto_adjust=False)
                if not bdf.empty:
                    bpx = bdf["Close"].dropna()
                    bench_rets = bpx.pct_change().dropna()
            except Exception:
                bench_rets = None

        if rets.empty:
            st.warning("Not enough data to backtest yet. Force refresh and try again.")
        else:
            # Align dates
            idx = rets.index
            if bench_rets is not None and not bench_rets.empty:
                bench_rets = bench_rets.reindex(idx).fillna(0.0)
            else:
                bench_rets = pd.Series(0.0, index=idx)

            # Build weights
            if strat == "Equal Weight (Buy & Hold)":
                w = pd.DataFrame(1.0 / len(pick), index=idx, columns=pick)

            elif strat == "Momentum (20D)":
                # weights proportional to momentum signal; normalized each day
                prices = (1.0 + rets).cumprod()
                sigs = pd.DataFrame({tk: momentum_signal(prices[tk], lookback=20) for tk in pick})
                sigs = sigs.reindex(idx).fillna(0.0)
                w = sigs.div(sigs.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

            else:
                # ML Ensemble: per-ticker walk-forward pred -> long only when conf >= threshold and pred>0
                sigs = pd.DataFrame(0.0, index=idx, columns=pick)
                confs = pd.DataFrame(0.0, index=idx, columns=pick)

                for tk in pick:
                    h = get_stock_history(tk, days=bt_days + 120)
                    if h.empty or len(h) < 90:
                        continue
                    out = walk_forward_ml_signal(h, min_train=60)
                    out = out.reindex(idx).fillna(0.0)
                    sig = ((out["pred"] > 0) & (out["conf"] >= conf_th)).astype(float)
                    sigs[tk] = sig
                    confs[tk] = out["conf"]

                # normalize weights
                w = sigs.div(sigs.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

            port_ret = compute_portfolio_returns(rets[pick], w[pick], tc_bps=tc_bps)
            bench_ret = bench_rets.reindex(port_ret.index).fillna(0.0)

            eq = (1.0 + port_ret).cumprod()
            bq = (1.0 + bench_ret).cumprod()

            # metrics
            cagr = annualize_return(port_ret)
            vol = annualize_vol(port_ret)
            sh = sharpe(port_ret)
            so = sortino(port_ret)
            mdd = max_drawdown(eq)
            ca = calmar(port_ret)
            turn = float(turnover_from_weights(w).mean())

            st.markdown("#### 📌 Performance Summary")
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("CAGR", f"{cagr*100:.2f}%")
            m2.metric("Vol", f"{vol*100:.2f}%")
            m3.metric("Sharpe", f"{sh:.2f}")
            m4.metric("Sortino", f"{so:.2f}")
            m5.metric("Max DD", f"{mdd*100:.2f}%")
            m6.metric("Avg Turnover", f"{turn:.2f}")

            st.markdown("---")
            st.markdown("#### 📈 Equity Curve vs Benchmark")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Portfolio"))
            fig.add_trace(go.Scatter(x=bq.index, y=bq.values, name=benchmark))
            fig.update_layout(height=420, xaxis_title="Date", yaxis_title="Growth of $1")
            st.plotly_chart(fig, width="stretch")

            st.markdown("#### 🧾 Weights (latest)")
            latest_w = w.iloc[-1].sort_values(ascending=False)
            st.dataframe(latest_w.to_frame("weight").style.format({"weight":"{:.2%}"}), width="stretch")

# ============================================================
# TAB 3: ML + Backtests (baselines + comparison)
# ============================================================
with tab3:
    st.markdown("### 🧠 ML & Backtests")
    st.markdown("Compare strategies: Buy&Hold vs Momentum vs ML Ensemble (walk-forward).")

    pick2 = st.multiselect("Tickers for strategy comparison", options=universe_list, default=universe_list[:6], key="cmp_tickers")
    bench2 = st.selectbox("Benchmark", ["QQQ", "SPY"], index=0, key="cmp_bench")
    days2 = st.slider("Comparison window (days)", 120, 720, 365, 30, key="cmp_days")
    tc2 = st.slider("Transaction cost (bps)", 0.0, 50.0, 5.0, 1.0, key="cmp_tc")
    conf2 = st.slider("ML confidence threshold", 0.0, 5.0, 1.0, 0.1, key="cmp_conf")

    if not pick2:
        st.info("Pick tickers to compare strategies.")
    else:
        rets2 = get_returns_matrix(pick2, days=days2)
        if rets2.empty:
            st.warning("Not enough data yet.")
        else:
            idx = rets2.index
            # benchmark
            try:
                bdf2 = yf.download(bench2, period=f"{days2+30}d", progress=False, auto_adjust=False)
                bpx2 = bdf2["Close"].dropna()
                br2 = bpx2.pct_change().dropna().reindex(idx).fillna(0.0)
            except Exception:
                br2 = pd.Series(0.0, index=idx)

            # Equal weight
            w_eq = pd.DataFrame(1.0/len(pick2), index=idx, columns=pick2)
            r_eq = compute_portfolio_returns(rets2[pick2], w_eq, tc_bps=0.0)

            # Momentum
            prices2 = (1.0 + rets2).cumprod()
            sig_m = pd.DataFrame({tk: momentum_signal(prices2[tk], lookback=20) for tk in pick2}).reindex(idx).fillna(0.0)
            w_m = sig_m.div(sig_m.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
            r_m = compute_portfolio_returns(rets2[pick2], w_m, tc_bps=tc2)

            # ML
            sig_ml = pd.DataFrame(0.0, index=idx, columns=pick2)
            for tk in pick2:
                h = get_stock_history(tk, days=days2 + 120)
                if h.empty or len(h) < 90:
                    continue
                out = walk_forward_ml_signal(h, min_train=60).reindex(idx).fillna(0.0)
                sig = ((out["pred"] > 0) & (out["conf"] >= conf2)).astype(float)
                sig_ml[tk] = sig
            w_ml = sig_ml.div(sig_ml.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
            r_ml = compute_portfolio_returns(rets2[pick2], w_ml, tc_bps=tc2)

            eq_eq = (1+r_eq).cumprod()
            eq_m  = (1+r_m).cumprod()
            eq_ml = (1+r_ml).cumprod()
            eq_b  = (1+br2).cumprod()

            # Metrics table
            def row(name, r):
                eq = (1+r).cumprod()
                return {
                    "Strategy": name,
                    "CAGR %": annualize_return(r)*100,
                    "Vol %": annualize_vol(r)*100,
                    "Sharpe": sharpe(r),
                    "Sortino": sortino(r),
                    "MaxDD %": max_drawdown(eq)*100,
                    "Calmar": calmar(r),
                }

            mtab = pd.DataFrame([
                row("Equal Weight (B&H)", r_eq),
                row("Momentum (20D)", r_m),
                row("ML Ensemble", r_ml),
                row(bench2, br2),
            ])

            st.markdown("#### 📊 Strategy Metrics")
            st.dataframe(mtab.style.format({
                "CAGR %":"{:.2f}", "Vol %":"{:.2f}", "Sharpe":"{:.2f}", "Sortino":"{:.2f}", "MaxDD %":"{:.2f}", "Calmar":"{:.2f}"
            }), width="stretch", hide_index=True)

            st.markdown("#### 📈 Equity Curves")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=eq_eq.index, y=eq_eq.values, name="Equal Weight"))
            fig.add_trace(go.Scatter(x=eq_m.index, y=eq_m.values, name="Momentum"))
            fig.add_trace(go.Scatter(x=eq_ml.index, y=eq_ml.values, name="ML Ensemble"))
            fig.add_trace(go.Scatter(x=eq_b.index, y=eq_b.values, name=bench2))
            fig.update_layout(height=450, xaxis_title="Date", yaxis_title="Growth of $1")
            st.plotly_chart(fig, width="stretch")

# ============================================================
# TAB 4: News Intelligence (ticker mapping + event impact)
# ============================================================
with tab4:
    st.markdown("### 📰 News Intelligence")
    st.markdown("News mapped to tickers + simple event impact (next 1d / 3d move from event date).")

    if news_df.empty:
        st.info("No news yet (refresh to fetch RSS).")
    else:
        # filter controls
        all_tickers = ["ALL"] + sorted(list(UNIVERSE.keys()))
        sel_tk = st.selectbox("Filter by ticker", all_tickers, index=0)
        kw = st.text_input("Keyword filter (optional)", "")

        df = news_df.copy()
        if sel_tk != "ALL":
            df = df[df["tickers"].apply(lambda x: isinstance(x, list) and sel_tk in x)]
        if kw.strip():
            k = kw.strip().lower()
            df = df[df.apply(lambda r: k in (str(r["title"]) + " " + str(r.get("description",""))).lower(), axis=1)]

        df = df.head(40)

        for _, a in df.iterrows():
            tks = a.get("tickers", [])
            tks_s = ", ".join(tks) if isinstance(tks, list) and tks else "N/A"
            with st.expander(f"🗞️ {a['published_date']} | {a['title']}"):
                st.markdown(f"**Source:** {a['source']}")
                st.markdown(f"**Tickers:** {tks_s}")
                if a.get("url"):
                    st.markdown(f"**Link:** {a['url']}")
                if a.get("description"):
                    st.markdown(a["description"][:900])

                # event impact table for mapped tickers
                if isinstance(tks, list) and tks:
                    rows = []
                    for tk in tks[:6]:
                        imp = event_impact_for_ticker(tk, a["published_date"])
                        rows.append({
                            "Ticker": tk,
                            "1D reaction %": (imp["r1d"]*100) if pd.notna(imp["r1d"]) else np.nan,
                            "3D reaction %": (imp["r3d"]*100) if pd.notna(imp["r3d"]) else np.nan
                        })
                    if rows:
                        st.markdown("**Event impact (simple):**")
                        st.dataframe(pd.DataFrame(rows).style.format({"1D reaction %":"{:.2f}", "3D reaction %":"{:.2f}"}), width="stretch")

# ============================================================
# TAB 5: Sector Index + Heatmap
# ============================================================
with tab5:
    st.markdown("### ⚛️ Quantum Sector Index")
    st.markdown("Equal-weight index for your selected universe + return heatmap.")

    uni_mode = st.selectbox("Index universe", ["Full", "Pure-play"], index=0, key="idx_uni")
    tickers_idx = sorted(list(UNIVERSE.keys())) if uni_mode == "Full" else sorted(list(PURE_PLAY))
    days_idx = st.slider("Index window (days)", 90, 720, 365, 30, key="idx_days")

    rets_i = get_returns_matrix(tickers_idx, days=days_idx)
    if rets_i.empty:
        st.info("Not enough data for index yet.")
    else:
        # equal-weight index returns
        idx_ret = rets_i.mean(axis=1).fillna(0.0)
        idx_eq = (1.0 + idx_ret).cumprod()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Index CAGR", f"{annualize_return(idx_ret)*100:.2f}%")
        c2.metric("Index Vol", f"{annualize_vol(idx_ret)*100:.2f}%")
        c3.metric("Index Sharpe", f"{sharpe(idx_ret):.2f}")
        c4.metric("Index MaxDD", f"{max_drawdown(idx_eq)*100:.2f}%")

        st.markdown("#### 📈 Sector Index Equity Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=idx_eq.index, y=idx_eq.values, name="Quantum Index"))
        fig.update_layout(height=420, xaxis_title="Date", yaxis_title="Growth of $1")
        st.plotly_chart(fig, width="stretch")

        st.markdown("#### 🧊 Return Heatmap (daily)")
        # heatmap on last ~60 days for readability
        hm = rets_i.tail(60).copy()
        hm = hm * 100.0
        fig2 = px.imshow(
            hm.T,
            aspect="auto",
            labels=dict(x="Date", y="Ticker", color="Daily %"),
        )
        fig2.update_layout(height=520)
        st.plotly_chart(fig2, width="stretch")

# Footer
st.markdown("---")
st.caption("Quantum Tracker • Streamlit Cloud-ready • Data: RSS + Yahoo Finance (yfinance) • Times in UTC")
