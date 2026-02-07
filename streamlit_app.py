import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import feedparser
import yfinance as yf
from datetime import datetime, timedelta
from anthropic import Anthropic
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px

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
# Custom CSS (professional)
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
    div[data-testid="stMetricValue"] { font-size: 28px; color: #667eea; }
    h1, h2, h3 { color: white !important; }
    .stMarkdown { color: white; }
</style>
""", unsafe_allow_html=True)

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

    # News table
    c.execute("""
        CREATE TABLE IF NOT EXISTS news_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            url TEXT UNIQUE,
            source TEXT,
            published_date TEXT,
            description TEXT,
            fetched_date TEXT,
            category TEXT
        )
    """)

    # Stock table (prevent duplicates)
    c.execute("""
        CREATE TABLE IF NOT EXISTS stock_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            company_name TEXT,
            price REAL,
            change REAL,
            change_percent REAL,
            volume INTEGER,
            market_cap REAL,
            timestamp TEXT,
            UNIQUE(ticker, timestamp)
        )
    """)

    conn.commit()
    return conn

conn = init_db()
st.sidebar.caption(f"DB: {DB_PATH}")

# ----------------------------
# Secrets / API keys
# ----------------------------
ANTHROPIC_API_KEY = ""
try:
    ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "")
except Exception:
    ANTHROPIC_API_KEY = ""

if not ANTHROPIC_API_KEY:
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ----------------------------
# News fetch (filtered)
# ----------------------------
@st.cache_data(ttl=3600)
def fetch_news():
    feeds = [
        ("https://thequantuminsider.com/feed/", "The Quantum Insider"),
        ("https://quantumcomputingreport.com/feed/", "Quantum Computing Report"),
    ]

    c = conn.cursor()
    articles_added = 0

    quantum_keywords = [
        "quantum", "qubit", "ionq", "rigetti", "ibm quantum", "google quantum",
        "d-wave", "dwave", "quantinuum", "psi quantum", "neutral atom", "superconducting"
    ]

    for feed_url, source in feeds:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:15]:
                title = (entry.get("title") or "").lower()
                summary = (entry.get("summary") or "").lower()

                if any(kw in title or kw in summary for kw in quantum_keywords):
                    category = "breakthrough" if "breakthrough" in title else "business"
                    published = entry.get("published") or datetime.utcnow().isoformat()

                    c.execute("""
                        INSERT OR IGNORE INTO news_articles
                        (title, url, source, published_date, description, fetched_date, category)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        entry.get("title", ""),
                        entry.get("link", ""),
                        source,
                        str(published)[:10],
                        (entry.get("summary") or "")[:600],
                        datetime.utcnow().isoformat(),
                        category
                    ))
                    articles_added += 1
        except Exception as e:
            st.sidebar.error(f"News fetch error ({source}): {e}")

    conn.commit()
    return articles_added

# ----------------------------
# Stocks fetch (historical)
# ----------------------------
@st.cache_data(ttl=900)
def fetch_stocks_historical(days=60):
    tickers = {
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

    c = conn.cursor()
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    for ticker, name in tickers.items():
        try:
            stock = yf.Ticker(ticker)

            # history can be empty sometimes; avoid crashing
            hist = stock.history(start=start_date, end=end_date, auto_adjust=False)
            if hist is None or hist.empty:
                continue

            # market cap sometimes fails; keep safe
            market_cap = 0
            try:
                info = stock.get_info()
                market_cap = info.get("marketCap", 0) or 0
            except Exception:
                market_cap = 0

            for date, row in hist.iterrows():
                open_p = float(row.get("Open", np.nan))
                close_p = float(row.get("Close", np.nan))
                vol = int(row.get("Volume", 0) or 0)

                if not np.isfinite(open_p) or not np.isfinite(close_p) or open_p == 0:
                    chg = 0.0
                    chg_pct = 0.0
                else:
                    chg = close_p - open_p
                    chg_pct = (chg / open_p) * 100.0

                ts = date.strftime("%Y-%m-%d %H:%M:%S")

                c.execute("""
                    INSERT OR IGNORE INTO stock_data
                    (ticker, company_name, price, change, change_percent, volume, market_cap, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (ticker, name, close_p, chg, chg_pct, vol, float(market_cap), ts))

        except Exception as e:
            st.sidebar.error(f"Stock fetch error ({ticker}): {e}")

    conn.commit()
    return True

# ----------------------------
# Queries
# ----------------------------
@st.cache_data(ttl=300)
def get_news(limit=20):
    return pd.read_sql_query(
        "SELECT * FROM news_articles ORDER BY id DESC LIMIT ?",
        conn,
        params=(int(limit),)
    )

@st.cache_data(ttl=300)
def get_latest_stocks():
    # Get latest timestamp row per ticker
    query = """
        SELECT s.ticker, s.company_name, s.price, s.change_percent, s.volume, s.market_cap, s.timestamp
        FROM stock_data s
        JOIN (
            SELECT ticker, MAX(timestamp) AS max_ts
            FROM stock_data
            GROUP BY ticker
        ) t
        ON s.ticker = t.ticker AND s.timestamp = t.max_ts
        ORDER BY s.ticker
    """
    return pd.read_sql_query(query, conn)

@st.cache_data(ttl=300)
def get_stock_history(ticker, days=60):
    query = """
        SELECT * FROM stock_data
        WHERE ticker = ?
          AND timestamp >= datetime('now', ?)
        ORDER BY timestamp
    """
    return pd.read_sql_query(query, conn, params=(ticker, f"-{int(days)} days"))

# ----------------------------
# AI Summary
# ----------------------------
@st.cache_data(ttl=3600)
def generate_ai_summary():
    if not ANTHROPIC_API_KEY:
        return "⚠️ Anthropic API key not configured. Add ANTHROPIC_API_KEY in Streamlit secrets or environment."

    news_df = get_news(10)
    if news_df.empty:
        return "No news available for summary."

    news_text = "\n\n".join(
        [f"{row['title']} - {row['source']}\n{(row['description'] or '')[:240]}"
         for _, row in news_df.iterrows()]
    )

    try:
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=700,
            messages=[{
                "role": "user",
                "content": (
                    "Analyze these quantum computing news articles and provide a structured investor summary.\n\n"
                    f"{news_text}\n\n"
                    "Provide:\n"
                    "1) Market Overview (2-3 sentences)\n"
                    "2) Key Company Developments (bullets)\n"
                    "3) Technology Breakthroughs (if any)\n"
                    "4) Investment & Financial News\n"
                    "5) Investment Implications\n"
                    "Be concise and actionable."
                )
            }]
        )
        return msg.content[0].text
    except Exception as e:
        return f"Error generating summary: {e}"

# ----------------------------
# ML Portfolio Analysis
# ----------------------------
@st.cache_data(ttl=3600)
def ml_portfolio_analysis():
    stocks_df = get_latest_stocks()
    if stocks_df.empty or len(stocks_df) < 5:
        return None

    results = []
    for ticker in stocks_df["ticker"].unique():
        hist_df = get_stock_history(ticker, days=90)
        if hist_df is None or hist_df.empty or len(hist_df) < 35:
            continue

        # Feature engineering
        hist_df = hist_df.copy()
        hist_df["returns"] = hist_df["price"].pct_change()
        hist_df["returns_1d"] = hist_df["price"].pct_change(1)
        hist_df["returns_5d"] = hist_df["price"].pct_change(5)
        hist_df["volatility_5d"] = hist_df["returns"].rolling(5).std()
        hist_df["volume_change"] = hist_df["volume"].pct_change()
        hist_df["price_ma5"] = hist_df["price"].rolling(5).mean()
        hist_df["price_to_ma5"] = hist_df["price"] / hist_df["price_ma5"]

        hist_df = hist_df.dropna()
        if len(hist_df) < 25:
            continue

        feature_cols = ["returns_1d", "returns_5d", "volatility_5d", "volume_change", "price_to_ma5"]
        X = hist_df[feature_cols].values
        y = hist_df["returns"].shift(-1).values

        # Drop last target NaN
        X = X[:-1]
        y = y[:-1]

        if len(X) < 20:
            continue

        split_idx = int(len(X) * 0.6)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        ridge = Ridge(alpha=1.0)
        rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)

        ridge.fit(X_train_s, y_train)
        rf.fit(X_train_s, y_train)

        ridge_pred = ridge.predict(X_test_s)
        rf_pred = rf.predict(X_test_s)
        ensemble_pred = (ridge_pred + rf_pred) / 2.0

        direction_accuracy = float(np.mean((ensemble_pred > 0) == (y_test > 0)) * 100.0)

        exp_ret = float(np.mean(ensemble_pred) * 252.0)       # annualized expected return
        vol = float(np.std(y_test) * np.sqrt(252.0))          # annualized vol
        sharpe = float(exp_ret / vol) if vol > 0 else 0.0

        results.append({
            "ticker": ticker,
            "company": hist_df.iloc[-1]["company_name"],
            "current_price": float(hist_df.iloc[-1]["price"]),
            "expected_return": exp_ret * 100.0,
            "volatility": vol * 100.0,
            "sharpe": sharpe,
            "direction_accuracy": direction_accuracy,
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
        })

    if not results:
        return None

    results_df = pd.DataFrame(results).sort_values("sharpe", ascending=False)

    # Simple inverse-vol weights with cap
    vols = results_df["volatility"].values / 100.0
    vols = np.where(vols <= 0, np.nan, vols)
    inv_vol = 1.0 / vols
    inv_vol = np.nan_to_num(inv_vol, nan=0.0, posinf=0.0, neginf=0.0)

    if inv_vol.sum() == 0:
        weights = np.ones(len(results_df)) / len(results_df)
    else:
        weights = inv_vol / inv_vol.sum()

    weights = np.minimum(weights, 0.40)
    weights = weights / weights.sum()

    results_df["weight"] = weights * 100.0
    results_df["allocation"] = results_df["weight"].apply(
        lambda x: "OVERWEIGHT" if x > 15 else "MARKET_WEIGHT" if x > 10 else "UNDERWEIGHT" if x > 5 else "AVOID"
    )

    returns = results_df["expected_return"].values / 100.0
    portfolio_return = float(np.sum(returns * weights) * 100.0)
    portfolio_vol = float(np.sqrt(np.sum((vols * weights) ** 2)) * 100.0)
    portfolio_sharpe = float(portfolio_return / portfolio_vol) if portfolio_vol > 0 else 0.0

    return {
        "stocks": results_df,
        "portfolio_return": portfolio_return,
        "portfolio_vol": portfolio_vol,
        "portfolio_sharpe": portfolio_sharpe
    }

# ----------------------------
# Initialize once per session
# ----------------------------
if "initialized" not in st.session_state:
    with st.spinner("🚀 Initializing Quantum Tracker..."):
        fetch_news()
        fetch_stocks_historical(90)
        st.session_state.initialized = True

# ----------------------------
# Header
# ----------------------------
st.title("⚛️ Quantum Tracker")
st.markdown("### Real-time ML-Powered Quantum Computing Portfolio Analysis")

colA, colB, colC = st.columns([6, 1, 1])
with colC:
    st.markdown('<span style="color: #00ff00;">● LIVE</span>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "💼 ML Portfolio", "📈 Stocks", "📰 News"])

# ----------------------------
# Overview tab
# ----------------------------
with tab1:
    c1, c2 = st.columns([2, 1])

    with c1:
        st.markdown("### 🤖 AI Market Summary")
        with st.spinner("Generating AI insights..."):
            summary = generate_ai_summary()
        st.markdown(summary)

    with c2:
        st.markdown("### 📊 Market Stats")
        stocks_df = get_latest_stocks()

        if not stocks_df.empty:
            avg_change = float(stocks_df["change_percent"].mean())
            gainers = int((stocks_df["change_percent"] > 0).sum())
            total_volume = int(stocks_df["volume"].sum())

            st.metric("Average Change", f"{avg_change:.2f}%", delta=f"{avg_change:.2f}%")
            st.metric("Gainers / Total", f"{gainers} / {len(stocks_df)}")
            st.metric("Total Volume", f"{total_volume/1e6:.1f}M")

            st.markdown("#### 🔥 Top Movers")
            top_movers = stocks_df.nlargest(3, "change_percent", keep="all")
            for _, stock in top_movers.iterrows():
                emoji = "🟢" if stock["change_percent"] > 0 else "🔴"
                st.markdown(f"{emoji} **{stock['ticker']}**: {float(stock['change_percent']):.2f}%")
        else:
            st.info("Loading stock data...")

# ----------------------------
# ML Portfolio tab
# ----------------------------
with tab2:
    st.markdown("### 💼 ML-Optimized Portfolio Analysis")
    st.markdown("*Ridge Regression + Random Forest Ensemble with out-of-sample evaluation*")

    with st.spinner("🧠 Running ML models..."):
        portfolio = ml_portfolio_analysis()

    if portfolio:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Expected Return", f"{portfolio['portfolio_return']:.2f}%")
        k2.metric("Volatility", f"{portfolio['portfolio_vol']:.2f}%")
        k3.metric("Sharpe Ratio", f"{portfolio['portfolio_sharpe']:.2f}")
        k4.metric("Stocks Analyzed", len(portfolio["stocks"]))

        st.markdown("---")
        st.markdown("#### 📈 Stock Recommendations")

        for _, stock in portfolio["stocks"].iterrows():
            with st.expander(f"**{stock['ticker']} - {stock['company']}** | {stock['allocation']} ({stock['weight']:.1f}%)"):
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Expected Return", f"{float(stock['expected_return']):.2f}%")
                m2.metric("Volatility", f"{float(stock['volatility']):.2f}%")
                m3.metric("Sharpe Ratio", f"{float(stock['sharpe']):.2f}")
                m4.metric("Direction Accuracy", f"{float(stock['direction_accuracy']):.1f}%")
                st.caption(f"Training: {int(stock['train_size'])} days | Testing: {int(stock['test_size'])} days")

        st.markdown("#### 📊 Risk-Return Profile")
        fig = px.scatter(
            portfolio["stocks"],
            x="volatility",
            y="expected_return",
            size="weight",
            color="sharpe",
            hover_data=["ticker", "direction_accuracy"],
            labels={"volatility": "Volatility (%)", "expected_return": "Expected Return (%)"},
            title="Stock Risk-Return Analysis"
        )
        # FIX: Streamlit deprecation
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("⏳ Collecting more data for ML analysis. Refresh in a few minutes.")

# ----------------------------
# Stocks tab
# ----------------------------
with tab3:
    st.markdown("### 📈 Quantum Computing Stocks")

    stocks_df = get_latest_stocks()
    if not stocks_df.empty:
        display_df = stocks_df[["ticker", "company_name", "price", "change_percent", "volume", "market_cap"]].copy()
        display_df["price"] = display_df["price"].apply(lambda x: f"${float(x):.2f}")
        display_df["change_percent"] = display_df["change_percent"].apply(lambda x: f"{float(x):.2f}%")
        display_df["volume"] = display_df["volume"].apply(lambda x: f"{int(x)/1e6:.1f}M")
        display_df["market_cap"] = display_df["market_cap"].apply(lambda x: f"${float(x)/1e9:.1f}B" if float(x) > 0 else "N/A")

        # FIX: Streamlit deprecation
        st.dataframe(display_df, width="stretch", hide_index=True)

        st.markdown("#### 📉 Price Changes")
        fig2 = go.Figure(data=[
            go.Bar(
                x=stocks_df["ticker"],
                y=stocks_df["change_percent"]
            )
        ])
        fig2.update_layout(title="Daily Price Changes", yaxis_title="Change (%)", height=420)
        st.plotly_chart(fig2, width="stretch")
    else:
        st.info("Loading stock data...")

# ----------------------------
# News tab
# ----------------------------
with tab4:
    st.markdown("### 📰 Latest Quantum Computing News")
    news_df = get_news(20)

    if not news_df.empty:
        for _, a in news_df.iterrows():
            badge = "🔬" if a.get("category") == "breakthrough" else "💼"
            st.markdown(f"{badge} **[{a['title']}]({a['url']})**")
            st.caption(f"{a['source']} • {a['published_date']}")
            if a.get("description"):
                st.markdown(f"> {(a['description'] or '')[:220]}...")
            st.divider()
    else:
        st.info("Loading news...")

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown("### ⚙️ Data Controls")

    if st.button("🔄 Refresh All Data", type="primary"):
        st.cache_data.clear()
        with st.spinner("Refreshing..."):
            fetch_news()
            fetch_stocks_historical(90)
        st.success("✅ Data refreshed!")
        st.rerun()

    st.markdown("---")

    news_count = len(get_news(1000))
    stocks_count = int(pd.read_sql_query("SELECT COUNT(*) AS count FROM stock_data", conn).iloc[0]["count"])

    st.metric("📰 News Articles", news_count)
    st.metric("📊 Stock Data Points", stocks_count)

    st.markdown("---")
    st.caption("Last updated (UTC): " + datetime.utcnow().strftime("%Y-%m-%d %H:%M"))
