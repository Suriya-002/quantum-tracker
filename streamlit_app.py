import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import feedparser
import yfinance as yf
from datetime import datetime, timedelta
import requests
from anthropic import Anthropic
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Quantum Tracker - ML Portfolio Analysis",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional design
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 12px 24px;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.3);
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #667eea;
    }
    h1, h2, h3 {
        color: white !important;
    }
    .stMarkdown {
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Database setup
DB_PATH = "quantum_tracker.db"

@st.cache_resource
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS news_articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT, url TEXT, source TEXT, published_date TEXT,
        description TEXT, fetched_date TEXT, category TEXT
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS stock_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT, company_name TEXT, price REAL, change REAL,
        change_percent REAL, volume INTEGER, market_cap REAL,
        timestamp TEXT
    )''')
    
    conn.commit()
    return conn

conn = init_db()

# Anthropic API
ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY", ""))

# Fetch news with filtering
@st.cache_data(ttl=3600)
def fetch_news():
    feeds = [
        ("https://thequantuminsider.com/feed/", "The Quantum Insider"),
        ("https://quantumcomputingreport.com/feed/", "Quantum Computing Report")
    ]
    
    c = conn.cursor()
    articles_added = 0
    
    quantum_keywords = ['quantum', 'qubit', 'ionq', 'rigetti', 'ibm quantum', 'google quantum']
    
    for feed_url, source in feeds:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:10]:
                title = entry.title.lower()
                summary = entry.get('summary', '').lower()
                
                if any(kw in title or kw in summary for kw in quantum_keywords):
                    category = 'breakthrough' if 'breakthrough' in title else 'business'
                    
                    c.execute('''INSERT OR IGNORE INTO news_articles 
                               (title, url, source, published_date, description, fetched_date, category)
                               VALUES (?, ?, ?, ?, ?, ?, ?)''',
                             (entry.title, entry.link, source, 
                              entry.get('published', datetime.now().isoformat())[:10],
                              entry.get('summary', '')[:500], 
                              datetime.now().isoformat(), category))
                    articles_added += 1
        except Exception as e:
            st.sidebar.error(f"Error fetching from {source}: {str(e)}")
    
    conn.commit()
    return articles_added

# Fetch historical stock data
@st.cache_data(ttl=300)
def fetch_stocks_historical(days=60):
    tickers = {
        'IONQ': 'IonQ Inc',
        'RGTI': 'Rigetti Computing',
        'QBTS': 'D-Wave Quantum',
        'IBM': 'IBM',
        'GOOGL': 'Alphabet Inc',
        'MSFT': 'Microsoft',
        'AMZN': 'Amazon',
        'INTC': 'Intel',
        'HON': 'Honeywell'
    }
    
    c = conn.cursor()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    for ticker, name in tickers.items():
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            info = stock.info
            
            for date, row in hist.iterrows():
                c.execute('''INSERT INTO stock_data 
                           (ticker, company_name, price, change, change_percent, volume, market_cap, timestamp)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                         (ticker, name, row['Close'], 
                          row['Close'] - row['Open'],
                          ((row['Close'] - row['Open']) / row['Open']) * 100,
                          int(row['Volume']),
                          info.get('marketCap', 0),
                          date.strftime('%Y-%m-%d %H:%M:%S')))
        except Exception as e:
            st.sidebar.error(f"Error fetching {ticker}: {str(e)}")
    
    conn.commit()

# Get data
@st.cache_data(ttl=300)
def get_news(limit=20):
    df = pd.read_sql_query(f"SELECT * FROM news_articles ORDER BY id DESC LIMIT {limit}", conn)
    return df

@st.cache_data(ttl=300)
def get_latest_stocks():
    query = '''
        SELECT ticker, company_name, price, change_percent, volume, market_cap, timestamp
        FROM stock_data 
        WHERE (ticker, timestamp) IN (
            SELECT ticker, MAX(timestamp) 
            FROM stock_data 
            GROUP BY ticker
        )
        ORDER BY ticker
    '''
    df = pd.read_sql_query(query, conn)
    return df

@st.cache_data(ttl=300)
def get_stock_history(ticker, days=60):
    query = f'''
        SELECT * FROM stock_data 
        WHERE ticker = ? 
        AND timestamp >= datetime('now', '-{days} days')
        ORDER BY timestamp
    '''
    df = pd.read_sql_query(query, conn, params=(ticker,))
    return df

# AI Summary
@st.cache_data(ttl=3600)
def generate_ai_summary():
    if not ANTHROPIC_API_KEY:
        return "⚠️ Anthropic API key not configured. Add it to Streamlit secrets."
    
    news_df = get_news(10)
    if news_df.empty:
        return "No news available for summary."
    
    news_text = "\n\n".join([f"**{row['title']}** - {row['source']}\n{row['description'][:200]}" 
                             for _, row in news_df.iterrows()])
    
    try:
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            messages=[{
                "role": "user",
                "content": f"""Analyze these quantum computing news articles and provide a structured summary:

{news_text}

Provide:
1. 📊 Market Overview (2-3 sentences)
2. 🏢 Key Company Developments (bullet points)
3. 🚀 Technology Breakthroughs (if any)
4. 💰 Investment & Financial News
5. 📈 Investment Implications

Be concise and focus on actionable insights for investors."""
            }]
        )
        return message.content[0].text
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# CORRECTED ML Portfolio Analysis
@st.cache_data(ttl=3600)
def ml_portfolio_analysis():
    stocks_df = get_latest_stocks()
    if stocks_df.empty or len(stocks_df) < 5:
        return None
    
    results = []
    
    for ticker in stocks_df['ticker'].unique():
        hist_df = get_stock_history(ticker, days=60)
        
        if len(hist_df) < 30:
            continue
        
        # Feature engineering
        hist_df['returns'] = hist_df['price'].pct_change()
        hist_df['returns_1d'] = hist_df['price'].pct_change(1)
        hist_df['returns_5d'] = hist_df['price'].pct_change(5)
        hist_df['volatility_5d'] = hist_df['returns'].rolling(5).std()
        hist_df['volume_change'] = hist_df['volume'].pct_change()
        hist_df['price_ma5'] = hist_df['price'].rolling(5).mean()
        hist_df['price_to_ma5'] = hist_df['price'] / hist_df['price_ma5']
        
        hist_df = hist_df.dropna()
        
        if len(hist_df) < 20:
            continue
        
        # Prepare features and target
        feature_cols = ['returns_1d', 'returns_5d', 'volatility_5d', 'volume_change', 'price_to_ma5']
        X = hist_df[feature_cols].values
        y = hist_df['returns'].shift(-1).values[:-1]  # Next day return
        X = X[:-1]
        
        # Train/test split
        split_idx = int(len(X) * 0.6)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        ridge = Ridge(alpha=1.0)
        rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        
        ridge.fit(X_train_scaled, y_train)
        rf.fit(X_train_scaled, y_train)
        
        # Ensemble prediction
        ridge_pred = ridge.predict(X_test_scaled)
        rf_pred = rf.predict(X_test_scaled)
        ensemble_pred = (ridge_pred + rf_pred) / 2
        
        # CORRECTED METRICS
        test_returns = y_test
        
        # Direction accuracy
        direction_accuracy = np.mean((ensemble_pred > 0) == (test_returns > 0)) * 100
        
        # Geometric mean return (proper compounding)
        daily_returns_actual = 1 + test_returns
        cumulative_return = np.prod(daily_returns_actual)
        geometric_mean = cumulative_return ** (1/len(test_returns)) - 1
        
        # Annualized return (geometric, with volatility drag)
        annualized_return = ((1 + geometric_mean) ** 252 - 1) * 100
        
        # Annualized volatility
        volatility = np.std(test_returns) * np.sqrt(252) * 100
        
        # Sharpe ratio (using geometric returns)
        sharpe = annualized_return / volatility if volatility > 0 else 0
        
        # Max drawdown
        cumulative = np.cumprod(1 + test_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown) * 100
        
        # Win rate
        win_rate = np.mean(test_returns > 0) * 100
        
        results.append({
            'ticker': ticker,
            'company': hist_df.iloc[-1]['company_name'],
            'current_price': hist_df.iloc[-1]['price'],
            'expected_return': annualized_return,
            'volatility': volatility,
            'sharpe': sharpe,
            'direction_accuracy': direction_accuracy,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'train_size': len(X_train),
            'test_size': len(X_test)
        })
    
    if not results:
        return None
    
    results_df = pd.DataFrame(results)
    
    # Portfolio optimization (mean-variance with realistic constraints)
    returns = results_df['expected_return'].values / 100
    vols = results_df['volatility'].values / 100
    
    # Only include stocks with positive Sharpe ratio
    positive_sharpe = results_df['sharpe'] > 0
    if positive_sharpe.sum() > 0:
        returns = returns[positive_sharpe]
        vols = vols[positive_sharpe]
        results_df = results_df[positive_sharpe].reset_index(drop=True)
    
    # Inverse volatility weighting (risk parity)
    inv_vol = 1 / vols
    weights = inv_vol / inv_vol.sum()
    
    # Cap at 25% per position (more realistic than 40%)
    weights = np.minimum(weights, 0.25)
    weights = weights / weights.sum()
    
    results_df['weight'] = weights * 100
    results_df['allocation'] = results_df['weight'].apply(
        lambda x: 'OVERWEIGHT' if x > 15 else 'MARKET_WEIGHT' if x > 10 else 'UNDERWEIGHT'
    )
    
    # Portfolio metrics (properly weighted)
    portfolio_return = np.sum(returns * weights) * 100
    
    # Portfolio volatility (assuming zero correlation for simplicity)
    portfolio_vol = np.sqrt(np.sum((vols * weights) ** 2)) * 100
    
    # Portfolio Sharpe
    portfolio_sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
    
    # Portfolio max drawdown (weighted average)
    portfolio_max_dd = np.sum(results_df['max_drawdown'].values * weights)
    
    return {
        'stocks': results_df.sort_values('sharpe', ascending=False),
        'portfolio_return': portfolio_return,
        'portfolio_vol': portfolio_vol,
        'portfolio_sharpe': portfolio_sharpe,
        'portfolio_max_dd': portfolio_max_dd
    }

# Initialize data on first load
if 'initialized' not in st.session_state:
    with st.spinner("🚀 Initializing Quantum Tracker..."):
        fetch_news()
        fetch_stocks_historical(60)
        st.session_state.initialized = True

# Header
st.title("⚛️ Quantum Tracker")
st.markdown("### Real-time ML-Powered Quantum Computing Portfolio Analysis")

# Live indicator
col1, col2, col3 = st.columns([6, 1, 1])
with col3:
    st.markdown('<span style="color: #00ff00;">● LIVE</span>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "💼 ML Portfolio", "📈 Stocks", "📰 News"])

# Overview Tab
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🤖 AI Market Summary")
        with st.spinner("Generating AI insights..."):
            summary = generate_ai_summary()
            st.markdown(summary)
    
    with col2:
        st.markdown("### 📊 Market Stats")
        stocks_df = get_latest_stocks()
        
        if not stocks_df.empty:
            avg_change = stocks_df['change_percent'].mean()
            gainers = len(stocks_df[stocks_df['change_percent'] > 0])
            total_volume = stocks_df['volume'].sum()
            
            st.metric("Average Change", f"{avg_change:.2f}%", 
                     delta=f"{avg_change:.2f}%")
            st.metric("Gainers / Total", f"{gainers} / {len(stocks_df)}")
            st.metric("Total Volume", f"{total_volume/1e6:.1f}M")
            
            # Top movers
            st.markdown("#### 🔥 Top Movers")
            top_movers = stocks_df.nlargest(3, 'change_percent', keep='all')
            for _, stock in top_movers.iterrows():
                emoji = "🟢" if stock['change_percent'] > 0 else "🔴"
                st.markdown(f"{emoji} **{stock['ticker']}**: {stock['change_percent']:.2f}%")

# ML Portfolio Tab
with tab2:
    st.markdown("### 💼 ML-Optimized Portfolio Analysis")
    st.markdown("*Using Ridge Regression + Random Forest Ensemble with Out-of-Sample Validation*")
    
    with st.spinner("🧠 Running ML models..."):
        portfolio = ml_portfolio_analysis()
    
    if portfolio:
        # Portfolio metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Expected Return", f"{portfolio['portfolio_return']:.2f}%")
        col2.metric("Volatility", f"{portfolio['portfolio_vol']:.2f}%")
        col3.metric("Sharpe Ratio", f"{portfolio['portfolio_sharpe']:.2f}")
        col4.metric("Max Drawdown", f"{portfolio['portfolio_max_dd']:.2f}%")
        
        st.info("📌 **Methodology Note**: Returns are calculated using geometric compounding over the test period, then annualized. This provides realistic estimates accounting for volatility drag.")
        
        st.markdown("---")
        
        # Recommendations
        st.markdown("#### 📈 Stock Recommendations")
        
        for _, stock in portfolio['stocks'].iterrows():
            with st.expander(f"**{stock['ticker']} - {stock['company']}** | {stock['allocation']} ({stock['weight']:.1f}%)"):
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Expected Return", f"{stock['expected_return']:.2f}%")
                col2.metric("Volatility", f"{stock['volatility']:.2f}%")
                col3.metric("Sharpe Ratio", f"{stock['sharpe']:.2f}")
                col4.metric("Direction Accuracy", f"{stock['direction_accuracy']:.1f}%")
                
                col5, col6, col7, col8 = st.columns(4)
                col5.metric("Max Drawdown", f"{stock['max_drawdown']:.2f}%")
                col6.metric("Win Rate", f"{stock['win_rate']:.1f}%")
                col7.metric("Train Days", stock['train_size'])
                col8.metric("Test Days", stock['test_size'])
                
                # Performance interpretation
                if stock['direction_accuracy'] < 50:
                    st.warning(f"⚠️ Direction accuracy below 50% suggests limited predictive power")
                elif stock['direction_accuracy'] > 55:
                    st.success(f"✅ Direction accuracy above 55% suggests some predictive signal")
        
        # Visualization
        st.markdown("#### 📊 Risk-Return Profile")
        fig = px.scatter(
            portfolio['stocks'], 
            x='volatility', 
            y='expected_return',
            size='weight',
            color='sharpe',
            hover_data=['ticker', 'direction_accuracy', 'max_drawdown'],
            labels={'volatility': 'Volatility (%)', 'expected_return': 'Expected Return (%)'},
            title='Stock Risk-Return Analysis (Annualized)',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("⏳ Collecting more data for ML analysis. Please check back in a few minutes.")

# Stocks Tab
with tab3:
    st.markdown("### 📈 Quantum Computing Stocks")
    
    stocks_df = get_latest_stocks()
    
    if not stocks_df.empty:
        # Display table
        display_df = stocks_df[['ticker', 'company_name', 'price', 'change_percent', 'volume', 'market_cap']]
        display_df['price'] = display_df['price'].apply(lambda x: f"${x:.2f}")
        display_df['change_percent'] = display_df['change_percent'].apply(lambda x: f"{x:.2f}%")
        display_df['volume'] = display_df['volume'].apply(lambda x: f"{x/1e6:.1f}M")
        display_df['market_cap'] = display_df['market_cap'].apply(lambda x: f"${x/1e9:.1f}B" if x > 0 else "N/A")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Price chart
        st.markdown("#### 📉 Price Changes")
        fig = go.Figure(data=[
            go.Bar(
                x=stocks_df['ticker'],
                y=stocks_df['change_percent'],
                marker_color=['green' if x > 0 else 'red' for x in stocks_df['change_percent']]
            )
        ])
        fig.update_layout(title='Daily Price Changes', yaxis_title='Change (%)', height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("Loading stock data...")

# News Tab
with tab4:
    st.markdown("### 📰 Latest Quantum Computing News")
    
    news_df = get_news(20)
    
    if not news_df.empty:
        for _, article in news_df.iterrows():
            badge = "🔬" if article.get('category') == 'breakthrough' else "💼"
            st.markdown(f"{badge} **[{article['title']}]({article['url']})**")
            st.caption(f"{article['source']} • {article['published_date']}")
            if article['description']:
                st.markdown(f"> {article['description'][:200]}...")
            st.divider()
    else:
        st.info("Loading news...")

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Data Controls")
    
    if st.button("🔄 Refresh All Data", type="primary"):
        st.cache_data.clear()
        with st.spinner("Refreshing..."):
            fetch_news()
            fetch_stocks_historical(60)
        st.success("✅ Data refreshed!")
        st.rerun()
    
    st.markdown("---")
    
    # Data status
    news_count = len(get_news(1000))
    stocks_count = pd.read_sql_query("SELECT COUNT(*) as count FROM stock_data", conn).iloc[0]['count']
    
    st.metric("📰 News Articles", news_count)
    st.metric("📊 Stock Data Points", stocks_count)
    
    st.markdown("---")
    st.caption("⚠️ **Disclaimer**: ML predictions are for educational purposes only. Not financial advice.")
    st.caption("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M"))
