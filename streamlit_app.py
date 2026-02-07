import streamlit as st
import pandas as pd
import sqlite3
import os
import feedparser
import yfinance as yf
from datetime import datetime, timedelta
import requests
from anthropic import Anthropic

# Page config
st.set_page_config(page_title="Quantum Tracker", page_icon="⚛️", layout="wide")

st.title("⚛️ Quantum Tracker")
st.markdown("Real-time Quantum Computing Industry News & Portfolio Analysis")

# Database setup
DB_PATH = "quantum_tracker.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS news_articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT, url TEXT, source TEXT, published_date TEXT,
        description TEXT, fetched_date TEXT
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS stock_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT, company_name TEXT, price REAL, change REAL,
        change_percent REAL, volume INTEGER, timestamp TEXT
    )''')
    
    conn.commit()
    conn.close()

init_db()

# Fetch news
def fetch_news():
    feeds = [
        "https://thequantuminsider.com/feed/",
        "https://quantumcomputingreport.com/feed/"
    ]
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    for feed_url in feeds:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:5]:
                c.execute('''INSERT INTO news_articles (title, url, source, published_date, description, fetched_date)
                           VALUES (?, ?, ?, ?, ?, ?)''',
                         (entry.title, entry.link, feed.feed.title, 
                          datetime.now().isoformat(), entry.get('summary', ''),
                          datetime.now().isoformat()))
        except:
            pass
    
    conn.commit()
    conn.close()

# Fetch stocks
def fetch_stocks():
    tickers = ['IONQ', 'RGTI', 'QBTS', 'IBM', 'GOOGL', 'MSFT', 'AMZN', 'INTC', 'HON']
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period='1d')
            
            if not hist.empty:
                price = hist['Close'].iloc[-1]
                change = price - hist['Open'].iloc[-1]
                change_pct = (change / hist['Open'].iloc[-1]) * 100
                
                c.execute('''INSERT INTO stock_data (ticker, company_name, price, change, change_percent, volume, timestamp)
                           VALUES (?, ?, ?, ?, ?, ?, ?)''',
                         (ticker, info.get('shortName', ticker), price, change, change_pct,
                          int(hist['Volume'].iloc[-1]), datetime.now().isoformat()))
        except:
            pass
    
    conn.commit()
    conn.close()

# Get data
def get_news():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM news_articles ORDER BY id DESC LIMIT 20", conn)
    conn.close()
    return df

def get_stocks():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('''SELECT ticker, company_name, price, change_percent, volume, timestamp
                              FROM stock_data 
                              GROUP BY ticker 
                              HAVING MAX(timestamp)
                              ORDER BY ticker''', conn)
    conn.close()
    return df

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    
    if st.button("🔄 Fetch News"):
        with st.spinner("Fetching news..."):
            fetch_news()
            st.success("✅ News updated!")
            st.rerun()
    
    if st.button("📊 Fetch Stocks"):
        with st.spinner("Fetching stocks..."):
            fetch_stocks()
            st.success("✅ Stocks updated!")
            st.rerun()

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Stocks", "📰 News"])

with tab1:
    st.subheader("Market Overview")
    
    stocks_df = get_stocks()
    if not stocks_df.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Stocks", len(stocks_df))
        
        avg_change = stocks_df['change_percent'].mean()
        col2.metric("Avg Change", f"{avg_change:.2f}%")
        
        gainers = len(stocks_df[stocks_df['change_percent'] > 0])
        col3.metric("Gainers", gainers)

with tab2:
    st.subheader("📈 Quantum Computing Stocks")
    
    stocks_df = get_stocks()
    if not stocks_df.empty:
        st.dataframe(stocks_df, use_container_width=True, hide_index=True)
        
        # Chart
        chart_data = stocks_df[['ticker', 'change_percent']].set_index('ticker')
        st.bar_chart(chart_data)
    else:
        st.info("No stock data. Click 'Fetch Stocks' in sidebar.")

with tab3:
    st.subheader("📰 Latest News")
    
    news_df = get_news()
    if not news_df.empty:
        for _, article in news_df.iterrows():
            st.markdown(f"**[{article['title']}]({article['url']})**")
            st.caption(f"{article['source']} - {article['published_date'][:10]}")
            st.divider()
    else:
        st.info("No news. Click 'Fetch News' in sidebar.")

# Auto-fetch on first load
if st.session_state.get('first_run', True):
    with st.spinner("Initializing data..."):
        try:
            fetch_news()
            fetch_stocks()
        except:
            pass
    st.session_state.first_run = False
