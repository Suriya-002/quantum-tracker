import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import sys
sys.path.append('backend')

from backend.src.utils.database import execute_query, init_db
from backend.src.services.rss_scraper import scrape_all_feeds
from backend.src.services.stock_fetcher import update_all_stocks
from backend.src.services.sentiment_analyzer import summarize_news_with_llm
from backend.src.services.quant_analyzer import analyze_portfolio
from backend.config import Config

# Page config
st.set_page_config(page_title="Quantum Tracker", page_icon="⚛️", layout="wide")

# Initialize
config = Config()
init_db(config.DATABASE_PATH)

# Header
st.title("⚛️ Quantum Tracker")
st.markdown("Real-time Quantum Computing Industry News & Portfolio Analysis")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "💼 Portfolio", "📈 Stocks", "📰 News"])

# Fetch data
@st.cache_data(ttl=300)
def get_news():
    return execute_query(config.DATABASE_PATH, "SELECT * FROM news_articles ORDER BY published_date DESC LIMIT 20")

@st.cache_data(ttl=300)
def get_stocks():
    return execute_query(config.DATABASE_PATH, 
        "SELECT * FROM stock_data WHERE (ticker, timestamp) IN (SELECT ticker, MAX(timestamp) FROM stock_data GROUP BY ticker)")

@st.cache_data(ttl=3600)
def get_ai_summary():
    news = get_news()
    return summarize_news_with_llm(news)

@st.cache_data(ttl=3600)
def get_portfolio():
    return analyze_portfolio(config.DATABASE_PATH)

# Overview Tab
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 AI Market Summary")
        with st.spinner("Generating AI summary..."):
            summary = get_ai_summary()
            st.markdown(summary)
    
    with col2:
        st.subheader("📈 Top Stock Movers")
        stocks = get_stocks()
        if stocks:
            df = pd.DataFrame(stocks)
            df = df.sort_values('change_percent', ascending=False, key=abs).head(5)
            for _, stock in df.iterrows():
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"**{stock['ticker']}** - {stock['company_name']}")
                with col_b:
                    change_color = "🟢" if stock['change_percent'] > 0 else "🔴"
                    st.write(f"{change_color} {stock['change_percent']:.2f}%")

# Portfolio Tab
with tab2:
    st.subheader("💼 ML-Optimized Portfolio Analysis")
    portfolio = get_portfolio()
    
    if portfolio and portfolio.get('recommendations'):
        st.markdown(portfolio['summary'])
        
        if portfolio.get('portfolio_metrics'):
            col1, col2, col3 = st.columns(3)
            col1.metric("Expected Return", f"+{portfolio['portfolio_metrics']['expected_return_pct']}%")
            col2.metric("Volatility", f"{portfolio['portfolio_metrics']['volatility_pct']}%")
            col3.metric("Sharpe Ratio", f"{portfolio['portfolio_metrics']['sharpe_ratio']}")
        
        for rec in portfolio['recommendations']:
            with st.expander(f"{rec['ticker']} - {rec['allocation']} ({rec['weight']}%)"):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Expected Return", f"{rec['expected_return']}%")
                col2.metric("Volatility", f"{rec['volatility']}%")
                col3.metric("Sharpe", f"{rec['sharpe']}")
                col4.metric("Direction Accuracy", f"{rec['direction_accuracy']}%")
    else:
        st.info("Insufficient data for portfolio analysis")

# Stocks Tab
with tab3:
    st.subheader("📈 Quantum Computing Stocks")
    stocks = get_stocks()
    if stocks:
        df = pd.DataFrame(stocks)
        st.dataframe(df[['ticker', 'price', 'change_percent', 'volume']], use_container_width=True)

# News Tab
with tab4:
    st.subheader("📰 Latest News")
    news = get_news()
    if news:
        for article in news[:10]:
            st.markdown(f"**[{article['title']}]({article['url']})**")
            st.caption(f"{article['source']} - {article['published_date']}")
            st.divider()

# Sidebar - Update Data
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🔄 Update News"):
        with st.spinner("Fetching news..."):
            scrape_all_feeds(config.__dict__)
            st.success("News updated!")
            st.cache_data.clear()
    
    if st.button("📊 Update Stocks"):
        with st.spinner("Fetching stocks..."):
            update_all_stocks(config.__dict__)
            st.success("Stocks updated!")
            st.cache_data.clear()
