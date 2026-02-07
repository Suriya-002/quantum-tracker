import streamlit as st
import pandas as pd
import sys
import os
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

# Check if database has data
@st.cache_data(ttl=60)
def check_data_exists():
    try:
        news_count = execute_query(config.DATABASE_PATH, "SELECT COUNT(*) as count FROM news_articles")
        stock_count = execute_query(config.DATABASE_PATH, "SELECT COUNT(*) as count FROM stock_data")
        return {
            'news': news_count[0]['count'] if news_count else 0,
            'stocks': stock_count[0]['count'] if stock_count else 0
        }
    except:
        return {'news': 0, 'stocks': 0}

data_status = check_data_exists()

# If no data, show initialization button
if data_status['news'] == 0 or data_status['stocks'] == 0:
    st.warning("⚠️ Database is empty. Please initialize data first!")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Fetch News", type="primary"):
            with st.spinner("Fetching news articles..."):
                try:
                    scrape_all_feeds(config.__dict__)
                    st.success("✅ News fetched successfully!")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error fetching news: {str(e)}")
    
    with col2:
        if st.button("📊 Fetch Stocks", type="primary"):
            with st.spinner("Fetching stock data..."):
                try:
                    update_all_stocks(config.__dict__)
                    st.success("✅ Stocks fetched successfully!")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error fetching stocks: {str(e)}")
    
    st.stop()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "💼 Portfolio", "📈 Stocks", "📰 News"])

# Fetch data functions
@st.cache_data(ttl=300)
def get_news():
    try:
        return execute_query(config.DATABASE_PATH, "SELECT * FROM news_articles ORDER BY published_date DESC LIMIT 20")
    except:
        return []

@st.cache_data(ttl=300)
def get_stocks():
    try:
        return execute_query(config.DATABASE_PATH, 
            "SELECT * FROM stock_data WHERE (ticker, timestamp) IN (SELECT ticker, MAX(timestamp) FROM stock_data GROUP BY ticker)")
    except:
        return []

@st.cache_data(ttl=3600)
def get_ai_summary():
    try:
        news = get_news()
        if not news:
            return "No news available for summary."
        return summarize_news_with_llm(news)
    except Exception as e:
        return f"Error generating summary: {str(e)}"

@st.cache_data(ttl=3600)
def get_portfolio():
    try:
        result = analyze_portfolio(config.DATABASE_PATH)
        return result if result else {'recommendations': [], 'summary': 'Insufficient data'}
    except Exception as e:
        st.error(f"Portfolio analysis error: {str(e)}")
        return {'recommendations': [], 'summary': 'Error analyzing portfolio'}

# Overview Tab
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 AI Market Summary")
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
                    st.write(f"**{stock['ticker']}** - ${stock['price']:.2f}")
                with col_b:
                    change_color = "🟢" if stock['change_percent'] > 0 else "🔴"
                    st.write(f"{change_color} {stock['change_percent']:.2f}%")
        else:
            st.info("No stock data available")

# Portfolio Tab
with tab2:
    st.subheader("💼 ML-Optimized Portfolio Analysis")
    portfolio = get_portfolio()
    
    if portfolio and portfolio.get('recommendations'):
        st.markdown(portfolio.get('summary', ''))
        
        if portfolio.get('portfolio_metrics'):
            col1, col2, col3 = st.columns(3)
            metrics = portfolio['portfolio_metrics']
            col1.metric("Expected Return", f"+{metrics.get('expected_return_pct', 0):.2f}%")
            col2.metric("Volatility", f"{metrics.get('volatility_pct', 0):.2f}%")
            col3.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
        
        for rec in portfolio['recommendations']:
            with st.expander(f"{rec['ticker']} - {rec['allocation']} ({rec['weight']}%)"):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Expected Return", f"{rec['expected_return']}%")
                col2.metric("Volatility", f"{rec['volatility']}%")
                col3.metric("Sharpe", f"{rec['sharpe']}")
                col4.metric("Direction Accuracy", f"{rec.get('direction_accuracy', 0)}%")
    else:
        st.info("Insufficient historical data for portfolio analysis. Please wait for more data to accumulate.")

# Stocks Tab
with tab3:
    st.subheader("📈 Quantum Computing Stocks")
    stocks = get_stocks()
    if stocks:
        df = pd.DataFrame(stocks)
        st.dataframe(df[['ticker', 'company_name', 'price', 'change_percent', 'volume']], 
                    use_container_width=True, hide_index=True)
    else:
        st.info("No stock data available")

# News Tab
with tab4:
    st.subheader("📰 Latest News")
    news = get_news()
    if news:
        for article in news[:15]:
            st.markdown(f"**[{article['title']}]({article['url']})**")
            st.caption(f"{article['source']} - {article.get('published_date', 'N/A')}")
            st.divider()
    else:
        st.info("No news articles available")

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    st.caption(f"News: {data_status['news']} articles")
    st.caption(f"Stocks: {data_status['stocks']} data points")
    
    if st.button("🔄 Refresh News"):
        with st.spinner("Fetching news..."):
            try:
                scrape_all_feeds(config.__dict__)
                st.success("✅ News updated!")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    if st.button("📊 Refresh Stocks"):
        with st.spinner("Fetching stocks..."):
            try:
                update_all_stocks(config.__dict__)
                st.success("✅ Stocks updated!")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
