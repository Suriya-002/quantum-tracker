import streamlit as st
import sys
import os
import json
from datetime import datetime

sys.path.append('backend')

from backend.src.utils.database import execute_query, init_db
from backend.src.services.rss_scraper import scrape_all_feeds
from backend.src.services.stock_fetcher import update_all_stocks
from backend.src.services.sentiment_analyzer import summarize_news_with_llm
from backend.src.services.quant_analyzer import analyze_portfolio
from backend.config import Config

# Initialize
config = Config()
init_db(config.DATABASE_PATH)

# Initialize data on startup
@st.cache_resource
def initialize_data():
    try:
        news_count = execute_query(config.DATABASE_PATH, "SELECT COUNT(*) as count FROM news_articles")
        if not news_count or news_count[0]['count'] == 0:
            scrape_all_feeds(config.__dict__)
        
        stock_count = execute_query(config.DATABASE_PATH, "SELECT COUNT(*) as count FROM stock_data")
        if not stock_count or stock_count[0]['count'] == 0:
            update_all_stocks(config.__dict__)
        
        return True
    except Exception as e:
        st.error(f"Init error: {str(e)}")
        return False

initialize_data()

# API Endpoints
st.set_page_config(page_title="Quantum Tracker API", layout="wide")

st.title("🔌 Quantum Tracker API Backend")
st.markdown("Backend API running on Streamlit for React frontend")

# Show API endpoints
st.subheader("Available Endpoints:")
st.code("""
GET /api/v1/news - Get latest news articles
GET /api/v1/stocks - Get current stock data  
GET /api/v1/analysis/news-summary - Get AI news summary
GET /api/v1/analysis/portfolio - Get ML portfolio analysis
""")

# API endpoint selector
endpoint = st.selectbox("Test Endpoint:", 
    ["news", "stocks", "news-summary", "portfolio"])

if st.button("🚀 Test Endpoint"):
    with st.spinner(f"Fetching {endpoint}..."):
        try:
            if endpoint == "news":
                data = execute_query(config.DATABASE_PATH, 
                    "SELECT * FROM news_articles ORDER BY published_date DESC LIMIT 20")
                st.json({"success": True, "count": len(data), "data": data})
            
            elif endpoint == "stocks":
                data = execute_query(config.DATABASE_PATH,
                    "SELECT * FROM stock_data WHERE (ticker, timestamp) IN (SELECT ticker, MAX(timestamp) FROM stock_data GROUP BY ticker)")
                st.json({"success": True, "count": len(data), "data": data})
            
            elif endpoint == "news-summary":
                news = execute_query(config.DATABASE_PATH, 
                    "SELECT * FROM news_articles ORDER BY published_date DESC LIMIT 10")
                summary = summarize_news_with_llm(news)
                st.json({"success": True, "data": {"summary": summary}})
            
            elif endpoint == "portfolio":
                result = analyze_portfolio(config.DATABASE_PATH)
                st.json({"success": True, "data": result})
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.json({"success": False, "error": str(e)})

# Data status
col1, col2 = st.columns(2)
with col1:
    if st.button("🔄 Update News"):
        with st.spinner("Fetching news..."):
            scrape_all_feeds(config.__dict__)
            st.success("✅ News updated!")

with col2:
    if st.button("📊 Update Stocks"):
        with st.spinner("Fetching stocks..."):
            update_all_stocks(config.__dict__)
            st.success("✅ Stocks updated!")

# Show data counts
try:
    news_count = execute_query(config.DATABASE_PATH, "SELECT COUNT(*) as count FROM news_articles")[0]['count']
    stock_count = execute_query(config.DATABASE_PATH, "SELECT COUNT(*) as count FROM stock_data")[0]['count']
    st.info(f"📰 {news_count} news articles | 📈 {stock_count} stock data points")
except:
    st.warning("Database not initialized")
