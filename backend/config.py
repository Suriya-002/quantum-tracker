import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

class Config:
    API_PREFIX = '/api/v1'
    CORS_ORIGINS = ['*']
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'quantum_tracker.log'
    
    # Use Streamlit secrets if available, otherwise environment variables
    DATABASE_PATH = st.secrets.get('DATABASE_PATH', 'backend/src/data/quantum_tracker.db') if hasattr(st, 'secrets') else os.getenv('DATABASE_PATH', 'backend/src/data/quantum_tracker.db')
    RSS_FEEDS_FILE = st.secrets.get('RSS_FEEDS_FILE', 'backend/src/data/rss_feeds.json') if hasattr(st, 'secrets') else os.getenv('RSS_FEEDS_FILE', 'backend/src/data/rss_feeds.json')
    ANTHROPIC_API_KEY = st.secrets.get('ANTHROPIC_API_KEY', '') if hasattr(st, 'secrets') else os.getenv('ANTHROPIC_API_KEY', '')
    
    QUANTUM_TICKERS = ['IONQ', 'RGTI', 'QBTS', 'IBM', 'GOOGL', 'MSFT', 'AMZN', 'INTC', 'HON']
    RSS_SCRAPE_INTERVAL = 3600
    RSS_UPDATE_INTERVAL = 3600
    STOCK_UPDATE_INTERVAL = 300
    STOCK_FETCH_INTERVAL = 300
