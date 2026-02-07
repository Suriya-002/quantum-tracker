import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    API_PREFIX = '/api/v1'
    CORS_ORIGINS = ['http://localhost:5173', 'https://quantum-tracker.onrender.com']
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'quantum_tracker.log'
    DATABASE_PATH = os.getenv('DATABASE_PATH', 'src/data/quantum_tracker.db')
    RSS_FEEDS_FILE = os.getenv('RSS_FEEDS_FILE', 'backend/src/data/rss_feeds.json')
    QUANTUM_TICKERS = ['IONQ', 'RGTI', 'QBTS', 'IBM', 'GOOGL', 'MSFT', 'AMZN', 'INTC', 'HON']
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    RSS_SCRAPE_INTERVAL = 3600
    RSS_UPDATE_INTERVAL = 3600
    STOCK_UPDATE_INTERVAL = 300
    STOCK_FETCH_INTERVAL = 300

