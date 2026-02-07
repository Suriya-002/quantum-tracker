import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Configuration
    API_PREFIX = '/api/v1'
    
    # CORS
    CORS_ORIGINS = ['http://localhost:5173', 'http://localhost:3000']
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'quantum_tracker.log'
    
    # Database
    DATABASE_PATH = os.getenv('DATABASE_PATH', 'src/data/quantum_tracker.db')
    
    # RSS Feeds
    RSS_FEEDS_FILE = os.getenv('RSS_FEEDS_FILE', 'src/data/rss_feeds.json')
    
    # Quantum Computing Companies
    QUANTUM_TICKERS = ['IONQ', 'RGTI', 'QBTS', 'IBM', 'GOOGL', 'MSFT', 'AMZN', 'INTC', 'HON']
    
    # Anthropic API
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    
    # Scheduler intervals (seconds)
    RSS_SCRAPE_INTERVAL = 3600
    RSS_UPDATE_INTERVAL = 3600
    STOCK_UPDATE_INTERVAL = 300
    STOCK_FETCH_INTERVAL = 300
