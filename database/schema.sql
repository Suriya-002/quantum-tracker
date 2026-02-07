-- Quantum Tracker Database Schema
-- SQLite Database Schema

-- News Articles Table
CREATE TABLE IF NOT EXISTS news_articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    content TEXT,
    url TEXT UNIQUE NOT NULL,
    source TEXT,
    category TEXT,
    published_date DATETIME,
    fetched_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    sentiment_score REAL,
    sentiment_label TEXT
);

CREATE INDEX idx_news_published ON news_articles(published_date DESC);
CREATE INDEX idx_news_category ON news_articles(category);
CREATE INDEX idx_news_source ON news_articles(source);

-- Stock Data Table
CREATE TABLE IF NOT EXISTS stock_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    company_name TEXT,
    price REAL,
    change REAL,
    change_percent REAL,
    volume INTEGER,
    market_cap REAL,
    pe_ratio REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, timestamp)
);

CREATE INDEX idx_stock_ticker ON stock_data(ticker);
CREATE INDEX idx_stock_timestamp ON stock_data(timestamp DESC);

-- Companies Table
CREATE TABLE IF NOT EXISTS companies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    ticker TEXT UNIQUE,
    description TEXT,
    website TEXT,
    rss_feed TEXT,
    is_public BOOLEAN DEFAULT 1,
    created_date DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_ticker ON companies(ticker);

-- Sentiment Cache Table
CREATE TABLE IF NOT EXISTS sentiment_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text_hash TEXT UNIQUE NOT NULL,
    sentiment_score REAL,
    sentiment_label TEXT,
    analyzed_date DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sentiment_hash ON sentiment_cache(text_hash);
