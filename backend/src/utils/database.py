import sqlite3
import os
from datetime import datetime

def get_connection(db_path):
    '''Create a database connection'''
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path):
    '''Initialize the database with tables'''
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    # News articles table
    cursor.execute('''
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
        )
    ''')
    
    # Stock data table
    cursor.execute('''
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
        )
    ''')
    
    # Companies table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS companies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            ticker TEXT UNIQUE,
            description TEXT,
            website TEXT,
            rss_feed TEXT,
            is_public BOOLEAN DEFAULT 1,
            created_date DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Sentiment analysis cache
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sentiment_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text_hash TEXT UNIQUE NOT NULL,
            sentiment_score REAL,
            sentiment_label TEXT,
            analyzed_date DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    
    print(f'Database initialized at {db_path}')

def execute_query(db_path, query, params=None):
    '''Execute a query and return results'''
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    if params:
        cursor.execute(query, params)
    else:
        cursor.execute(query)
    
    if query.strip().upper().startswith('SELECT'):
        results = cursor.fetchall()
        conn.close()
        return [dict(row) for row in results]
    else:
        conn.commit()
        last_id = cursor.lastrowid
        conn.close()
        return last_id
