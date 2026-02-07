from src.utils.database import execute_query
from datetime import datetime, timedelta
import logging

logger = logging.getLogger('quantum_tracker')

def get_trending_topics(db_path, days=7, limit=10):
    '''Get trending topics from news articles'''
    
    cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
    
    query = '''
        SELECT source, category, COUNT(*) as article_count
        FROM news_articles
        WHERE published_date >= ?
        GROUP BY source, category
        ORDER BY article_count DESC
        LIMIT ?
    '''
    
    return execute_query(db_path, query, (cutoff_date, limit))

def get_sentiment_distribution(db_path, days=30):
    '''Get sentiment distribution over time'''
    
    cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
    
    query = '''
        SELECT 
            sentiment_label,
            COUNT(*) as count,
            AVG(sentiment_score) as avg_score
        FROM news_articles
        WHERE published_date >= ?
        GROUP BY sentiment_label
    '''
    
    return execute_query(db_path, query, (cutoff_date,))

def get_recent_news_summary(db_path, hours=24, limit=50):
    '''Get summary of recent news'''
    
    cutoff_date = (datetime.now() - timedelta(hours=hours)).isoformat()
    
    query = '''
        SELECT 
            COUNT(*) as total_articles,
            COUNT(DISTINCT source) as unique_sources,
            AVG(sentiment_score) as avg_sentiment
        FROM news_articles
        WHERE published_date >= ?
    '''
    
    summary = execute_query(db_path, query, (cutoff_date,))
    
    # Get latest articles
    latest_query = '''
        SELECT * FROM news_articles
        WHERE published_date >= ?
        ORDER BY published_date DESC
        LIMIT ?
    '''
    
    latest_articles = execute_query(db_path, latest_query, (cutoff_date, limit))
    
    return {
        'summary': summary[0] if summary else {},
        'articles': latest_articles
    }
