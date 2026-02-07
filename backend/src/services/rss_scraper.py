import feedparser
from datetime import datetime
from src.utils.database import execute_query
import logging
import json
import socket

logger = logging.getLogger('quantum_tracker')

# Set global timeout for feedparser
socket.setdefaulttimeout(10)

def scrape_all_feeds(config):
    feeds_file = config['RSS_FEEDS_FILE']
    db_path = config['DATABASE_PATH']
    
    try:
        with open(feeds_file, 'r', encoding='utf-8') as f:
            feeds_data = json.load(f)
    except:
        return
    
    total = 0
    keywords = ['quantum', 'qubit', 'ionq', 'rigetti', 'ibm quantum', 'd-wave', 'dwave']
    business = ['funding', 'partnership', 'million', 'company', 'announce', 'launch', 'investment', 'market', 'raise', 'deal']
    
    for feed_info in feeds_data.get('quantum_news', []):
        try:
            logger.info(f'Fetching {feed_info["name"]}...')
            feed = feedparser.parse(feed_info['url'])
            added = 0
            
            for entry in feed.entries[:10]:
                title = entry.get('title', '').lower()
                desc = entry.get('summary', '').lower()
                text = title + ' ' + desc
                
                if any(k in text for k in keywords) and any(b in text for b in business):
                    execute_query(db_path,
                        "INSERT OR IGNORE INTO news_articles (title, description, url, source, category, published_date, sentiment_score, sentiment_label) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (entry.get('title','')[:200], entry.get('summary','')[:500], entry.get('link',''), feed_info['name'], 'industry', datetime.now().isoformat(), 0.5, 'NEUTRAL'))
                    added += 1
            
            total += added
            logger.info(f'Added {added} from {feed_info["name"]}')
        except Exception as e:
            logger.warning(f'Failed {feed_info["name"]}: timeout')
    
    logger.info(f'Total: {total}')
    return total
