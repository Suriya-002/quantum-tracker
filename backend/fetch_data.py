import sys
sys.path.insert(0, '.')

from config import Config
from src.services.stock_fetcher import update_all_stocks
from src.utils.database import execute_query
import feedparser
from datetime import datetime

print('='*60)
print('FETCHING REAL DATA')
print('='*60)

# Fetch stocks
print('\n[1/2] Fetching stock data...')
config_dict = {
    'QUANTUM_TICKERS': Config.QUANTUM_TICKERS,
    'DATABASE_PATH': Config.DATABASE_PATH
}
updated = update_all_stocks(config_dict)
print(f'✓ Updated {updated} stocks')

# Fetch news manually (skip sentiment for now - it's broken)
print('\n[2/2] Fetching news...')
feeds = [
    ('https://quantumcomputingreport.com/feed/', 'Quantum Computing Report'),
    ('https://www.sciencedaily.com/rss/matter_energy/quantum_physics.xml', 'ScienceDaily'),
]

total = 0
for url, source in feeds:
    try:
        feed = feedparser.parse(url)
        for entry in feed.entries[:5]:
            try:
                execute_query(
                    Config.DATABASE_PATH,
                    """INSERT OR IGNORE INTO news_articles 
                       (title, description, url, source, category, published_date, sentiment_score, sentiment_label)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        entry.get('title', '')[:200],
                        entry.get('summary', '')[:500],
                        entry.get('link', ''),
                        source,
                        'news',
                        datetime.now().isoformat(),
                        0.5,
                        'NEUTRAL'
                    )
                )
                total += 1
            except:
                pass
        print(f'✓ Fetched from {source}')
    except Exception as e:
        print(f'✗ Failed: {source} - {e}')

print(f'\n✓ Added {total} news articles')
print('\n' + '='*60)
print('DONE! Now starting server...')
print('='*60 + '\n')
