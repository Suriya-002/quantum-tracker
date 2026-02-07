import sys
sys.path.insert(0, '.')

from config import Config
from src.utils.database import execute_query
import feedparser
from datetime import datetime
import requests

print('='*60)
print('FETCHING QUANTUM INDUSTRY NEWS & STOCKS')
print('='*60)

# Fetch stocks
print('\n[1/2] Fetching stock data...')
tickers = ['IONQ', 'RGTI', 'QBTS', 'IBM', 'GOOGL', 'MSFT', 'AMZN', 'INTC', 'HON']

for ticker in tickers:
    try:
        url = f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=5d'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        
        result = data['chart']['result'][0]
        meta = result['meta']
        quote = result['indicators']['quote'][0]
        
        current_price = meta['regularMarketPrice']
        prev_close = meta['chartPreviousClose']
        change = current_price - prev_close
        change_percent = (change / prev_close) * 100
        
        execute_query(
            Config.DATABASE_PATH,
            """INSERT INTO stock_data 
               (ticker, company_name, price, change, change_percent, volume, market_cap, pe_ratio, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ticker, ticker, current_price, change, change_percent,
             quote['volume'][-1] if quote['volume'] else 0, None, None,
             datetime.now().isoformat())
        )
        print(f'✓ {ticker}: ${current_price:.2f} ({change_percent:+.2f}%)')
    except Exception as e:
        print(f'✗ {ticker}: {str(e)[:50]}')

# Fetch industry-focused news
print('\n[2/2] Fetching quantum industry news...')
feeds = [
    ('https://quantumcomputingreport.com/feed/', 'Quantum Computing Report'),
    ('https://thequantuminsider.com/feed/', 'The Quantum Insider'),
    ('https://www.prnewswire.com/rss/technology-latest-news/technology-latest-news-list.rss', 'PR Newswire Tech'),
]

# Keywords to filter for quantum-related industry news
keywords = ['quantum', 'qubit', 'ionq', 'rigetti', 'dwave', 'd-wave', 'ibm quantum', 
            'google quantum', 'atom computing', 'xanadu', 'psiquantum', 'pasqal']

total = 0
for url, source in feeds:
    try:
        feed = feedparser.parse(url)
        count = 0
        for entry in feed.entries[:20]:
            title = entry.get('title', '').lower()
            desc = entry.get('summary', '').lower()
            
            # Filter for quantum industry news
            if any(kw in title or kw in desc for kw in keywords):
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
                            'industry',
                            datetime.now().isoformat(),
                            0.5,
                            'NEUTRAL'
                        )
                    )
                    count += 1
                except:
                    pass
        total += count
        print(f'✓ {source}: {count} articles')
    except Exception as e:
        print(f'✗ {source}: {str(e)[:50]}')

print(f'\n✅ COMPLETE: {len(tickers)} stocks, {total} industry news articles')
print('='*60 + '\n')
