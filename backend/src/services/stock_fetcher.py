import yfinance as yf
from datetime import datetime
from src.utils.database import execute_query
import logging
import requests

logger = logging.getLogger('quantum_tracker')

def update_all_stocks(config):
    tickers = config['QUANTUM_TICKERS']
    db_path = config['DATABASE_PATH']
    
    updated = 0
    for ticker in tickers:
        try:
            url = f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=5d'
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
            data = response.json()
            
            result = data['chart']['result'][0]
            meta = result['meta']
            quote = result['indicators']['quote'][0]
            
            current_price = meta['regularMarketPrice']
            prev_close = meta['chartPreviousClose']
            change = current_price - prev_close
            change_percent = (change / prev_close) * 100
            
            execute_query(db_path,
                "INSERT INTO stock_data (ticker, company_name, price, change, change_percent, volume, market_cap, pe_ratio, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ticker, ticker, current_price, change, change_percent, quote['volume'][-1] if quote['volume'] else 0, None, None, datetime.now().isoformat()))
            
            updated += 1
            logger.info(f'{ticker}: ${current_price:.2f} ({change_percent:+.2f}%)')
        except Exception as e:
            logger.warning(f'Skip {ticker}: timeout')
    
    logger.info(f'Updated {updated}/{len(tickers)} stocks')
    return updated

def update_stock(ticker, db_path):
    return True

def get_latest_stocks(db_path, limit=10):
    return execute_query(db_path, 
        "SELECT ticker, company_name, price, change, change_percent, volume, market_cap, pe_ratio, timestamp FROM stock_data WHERE (ticker, timestamp) IN (SELECT ticker, MAX(timestamp) FROM stock_data GROUP BY ticker) ORDER BY ticker LIMIT ?",
        (limit,))
