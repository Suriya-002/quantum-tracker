from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import atexit

def start_scheduler(app):
    '''Start background scheduler for periodic tasks'''
    
    scheduler = BackgroundScheduler()
    
    # Import services (to avoid circular imports)
    from src.services.rss_scraper import scrape_all_feeds
    from src.services.stock_fetcher import update_all_stocks
    
    # Schedule RSS scraping
    scheduler.add_job(
        func=lambda: scrape_all_feeds(app.config),
        trigger=IntervalTrigger(seconds=app.config['RSS_UPDATE_INTERVAL']),
        id='rss_scraper',
        name='Scrape RSS feeds',
        replace_existing=True
    )
    
    # Schedule stock updates
    scheduler.add_job(
        func=lambda: update_all_stocks(app.config),
        trigger=IntervalTrigger(seconds=app.config['STOCK_UPDATE_INTERVAL']),
        id='stock_updater',
        name='Update stock data',
        replace_existing=True
    )
    
    scheduler.start()
    app.logger.info('Background scheduler started')
    
    # Run immediately on startup
    with app.app_context():
        app.logger.info('Running initial data fetch...')
        scrape_all_feeds(app.config)
        update_all_stocks(app.config)
    
    # Shut down scheduler on exit
    atexit.register(lambda: scheduler.shutdown())
    
    return scheduler
