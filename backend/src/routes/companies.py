from flask import Blueprint, jsonify, request, current_app
from src.utils.database import execute_query
import json
import os

companies_bp = Blueprint('companies', __name__)

@companies_bp.route('/', methods=['GET'])
def get_companies():
    '''Get all quantum companies'''
    
    try:
        # Load from JSON file
        companies_file = current_app.config['RSS_FEEDS_FILE'].replace('rss_feeds.json', 'quantum_companies.json')
        
        with open(companies_file, 'r') as f:
            companies_data = json.load(f)
        
        # Also get stock data if available
        public_tickers = [c['ticker'] for c in companies_data['public_companies'] if 'ticker' in c]
        
        stocks = {}
        if public_tickers:
            from src.services.stock_fetcher import get_latest_stocks
            stock_data = get_latest_stocks(current_app.config['DATABASE_PATH'], limit=20)
            stocks = {s['ticker']: s for s in stock_data}
        
        # Enrich company data with stock info
        for company in companies_data['public_companies']:
            ticker = company.get('ticker')
            if ticker and ticker in stocks:
                company['stock_data'] = stocks[ticker]
        
        return jsonify({
            'success': True,
            'data': companies_data
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@companies_bp.route('/<string:ticker>', methods=['GET'])
def get_company(ticker):
    '''Get a single company by ticker'''
    
    try:
        companies_file = current_app.config['RSS_FEEDS_FILE'].replace('rss_feeds.json', 'quantum_companies.json')
        
        with open(companies_file, 'r') as f:
            companies_data = json.load(f)
        
        # Find company
        company = None
        for c in companies_data['public_companies']:
            if c.get('ticker', '').upper() == ticker.upper():
                company = c
                break
        
        if not company:
            return jsonify({
                'success': False,
                'error': 'Company not found'
            }), 404
        
        # Get stock data
        from src.services.stock_fetcher import get_latest_stocks
        stock_data = get_latest_stocks(current_app.config['DATABASE_PATH'], limit=20)
        stocks = {s['ticker']: s for s in stock_data}
        
        if ticker.upper() in stocks:
            company['stock_data'] = stocks[ticker.upper()]
        
        # Get related news
        news = execute_query(
            current_app.config['DATABASE_PATH'],
            '''SELECT * FROM news_articles 
               WHERE title LIKE ? OR description LIKE ?
               ORDER BY published_date DESC 
               LIMIT 10''',
            (f'%{company["name"]}%', f'%{company["name"]}%')
        )
        
        company['recent_news'] = news
        
        return jsonify({
            'success': True,
            'data': company
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@companies_bp.route('/search', methods=['GET'])
def search_companies():
    '''Search companies by name or ticker'''
    
    query = request.args.get('q', '')
    
    if not query:
        return jsonify({
            'success': False,
            'error': 'Query parameter required'
        }), 400
    
    try:
        companies_file = current_app.config['RSS_FEEDS_FILE'].replace('rss_feeds.json', 'quantum_companies.json')
        
        with open(companies_file, 'r') as f:
            companies_data = json.load(f)
        
        # Search in both public and private companies
        results = []
        query_lower = query.lower()
        
        for company in companies_data['public_companies'] + companies_data['private_companies']:
            if (query_lower in company['name'].lower() or 
                (company.get('ticker') and query_lower in company['ticker'].lower())):
                results.append(company)
        
        return jsonify({
            'success': True,
            'count': len(results),
            'data': results
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
