from flask import Blueprint, jsonify, request, current_app
from src.services.stock_fetcher import get_latest_stocks
from src.utils.database import execute_query

stocks_bp = Blueprint('stocks', __name__)

@stocks_bp.route('/', methods=['GET'])
def get_stocks():
    '''Get latest stock data for all quantum companies'''
    
    try:
        stocks = get_latest_stocks(current_app.config['DATABASE_PATH'], limit=20)
        
        return jsonify({
            'success': True,
            'count': len(stocks),
            'data': stocks
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@stocks_bp.route('/<ticker>', methods=['GET'])
def get_stock(ticker):
    '''Get stock data for a specific ticker'''
    
    try:
        stock = execute_query(
            current_app.config['DATABASE_PATH'],
            '''SELECT * FROM stock_data 
               WHERE ticker = ? 
               ORDER BY timestamp DESC 
               LIMIT 1''',
            (ticker.upper(),)
        )
        
        if not stock:
            return jsonify({
                'success': False,
                'error': f'Stock data not found for {ticker}'
            }), 404
        
        return jsonify({
            'success': True,
            'data': stock[0]
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@stocks_bp.route('/<ticker>/history', methods=['GET'])
def get_stock_history(ticker):
    '''Get historical stock data'''
    
    limit = request.args.get('limit', 100, type=int)
    
    try:
        history = execute_query(
            current_app.config['DATABASE_PATH'],
            '''SELECT * FROM stock_data 
               WHERE ticker = ? 
               ORDER BY timestamp DESC 
               LIMIT ?''',
            (ticker.upper(), limit)
        )
        
        return jsonify({
            'success': True,
            'count': len(history),
            'data': history
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
