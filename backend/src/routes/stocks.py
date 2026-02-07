from flask import Blueprint, jsonify, request, current_app
from src.utils.database import execute_query

stocks_bp = Blueprint('stocks', __name__)

@stocks_bp.route('/stocks', methods=['GET'], strict_slashes=False)
@stocks_bp.route('', methods=['GET'], strict_slashes=False)
def get_stocks():
    try:
        query = """SELECT ticker, company_name, price, change, change_percent, volume, market_cap, pe_ratio, timestamp
                   FROM stock_data 
                   WHERE (ticker, timestamp) IN (SELECT ticker, MAX(timestamp) FROM stock_data GROUP BY ticker)
                   ORDER BY ticker"""
        stocks = execute_query(current_app.config['DATABASE_PATH'], query)
        return jsonify({'success': True, 'data': stocks, 'count': len(stocks)}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@stocks_bp.route('/<ticker>', methods=['GET'], strict_slashes=False)
def get_stock(ticker):
    try:
        query = """SELECT * FROM stock_data WHERE ticker = ? ORDER BY timestamp DESC LIMIT 1"""
        stock = execute_query(current_app.config['DATABASE_PATH'], query, (ticker.upper(),))
        if stock:
            return jsonify({'success': True, 'data': stock[0]}), 200
        return jsonify({'success': False, 'error': 'Stock not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@stocks_bp.route('/<ticker>/history', methods=['GET'], strict_slashes=False)
def get_stock_history(ticker):
    try:
        limit = request.args.get('limit', default=30, type=int)
        query = """SELECT * FROM stock_data WHERE ticker = ? ORDER BY timestamp DESC LIMIT ?"""
        history = execute_query(current_app.config['DATABASE_PATH'], query, (ticker.upper(), limit))
        return jsonify({'success': True, 'data': history, 'count': len(history)}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500



