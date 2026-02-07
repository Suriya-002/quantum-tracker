from flask import Blueprint, jsonify, request, current_app
from src.services.sentiment_analyzer import analyze_sentiment
from src.services.news_aggregator import get_sentiment_distribution

sentiment_bp = Blueprint('sentiment', __name__)

@sentiment_bp.route('/analyze', methods=['POST'])
def analyze():
    '''Analyze sentiment of provided text'''
    
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({
            'success': False,
            'error': 'Text is required'
        }), 400
    
    text = data['text']
    
    if len(text.strip()) < 10:
        return jsonify({
            'success': False,
            'error': 'Text too short (minimum 10 characters)'
        }), 400
    
    try:
        sentiment = analyze_sentiment(text)
        
        return jsonify({
            'success': True,
            'sentiment': sentiment,
            'text_length': len(text)
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@sentiment_bp.route('/distribution', methods=['GET'])
def get_distribution():
    '''Get sentiment distribution of recent news'''
    
    days = request.args.get('days', 30, type=int)
    
    try:
        distribution = get_sentiment_distribution(
            current_app.config['DATABASE_PATH'],
            days=days
        )
        
        return jsonify({
            'success': True,
            'days': days,
            'data': distribution
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@sentiment_bp.route('/trends', methods=['GET'])
def get_trends():
    '''Get sentiment trends over time'''
    
    days = request.args.get('days', 30, type=int)
    
    try:
        from src.utils.database import execute_query
        from datetime import datetime, timedelta
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        query = '''
            SELECT 
                DATE(published_date) as date,
                sentiment_label,
                COUNT(*) as count,
                AVG(sentiment_score) as avg_score
            FROM news_articles
            WHERE published_date >= ?
            GROUP BY DATE(published_date), sentiment_label
            ORDER BY date DESC
        '''
        
        trends = execute_query(
            current_app.config['DATABASE_PATH'],
            query,
            (cutoff_date,)
        )
        
        return jsonify({
            'success': True,
            'days': days,
            'data': trends
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
