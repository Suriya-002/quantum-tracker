from flask import Blueprint, jsonify, request, current_app
from src.utils.database import execute_query

news_bp = Blueprint('news', __name__)

@news_bp.route('/', methods=['GET'])
def get_news():
    '''Get news articles with optional filters'''
    
    # Query parameters
    category = request.args.get('category')
    source = request.args.get('source')
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    sentiment = request.args.get('sentiment')  # positive, negative, neutral
    
    # Build query
    query = 'SELECT * FROM news_articles WHERE 1=1'
    params = []
    
    if category:
        query += ' AND category = ?'
        params.append(category)
    
    if source:
        query += ' AND source = ?'
        params.append(source)
    
    if sentiment:
        query += ' AND sentiment_label = ?'
        params.append(sentiment)
    
    query += ' ORDER BY published_date DESC LIMIT ? OFFSET ?'
    params.extend([limit, offset])
    
    try:
        articles = execute_query(current_app.config['DATABASE_PATH'], query, tuple(params))
        
        return jsonify({
            'success': True,
            'count': len(articles),
            'data': articles
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@news_bp.route('/<int:article_id>', methods=['GET'])
def get_article(article_id):
    '''Get a single article by ID'''
    
    try:
        article = execute_query(
            current_app.config['DATABASE_PATH'],
            'SELECT * FROM news_articles WHERE id = ?',
            (article_id,)
        )
        
        if not article:
            return jsonify({
                'success': False,
                'error': 'Article not found'
            }), 404
        
        return jsonify({
            'success': True,
            'data': article[0]
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@news_bp.route('/sources', methods=['GET'])
def get_sources():
    '''Get list of all news sources'''
    
    try:
        sources = execute_query(
            current_app.config['DATABASE_PATH'],
            'SELECT DISTINCT source, category FROM news_articles ORDER BY source'
        )
        
        return jsonify({
            'success': True,
            'data': sources
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
