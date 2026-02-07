from flask import Blueprint, jsonify, request, current_app
from src.utils.database import execute_query

news_bp = Blueprint('news', __name__)

@news_bp.route('/', methods=['GET'])
@news_bp.route('', methods=['GET'])
def get_news():
    try:
        limit = request.args.get('limit', default=10, type=int)
        query = "SELECT * FROM news_articles ORDER BY published_date DESC LIMIT ?"
        articles = execute_query(current_app.config['DATABASE_PATH'], query, (limit,))
        return jsonify({'success': True, 'data': articles, 'count': len(articles)}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@news_bp.route('/<int:article_id>', methods=['GET'])
def get_article(article_id):
    try:
        query = "SELECT * FROM news_articles WHERE id = ?"
        article = execute_query(current_app.config['DATABASE_PATH'], query, (article_id,))
        if article:
            return jsonify({'success': True, 'data': article[0]}), 200
        return jsonify({'success': False, 'error': 'Article not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@news_bp.route('/sources', methods=['GET'])
def get_sources():
    try:
        query = "SELECT DISTINCT source FROM news_articles"
        sources = execute_query(current_app.config['DATABASE_PATH'], query)
        return jsonify({'success': True, 'data': [s['source'] for s in sources]}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
