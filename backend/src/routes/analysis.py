from flask import Blueprint, jsonify, current_app
from src.services.sentiment_analyzer import summarize_news_with_llm, predict_stock_impact
from src.services.quant_analyzer import analyze_portfolio
from src.utils.database import execute_query

analysis_bp = Blueprint('analysis', __name__)

@analysis_bp.route('/news-summary', methods=['GET'])
def get_news_summary():
    try:
        news = execute_query(
            current_app.config['DATABASE_PATH'],
            "SELECT title, description, source, url FROM news_articles ORDER BY published_date DESC LIMIT 20"
        )
        
        stocks = execute_query(
            current_app.config['DATABASE_PATH'],
            "SELECT ticker FROM stock_data WHERE (ticker, timestamp) IN (SELECT ticker, MAX(timestamp) FROM stock_data GROUP BY ticker)"
        )
        
        summary = summarize_news_with_llm(news)
        predictions = predict_stock_impact(news, stocks) if stocks else None
        
        return jsonify({
            'success': True,
            'summary': summary,
            'predictions': predictions,
            'article_count': len(news)
        }), 200
        
    except Exception as e:
        current_app.logger.error(f'Summary error: {e}')
        return jsonify({'success': False, 'error': str(e)}), 500

@analysis_bp.route('/portfolio', methods=['GET'])
def get_portfolio_analysis():
    try:
        analysis = analyze_portfolio(current_app.config['DATABASE_PATH'])
        return jsonify({'success': True, 'data': analysis}), 200
    except Exception as e:
        current_app.logger.error(f'Portfolio error: {e}')
        return jsonify({'success': False, 'error': str(e)}), 500
