from flask import Flask, jsonify
from flask_cors import CORS
from config import Config
from src.utils.database import init_db
from src.utils.logger import setup_logger
from src.utils.scheduler import start_scheduler
import os

# Import routes
from src.routes.news import news_bp
from src.routes.stocks import stocks_bp
from src.routes.companies import companies_bp
from src.routes.sentiment import sentiment_bp
from src.routes.analysis import analysis_bp

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Setup CORS
    CORS(app, resources={
        r"/api/*": {
            "origins": app.config['CORS_ORIGINS']
        }
    })
    
    # Setup logging
    logger = setup_logger(app.config['LOG_LEVEL'], app.config['LOG_FILE'])
    app.logger.handlers = logger.handlers
    app.logger.setLevel(logger.level)
    
    # Initialize database
    with app.app_context():
        init_db(app.config['DATABASE_PATH'])
        app.logger.info('Database initialized')
    
    # Register blueprints
    app.register_blueprint(news_bp, url_prefix=f"{app.config['API_PREFIX']}/news")
    app.register_blueprint(stocks_bp, url_prefix=f"{app.config['API_PREFIX']}/stocks")
    app.register_blueprint(companies_bp, url_prefix=f"{app.config['API_PREFIX']}/companies")
    app.register_blueprint(sentiment_bp, url_prefix=f"{app.config['API_PREFIX']}/sentiment")
    app.register_blueprint(analysis_bp, url_prefix=f"{app.config['API_PREFIX']}/analysis")
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        return jsonify({'status': 'healthy', 'message': 'Quantum Tracker API is running'}), 200
    
    # Root endpoint
    @app.route('/')
    def index():
        return jsonify({
            'message': 'Quantum Tracker API',
            'version': '1.0.0',
            'endpoints': {
                'news': f"{app.config['API_PREFIX']}/news",
                'stocks': f"{app.config['API_PREFIX']}/stocks",
                'companies': f"{app.config['API_PREFIX']}/companies",
                'sentiment': f"{app.config['API_PREFIX']}/sentiment"
            }
        }), 200
    

    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', 'https://quantum-tracker.onrender.com')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response

    # Start background scheduler
    start_scheduler(app)
    
    return app

# Create app instance for gunicorn
app = create_app()

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)



