from flask import Flask, jsonify, request
from flask_cors import CORS
from config import Config
from src.utils.database import init_db
from src.utils.scheduler import start_scheduler
from src.routes.news import news_bp
from src.routes.stocks import stocks_bp
from src.routes.companies import companies_bp
from src.routes.analysis import analysis_bp
import logging

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # CRITICAL: Enable CORS for ALL routes, ALL origins during testing
    CORS(app, resources={r"/*": {"origins": "*"}})
    
    # Also add CORS headers directly to every response
    @app.after_request
    def after_request(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        if request.method == 'OPTIONS':
            return response
        return response
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, app.config['LOG_LEVEL']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(app.config['LOG_FILE']),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('app')
    
    # Initialize database
    init_db(app.config['DATABASE_PATH'])
    logger.info('Database initialized')
    
    # Register blueprints
    app.register_blueprint(news_bp, url_prefix=app.config['API_PREFIX'])
    app.register_blueprint(stocks_bp, url_prefix=app.config['API_PREFIX'])
    app.register_blueprint(companies_bp, url_prefix=app.config['API_PREFIX'])
    app.register_blueprint(analysis_bp, url_prefix=f"{app.config['API_PREFIX']}/analysis")
    
    @app.route('/')
    def index():
        return jsonify({
            'message': 'Quantum Tracker API',
            'version': '1.0.0',
            'status': 'live'
        }), 200
    
    # Start background scheduler
    start_scheduler(app)
    
    return app

# Create app instance for gunicorn
app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
