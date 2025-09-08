#!/usr/bin/env python3
"""
Main Flask application for Resource-Based Web Server Cost Forecasting.
INDT 4216 - Industrial Training Programme at Wayamba University of Sri Lanka.
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import logging
from datetime import datetime, date

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.config import config
from app.models.database import init_database
from app.routes.dashboard import dashboard_bp
from app.routes.api import api_bp

def create_app(config_name: str = None) -> Flask:
    """Create and configure the Flask application."""
    
    # Determine configuration
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(config.get(config_name, config['default']))
    config[config_name].init_app(app)
    
    # Setup logging
    setup_logging(app)
    
    # Initialize database
    try:
        init_database(app.config['DATABASE_PATH'])
        app.logger.info(f"Database initialized: {app.config['DATABASE_PATH']}")
    except Exception as e:
        app.logger.error(f"Database initialization failed: {e}")
        raise
    
    # Register blueprints
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(api_bp)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Add template filters
    register_template_filters(app)
    
    # Add context processors
    register_context_processors(app)
    
    app.logger.info(f"Flask app created with config: {config_name}")
    
    return app

def setup_logging(app: Flask):
    """Configure logging for the application."""
    
    # Don't add handlers if they already exist (prevents duplicate logs)
    if app.logger.handlers:
        return
    
    # Set log level based on debug mode
    log_level = logging.DEBUG if app.debug else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s [%(name)s]: %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Add handlers to app logger
    app.logger.addHandler(console_handler)
    app.logger.setLevel(log_level)
    
    # Suppress excessive logs from some libraries
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    
    app.logger.info("Logging configured")

def register_error_handlers(app: Flask):
    """Register error handlers for common HTTP errors."""
    
    @app.errorhandler(404)
    def not_found_error(error):
        app.logger.warning(f"404 error: {request.url}")
        return render_template('error.html', 
                             error="Page not found", 
                             error_code=404,
                             error_description="The requested page could not be found."), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f"500 error: {error}")
        return render_template('error.html',
                             error="Internal server error",
                             error_code=500,
                             error_description="An internal server error occurred."), 500
    
    @app.errorhandler(400)
    def bad_request(error):
        app.logger.warning(f"400 error: {error}")
        return render_template('error.html',
                             error="Bad request",
                             error_code=400,
                             error_description="The request was invalid."), 400
    
    app.logger.info("Error handlers registered")

def register_template_filters(app: Flask):
    """Register custom template filters."""
    
    @app.template_filter('currency')
    def currency_filter(amount):
        """Format number as currency."""
        if amount is None:
            return "$0.00"
        try:
            return f"${float(amount):.2f}"
        except (ValueError, TypeError):
            return "$0.00"
    
    @app.template_filter('percentage')
    def percentage_filter(value, decimals=1):
        """Format number as percentage."""
        if value is None:
            return "0%"
        try:
            return f"{float(value):.{decimals}f}%"
        except (ValueError, TypeError):
            return "0%"
    
    @app.template_filter('round_decimal')
    def round_decimal_filter(value, decimals=2):
        """Round number to specified decimal places."""
        if value is None:
            return 0
        try:
            return round(float(value), decimals)
        except (ValueError, TypeError):
            return 0
    
    @app.template_filter('format_date')
    def format_date_filter(date_value, format='%Y-%m-%d'):
        """Format date with specified format."""
        if date_value is None:
            return ""
        try:
            if isinstance(date_value, str):
                date_value = datetime.fromisoformat(date_value).date()
            return date_value.strftime(format)
        except (ValueError, AttributeError):
            return str(date_value)
    
    @app.template_filter('format_datetime')
    def format_datetime_filter(datetime_value, format='%Y-%m-%d %H:%M'):
        """Format datetime with specified format."""
        if datetime_value is None:
            return ""
        try:
            if isinstance(datetime_value, str):
                datetime_value = datetime.fromisoformat(datetime_value)
            return datetime_value.strftime(format)
        except (ValueError, AttributeError):
            return str(datetime_value)
    
    @app.template_filter('size_mb')
    def size_mb_filter(size_mb):
        """Format size in MB with appropriate units."""
        if size_mb is None:
            return "0 MB"
        try:
            size_mb = float(size_mb)
            if size_mb >= 1024:
                return f"{size_mb/1024:.1f} GB"
            else:
                return f"{size_mb:.1f} MB"
        except (ValueError, TypeError):
            return "0 MB"
    
    app.logger.info("Template filters registered")

def register_context_processors(app: Flask):
    """Register context processors to make variables available in all templates."""
    
    @app.context_processor
    def inject_app_info():
        """Inject application information into template context."""
        return {
            'app_name': 'Cost Forecasting Dashboard',
            'app_version': '1.0.0',
            'current_year': datetime.now().year,
            'environment': app.config.get('FLASK_ENV', 'development')
        }
    
    @app.context_processor
    def inject_navigation():
        """Inject navigation information."""
        servers = [
            {'id': 'web-frontend', 'name': 'Web Frontend'},
            {'id': 'api-backend', 'name': 'API Backend'},
            {'id': 'database', 'name': 'Database'},
            {'id': 'cache-server', 'name': 'Cache Server'}
        ]
        return {'servers': servers}
    
    app.logger.info("Context processors registered")

# Additional routes for the main application
def register_main_routes(app: Flask):
    """Register main application routes."""
    
    @app.route('/forecasting')
    def forecasting():
        """Cost forecasting interface."""
        try:
            # This would typically load forecasting models and generate predictions
            # For now, return a basic forecasting template
            return render_template('forecasting.html')
        except Exception as e:
            app.logger.error(f"Forecasting page error: {e}")
            return render_template('error.html', error="Failed to load forecasting page"), 500
    
    @app.route('/health')
    def health_check():
        """Application health check endpoint."""
        try:
            # Check database connectivity
            from app.models.database import get_database
            db = get_database()
            db.execute_query("SELECT 1")
            
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0',
                'database': 'connected'
            })
        except Exception as e:
            app.logger.error(f"Health check failed: {e}")
            return jsonify({
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 503

# Create the Flask application instance
app = create_app()

# Register main routes
register_main_routes(app)

if __name__ == '__main__':
    # Development server configuration
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    app.logger.info(f"Starting Flask development server on {host}:{port}")
    app.logger.info(f"Debug mode: {debug}")
    app.logger.info(f"Database: {app.config['DATABASE_PATH']}")
    
    # Check if database exists and has data
    try:
        from app.models.database import get_database
        db = get_database()
        
        # Check for sample data
        result = db.execute_query("SELECT COUNT(*) as count FROM resource_metrics")
        metrics_count = result[0]['count'] if result else 0
        
        if metrics_count == 0:
            app.logger.warning("No sample data found in database")
            app.logger.info("Run: python scripts/setup_database.py")
            app.logger.info("Then: python scripts/generate_sample_data.py")
        else:
            app.logger.info(f"Database ready with {metrics_count} metric records")
            
    except Exception as e:
        app.logger.error(f"Database check failed: {e}")
    
    # Start the development server
    try:
        app.run(host=host, port=port, debug=debug, threaded=True)
    except KeyboardInterrupt:
        app.logger.info("Server stopped by user")
    except Exception as e:
        app.logger.error(f"Server error: {e}")
        raise