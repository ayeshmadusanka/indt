"""
Configuration settings for the Flask application.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration class."""
    
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    FLASK_ENV = os.environ.get('FLASK_ENV') or 'development'
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Database configuration
    BASE_DIR = Path(__file__).parent.parent
    DATABASE_PATH = os.environ.get('DATABASE_PATH') or str(BASE_DIR / 'database' / 'cost_forecasting.db')
    DATABASE_URL = f'sqlite:///{DATABASE_PATH}'
    
    # Cost calculation parameters
    CPU_HOURLY_RATE = float(os.environ.get('CPU_HOURLY_RATE', '0.0116'))
    RAM_HOURLY_RATE = float(os.environ.get('RAM_HOURLY_RATE', '0.0058'))
    
    BANDWIDTH_TIER_1_LIMIT = int(os.environ.get('BANDWIDTH_TIER_1_LIMIT', '10000'))
    BANDWIDTH_TIER_1_PRICE = float(os.environ.get('BANDWIDTH_TIER_1_PRICE', '0.09'))
    BANDWIDTH_TIER_2_LIMIT = int(os.environ.get('BANDWIDTH_TIER_2_LIMIT', '40000'))
    BANDWIDTH_TIER_2_PRICE = float(os.environ.get('BANDWIDTH_TIER_2_PRICE', '0.085'))
    BANDWIDTH_TIER_3_PRICE = float(os.environ.get('BANDWIDTH_TIER_3_PRICE', '0.07'))
    
    # Data generation parameters
    SIMULATION_DAYS = int(os.environ.get('SIMULATION_DAYS', '90'))
    DATA_INTERVAL_MINUTES = int(os.environ.get('DATA_INTERVAL_MINUTES', '5'))
    
    # Forecasting parameters
    FORECAST_HORIZON_DAYS = int(os.environ.get('FORECAST_HORIZON_DAYS', '30'))
    CONFIDENCE_INTERVAL = float(os.environ.get('CONFIDENCE_INTERVAL', '0.95'))
    MIN_TRAINING_DAYS = int(os.environ.get('MIN_TRAINING_DAYS', '30'))
    
    # Application settings
    ITEMS_PER_PAGE = 50
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file upload
    
    # Chart and visualization settings
    DEFAULT_CHART_COLORS = [
        '#3b82f6', '#ef4444', '#10b981', '#f59e0b', 
        '#8b5cf6', '#06b6d4', '#f97316', '#ec4899'
    ]
    
    @staticmethod
    def init_app(app):
        """Initialize application with configuration."""
        pass

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False
    
    @classmethod
    def init_app(cls, app):
        """Initialize production app configuration."""
        Config.init_app(app)
        
        # Log to stderr in production
        import logging
        from logging import StreamHandler
        file_handler = StreamHandler()
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True
    DATABASE_PATH = ':memory:'  # In-memory database for testing
    WTF_CSRF_ENABLED = False

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}