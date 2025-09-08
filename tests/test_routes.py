"""
Tests for Flask routes and API endpoints.
"""

import unittest
import tempfile
import os
import json
from datetime import datetime, date, timedelta
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.app import create_app
from app.models.database import init_database

class TestFlaskRoutes(unittest.TestCase):
    """Test Flask application routes."""
    
    def setUp(self):
        """Set up test Flask application."""
        self.db_fd, self.db_path = tempfile.mkstemp()
        
        # Create test app
        self.app = create_app('testing')
        self.app.config['DATABASE_PATH'] = self.db_path
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        # Initialize database
        init_database(self.db_path)
        
        with self.app.app_context():
            # Set up test database schema
            from app.models.database import get_database
            db = get_database()
            
            # Create basic schema
            schema = """
            CREATE TABLE IF NOT EXISTS servers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                server_id TEXT UNIQUE NOT NULL,
                server_name TEXT NOT NULL,
                server_type TEXT NOT NULL DEFAULT 'web',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS resource_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                server_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                cpu_usage_percent REAL NOT NULL,
                ram_usage_mb REAL NOT NULL,
                ram_total_mb REAL NOT NULL,
                bandwidth_in_mb REAL NOT NULL,
                bandwidth_out_mb REAL NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS daily_costs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                server_id TEXT NOT NULL,
                date DATE NOT NULL,
                cpu_hours REAL NOT NULL,
                cpu_cost REAL NOT NULL,
                ram_gb_hours REAL NOT NULL,
                ram_cost REAL NOT NULL,
                bandwidth_in_gb REAL NOT NULL,
                bandwidth_out_gb REAL NOT NULL,
                bandwidth_cost REAL NOT NULL,
                total_cost REAL NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            with db.get_connection() as conn:
                conn.executescript(schema)
                
                # Insert test data
                test_servers = [
                    ('web-frontend', 'Web Frontend Server', 'web'),
                    ('api-backend', 'API Backend Server', 'api')
                ]
                
                conn.executemany(
                    "INSERT OR IGNORE INTO servers (server_id, server_name, server_type) VALUES (?, ?, ?)",
                    test_servers
                )
                
                # Insert sample resource metrics
                now = datetime.now()
                test_metrics = []
                for i in range(24):  # 24 hours of data
                    timestamp = now - timedelta(hours=i)
                    test_metrics.extend([
                        ('web-frontend', timestamp, 75.0 + i, 4096.0, 8192.0, 100.0 + i, 150.0 + i),
                        ('api-backend', timestamp, 60.0 + i, 2048.0, 4096.0, 80.0 + i, 120.0 + i)
                    ])
                
                conn.executemany(
                    "INSERT INTO resource_metrics (server_id, timestamp, cpu_usage_percent, ram_usage_mb, ram_total_mb, bandwidth_in_mb, bandwidth_out_mb) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    test_metrics
                )
                
                # Insert sample daily costs
                test_costs = []
                for i in range(7):  # 7 days of cost data
                    cost_date = date.today() - timedelta(days=i)
                    test_costs.extend([
                        ('web-frontend', cost_date, 18.0, 0.20 + i*0.01, 96.0, 0.50 + i*0.02, 2.4, 3.6, 0.30 + i*0.01, 1.00 + i*0.04),
                        ('api-backend', cost_date, 14.4, 0.16 + i*0.01, 48.0, 0.25 + i*0.01, 1.9, 2.9, 0.25 + i*0.01, 0.66 + i*0.03)
                    ])
                
                conn.executemany(
                    "INSERT INTO daily_costs (server_id, date, cpu_hours, cpu_cost, ram_gb_hours, ram_cost, bandwidth_in_gb, bandwidth_out_gb, bandwidth_cost, total_cost) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    test_costs
                )
                
                conn.commit()
    
    def tearDown(self):
        """Clean up test database."""
        os.close(self.db_fd)
        os.unlink(self.db_path)
    
    def test_dashboard_index(self):
        """Test main dashboard route."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Cost Forecasting Dashboard', response.data)
        self.assertIn(b'dashboard-card', response.data)
    
    def test_server_detail_valid(self):
        """Test server detail route with valid server."""
        response = self.client.get('/server/web-frontend')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Web Frontend', response.data)
        self.assertIn(b'analytics', response.data)
    
    def test_server_detail_invalid(self):
        """Test server detail route with invalid server."""
        response = self.client.get('/server/invalid-server')
        self.assertEqual(response.status_code, 404)
    
    def test_compare_servers(self):
        """Test server comparison route."""
        response = self.client.get('/compare')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'comparison', response.data)
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('database', data)

class TestAPIRoutes(unittest.TestCase):
    """Test API endpoints."""
    
    def setUp(self):
        """Set up test Flask application for API testing."""
        self.db_fd, self.db_path = tempfile.mkstemp()
        
        # Create test app
        self.app = create_app('testing')
        self.app.config['DATABASE_PATH'] = self.db_path
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        # Initialize database
        init_database(self.db_path)
        
        with self.app.app_context():
            from app.models.database import get_database
            db = get_database()
            
            # Create schema and test data (similar to TestFlaskRoutes)
            schema = """
            CREATE TABLE IF NOT EXISTS servers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                server_id TEXT UNIQUE NOT NULL,
                server_name TEXT NOT NULL,
                server_type TEXT NOT NULL DEFAULT 'web'
            );
            
            CREATE TABLE IF NOT EXISTS resource_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                server_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                cpu_usage_percent REAL NOT NULL,
                ram_usage_mb REAL NOT NULL,
                ram_total_mb REAL NOT NULL,
                bandwidth_in_mb REAL NOT NULL,
                bandwidth_out_mb REAL NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS daily_costs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                server_id TEXT NOT NULL,
                date DATE NOT NULL,
                cpu_cost REAL NOT NULL,
                ram_cost REAL NOT NULL,
                bandwidth_cost REAL NOT NULL,
                total_cost REAL NOT NULL
            );
            """
            
            with db.get_connection() as conn:
                conn.executescript(schema)
                
                # Insert test servers
                conn.execute("INSERT INTO servers (server_id, server_name, server_type) VALUES ('web-frontend', 'Web Frontend Server', 'web')")
                
                # Insert test metrics
                now = datetime.now()
                for i in range(10):
                    timestamp = now - timedelta(hours=i)
                    conn.execute(
                        "INSERT INTO resource_metrics (server_id, timestamp, cpu_usage_percent, ram_usage_mb, ram_total_mb, bandwidth_in_mb, bandwidth_out_mb) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        ('web-frontend', timestamp, 75.0, 4096.0, 8192.0, 100.0, 150.0)
                    )
                
                # Insert test costs
                for i in range(5):
                    cost_date = date.today() - timedelta(days=i)
                    conn.execute(
                        "INSERT INTO daily_costs (server_id, date, cpu_cost, ram_cost, bandwidth_cost, total_cost) VALUES (?, ?, ?, ?, ?, ?)",
                        ('web-frontend', cost_date, 0.20, 0.50, 0.30, 1.00)
                    )
                
                conn.commit()
    
    def tearDown(self):
        """Clean up test database."""
        os.close(self.db_fd)
        os.unlink(self.db_path)
    
    def test_api_servers(self):
        """Test /api/servers endpoint."""
        response = self.client.get('/api/servers')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('servers', data)
        self.assertGreater(len(data['servers']), 0)
        
        # Check server structure
        server = data['servers'][0]
        self.assertIn('id', server)
        self.assertIn('name', server)
        self.assertIn('type', server)
    
    def test_api_server_metrics(self):
        """Test /api/servers/<id>/metrics endpoint."""
        response = self.client.get('/api/servers/web-frontend/metrics')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('server_id', data)
        self.assertIn('metrics', data)
        self.assertIn('count', data)
        
        self.assertEqual(data['server_id'], 'web-frontend')
        self.assertGreater(data['count'], 0)
        
        # Check metric structure
        if data['metrics']:
            metric = data['metrics'][0]
            required_fields = ['timestamp', 'cpu_usage_percent', 'ram_usage_mb', 'bandwidth_out_mb']
            for field in required_fields:
                self.assertIn(field, metric)
    
    def test_api_server_costs(self):
        """Test /api/servers/<id>/costs endpoint."""
        response = self.client.get('/api/servers/web-frontend/costs')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('server_id', data)
        self.assertIn('costs', data)
        self.assertIn('summary', data)
        
        self.assertEqual(data['server_id'], 'web-frontend')
        
        # Check cost structure
        if data['costs']:
            cost = data['costs'][0]
            required_fields = ['date', 'cpu_cost', 'ram_cost', 'bandwidth_cost', 'total_cost']
            for field in required_fields:
                self.assertIn(field, cost)
        
        # Check summary structure
        summary = data['summary']
        summary_fields = ['total_cost', 'avg_daily_cost', 'max_daily_cost', 'min_daily_cost']
        for field in summary_fields:
            self.assertIn(field, summary)
    
    def test_api_invalid_server(self):
        """Test API endpoints with invalid server ID."""
        response = self.client.get('/api/servers/invalid-server/metrics')
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'Invalid server ID')
    
    def test_api_cost_comparison(self):
        """Test /api/costs/comparison endpoint."""
        response = self.client.get('/api/costs/comparison')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('servers', data)
        self.assertIn('comparison', data)
        self.assertIn('totals', data)
    
    def test_api_cost_trends(self):
        """Test /api/costs/trends endpoint."""
        response = self.client.get('/api/costs/trends')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('dates', data)
        self.assertIn('trends', data)
        self.assertIn('server_trends', data)
        
        trends = data['trends']
        trend_fields = ['total', 'cpu', 'ram', 'bandwidth']
        for field in trend_fields:
            self.assertIn(field, trends)
    
    def test_api_health(self):
        """Test /api/health endpoint."""
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertEqual(data['database'], 'connected')
        self.assertIn('servers_configured', data)
        self.assertIn('timestamp', data)
    
    def test_api_query_parameters(self):
        """Test API endpoints with query parameters."""
        # Test metrics with hours parameter
        response = self.client.get('/api/servers/web-frontend/metrics?hours=12')
        self.assertEqual(response.status_code, 200)
        
        # Test costs with days parameter
        response = self.client.get('/api/servers/web-frontend/costs?days=7')
        self.assertEqual(response.status_code, 200)
        
        # Test trends with granularity parameter
        response = self.client.get('/api/costs/trends?days=30&granularity=weekly')
        self.assertEqual(response.status_code, 200)

class TestErrorHandling(unittest.TestCase):
    """Test error handling in routes."""
    
    def setUp(self):
        """Set up test Flask application."""
        self.app = create_app('testing')
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
    
    def test_404_error(self):
        """Test 404 error handling."""
        response = self.client.get('/nonexistent-page')
        self.assertEqual(response.status_code, 404)
        self.assertIn(b'Page not found', response.data)
    
    def test_invalid_server_id(self):
        """Test handling of invalid server IDs."""
        response = self.client.get('/server/nonexistent-server')
        self.assertEqual(response.status_code, 404)

if __name__ == '__main__':
    unittest.main(verbosity=2)