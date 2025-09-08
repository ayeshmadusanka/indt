"""
Tests for database models and cost calculation logic.
"""

import unittest
import tempfile
import os
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.models.database import DatabaseManager, ResourceMetricsDAO, DailyCostsDAO
from app.models.cost_models import DailyCost, ResourceMetric, CostAnalyzer
from data_generation.cost_calculator import CostCalculator, PricingConfig
from data_generation.resource_simulator import ResourceSimulator

class TestDatabaseManager(unittest.TestCase):
    """Test database connection and basic operations."""
    
    def setUp(self):
        """Set up test database."""
        self.db_fd, self.db_path = tempfile.mkstemp()
        self.db_manager = DatabaseManager(self.db_path)
        
        # Create basic tables for testing
        schema = """
        CREATE TABLE IF NOT EXISTS test_table (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            value REAL
        );
        """
        
        with self.db_manager.get_connection() as conn:
            conn.executescript(schema)
    
    def tearDown(self):
        """Clean up test database."""
        os.close(self.db_fd)
        os.unlink(self.db_path)
    
    def test_connection(self):
        """Test database connection."""
        with self.db_manager.get_connection() as conn:
            self.assertIsNotNone(conn)
    
    def test_execute_query(self):
        """Test query execution."""
        # Insert test data
        result = self.db_manager.execute_update(
            "INSERT INTO test_table (name, value) VALUES (?, ?)",
            ("test", 123.45)
        )
        self.assertEqual(result, 1)
        
        # Query data
        rows = self.db_manager.execute_query("SELECT * FROM test_table")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['name'], 'test')
        self.assertEqual(rows[0]['value'], 123.45)
    
    def test_execute_query_df(self):
        """Test DataFrame query execution."""
        # Insert test data
        test_data = [("item1", 1.0), ("item2", 2.0), ("item3", 3.0)]
        self.db_manager.execute_many(
            "INSERT INTO test_table (name, value) VALUES (?, ?)",
            test_data
        )
        
        # Query as DataFrame
        df = self.db_manager.execute_query_df("SELECT * FROM test_table ORDER BY id")
        
        self.assertEqual(len(df), 3)
        self.assertListEqual(df['name'].tolist(), ['item1', 'item2', 'item3'])
        self.assertListEqual(df['value'].tolist(), [1.0, 2.0, 3.0])
    
    def test_table_exists(self):
        """Test table existence check."""
        self.assertTrue(self.db_manager.table_exists('test_table'))
        self.assertFalse(self.db_manager.table_exists('nonexistent_table'))

class TestCostCalculator(unittest.TestCase):
    """Test cost calculation logic."""
    
    def setUp(self):
        """Set up cost calculator with test pricing."""
        self.pricing = PricingConfig(
            cpu_hourly_rate=0.01,
            ram_hourly_rate=0.005,
            bandwidth_tier_1_limit=1000,  # 1GB in MB
            bandwidth_tier_1_price=0.0001,  # $0.1/GB
            bandwidth_tier_2_limit=5000,   # 5GB in MB
            bandwidth_tier_2_price=0.00008, # $0.08/GB
            bandwidth_tier_3_price=0.00006  # $0.06/GB
        )
        self.calculator = CostCalculator(self.pricing)
    
    def test_cpu_cost_calculation(self):
        """Test CPU cost calculation."""
        # 50% CPU for 1 hour
        cost = self.calculator.calculate_cpu_cost(50, 1)
        expected = 0.5 * 1 * 0.01  # 50% * 1 hour * $0.01
        self.assertAlmostEqual(cost, expected, places=6)
        
        # 100% CPU for 0.5 hours
        cost = self.calculator.calculate_cpu_cost(100, 0.5)
        expected = 1.0 * 0.5 * 0.01
        self.assertAlmostEqual(cost, expected, places=6)
    
    def test_ram_cost_calculation(self):
        """Test RAM cost calculation."""
        # 2GB for 1 hour
        cost = self.calculator.calculate_ram_cost(2048, 1)  # 2048 MB
        expected = 2 * 1 * 0.005  # 2GB * 1 hour * $0.005
        self.assertAlmostEqual(cost, expected, places=6)
    
    def test_tiered_bandwidth_calculation(self):
        """Test tiered bandwidth pricing."""
        # Test tier 1 (under 1GB)
        cost = self.calculator.calculate_tiered_bandwidth_cost(500)  # 0.5GB
        expected = 500 * 0.0001  # 500MB * $0.0001/MB
        self.assertAlmostEqual(cost, expected, places=6)
        
        # Test tier 2 (1-5GB)
        cost = self.calculator.calculate_tiered_bandwidth_cost(3000)  # 3GB
        expected = (1000 * 0.0001) + (2000 * 0.00008)  # Tier 1 + Tier 2
        self.assertAlmostEqual(cost, expected, places=6)
        
        # Test tier 3 (over 5GB)
        cost = self.calculator.calculate_tiered_bandwidth_cost(8000)  # 8GB
        expected = (1000 * 0.0001) + (4000 * 0.00008) + (3000 * 0.00006)
        self.assertAlmostEqual(cost, expected, places=6)
    
    def test_interval_cost_calculation(self):
        """Test complete interval cost calculation."""
        result = self.calculator.calculate_interval_cost(
            cpu_percent=75,
            ram_mb=4096,  # 4GB
            bandwidth_in_mb=200,
            bandwidth_out_mb=300,
            interval_minutes=5
        )
        
        # Verify all components are present
        self.assertIn('cpu_cost', result)
        self.assertIn('ram_cost', result)
        self.assertIn('bandwidth_cost', result)
        self.assertIn('total_cost', result)
        
        # Verify total is sum of components
        expected_total = result['cpu_cost'] + result['ram_cost'] + result['bandwidth_cost']
        self.assertAlmostEqual(result['total_cost'], expected_total, places=6)
        
        # Verify positive costs
        self.assertGreaterEqual(result['cpu_cost'], 0)
        self.assertGreaterEqual(result['ram_cost'], 0)
        self.assertGreaterEqual(result['bandwidth_cost'], 0)
    
    def test_daily_costs_calculation(self):
        """Test daily cost aggregation."""
        # Create sample metrics data
        dates = pd.date_range('2024-01-01', periods=2, freq='D')
        timestamps = []
        for date in dates:
            for hour in [0, 6, 12, 18]:  # 4 measurements per day
                timestamps.append(date + timedelta(hours=hour))
        
        metrics_data = []
        for i, timestamp in enumerate(timestamps):
            metrics_data.append({
                'server_id': 'test-server',
                'timestamp': timestamp,
                'cpu_usage_percent': 50 + i * 5,
                'ram_usage_mb': 2048 + i * 100,
                'bandwidth_in_mb': 100 + i * 10,
                'bandwidth_out_mb': 150 + i * 15
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        daily_costs_df = self.calculator.calculate_daily_costs(metrics_df)
        
        self.assertEqual(len(daily_costs_df), 2)  # 2 days
        self.assertTrue(all(daily_costs_df['server_id'] == 'test-server'))
        self.assertTrue(all(daily_costs_df['total_cost'] > 0))

class TestResourceSimulator(unittest.TestCase):
    """Test resource usage simulation."""
    
    def setUp(self):
        """Set up resource simulator."""
        self.simulator = ResourceSimulator(seed=42)
    
    def test_business_hours_factor(self):
        """Test business hours calculation."""
        # Business hour
        business_time = datetime(2024, 1, 15, 14, 0)  # Monday 2 PM
        factor = self.simulator.generate_business_hours_factor(business_time)
        self.assertGreater(factor, 0.5)  # Should be elevated
        
        # Night time
        night_time = datetime(2024, 1, 15, 2, 0)  # Monday 2 AM
        factor = self.simulator.generate_business_hours_factor(night_time)
        self.assertLess(factor, 0.5)  # Should be low
        
        # Weekend
        weekend_time = datetime(2024, 1, 13, 14, 0)  # Saturday 2 PM
        factor = self.simulator.generate_business_hours_factor(weekend_time)
        self.assertLess(factor, 1.0)  # Should be reduced
    
    def test_cpu_usage_generation(self):
        """Test CPU usage generation."""
        timestamp = datetime(2024, 1, 15, 14, 0)
        cpu_usage = self.simulator.generate_cpu_usage('web-frontend', timestamp)
        
        self.assertGreaterEqual(cpu_usage, 0)
        self.assertLessEqual(cpu_usage, 100)
        self.assertIsInstance(cpu_usage, float)
    
    def test_ram_usage_generation(self):
        """Test RAM usage generation."""
        timestamp = datetime(2024, 1, 15, 14, 0)
        cpu_usage = 75.0
        ram_usage_mb, ram_total_mb = self.simulator.generate_ram_usage(
            'web-frontend', timestamp, cpu_usage
        )
        
        self.assertGreater(ram_usage_mb, 0)
        self.assertGreater(ram_total_mb, ram_usage_mb)
        self.assertIsInstance(ram_usage_mb, float)
        self.assertIsInstance(ram_total_mb, float)
    
    def test_bandwidth_usage_generation(self):
        """Test bandwidth usage generation."""
        timestamp = datetime(2024, 1, 15, 14, 0)
        cpu_usage = 75.0
        inbound_mb, outbound_mb = self.simulator.generate_bandwidth_usage(
            'web-frontend', timestamp, cpu_usage
        )
        
        self.assertGreaterEqual(inbound_mb, 0)
        self.assertGreaterEqual(outbound_mb, 0)
        self.assertIsInstance(inbound_mb, float)
        self.assertIsInstance(outbound_mb, float)
    
    def test_server_data_generation(self):
        """Test complete server data generation."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 3)  # 2 days
        
        data = self.simulator.generate_server_data(
            'web-frontend', start_date, end_date, interval_minutes=60
        )
        
        # Should have data points for each hour
        expected_points = int((end_date - start_date).total_seconds() / 3600)
        self.assertEqual(len(data), expected_points)
        
        # Check required columns
        required_columns = [
            'server_id', 'timestamp', 'cpu_usage_percent',
            'ram_usage_mb', 'ram_total_mb', 'bandwidth_in_mb', 'bandwidth_out_mb'
        ]
        for col in required_columns:
            self.assertIn(col, data.columns)
        
        # Check data validity
        self.assertTrue(all(data['server_id'] == 'web-frontend'))
        self.assertTrue(all(data['cpu_usage_percent'] >= 0))
        self.assertTrue(all(data['cpu_usage_percent'] <= 100))
        self.assertTrue(all(data['ram_usage_mb'] > 0))
        self.assertTrue(all(data['bandwidth_in_mb'] >= 0))

class TestCostModels(unittest.TestCase):
    """Test cost model data structures."""
    
    def test_resource_metric_creation(self):
        """Test ResourceMetric data class."""
        metric = ResourceMetric(
            server_id='test-server',
            timestamp=datetime.now(),
            cpu_usage_percent=75.5,
            ram_usage_mb=4096,
            ram_total_mb=8192,
            bandwidth_in_mb=100,
            bandwidth_out_mb=150
        )
        
        self.assertEqual(metric.ram_usage_percent, 50.0)
        self.assertEqual(metric.ram_usage_gb, 4.0)
        self.assertEqual(metric.ram_total_gb, 8.0)
    
    def test_daily_cost_creation(self):
        """Test DailyCost data class."""
        cost = DailyCost(
            server_id='test-server',
            date=date.today(),
            cpu_hours=12.0,
            cpu_cost=0.10,
            ram_gb_hours=48.0,
            ram_cost=0.20,
            bandwidth_in_gb=5.0,
            bandwidth_out_gb=8.0,
            bandwidth_cost=0.30,
            total_cost=0.60
        )
        
        self.assertAlmostEqual(cost.cpu_percentage, 16.67, places=1)
        self.assertAlmostEqual(cost.ram_percentage, 33.33, places=1)
        self.assertAlmostEqual(cost.bandwidth_percentage, 50.0, places=1)
    
    def test_cost_analyzer_trends(self):
        """Test cost trend analysis."""
        # Create sample daily costs
        daily_costs = []
        base_date = date(2024, 1, 1)
        
        for i in range(14):  # 14 days
            cost = DailyCost(
                server_id='test-server',
                date=base_date + timedelta(days=i),
                cpu_hours=10.0,
                cpu_cost=0.05 + i * 0.01,  # Increasing trend
                ram_gb_hours=20.0,
                ram_cost=0.10,
                bandwidth_in_gb=5.0,
                bandwidth_out_gb=8.0,
                bandwidth_cost=0.15,
                total_cost=0.30 + i * 0.01  # Increasing trend
            )
            daily_costs.append(cost)
        
        trends = CostAnalyzer.calculate_cost_trends(daily_costs, window_days=7)
        
        self.assertIn('trend', trends)
        self.assertIn('change_percent', trends)
        self.assertGreater(trends['change_percent'], 0)  # Should show increasing trend
    
    def test_cost_anomaly_detection(self):
        """Test cost anomaly detection."""
        # Create costs with one anomaly
        daily_costs = []
        base_date = date(2024, 1, 1)
        
        for i in range(10):
            if i == 5:  # Anomaly day
                total_cost = 10.0  # Much higher than normal
            else:
                total_cost = 1.0  # Normal cost
                
            cost = DailyCost(
                server_id='test-server',
                date=base_date + timedelta(days=i),
                cpu_hours=10.0,
                cpu_cost=total_cost * 0.3,
                ram_gb_hours=20.0,
                ram_cost=total_cost * 0.3,
                bandwidth_in_gb=5.0,
                bandwidth_out_gb=8.0,
                bandwidth_cost=total_cost * 0.4,
                total_cost=total_cost
            )
            daily_costs.append(cost)
        
        anomalies = CostAnalyzer.identify_cost_anomalies(daily_costs, std_threshold=2.0)
        
        self.assertGreater(len(anomalies), 0)
        anomaly_cost, anomaly_type = anomalies[0]
        self.assertEqual(anomaly_cost.total_cost, 10.0)
        self.assertIn('High cost spike', anomaly_type)

if __name__ == '__main__':
    unittest.main(verbosity=2)