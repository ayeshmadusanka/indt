"""
Tests for forecasting services and machine learning models.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from pathlib import Path
import sys
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml_models.arima_forecaster import ARIMAForecaster
from data_generation.resource_simulator import ResourceSimulator
from data_generation.cost_calculator import CostCalculator

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

class TestARIMAForecaster(unittest.TestCase):
    """Test ARIMA forecasting model."""
    
    def setUp(self):
        """Set up test data and forecaster."""
        self.forecaster = ARIMAForecaster(max_p=3, max_q=3, max_d=2)
        
        # Generate synthetic time series data
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
        
        # Create trend + seasonality + noise
        trend = np.linspace(1, 2, 60)
        seasonal = 0.2 * np.sin(2 * np.pi * np.arange(60) / 7)  # Weekly pattern
        noise = np.random.normal(0, 0.1, 60)
        
        self.cost_data = pd.DataFrame({
            'date': dates,
            'total_cost': trend + seasonal + noise
        })
        
        # Split into train/test
        self.train_data = self.cost_data.iloc[:45]  # 45 days for training
        self.test_data = self.cost_data.iloc[45:]   # 15 days for testing
    
    def test_stationarity_check(self):
        """Test stationarity checking functionality."""
        # Create non-stationary series (with trend)
        non_stationary = pd.Series(np.cumsum(np.random.randn(100)))
        is_stationary, p_value = self.forecaster._check_stationarity(non_stationary)
        
        # Should detect non-stationarity
        self.assertFalse(is_stationary or p_value > 0.1)
        
        # Create stationary series
        stationary = pd.Series(np.random.randn(100))
        is_stationary, p_value = self.forecaster._check_stationarity(stationary)
        
        # Should be more likely to be stationary
        self.assertTrue(is_stationary or p_value < 0.05)
    
    def test_differencing(self):
        """Test differencing functionality."""
        # Create series with trend
        trend_series = pd.Series(np.cumsum(np.random.randn(50)))
        
        differenced, diff_count = self.forecaster._difference_series(trend_series, max_diff=2)
        
        self.assertGreaterEqual(diff_count, 0)
        self.assertLessEqual(diff_count, 2)
        self.assertLess(len(differenced), len(trend_series))  # Should be shorter due to differencing
    
    def test_model_fitting(self):
        """Test ARIMA model fitting."""
        success = self.forecaster.fit(self.train_data, 'test-server')
        
        self.assertTrue(success)
        self.assertIsNotNone(self.forecaster.fitted_model)
        self.assertIsNotNone(self.forecaster.best_params)
        
        # Check that parameters are reasonable
        p, d, q = self.forecaster.best_params
        self.assertGreaterEqual(p, 0)
        self.assertGreaterEqual(d, 0)
        self.assertGreaterEqual(q, 0)
        self.assertLessEqual(p, 3)
        self.assertLessEqual(d, 2)
        self.assertLessEqual(q, 3)
    
    def test_prediction(self):
        """Test ARIMA prediction functionality."""
        # First fit the model
        success = self.forecaster.fit(self.train_data, 'test-server')
        self.assertTrue(success)
        
        # Generate forecast
        forecast_days = 10
        forecast_result = self.forecaster.predict(forecast_days, confidence_level=0.95)
        
        self.assertIn('dates', forecast_result)
        self.assertIn('forecasts', forecast_result)
        self.assertIn('lower_bound', forecast_result)
        self.assertIn('upper_bound', forecast_result)
        
        # Check forecast length
        self.assertEqual(len(forecast_result['forecasts']), forecast_days)
        self.assertEqual(len(forecast_result['dates']), forecast_days)
        
        # Check that forecasts are non-negative
        self.assertTrue(all(f >= 0 for f in forecast_result['forecasts']))
        
        # Check that confidence intervals make sense
        for i in range(forecast_days):
            self.assertLessEqual(forecast_result['lower_bound'][i], forecast_result['forecasts'][i])
            self.assertLessEqual(forecast_result['forecasts'][i], forecast_result['upper_bound'][i])
    
    def test_model_validation(self):
        """Test ARIMA model validation."""
        # Fit model on training data
        success = self.forecaster.fit(self.train_data, 'test-server')
        self.assertTrue(success)
        
        # Validate on test data
        metrics = self.forecaster.validate_model(self.test_data)
        
        # Check that all expected metrics are present
        expected_metrics = ['mae', 'rmse', 'mape', 'r2_score', 'model_name', 'validation_points']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Check that metrics are reasonable
        self.assertGreater(metrics['mae'], 0)
        self.assertGreater(metrics['rmse'], 0)
        self.assertGreater(metrics['mape'], 0)
        self.assertEqual(metrics['validation_points'], len(self.test_data))
    
    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        # Create very small dataset
        small_data = self.cost_data.head(5)  # Only 5 data points
        
        success = self.forecaster.fit(small_data, 'test-server')
        self.assertFalse(success)  # Should fail due to insufficient data
    
    def test_residual_diagnosis(self):
        """Test residual diagnosis functionality."""
        # Fit model first
        success = self.forecaster.fit(self.train_data, 'test-server')
        self.assertTrue(success)
        
        diagnosis = self.forecaster.diagnose_residuals()
        
        # Check that diagnosis contains expected keys
        expected_keys = ['ljung_box_pvalue', 'residuals_autocorrelated', 'residual_mean', 'model_aic']
        for key in expected_keys:
            self.assertIn(key, diagnosis)
        
        # Check that values are reasonable
        self.assertIsInstance(diagnosis['residuals_autocorrelated'], bool)
        self.assertGreater(diagnosis['model_aic'], 0)

class TestProphetForecaster(unittest.TestCase):
    """Test Facebook Prophet forecasting model."""
    
    def setUp(self):
        """Set up test data and forecaster."""
        try:
            from ml_models.prophet_forecaster import ProphetForecaster, PROPHET_AVAILABLE
            if not PROPHET_AVAILABLE:
                self.skipTest("Prophet not available")
            
            self.forecaster = ProphetForecaster()
            
            # Generate synthetic time series data with stronger patterns
            np.random.seed(42)
            dates = pd.date_range(start='2024-01-01', periods=90, freq='D')
            
            # Create trend + weekly seasonality + noise
            trend = np.linspace(1, 3, 90)
            weekly_pattern = 0.5 * np.sin(2 * np.pi * np.arange(90) / 7)
            noise = np.random.normal(0, 0.2, 90)
            
            self.cost_data = pd.DataFrame({
                'date': dates,
                'total_cost': trend + weekly_pattern + noise
            })
            
            # Split into train/test
            self.train_data = self.cost_data.iloc[:70]  # 70 days for training
            self.test_data = self.cost_data.iloc[70:]   # 20 days for testing
            
        except ImportError:
            self.skipTest("Prophet not available")
    
    def test_data_preparation(self):
        """Test Prophet data preparation."""
        prepared_data = self.forecaster._prepare_data(self.cost_data)
        
        self.assertIn('ds', prepared_data.columns)
        self.assertIn('y', prepared_data.columns)
        self.assertEqual(len(prepared_data), len(self.cost_data))
        
        # Check that all y values are positive
        self.assertTrue(all(prepared_data['y'] > 0))
    
    def test_holidays_dataframe(self):
        """Test holidays dataframe creation."""
        holidays_df = self.forecaster._create_holidays_dataframe()
        
        self.assertIn('holiday', holidays_df.columns)
        self.assertIn('ds', holidays_df.columns)
        self.assertGreater(len(holidays_df), 0)
        
        # Check that holidays span multiple years
        holiday_years = pd.to_datetime(holidays_df['ds']).dt.year.unique()
        self.assertGreater(len(holiday_years), 1)
    
    def test_model_fitting(self):
        """Test Prophet model fitting."""
        success = self.forecaster.fit(self.train_data, 'test-server')
        
        self.assertTrue(success)
        self.assertIsNotNone(self.forecaster.fitted_model)
        self.assertIsNotNone(self.forecaster.training_data)
    
    def test_prediction(self):
        """Test Prophet prediction functionality."""
        # First fit the model
        success = self.forecaster.fit(self.train_data, 'test-server')
        self.assertTrue(success)
        
        # Generate forecast
        forecast_days = 15
        forecast_result = self.forecaster.predict(forecast_days, confidence_level=0.95)
        
        self.assertIn('dates', forecast_result)
        self.assertIn('forecasts', forecast_result)
        self.assertIn('lower_bound', forecast_result)
        self.assertIn('upper_bound', forecast_result)
        self.assertIn('trend', forecast_result)
        
        # Check forecast length
        self.assertEqual(len(forecast_result['forecasts']), forecast_days)
        
        # Check that forecasts are non-negative
        self.assertTrue(all(f >= 0 for f in forecast_result['forecasts']))
    
    def test_model_validation(self):
        """Test Prophet model validation."""
        # Fit model on training data
        success = self.forecaster.fit(self.train_data, 'test-server')
        self.assertTrue(success)
        
        # Validate on test data
        metrics = self.forecaster.validate_model(self.test_data)
        
        # Check that all expected metrics are present
        expected_metrics = ['mae', 'rmse', 'mape', 'r2_score', 'model_name', 'validation_points']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Check that metrics are reasonable
        self.assertGreater(metrics['mae'], 0)
        self.assertGreater(metrics['rmse'], 0)
        self.assertGreater(metrics['mape'], 0)
        self.assertEqual(metrics['model_name'], 'Prophet')
    
    def test_component_analysis(self):
        """Test Prophet component analysis."""
        # Fit model first
        success = self.forecaster.fit(self.train_data, 'test-server')
        self.assertTrue(success)
        
        components = self.forecaster.get_component_analysis(forecast_days=10)
        
        self.assertIn('historical', components)
        self.assertIn('forecast', components)
        self.assertIn('full_forecast', components)
        
        # Check that forecast has expected length
        self.assertEqual(len(components['forecast']), 10)
    
    def test_changepoint_detection(self):
        """Test changepoint detection."""
        # Fit model first
        success = self.forecaster.fit(self.train_data, 'test-server')
        self.assertTrue(success)
        
        changepoints = self.forecaster.detect_changepoints()
        
        self.assertIn('all_changepoints', changepoints)
        self.assertIn('significant_changepoints', changepoints)
        self.assertIn('changepoint_count', changepoints)
        
        # Check that changepoint count is reasonable
        self.assertGreaterEqual(changepoints['changepoint_count'], 0)
    
    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        # Create very small dataset
        small_data = self.cost_data.head(10)  # Only 10 data points
        
        success = self.forecaster.fit(small_data, 'test-server')
        self.assertFalse(success)  # Should fail due to insufficient data

class TestResourceSimulatorIntegration(unittest.TestCase):
    """Test integration between resource simulator and cost calculator."""
    
    def setUp(self):
        """Set up simulator and calculator."""
        self.simulator = ResourceSimulator(seed=42)
        self.calculator = CostCalculator()
    
    def test_end_to_end_simulation(self):
        """Test complete simulation and cost calculation pipeline."""
        # Generate resource data
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 8)  # 7 days
        
        metrics_df = self.simulator.generate_server_data(
            'web-frontend', start_date, end_date, interval_minutes=60
        )
        
        # Calculate costs
        daily_costs_df = self.calculator.calculate_daily_costs(metrics_df)
        
        # Verify integration
        self.assertGreater(len(metrics_df), 0)
        self.assertGreater(len(daily_costs_df), 0)
        
        # Should have approximately 7 days of cost data
        self.assertEqual(len(daily_costs_df), 7)
        
        # All costs should be positive
        self.assertTrue(all(daily_costs_df['total_cost'] > 0))
        self.assertTrue(all(daily_costs_df['cpu_cost'] >= 0))
        self.assertTrue(all(daily_costs_df['ram_cost'] >= 0))
        self.assertTrue(all(daily_costs_df['bandwidth_cost'] >= 0))
    
    def test_multiple_servers_simulation(self):
        """Test simulation with multiple servers."""
        # Generate data for all servers
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 5)  # 4 days
        
        all_metrics = self.simulator.generate_all_servers_data(
            start_date, end_date, interval_minutes=120
        )
        
        # Calculate costs
        daily_costs = self.calculator.calculate_daily_costs(all_metrics)
        
        # Verify results
        servers = all_metrics['server_id'].unique()
        self.assertEqual(len(servers), 4)  # 4 configured servers
        
        # Should have cost data for each server
        cost_servers = daily_costs['server_id'].unique()
        self.assertEqual(len(cost_servers), 4)
        
        # Each server should have 4 days of data
        for server in servers:
            server_costs = daily_costs[daily_costs['server_id'] == server]
            self.assertEqual(len(server_costs), 4)
    
    def test_cost_forecasting_pipeline(self):
        """Test complete pipeline from simulation to forecasting."""
        # Generate historical data
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 2, 15)  # 45 days
        
        metrics_df = self.simulator.generate_server_data(
            'api-backend', start_date, end_date, interval_minutes=60
        )
        
        # Calculate daily costs
        daily_costs_df = self.calculator.calculate_daily_costs(metrics_df)
        
        # Prepare data for forecasting
        forecast_data = daily_costs_df[['date', 'total_cost']].copy()
        
        # Test with ARIMA forecaster
        forecaster = ARIMAForecaster(max_p=2, max_q=2, max_d=1)
        success = forecaster.fit(forecast_data, 'api-backend')
        
        if success:
            # Generate forecast
            forecast_result = forecaster.predict(10)
            
            # Verify forecast
            self.assertEqual(len(forecast_result['forecasts']), 10)
            self.assertTrue(all(f > 0 for f in forecast_result['forecasts']))
            
            # Forecasts should be in reasonable range compared to historical data
            historical_mean = forecast_data['total_cost'].mean()
            forecast_mean = np.mean(forecast_result['forecasts'])
            
            # Forecast should be within reasonable bounds of historical mean
            self.assertLess(abs(forecast_mean - historical_mean) / historical_mean, 2.0)

if __name__ == '__main__':
    unittest.main(verbosity=2)