"""
Facebook Prophet Forecasting Model for Cost Prediction
Implements Prophet for seasonal trend analysis and robust forecasting with holidays.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional, Union
import warnings
import logging

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class ProphetForecaster:
    """Facebook Prophet-based cost forecasting model."""
    
    def __init__(self, 
                 growth: str = 'linear',
                 seasonality_mode: str = 'additive',
                 yearly_seasonality: bool = True,
                 weekly_seasonality: bool = True,
                 daily_seasonality: bool = False,
                 seasonality_prior_scale: float = 10.0,
                 holidays_prior_scale: float = 10.0,
                 changepoint_prior_scale: float = 0.05,
                 interval_width: float = 0.95):
        """
        Initialize Prophet forecaster.
        
        Args:
            growth: 'linear' or 'logistic' growth
            seasonality_mode: 'additive' or 'multiplicative'
            yearly_seasonality: Include yearly seasonality
            weekly_seasonality: Include weekly seasonality  
            daily_seasonality: Include daily seasonality
            seasonality_prior_scale: Prior scale for seasonality
            holidays_prior_scale: Prior scale for holidays
            changepoint_prior_scale: Prior scale for changepoint detection
            interval_width: Confidence interval width
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not available. Install with: pip install prophet")
        
        self.growth = growth
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.changepoint_prior_scale = changepoint_prior_scale
        self.interval_width = interval_width
        
        self.model = None
        self.fitted_model = None
        self.training_data = None
        self.holidays_df = None
        
    def _create_holidays_dataframe(self) -> pd.DataFrame:
        """Create holidays dataframe for common business holidays."""
        holidays = []
        
        # Define common holidays that might affect server costs
        yearly_holidays = [
            {'holiday': 'New Year', 'ds': '2024-01-01', 'lower_window': 0, 'upper_window': 1},
            {'holiday': 'Christmas', 'ds': '2024-12-25', 'lower_window': -1, 'upper_window': 1},
            {'holiday': 'Thanksgiving', 'ds': '2024-11-28', 'lower_window': 0, 'upper_window': 3},
            {'holiday': 'Black Friday', 'ds': '2024-11-29', 'lower_window': 0, 'upper_window': 3},
            {'holiday': 'Independence Day', 'ds': '2024-07-04', 'lower_window': 0, 'upper_window': 1},
        ]
        
        # Extend holidays for multiple years
        base_year = 2023
        for year_offset in range(3):  # 2023, 2024, 2025
            current_year = base_year + year_offset
            for holiday in yearly_holidays:
                holiday_copy = holiday.copy()
                original_date = datetime.strptime(holiday['ds'], '%Y-%m-%d')
                new_date = original_date.replace(year=current_year)
                holiday_copy['ds'] = new_date.strftime('%Y-%m-%d')
                holidays.append(holiday_copy)
        
        # Add end-of-month spikes (common in business)
        for year in range(2023, 2026):
            for month in range(1, 13):
                if month == 12:
                    next_month = 1
                    next_year = year + 1
                else:
                    next_month = month + 1
                    next_year = year
                
                # Last day of month
                last_day = (datetime(next_year, next_month, 1) - timedelta(days=1)).strftime('%Y-%m-%d')
                holidays.append({
                    'holiday': 'Month End',
                    'ds': last_day,
                    'lower_window': 0,
                    'upper_window': 0
                })
        
        return pd.DataFrame(holidays)
    
    def _prepare_data(self, cost_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data in Prophet format (ds, y columns)."""
        if 'date' not in cost_data.columns or 'total_cost' not in cost_data.columns:
            raise ValueError("cost_data must have 'date' and 'total_cost' columns")
        
        # Prepare Prophet format
        prophet_data = cost_data.copy()
        prophet_data['ds'] = pd.to_datetime(prophet_data['date'])
        prophet_data['y'] = prophet_data['total_cost']
        
        # Sort by date
        prophet_data = prophet_data.sort_values('ds').reset_index(drop=True)
        
        # Handle missing dates by resampling
        prophet_data = prophet_data.set_index('ds')
        prophet_data = prophet_data.resample('D').mean()
        prophet_data = prophet_data.interpolate(method='linear')
        prophet_data = prophet_data.fillna(method='bfill').fillna(method='ffill')
        prophet_data = prophet_data.reset_index()
        
        # Ensure positive values (Prophet works better with positive values)
        prophet_data['y'] = np.maximum(prophet_data['y'], 0.001)
        
        return prophet_data[['ds', 'y']]
    
    def _add_custom_regressors(self, model: Prophet, data: pd.DataFrame) -> Prophet:
        """Add custom regressors for business context."""
        # Add day of week effect
        data['dow'] = data['ds'].dt.dayofweek
        data['is_weekend'] = (data['dow'] >= 5).astype(int)
        model.add_regressor('is_weekend')
        
        # Add month effect for seasonal patterns
        data['month'] = data['ds'].dt.month
        data['is_high_activity_month'] = data['month'].isin([11, 12, 1]).astype(int)  # Holiday season
        model.add_regressor('is_high_activity_month')
        
        # Add day of month effect (end of month spikes)
        data['day_of_month'] = data['ds'].dt.day
        data['is_month_end'] = (data['day_of_month'] >= 28).astype(int)
        model.add_regressor('is_month_end')
        
        return model
    
    def fit(self, cost_data: pd.DataFrame, server_id: str) -> bool:
        """
        Fit Prophet model to historical cost data.
        
        Args:
            cost_data: DataFrame with columns ['date', 'total_cost']  
            server_id: Server identifier for logging
            
        Returns:
            bool: True if fitting was successful
        """
        try:
            logger.info(f"Fitting Prophet model for {server_id}")
            
            # Prepare data
            prophet_data = self._prepare_data(cost_data)
            
            if len(prophet_data) < 14:
                raise ValueError(f"Insufficient data points: {len(prophet_data)} (minimum: 14)")
            
            self.training_data = prophet_data
            
            # Create holidays dataframe
            self.holidays_df = self._create_holidays_dataframe()
            
            # Initialize Prophet model
            self.model = Prophet(
                growth=self.growth,
                seasonality_mode=self.seasonality_mode,
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
                seasonality_prior_scale=self.seasonality_prior_scale,
                holidays_prior_scale=self.holidays_prior_scale,
                changepoint_prior_scale=self.changepoint_prior_scale,
                interval_width=self.interval_width,
                holidays=self.holidays_df
            )
            
            # Add custom regressors
            self.model = self._add_custom_regressors(self.model, prophet_data)
            
            # Fit the model
            logger.info("Fitting Prophet model...")
            self.fitted_model = self.model.fit(prophet_data)
            
            logger.info(f"Prophet model fitted successfully for {server_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Prophet fitting failed for {server_id}: {e}")
            return False
    
    def predict(self, forecast_days: int, confidence_level: float = 0.95) -> Dict[str, Union[List, np.ndarray]]:
        """
        Generate cost forecasts for specified number of days.
        
        Args:
            forecast_days: Number of days to forecast
            confidence_level: Confidence level for prediction intervals
            
        Returns:
            Dictionary with forecasts, confidence intervals, and dates
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=forecast_days)
            
            # Add custom regressors to future dataframe
            future['dow'] = future['ds'].dt.dayofweek
            future['is_weekend'] = (future['dow'] >= 5).astype(int)
            
            future['month'] = future['ds'].dt.month
            future['is_high_activity_month'] = future['month'].isin([11, 12, 1]).astype(int)
            
            future['day_of_month'] = future['ds'].dt.day
            future['is_month_end'] = (future['day_of_month'] >= 28).astype(int)
            
            # Generate forecast
            forecast = self.fitted_model.predict(future)
            
            # Extract forecast period only
            forecast_period = forecast.tail(forecast_days)
            
            # Ensure non-negative forecasts
            forecast_values = np.maximum(forecast_period['yhat'].values, 0)
            lower_bound = np.maximum(forecast_period['yhat_lower'].values, 0)
            upper_bound = np.maximum(forecast_period['yhat_upper'].values, 0)
            
            return {
                'dates': forecast_period['ds'].dt.date,
                'forecasts': forecast_values,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'trend': forecast_period['trend'].values,
                'seasonal': forecast_period.get('seasonal', np.zeros(len(forecast_period))).values,
                'weekly': forecast_period.get('weekly', np.zeros(len(forecast_period))).values
            }
            
        except Exception as e:
            logger.error(f"Prophet prediction failed: {e}")
            raise
    
    def validate_model(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Validate model performance against test data.
        
        Args:
            test_data: Test dataset with same structure as training data
            
        Returns:
            Dictionary with validation metrics (MAE, RMSE, MAPE)
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before validation")
        
        try:
            # Prepare test data
            test_prophet = self._prepare_data(test_data)
            
            # Create future dataframe for test period
            last_train_date = self.training_data['ds'].iloc[-1]
            test_future = test_prophet.copy()
            test_future = test_future[test_future['ds'] > last_train_date]
            
            if len(test_future) == 0:
                raise ValueError("Test data must be after training data")
            
            # Add regressors to test data
            test_future['dow'] = test_future['ds'].dt.dayofweek
            test_future['is_weekend'] = (test_future['dow'] >= 5).astype(int)
            test_future['month'] = test_future['ds'].dt.month
            test_future['is_high_activity_month'] = test_future['month'].isin([11, 12, 1]).astype(int)
            test_future['day_of_month'] = test_future['ds'].dt.day
            test_future['is_month_end'] = (test_future['day_of_month'] >= 28).astype(int)
            
            # Generate predictions
            predictions = self.fitted_model.predict(test_future)
            predicted_values = np.maximum(predictions['yhat'].values, 0)
            
            # Calculate metrics
            actual = test_future['y'].values
            predicted = predicted_values
            
            mae = np.mean(np.abs(actual - predicted))
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            
            # MAPE with handling for zero values
            non_zero_mask = actual != 0
            if np.any(non_zero_mask):
                mape = np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100
            else:
                mape = float('inf')
            
            # R-squared
            ss_res = np.sum((actual - predicted) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            metrics = {
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'r2_score': float(r2),
                'model_name': 'Prophet',
                'validation_points': len(actual)
            }
            
            logger.info(f"Prophet validation: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Prophet validation failed: {e}")
            raise
    
    def get_component_analysis(self, forecast_days: int = 30) -> Dict[str, pd.DataFrame]:
        """Get detailed component analysis (trend, seasonality, holidays)."""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before analysis")
        
        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=forecast_days)
            
            # Add regressors
            future['dow'] = future['ds'].dt.dayofweek
            future['is_weekend'] = (future['dow'] >= 5).astype(int)
            future['month'] = future['ds'].dt.month
            future['is_high_activity_month'] = future['month'].isin([11, 12, 1]).astype(int)
            future['day_of_month'] = future['ds'].dt.day
            future['is_month_end'] = (future['day_of_month'] >= 28).astype(int)
            
            # Generate forecast with components
            forecast = self.fitted_model.predict(future)
            
            # Separate historical and forecast components
            train_length = len(self.training_data)
            
            components = {
                'historical': forecast.head(train_length),
                'forecast': forecast.tail(forecast_days),
                'full_forecast': forecast
            }
            
            return components
            
        except Exception as e:
            logger.error(f"Component analysis failed: {e}")
            return {}
    
    def detect_changepoints(self) -> Dict[str, Union[List, pd.DataFrame]]:
        """Detect and analyze trend changepoints."""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before changepoint detection")
        
        try:
            # Get changepoint information
            changepoints = self.fitted_model.changepoints
            changepoint_effects = self.fitted_model.params['delta'].mean(axis=0)
            
            # Create changepoint analysis
            changepoint_data = []
            for i, (cp_date, effect) in enumerate(zip(changepoints, changepoint_effects)):
                changepoint_data.append({
                    'date': cp_date,
                    'effect_magnitude': abs(effect),
                    'effect_direction': 'increase' if effect > 0 else 'decrease',
                    'significance': abs(effect) > np.std(changepoint_effects)
                })
            
            changepoint_df = pd.DataFrame(changepoint_data)
            
            # Get significant changepoints only
            significant_changepoints = changepoint_df[changepoint_df['significance']].copy()
            
            return {
                'all_changepoints': changepoint_df,
                'significant_changepoints': significant_changepoints,
                'changepoint_count': len(changepoints),
                'significant_count': len(significant_changepoints)
            }
            
        except Exception as e:
            logger.error(f"Changepoint detection failed: {e}")
            return {}
    
    def get_model_summary(self) -> str:
        """Get model configuration summary."""
        if self.fitted_model is None:
            return "Model not fitted"
        
        summary = f"""Prophet Model Summary:
        Growth: {self.growth}
        Seasonality Mode: {self.seasonality_mode}
        Yearly Seasonality: {self.yearly_seasonality}
        Weekly Seasonality: {self.weekly_seasonality}
        Daily Seasonality: {self.daily_seasonality}
        Holidays Included: {len(self.holidays_df) if self.holidays_df is not None else 0}
        Training Data Points: {len(self.training_data) if self.training_data is not None else 0}
        """
        
        return summary

if __name__ == "__main__":
    # Test the Prophet forecaster
    if not PROPHET_AVAILABLE:
        print("Prophet not available for testing")
        exit(1)
        
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data with trend and seasonality
    dates = pd.date_range(start='2024-01-01', periods=90, freq='D')
    trend = np.linspace(1, 3, 90)
    weekly_pattern = 0.3 * np.sin(2 * np.pi * np.arange(90) / 7)
    monthly_pattern = 0.2 * np.sin(2 * np.pi * np.arange(90) / 30)
    noise = np.random.normal(0, 0.1, 90)
    costs = trend + weekly_pattern + monthly_pattern + noise
    
    sample_data = pd.DataFrame({
        'date': dates,
        'total_cost': costs
    })
    
    # Split into train and test
    train_data = sample_data.iloc[:70]
    test_data = sample_data.iloc[70:]
    
    print("Testing Prophet Forecaster...")
    
    # Initialize and fit model
    forecaster = ProphetForecaster()
    success = forecaster.fit(train_data, 'test-server')
    
    if success:
        print("Prophet model fitted successfully")
        
        # Generate forecast
        forecast_result = forecaster.predict(20, confidence_level=0.95)
        print(f"Generated {len(forecast_result['forecasts'])} day forecast")
        
        # Validate model
        validation_metrics = forecaster.validate_model(test_data)
        print(f"Validation metrics: MAE={validation_metrics['mae']:.4f}, MAPE={validation_metrics['mape']:.2f}%")
        
        # Analyze components
        components = forecaster.get_component_analysis()
        print(f"Component analysis completed: {len(components)} components")
        
        # Detect changepoints
        changepoints = forecaster.detect_changepoints()
        print(f"Changepoint analysis: {changepoints.get('changepoint_count', 0)} total, {changepoints.get('significant_count', 0)} significant")
    else:
        print("Prophet model fitting failed")