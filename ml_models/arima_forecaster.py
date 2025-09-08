"""
ARIMA Time Series Forecasting Model for Cost Prediction
Implements Auto-ARIMA for automatic parameter selection and robust forecasting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional, Union
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class ARIMAForecaster:
    """ARIMA-based cost forecasting model with automatic parameter selection."""
    
    def __init__(self, max_p: int = 5, max_d: int = 2, max_q: int = 5, 
                 seasonal: bool = False, m: int = 7):
        """
        Initialize ARIMA forecaster.
        
        Args:
            max_p: Maximum AR order to test
            max_d: Maximum differencing order
            max_q: Maximum MA order to test
            seasonal: Whether to use seasonal ARIMA
            m: Seasonal period (7 for weekly seasonality)
        """
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.seasonal = seasonal
        self.m = m
        self.model = None
        self.fitted_model = None
        self.best_params = None
        self.training_data = None
        self.training_dates = None
        
    def _check_stationarity(self, series: pd.Series, significance: float = 0.05) -> Tuple[bool, float]:
        """Check if series is stationary using Augmented Dickey-Fuller test."""
        try:
            result = adfuller(series.dropna())
            adf_statistic = result[0]
            p_value = result[1]
            is_stationary = p_value < significance
            
            logger.debug(f"ADF test: statistic={adf_statistic:.4f}, p-value={p_value:.4f}, stationary={is_stationary}")
            return is_stationary, p_value
            
        except Exception as e:
            logger.warning(f"Stationarity test failed: {e}")
            return False, 1.0
    
    def _difference_series(self, series: pd.Series, max_diff: int = 2) -> Tuple[pd.Series, int]:
        """Apply differencing to make series stationary."""
        original_series = series.copy()
        diff_count = 0
        
        while diff_count < max_diff:
            is_stationary, p_value = self._check_stationarity(series)
            if is_stationary:
                break
                
            series = series.diff().dropna()
            diff_count += 1
            
            logger.debug(f"Applied {diff_count} differences, series length: {len(series)}")
        
        if diff_count == max_diff and not is_stationary:
            logger.warning("Series may still not be stationary after maximum differencing")
        
        return series, diff_count
    
    def _evaluate_arima_model(self, data: pd.Series, p: int, d: int, q: int) -> Optional[float]:
        """Evaluate ARIMA model with given parameters."""
        try:
            model = ARIMA(data, order=(p, d, q))
            fitted_model = model.fit()
            
            # Use AIC for model selection
            return fitted_model.aic
            
        except Exception as e:
            logger.debug(f"ARIMA({p},{d},{q}) failed: {e}")
            return None
    
    def _auto_arima(self, data: pd.Series) -> Tuple[int, int, int]:
        """Automatic ARIMA parameter selection using grid search."""
        logger.info("Starting Auto-ARIMA parameter selection")
        
        # Determine d (differencing order)
        _, d_optimal = self._difference_series(data, self.max_d)
        d_optimal = min(d_optimal, self.max_d)
        
        best_aic = float('inf')
        best_params = (1, d_optimal, 1)  # Default fallback
        
        # Grid search for p and q
        for p in range(0, self.max_p + 1):
            for q in range(0, self.max_q + 1):
                if p == 0 and q == 0:
                    continue  # Skip (0,d,0) model
                
                aic = self._evaluate_arima_model(data, p, d_optimal, q)
                
                if aic is not None and aic < best_aic:
                    best_aic = aic
                    best_params = (p, d_optimal, q)
                    logger.debug(f"New best model: ARIMA{best_params} with AIC={best_aic:.2f}")
        
        logger.info(f"Selected ARIMA{best_params} with AIC={best_aic:.2f}")
        return best_params
    
    def fit(self, cost_data: pd.DataFrame, server_id: str) -> bool:
        """
        Fit ARIMA model to historical cost data.
        
        Args:
            cost_data: DataFrame with columns ['date', 'total_cost']
            server_id: Server identifier for logging
            
        Returns:
            bool: True if fitting was successful
        """
        try:
            logger.info(f"Fitting ARIMA model for {server_id}")
            
            # Prepare data
            if 'date' not in cost_data.columns or 'total_cost' not in cost_data.columns:
                raise ValueError("cost_data must have 'date' and 'total_cost' columns")
            
            # Sort by date and create time series
            cost_data = cost_data.sort_values('date').copy()
            cost_data['date'] = pd.to_datetime(cost_data['date'])
            
            # Remove any duplicates and handle missing dates
            cost_data = cost_data.drop_duplicates('date').set_index('date')
            
            # Resample to ensure daily frequency (fill missing with interpolation)
            ts = cost_data['total_cost'].resample('D').mean()
            ts = ts.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
            
            if len(ts) < 10:
                raise ValueError(f"Insufficient data points: {len(ts)} (minimum: 10)")
            
            self.training_data = ts
            self.training_dates = ts.index
            
            # Automatic parameter selection
            self.best_params = self._auto_arima(ts)
            
            # Fit the best model
            self.model = ARIMA(ts, order=self.best_params)
            self.fitted_model = self.model.fit()
            
            logger.info(f"ARIMA{self.best_params} fitted successfully for {server_id}")
            logger.info(f"Model AIC: {self.fitted_model.aic:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"ARIMA fitting failed for {server_id}: {e}")
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
            # Generate forecasts
            forecast_result = self.fitted_model.forecast(steps=forecast_days)
            forecast_values = forecast_result
            
            # Get confidence intervals
            forecast_ci = self.fitted_model.get_forecast(steps=forecast_days).conf_int(alpha=1-confidence_level)
            
            # Generate forecast dates
            last_date = self.training_dates[-1]
            forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='D')
            
            # Ensure non-negative forecasts
            forecast_values = np.maximum(forecast_values, 0)
            lower_bound = np.maximum(forecast_ci.iloc[:, 0].values, 0)
            upper_bound = np.maximum(forecast_ci.iloc[:, 1].values, 0)
            
            return {
                'dates': forecast_dates,
                'forecasts': forecast_values,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'model_params': self.best_params
            }
            
        except Exception as e:
            logger.error(f"ARIMA prediction failed: {e}")
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
            test_data = test_data.sort_values('date').copy()
            test_data['date'] = pd.to_datetime(test_data['date'])
            test_data = test_data.set_index('date')
            
            test_ts = test_data['total_cost'].resample('D').mean()
            test_ts = test_ts.interpolate().fillna(method='bfill').fillna(method='ffill')
            
            # Generate predictions for test period
            forecast_steps = len(test_ts)
            predictions = self.fitted_model.forecast(steps=forecast_steps)
            predictions = np.maximum(predictions, 0)  # Ensure non-negative
            
            # Calculate metrics
            actual = test_ts.values
            predicted = predictions
            
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
                'model_name': f'ARIMA{self.best_params}',
                'validation_points': len(actual)
            }
            
            logger.info(f"ARIMA validation: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")
            
            return metrics
            
        except Exception as e:
            logger.error(f"ARIMA validation failed: {e}")
            raise
    
    def diagnose_residuals(self) -> Dict[str, Union[bool, float]]:
        """Diagnose model residuals for goodness of fit."""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before diagnosis")
        
        try:
            residuals = self.fitted_model.resid
            
            # Ljung-Box test for autocorrelation in residuals
            lb_stat, lb_pvalue = acorr_ljungbox(residuals, lags=10, return_df=False)
            
            # Residual statistics
            residual_mean = np.mean(residuals)
            residual_std = np.std(residuals)
            
            # Normality test (simple check)
            residual_skewness = pd.Series(residuals).skew()
            residual_kurtosis = pd.Series(residuals).kurtosis()
            
            diagnosis = {
                'ljung_box_pvalue': float(lb_pvalue[-1]),  # Use last lag p-value
                'residuals_autocorrelated': bool(lb_pvalue[-1] < 0.05),
                'residual_mean': float(residual_mean),
                'residual_std': float(residual_std),
                'residual_skewness': float(residual_skewness),
                'residual_kurtosis': float(residual_kurtosis),
                'model_aic': float(self.fitted_model.aic),
                'model_bic': float(self.fitted_model.bic)
            }
            
            return diagnosis
            
        except Exception as e:
            logger.error(f"Residual diagnosis failed: {e}")
            return {}
    
    def get_model_summary(self) -> str:
        """Get detailed model summary."""
        if self.fitted_model is None:
            return "Model not fitted"
        
        try:
            return str(self.fitted_model.summary())
        except Exception as e:
            logger.error(f"Failed to generate model summary: {e}")
            return f"Model: ARIMA{self.best_params}, Error: {str(e)}"

if __name__ == "__main__":
    # Test the ARIMA forecaster
    logger.basicConfig(level=logging.INFO)
    
    # Generate sample data
    dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
    trend = np.linspace(1, 2, 60)
    seasonal = 0.1 * np.sin(2 * np.pi * np.arange(60) / 7)
    noise = np.random.normal(0, 0.05, 60)
    costs = trend + seasonal + noise
    
    sample_data = pd.DataFrame({
        'date': dates,
        'total_cost': costs
    })
    
    # Split into train and test
    train_data = sample_data.iloc[:50]
    test_data = sample_data.iloc[50:]
    
    print("Testing ARIMA Forecaster...")
    
    # Initialize and fit model
    forecaster = ARIMAForecaster()
    success = forecaster.fit(train_data, 'test-server')
    
    if success:
        print(f"Model fitted successfully: ARIMA{forecaster.best_params}")
        
        # Generate forecast
        forecast_result = forecaster.predict(10, confidence_level=0.95)
        print(f"Generated {len(forecast_result['forecasts'])} day forecast")
        
        # Validate model
        validation_metrics = forecaster.validate_model(test_data)
        print(f"Validation metrics: MAE={validation_metrics['mae']:.4f}, MAPE={validation_metrics['mape']:.2f}%")
        
        # Diagnose residuals
        diagnosis = forecaster.diagnose_residuals()
        print(f"Model diagnosis: AIC={diagnosis.get('model_aic', 'N/A'):.2f}")
    else:
        print("Model fitting failed")