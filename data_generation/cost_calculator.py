"""
Cost Calculation Engine for Resource-Based Web Server Cost Forecasting
Implements tiered pricing model for CPU, RAM, and bandwidth usage.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class PricingConfig:
    """Configuration for resource pricing."""
    cpu_hourly_rate: float
    ram_hourly_rate: float  # per GB per hour
    bandwidth_tier_1_limit: int  # MB
    bandwidth_tier_1_price: float  # per MB
    bandwidth_tier_2_limit: int  # MB
    bandwidth_tier_2_price: float  # per MB
    bandwidth_tier_3_price: float  # per MB

class CostCalculator:
    """Calculate costs for web server resources using tiered pricing."""
    
    def __init__(self, pricing_config: Optional[PricingConfig] = None):
        """Initialize with pricing configuration."""
        self.pricing = pricing_config or PricingConfig(
            cpu_hourly_rate=0.0116,
            ram_hourly_rate=0.0058,
            bandwidth_tier_1_limit=10000,    # 10GB in MB
            bandwidth_tier_1_price=0.00009,  # $0.09/GB = $0.00009/MB
            bandwidth_tier_2_limit=40000,    # 40GB in MB
            bandwidth_tier_2_price=0.000085, # $0.085/GB = $0.000085/MB
            bandwidth_tier_3_price=0.00007   # $0.07/GB = $0.00007/MB
        )
    
    def calculate_cpu_cost(self, cpu_usage_percent: float, hours: float) -> float:
        """Calculate CPU cost based on usage percentage and time."""
        cpu_fraction = cpu_usage_percent / 100.0
        cost = cpu_fraction * hours * self.pricing.cpu_hourly_rate
        return max(0, cost)
    
    def calculate_ram_cost(self, ram_usage_mb: float, hours: float) -> float:
        """Calculate RAM cost based on usage in MB and time."""
        ram_gb = ram_usage_mb / 1024.0
        cost = ram_gb * hours * self.pricing.ram_hourly_rate
        return max(0, cost)
    
    def calculate_tiered_bandwidth_cost(self, bandwidth_mb: float) -> float:
        """Calculate bandwidth cost using tiered pricing model."""
        if bandwidth_mb <= 0:
            return 0.0
        
        total_cost = 0.0
        remaining_bandwidth = bandwidth_mb
        
        # Tier 1: First 10GB
        if remaining_bandwidth > 0:
            tier_1_usage = min(remaining_bandwidth, self.pricing.bandwidth_tier_1_limit)
            total_cost += tier_1_usage * self.pricing.bandwidth_tier_1_price
            remaining_bandwidth -= tier_1_usage
        
        # Tier 2: Next 30GB (10-40GB)
        if remaining_bandwidth > 0:
            tier_2_limit = self.pricing.bandwidth_tier_2_limit - self.pricing.bandwidth_tier_1_limit
            tier_2_usage = min(remaining_bandwidth, tier_2_limit)
            total_cost += tier_2_usage * self.pricing.bandwidth_tier_2_price
            remaining_bandwidth -= tier_2_usage
        
        # Tier 3: Over 40GB
        if remaining_bandwidth > 0:
            total_cost += remaining_bandwidth * self.pricing.bandwidth_tier_3_price
        
        return total_cost
    
    def calculate_interval_cost(self, cpu_percent: float, ram_mb: float, 
                              bandwidth_in_mb: float, bandwidth_out_mb: float,
                              interval_minutes: int = 5) -> Dict[str, float]:
        """Calculate cost for a single time interval (default 5 minutes)."""
        hours = interval_minutes / 60.0
        
        # Calculate individual component costs
        cpu_cost = self.calculate_cpu_cost(cpu_percent, hours)
        ram_cost = self.calculate_ram_cost(ram_mb, hours)
        bandwidth_cost = self.calculate_tiered_bandwidth_cost(bandwidth_out_mb)  # Only charge for outbound
        
        total_cost = cpu_cost + ram_cost + bandwidth_cost
        
        return {
            'cpu_cost': cpu_cost,
            'ram_cost': ram_cost,
            'bandwidth_cost': bandwidth_cost,
            'total_cost': total_cost,
            'cpu_hours': cpu_percent / 100.0 * hours,
            'ram_gb_hours': (ram_mb / 1024.0) * hours,
            'bandwidth_out_mb': bandwidth_out_mb
        }
    
    def calculate_daily_costs(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily aggregated costs from resource metrics."""
        daily_costs = []
        
        # Group by server and date
        metrics_df['date'] = pd.to_datetime(metrics_df['timestamp']).dt.date
        
        for (server_id, date_val), group in metrics_df.groupby(['server_id', 'date']):
            # Calculate costs for each interval
            interval_costs = []
            
            for _, row in group.iterrows():
                cost_data = self.calculate_interval_cost(
                    cpu_percent=row['cpu_usage_percent'],
                    ram_mb=row['ram_usage_mb'],
                    bandwidth_in_mb=row['bandwidth_in_mb'],
                    bandwidth_out_mb=row['bandwidth_out_mb'],
                    interval_minutes=5  # Assuming 5-minute intervals
                )
                interval_costs.append(cost_data)
            
            # Aggregate for the day
            total_cpu_cost = sum(cost['cpu_cost'] for cost in interval_costs)
            total_ram_cost = sum(cost['ram_cost'] for cost in interval_costs)
            total_bandwidth_cost = sum(cost['bandwidth_cost'] for cost in interval_costs)
            total_cpu_hours = sum(cost['cpu_hours'] for cost in interval_costs)
            total_ram_gb_hours = sum(cost['ram_gb_hours'] for cost in interval_costs)
            total_bandwidth_out_gb = sum(cost['bandwidth_out_mb'] for cost in interval_costs) / 1024.0
            
            daily_cost = {
                'server_id': server_id,
                'date': date_val,
                'cpu_hours': round(total_cpu_hours, 4),
                'cpu_cost': round(total_cpu_cost, 6),
                'ram_gb_hours': round(total_ram_gb_hours, 4),
                'ram_cost': round(total_ram_cost, 6),
                'bandwidth_in_gb': round(group['bandwidth_in_mb'].sum() / 1024.0, 4),
                'bandwidth_out_gb': round(total_bandwidth_out_gb, 4),
                'bandwidth_cost': round(total_bandwidth_cost, 6),
                'total_cost': round(total_cpu_cost + total_ram_cost + total_bandwidth_cost, 6)
            }
            
            daily_costs.append(daily_cost)
        
        return pd.DataFrame(daily_costs).sort_values(['date', 'server_id']).reset_index(drop=True)
    
    def calculate_monthly_forecast(self, daily_costs_df: pd.DataFrame, 
                                 target_month: int, target_year: int) -> Dict[str, Dict[str, float]]:
        """Calculate monthly cost forecast based on historical daily costs."""
        # Filter data for the target month
        daily_costs_df['date'] = pd.to_datetime(daily_costs_df['date'])
        monthly_data = daily_costs_df[
            (daily_costs_df['date'].dt.month == target_month) & 
            (daily_costs_df['date'].dt.year == target_year)
        ]
        
        if monthly_data.empty:
            return {}
        
        monthly_forecasts = {}
        
        for server_id in monthly_data['server_id'].unique():
            server_data = monthly_data[monthly_data['server_id'] == server_id]
            
            # Calculate statistics
            avg_daily_cost = server_data['total_cost'].mean()
            max_daily_cost = server_data['total_cost'].max()
            min_daily_cost = server_data['total_cost'].min()
            std_daily_cost = server_data['total_cost'].std()
            
            # Days in target month
            if target_month in [1, 3, 5, 7, 8, 10, 12]:
                days_in_month = 31
            elif target_month in [4, 6, 9, 11]:
                days_in_month = 30
            else:  # February
                if target_year % 4 == 0 and (target_year % 100 != 0 or target_year % 400 == 0):
                    days_in_month = 29
                else:
                    days_in_month = 28
            
            # Calculate forecast
            monthly_forecast = avg_daily_cost * days_in_month
            monthly_min = min_daily_cost * days_in_month
            monthly_max = max_daily_cost * days_in_month
            
            # Confidence intervals (assuming normal distribution)
            confidence_margin = 1.96 * std_daily_cost * np.sqrt(days_in_month)
            confidence_lower = max(0, monthly_forecast - confidence_margin)
            confidence_upper = monthly_forecast + confidence_margin
            
            monthly_forecasts[server_id] = {
                'forecast_cost': round(monthly_forecast, 2),
                'min_cost': round(monthly_min, 2),
                'max_cost': round(monthly_max, 2),
                'confidence_lower': round(confidence_lower, 2),
                'confidence_upper': round(confidence_upper, 2),
                'avg_daily_cost': round(avg_daily_cost, 4),
                'std_daily_cost': round(std_daily_cost, 4),
                'days_in_month': days_in_month,
                'data_points': len(server_data)
            }
        
        return monthly_forecasts
    
    def get_cost_breakdown(self, daily_costs_df: pd.DataFrame, 
                          server_id: Optional[str] = None,
                          start_date: Optional[date] = None,
                          end_date: Optional[date] = None) -> Dict[str, float]:
        """Get cost breakdown by component for analysis."""
        df = daily_costs_df.copy()
        
        # Apply filters
        if server_id:
            df = df[df['server_id'] == server_id]
        
        if start_date:
            df = df[pd.to_datetime(df['date']).dt.date >= start_date]
        
        if end_date:
            df = df[pd.to_datetime(df['date']).dt.date <= end_date]
        
        if df.empty:
            return {'cpu_cost': 0, 'ram_cost': 0, 'bandwidth_cost': 0, 'total_cost': 0}
        
        return {
            'cpu_cost': round(df['cpu_cost'].sum(), 4),
            'ram_cost': round(df['ram_cost'].sum(), 4),
            'bandwidth_cost': round(df['bandwidth_cost'].sum(), 4),
            'total_cost': round(df['total_cost'].sum(), 4),
            'cpu_percentage': round(df['cpu_cost'].sum() / df['total_cost'].sum() * 100, 1),
            'ram_percentage': round(df['ram_cost'].sum() / df['total_cost'].sum() * 100, 1),
            'bandwidth_percentage': round(df['bandwidth_cost'].sum() / df['total_cost'].sum() * 100, 1)
        }
    
    def optimize_cost_recommendations(self, daily_costs_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Generate cost optimization recommendations based on usage patterns."""
        recommendations = {}
        
        for server_id in daily_costs_df['server_id'].unique():
            server_data = daily_costs_df[daily_costs_df['server_id'] == server_id]
            server_recommendations = []
            
            # Calculate average costs and percentages
            breakdown = self.get_cost_breakdown(daily_costs_df, server_id)
            avg_daily_cost = breakdown['total_cost'] / len(server_data)
            
            # High bandwidth cost recommendation
            if breakdown['bandwidth_percentage'] > 50:
                server_recommendations.append(
                    f"Consider implementing CDN or caching to reduce bandwidth costs "
                    f"({breakdown['bandwidth_percentage']:.1f}% of total cost)"
                )
            
            # High CPU cost recommendation
            if breakdown['cpu_percentage'] > 40:
                server_recommendations.append(
                    f"CPU usage is high ({breakdown['cpu_percentage']:.1f}% of cost). "
                    f"Consider optimizing application performance or scaling horizontally"
                )
            
            # High RAM cost recommendation
            if breakdown['ram_percentage'] > 35:
                server_recommendations.append(
                    f"Memory usage is significant ({breakdown['ram_percentage']:.1f}% of cost). "
                    f"Review for memory leaks or consider memory optimization"
                )
            
            # Cost trend analysis
            if len(server_data) >= 7:
                recent_costs = server_data.tail(7)['total_cost'].mean()
                earlier_costs = server_data.head(7)['total_cost'].mean()
                
                if recent_costs > earlier_costs * 1.2:
                    server_recommendations.append(
                        f"Cost trend is increasing. Recent average: ${recent_costs:.4f}/day vs "
                        f"earlier: ${earlier_costs:.4f}/day"
                    )
            
            # Daily cost threshold recommendations
            if avg_daily_cost > 10:
                server_recommendations.append(
                    f"High daily cost (${avg_daily_cost:.2f}). Consider rightsizing resources"
                )
            
            recommendations[server_id] = server_recommendations
        
        return recommendations

if __name__ == "__main__":
    # Test the cost calculator
    calculator = CostCalculator()
    
    # Test single interval calculation
    print("Testing single interval cost calculation:")
    cost = calculator.calculate_interval_cost(
        cpu_percent=75, 
        ram_mb=8192, 
        bandwidth_in_mb=100, 
        bandwidth_out_mb=150
    )
    print(f"5-minute interval cost: ${cost['total_cost']:.6f}")
    print(f"  CPU: ${cost['cpu_cost']:.6f}")
    print(f"  RAM: ${cost['ram_cost']:.6f}")
    print(f"  Bandwidth: ${cost['bandwidth_cost']:.6f}")
    
    # Test tiered bandwidth pricing
    print(f"\nTesting tiered bandwidth pricing:")
    test_bandwidths = [5000, 15000, 50000]  # 5GB, 15GB, 50GB in MB
    for bw in test_bandwidths:
        cost = calculator.calculate_tiered_bandwidth_cost(bw)
        print(f"  {bw/1024:.1f}GB: ${cost:.4f}")
    
    print(f"\nPricing configuration:")
    print(f"  CPU: ${calculator.pricing.cpu_hourly_rate}/hour")
    print(f"  RAM: ${calculator.pricing.ram_hourly_rate}/GB/hour")
    print(f"  Bandwidth Tier 1 (0-{calculator.pricing.bandwidth_tier_1_limit/1024:.0f}GB): ${calculator.pricing.bandwidth_tier_1_price*1024:.3f}/GB")
    print(f"  Bandwidth Tier 2 ({calculator.pricing.bandwidth_tier_1_limit/1024:.0f}-{calculator.pricing.bandwidth_tier_2_limit/1024:.0f}GB): ${calculator.pricing.bandwidth_tier_2_price*1024:.3f}/GB")
    print(f"  Bandwidth Tier 3 (>{calculator.pricing.bandwidth_tier_2_limit/1024:.0f}GB): ${calculator.pricing.bandwidth_tier_3_price*1024:.3f}/GB")