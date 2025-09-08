"""
Cost models and data structures for the forecasting application.
Provides data classes and utilities for handling cost-related data.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

@dataclass
class ResourceMetric:
    """Represents a single resource usage measurement."""
    server_id: str
    timestamp: datetime
    cpu_usage_percent: float
    ram_usage_mb: float
    ram_total_mb: float
    bandwidth_in_mb: float
    bandwidth_out_mb: float
    
    @property
    def ram_usage_percent(self) -> float:
        """Calculate RAM usage percentage."""
        if self.ram_total_mb > 0:
            return (self.ram_usage_mb / self.ram_total_mb) * 100
        return 0.0
    
    @property
    def ram_usage_gb(self) -> float:
        """Get RAM usage in GB."""
        return self.ram_usage_mb / 1024.0
    
    @property
    def ram_total_gb(self) -> float:
        """Get total RAM in GB."""
        return self.ram_total_mb / 1024.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

@dataclass
class DailyCost:
    """Represents daily aggregated costs for a server."""
    server_id: str
    date: date
    cpu_hours: float
    cpu_cost: float
    ram_gb_hours: float
    ram_cost: float
    bandwidth_in_gb: float
    bandwidth_out_gb: float
    bandwidth_cost: float
    total_cost: float
    
    @property
    def cpu_percentage(self) -> float:
        """CPU cost as percentage of total."""
        return (self.cpu_cost / self.total_cost * 100) if self.total_cost > 0 else 0
    
    @property
    def ram_percentage(self) -> float:
        """RAM cost as percentage of total."""
        return (self.ram_cost / self.total_cost * 100) if self.total_cost > 0 else 0
    
    @property
    def bandwidth_percentage(self) -> float:
        """Bandwidth cost as percentage of total."""
        return (self.bandwidth_cost / self.total_cost * 100) if self.total_cost > 0 else 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        data = asdict(self)
        data.update({
            'cpu_percentage': self.cpu_percentage,
            'ram_percentage': self.ram_percentage,
            'bandwidth_percentage': self.bandwidth_percentage
        })
        return data

@dataclass
class CostForecast:
    """Represents a cost forecast for a specific date."""
    server_id: str
    forecast_date: date
    predicted_cost: float
    confidence_interval_lower: Optional[float]
    confidence_interval_upper: Optional[float]
    model_used: str
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None
    created_at: Optional[datetime] = None
    
    @property
    def confidence_range(self) -> Optional[float]:
        """Calculate confidence interval range."""
        if self.confidence_interval_lower is not None and self.confidence_interval_upper is not None:
            return self.confidence_interval_upper - self.confidence_interval_lower
        return None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

@dataclass
class ServerSummary:
    """Summary statistics for a server."""
    server_id: str
    server_name: str
    server_type: str
    days_monitored: int
    first_metric: Optional[datetime]
    last_metric: Optional[datetime]
    avg_cpu_usage: float
    avg_ram_usage: float
    total_bandwidth_gb: float
    total_cost_to_date: float
    
    @property
    def days_since_first_metric(self) -> Optional[int]:
        """Days since first metric was recorded."""
        if self.first_metric:
            return (datetime.now() - self.first_metric).days
        return None
    
    @property
    def is_active(self) -> bool:
        """Check if server has recent data (within 24 hours)."""
        if self.last_metric:
            return (datetime.now() - self.last_metric).total_seconds() < 86400  # 24 hours
        return False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        data = asdict(self)
        data.update({
            'days_since_first_metric': self.days_since_first_metric,
            'is_active': self.is_active
        })
        return data

@dataclass
class ModelPerformance:
    """Model performance metrics."""
    model_name: str
    server_id: str
    training_start_date: date
    training_end_date: date
    test_start_date: date
    test_end_date: date
    mae: float
    rmse: float
    mape: float
    r2_score: Optional[float] = None
    
    @property
    def training_days(self) -> int:
        """Number of training days."""
        return (self.training_end_date - self.training_start_date).days
    
    @property
    def test_days(self) -> int:
        """Number of test days."""
        return (self.test_end_date - self.test_start_date).days
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        data = asdict(self)
        data.update({
            'training_days': self.training_days,
            'test_days': self.test_days
        })
        return data

class CostAnalyzer:
    """Utility class for analyzing costs and generating insights."""
    
    @staticmethod
    def calculate_cost_trends(daily_costs: List[DailyCost], 
                            window_days: int = 7) -> Dict[str, float]:
        """Calculate cost trends over a rolling window."""
        if len(daily_costs) < window_days * 2:
            return {'trend': 0.0, 'change_percent': 0.0}
        
        # Sort by date
        costs_sorted = sorted(daily_costs, key=lambda x: x.date)
        
        # Calculate recent and previous averages
        recent_costs = [c.total_cost for c in costs_sorted[-window_days:]]
        previous_costs = [c.total_cost for c in costs_sorted[-window_days*2:-window_days]]
        
        recent_avg = np.mean(recent_costs)
        previous_avg = np.mean(previous_costs)
        
        if previous_avg > 0:
            change_percent = ((recent_avg - previous_avg) / previous_avg) * 100
        else:
            change_percent = 0.0
        
        return {
            'trend': recent_avg - previous_avg,
            'change_percent': change_percent,
            'recent_avg': recent_avg,
            'previous_avg': previous_avg
        }
    
    @staticmethod
    def identify_cost_anomalies(daily_costs: List[DailyCost], 
                               std_threshold: float = 2.0) -> List[Tuple[DailyCost, str]]:
        """Identify cost anomalies using standard deviation threshold."""
        if len(daily_costs) < 7:
            return []
        
        costs = [c.total_cost for c in daily_costs]
        mean_cost = np.mean(costs)
        std_cost = np.std(costs)
        
        anomalies = []
        
        for daily_cost in daily_costs:
            deviation = abs(daily_cost.total_cost - mean_cost) / std_cost
            
            if deviation > std_threshold:
                if daily_cost.total_cost > mean_cost:
                    anomaly_type = f"High cost spike ({deviation:.1f}σ above mean)"
                else:
                    anomaly_type = f"Low cost anomaly ({deviation:.1f}σ below mean)"
                
                anomalies.append((daily_cost, anomaly_type))
        
        return anomalies
    
    @staticmethod
    def generate_cost_insights(daily_costs: List[DailyCost]) -> List[str]:
        """Generate insights and recommendations based on cost patterns."""
        if not daily_costs:
            return ["No cost data available for analysis"]
        
        insights = []
        
        # Calculate overall statistics
        total_costs = [c.total_cost for c in daily_costs]
        avg_daily_cost = np.mean(total_costs)
        max_daily_cost = max(total_costs)
        min_daily_cost = min(total_costs)
        
        # Component analysis
        cpu_costs = [c.cpu_cost for c in daily_costs]
        ram_costs = [c.ram_cost for c in daily_costs]
        bandwidth_costs = [c.bandwidth_cost for c in daily_costs]
        
        total_cpu = sum(cpu_costs)
        total_ram = sum(ram_costs)
        total_bandwidth = sum(bandwidth_costs)
        total_all = total_cpu + total_ram + total_bandwidth
        
        if total_all > 0:
            cpu_pct = (total_cpu / total_all) * 100
            ram_pct = (total_ram / total_all) * 100
            bandwidth_pct = (total_bandwidth / total_all) * 100
            
            insights.append(f"Cost breakdown: CPU {cpu_pct:.1f}%, RAM {ram_pct:.1f}%, Bandwidth {bandwidth_pct:.1f}%")
            
            # Dominant cost component
            if bandwidth_pct > 50:
                insights.append("Bandwidth is the dominant cost driver - consider CDN or caching optimization")
            elif cpu_pct > 40:
                insights.append("CPU usage drives significant costs - review application performance")
            elif ram_pct > 35:
                insights.append("Memory usage is substantial - check for memory leaks or optimization opportunities")
        
        # Variability analysis
        cost_cv = np.std(total_costs) / avg_daily_cost if avg_daily_cost > 0 else 0
        
        if cost_cv > 0.5:
            insights.append("High cost variability detected - usage patterns are inconsistent")
        elif cost_cv < 0.1:
            insights.append("Very stable cost patterns - good for predictable budgeting")
        
        # Trend analysis
        if len(daily_costs) >= 14:
            trend_analysis = CostAnalyzer.calculate_cost_trends(daily_costs, window_days=7)
            change_pct = trend_analysis['change_percent']
            
            if abs(change_pct) > 20:
                direction = "increasing" if change_pct > 0 else "decreasing"
                insights.append(f"Significant cost trend: {direction} by {abs(change_pct):.1f}% over recent period")
        
        # Cost level insights
        if avg_daily_cost > 5:
            insights.append(f"High daily cost average (${avg_daily_cost:.2f}) - consider resource optimization")
        elif avg_daily_cost < 0.50:
            insights.append(f"Low daily cost average (${avg_daily_cost:.2f}) - resources may be underutilized")
        
        return insights
    
    @staticmethod
    def compare_servers(server_costs: Dict[str, List[DailyCost]]) -> Dict[str, Dict[str, float]]:
        """Compare costs across multiple servers."""
        comparison = {}
        
        for server_id, costs in server_costs.items():
            if not costs:
                continue
            
            total_costs = [c.total_cost for c in costs]
            comparison[server_id] = {
                'total_cost': sum(total_costs),
                'avg_daily_cost': np.mean(total_costs),
                'max_daily_cost': max(total_costs),
                'min_daily_cost': min(total_costs),
                'cost_variability': np.std(total_costs),
                'days_analyzed': len(costs)
            }
        
        # Add rankings
        if len(comparison) > 1:
            # Rank by total cost
            total_costs_ranking = sorted(comparison.keys(), 
                                       key=lambda s: comparison[s]['total_cost'], 
                                       reverse=True)
            
            for i, server_id in enumerate(total_costs_ranking):
                comparison[server_id]['cost_rank'] = i + 1
        
        return comparison