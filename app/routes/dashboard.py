"""
Dashboard routes for the cost forecasting application.
Provides main dashboard views and server-specific analytics.
"""

from flask import Blueprint, render_template, request, jsonify, current_app
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any

from app.models.database import get_database, ResourceMetricsDAO, DailyCostsDAO
from app.models.cost_models import CostAnalyzer

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/')
def index():
    """Main dashboard showing overview of all servers."""
    try:
        db = get_database()
        metrics_dao = ResourceMetricsDAO(db)
        costs_dao = DailyCostsDAO(db)
        
        # Get date range for analysis (last 30 days)
        end_date = date.today()
        start_date = end_date - timedelta(days=30)
        
        # Get daily costs for all servers
        daily_costs_df = costs_dao.get_costs_by_date_range(start_date, end_date)
        
        # Calculate dashboard metrics
        dashboard_data = calculate_dashboard_metrics(daily_costs_df)
        
        # Get server summaries
        servers = ['web-frontend', 'api-backend', 'database', 'cache-server']
        server_summaries = []
        
        for server_id in servers:
            summary = metrics_dao.get_server_summary(server_id, days=7)
            if summary and summary.get('data_points', 0) > 0:
                server_costs = daily_costs_df[daily_costs_df['server_id'] == server_id]
                if not server_costs.empty:
                    summary['total_cost_7days'] = server_costs['total_cost'].sum()
                    summary['avg_daily_cost'] = server_costs['total_cost'].mean()
                else:
                    summary['total_cost_7days'] = 0
                    summary['avg_daily_cost'] = 0
                
                summary['server_id'] = server_id
                server_summaries.append(summary)
        
        # Get cost trends
        cost_trends = get_cost_trends_data(daily_costs_df)
        
        return render_template('dashboard.html', 
                             dashboard_data=dashboard_data,
                             server_summaries=server_summaries,
                             cost_trends=cost_trends)
    
    except Exception as e:
        current_app.logger.error(f"Dashboard error: {e}")
        return render_template('error.html', error="Failed to load dashboard data"), 500

@dashboard_bp.route('/server/<server_id>')
def server_detail(server_id: str):
    """Detailed analytics for a specific server."""
    try:
        db = get_database()
        metrics_dao = ResourceMetricsDAO(db)
        costs_dao = DailyCostsDAO(db)
        
        # Validate server_id
        valid_servers = ['web-frontend', 'api-backend', 'database', 'cache-server']
        if server_id not in valid_servers:
            return render_template('error.html', error="Server not found"), 404
        
        # Get date range for analysis
        days = int(request.args.get('days', 30))
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        # Get server metrics and costs
        server_summary = metrics_dao.get_server_summary(server_id, days=7)
        daily_costs = costs_dao.get_costs_by_date_range(start_date, end_date, server_id)
        recent_metrics = metrics_dao.get_latest_metrics(server_id, limit=288)  # 24 hours of 5-min data
        
        # Calculate server analytics
        server_analytics = calculate_server_analytics(server_id, daily_costs, recent_metrics)
        
        # Get cost breakdown
        cost_breakdown = calculate_cost_breakdown(daily_costs)
        
        return render_template('analytics.html',
                             server_id=server_id,
                             server_summary=server_summary,
                             server_analytics=server_analytics,
                             cost_breakdown=cost_breakdown,
                             days=days)
    
    except Exception as e:
        current_app.logger.error(f"Server detail error: {e}")
        return render_template('error.html', error="Failed to load server data"), 500

@dashboard_bp.route('/compare')
def compare_servers():
    """Compare multiple servers side by side."""
    try:
        db = get_database()
        costs_dao = DailyCostsDAO(db)
        
        # Get comparison period
        days = int(request.args.get('days', 30))
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        # Get costs for all servers
        daily_costs_df = costs_dao.get_costs_by_date_range(start_date, end_date)
        
        # Calculate comparison data
        server_comparison = calculate_server_comparison(daily_costs_df)
        
        return render_template('comparison.html',
                             server_comparison=server_comparison,
                             days=days)
    
    except Exception as e:
        current_app.logger.error(f"Server comparison error: {e}")
        return render_template('error.html', error="Failed to load comparison data"), 500

def calculate_dashboard_metrics(daily_costs_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate metrics for the main dashboard."""
    if daily_costs_df.empty:
        return {
            'total_cost': 0,
            'daily_cost': 0,
            'monthly_forecast': 0,
            'cost_change': 0,
            'active_servers': 0,
            'bandwidth_cost': 0,
            'cpu_cost': 0,
            'ram_cost': 0
        }
    
    # Calculate totals
    total_cost = daily_costs_df['total_cost'].sum()
    total_bandwidth_cost = daily_costs_df['bandwidth_cost'].sum()
    total_cpu_cost = daily_costs_df['cpu_cost'].sum()
    total_ram_cost = daily_costs_df['ram_cost'].sum()
    
    # Calculate recent daily average
    # Convert date column to datetime for comparison
    daily_costs_df['date_parsed'] = pd.to_datetime(daily_costs_df['date']).dt.date
    recent_7_days = daily_costs_df[daily_costs_df['date_parsed'] >= (date.today() - timedelta(days=7))]
    if not recent_7_days.empty:
        daily_totals = recent_7_days.groupby('date')['total_cost'].sum()
        daily_cost = daily_totals.mean()
        
        # Calculate cost change (recent 7 days vs previous 7 days)
        if len(daily_costs_df) >= 14:
            previous_7_days = daily_costs_df[
                (daily_costs_df['date_parsed'] >= (date.today() - timedelta(days=14))) &
                (daily_costs_df['date_parsed'] < (date.today() - timedelta(days=7)))
            ]
            if not previous_7_days.empty:
                prev_daily_totals = previous_7_days.groupby('date')['total_cost'].sum()
                prev_daily_cost = prev_daily_totals.mean()
                cost_change = ((daily_cost - prev_daily_cost) / prev_daily_cost * 100) if prev_daily_cost > 0 else 0
            else:
                cost_change = 0
        else:
            cost_change = 0
    else:
        daily_cost = 0
        cost_change = 0
    
    # Monthly forecast (simple projection based on recent average)
    monthly_forecast = daily_cost * 30
    
    # Active servers
    active_servers = daily_costs_df['server_id'].nunique()
    
    return {
        'total_cost': round(total_cost, 2),
        'daily_cost': round(daily_cost, 2),
        'monthly_forecast': round(monthly_forecast, 2),
        'cost_change': round(cost_change, 1),
        'active_servers': active_servers,
        'bandwidth_cost': round(total_bandwidth_cost, 2),
        'cpu_cost': round(total_cpu_cost, 2),
        'ram_cost': round(total_ram_cost, 2)
    }

def get_cost_trends_data(daily_costs_df: pd.DataFrame) -> Dict[str, Any]:
    """Get cost trends data for charts."""
    if daily_costs_df.empty:
        return {'dates': [], 'total_costs': [], 'server_costs': {}}
    
    # Group by date and calculate daily totals
    daily_totals = daily_costs_df.groupby('date').agg({
        'total_cost': 'sum',
        'cpu_cost': 'sum',
        'ram_cost': 'sum',
        'bandwidth_cost': 'sum'
    }).reset_index()
    
    # Sort by date
    daily_totals = daily_totals.sort_values('date')
    
    # Prepare data for charts
    dates = []
    for d in daily_totals['date']:
        if isinstance(d, str):
            dates.append(d)
        else:
            dates.append(d.strftime('%Y-%m-%d'))
    total_costs = daily_totals['total_cost'].round(4).tolist()
    cpu_costs = daily_totals['cpu_cost'].round(4).tolist()
    ram_costs = daily_totals['ram_cost'].round(4).tolist()
    bandwidth_costs = daily_totals['bandwidth_cost'].round(4).tolist()
    
    # Server-wise costs
    server_costs = {}
    for server_id in daily_costs_df['server_id'].unique():
        server_data = daily_costs_df[daily_costs_df['server_id'] == server_id]
        server_daily = server_data.groupby('date')['total_cost'].sum().reset_index()
        server_daily = server_daily.sort_values('date')
        
        # Align with main dates (fill missing dates with 0)
        server_costs[server_id] = []
        server_dict = dict(zip(server_daily['date'], server_daily['total_cost']))
        
        for date_str in dates:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
            cost = server_dict.get(date_obj, 0)
            server_costs[server_id].append(round(cost, 4))
    
    return {
        'dates': dates,
        'total_costs': total_costs,
        'cpu_costs': cpu_costs,
        'ram_costs': ram_costs,
        'bandwidth_costs': bandwidth_costs,
        'server_costs': server_costs
    }

def calculate_server_analytics(server_id: str, daily_costs: pd.DataFrame, 
                             recent_metrics: pd.DataFrame) -> Dict[str, Any]:
    """Calculate detailed analytics for a specific server."""
    analytics = {
        'total_cost': 0,
        'avg_daily_cost': 0,
        'cost_trend': 0,
        'peak_usage_hour': 'N/A',
        'efficiency_score': 0,
        'recommendations': []
    }
    
    if not daily_costs.empty:
        analytics['total_cost'] = daily_costs['total_cost'].sum()
        analytics['avg_daily_cost'] = daily_costs['total_cost'].mean()
        
        # Calculate trend (recent vs older data)
        if len(daily_costs) >= 14:
            recent_half = daily_costs.tail(len(daily_costs) // 2)
            older_half = daily_costs.head(len(daily_costs) // 2)
            
            recent_avg = recent_half['total_cost'].mean()
            older_avg = older_half['total_cost'].mean()
            
            if older_avg > 0:
                analytics['cost_trend'] = ((recent_avg - older_avg) / older_avg) * 100
    
    if not recent_metrics.empty:
        # Find peak usage hour
        recent_metrics['hour'] = pd.to_datetime(recent_metrics['timestamp']).dt.hour
        hourly_cpu = recent_metrics.groupby('hour')['cpu_usage_percent'].mean()
        peak_hour = hourly_cpu.idxmax()
        analytics['peak_usage_hour'] = f"{peak_hour:02d}:00"
        
        # Simple efficiency score (inverse of resource waste)
        avg_cpu = recent_metrics['cpu_usage_percent'].mean()
        avg_ram_pct = (recent_metrics['ram_usage_mb'] / recent_metrics['ram_total_mb'] * 100).mean()
        
        # Efficiency: higher usage = more efficient (up to 80%)
        cpu_efficiency = min(avg_cpu / 80 * 100, 100)
        ram_efficiency = min(avg_ram_pct / 80 * 100, 100)
        analytics['efficiency_score'] = (cpu_efficiency + ram_efficiency) / 2
    
    # Generate recommendations
    if not daily_costs.empty:
        cost_breakdown = calculate_cost_breakdown(daily_costs)
        
        if cost_breakdown['bandwidth_percentage'] > 50:
            analytics['recommendations'].append("High bandwidth costs - consider CDN implementation")
        
        if cost_breakdown['cpu_percentage'] > 45:
            analytics['recommendations'].append("CPU-heavy workload - review performance optimization")
        
        if analytics['cost_trend'] > 20:
            analytics['recommendations'].append("Increasing cost trend - monitor resource usage")
    
    return analytics

def calculate_cost_breakdown(daily_costs: pd.DataFrame) -> Dict[str, float]:
    """Calculate cost breakdown by component."""
    if daily_costs.empty:
        return {
            'cpu_cost': 0,
            'ram_cost': 0,
            'bandwidth_cost': 0,
            'total_cost': 0,
            'cpu_percentage': 0,
            'ram_percentage': 0,
            'bandwidth_percentage': 0
        }
    
    total_cpu = daily_costs['cpu_cost'].sum()
    total_ram = daily_costs['ram_cost'].sum()
    total_bandwidth = daily_costs['bandwidth_cost'].sum()
    total_cost = daily_costs['total_cost'].sum()
    
    if total_cost > 0:
        cpu_pct = (total_cpu / total_cost) * 100
        ram_pct = (total_ram / total_cost) * 100
        bandwidth_pct = (total_bandwidth / total_cost) * 100
    else:
        cpu_pct = ram_pct = bandwidth_pct = 0
    
    return {
        'cpu_cost': round(total_cpu, 4),
        'ram_cost': round(total_ram, 4),
        'bandwidth_cost': round(total_bandwidth, 4),
        'total_cost': round(total_cost, 4),
        'cpu_percentage': round(cpu_pct, 1),
        'ram_percentage': round(ram_pct, 1),
        'bandwidth_percentage': round(bandwidth_pct, 1)
    }

def calculate_server_comparison(daily_costs_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate data for server comparison."""
    if daily_costs_df.empty:
        return {'servers': [], 'comparison_data': {}}
    
    servers = daily_costs_df['server_id'].unique()
    comparison_data = {}
    
    for server_id in servers:
        server_costs = daily_costs_df[daily_costs_df['server_id'] == server_id]
        
        comparison_data[server_id] = {
            'total_cost': server_costs['total_cost'].sum(),
            'avg_daily_cost': server_costs['total_cost'].mean(),
            'max_daily_cost': server_costs['total_cost'].max(),
            'min_daily_cost': server_costs['total_cost'].min(),
            'days_active': len(server_costs),
            'cost_std': server_costs['total_cost'].std(),
            'cpu_percentage': (server_costs['cpu_cost'].sum() / server_costs['total_cost'].sum() * 100) if server_costs['total_cost'].sum() > 0 else 0,
            'ram_percentage': (server_costs['ram_cost'].sum() / server_costs['total_cost'].sum() * 100) if server_costs['total_cost'].sum() > 0 else 0,
            'bandwidth_percentage': (server_costs['bandwidth_cost'].sum() / server_costs['total_cost'].sum() * 100) if server_costs['total_cost'].sum() > 0 else 0
        }
    
    # Sort servers by total cost
    servers_sorted = sorted(servers, key=lambda s: comparison_data[s]['total_cost'], reverse=True)
    
    return {
        'servers': servers_sorted,
        'comparison_data': comparison_data
    }