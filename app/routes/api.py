"""
API routes for AJAX data requests.
Provides JSON endpoints for charts, real-time data, and analytics.
"""

from flask import Blueprint, jsonify, request, current_app
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

from app.models.database import get_database, ResourceMetricsDAO, DailyCostsDAO, ForecastDAO
from app.models.cost_models import CostAnalyzer

api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/servers')
def get_servers():
    """Get list of all servers with basic info."""
    try:
        servers = [
            {
                'id': 'web-frontend',
                'name': 'Web Frontend Server',
                'type': 'web',
                'description': 'Handles web requests and serves static content'
            },
            {
                'id': 'api-backend',
                'name': 'API Backend Server',
                'type': 'api',
                'description': 'Processes API requests and business logic'
            },
            {
                'id': 'database',
                'name': 'Database Server',
                'type': 'database',
                'description': 'Stores and manages application data'
            },
            {
                'id': 'cache-server',
                'name': 'Cache Server',
                'type': 'cache',
                'description': 'Caches frequently accessed data'
            }
        ]
        
        return jsonify({'servers': servers})
    
    except Exception as e:
        current_app.logger.error(f"Get servers error: {e}")
        return jsonify({'error': 'Failed to fetch servers'}), 500

@api_bp.route('/servers/<server_id>/metrics')
def get_server_metrics(server_id: str):
    """Get recent resource metrics for a server."""
    try:
        # Validate server_id
        valid_servers = ['web-frontend', 'api-backend', 'database', 'cache-server']
        if server_id not in valid_servers:
            return jsonify({'error': 'Invalid server ID'}), 400
        
        # Get query parameters
        hours = int(request.args.get('hours', 24))
        limit = min(int(request.args.get('limit', 500)), 1000)  # Cap at 1000 records
        
        db = get_database()
        metrics_dao = ResourceMetricsDAO(db)
        
        # Get recent metrics
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        metrics_df = metrics_dao.get_metrics_by_date_range(start_time, end_time, server_id)
        
        # Limit results if needed
        if len(metrics_df) > limit:
            # Sample evenly across the time range
            step = len(metrics_df) // limit
            metrics_df = metrics_df.iloc[::step]
        
        # Convert to JSON-friendly format
        metrics_data = []
        for _, row in metrics_df.iterrows():
            metrics_data.append({
                'timestamp': row['timestamp'],
                'cpu_usage_percent': round(row['cpu_usage_percent'], 2),
                'ram_usage_mb': round(row['ram_usage_mb'], 2),
                'ram_total_mb': round(row['ram_total_mb'], 2),
                'ram_usage_percent': round((row['ram_usage_mb'] / row['ram_total_mb']) * 100, 2),
                'bandwidth_in_mb': round(row['bandwidth_in_mb'], 2),
                'bandwidth_out_mb': round(row['bandwidth_out_mb'], 2)
            })
        
        return jsonify({
            'server_id': server_id,
            'metrics': metrics_data,
            'count': len(metrics_data),
            'time_range': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            }
        })
    
    except Exception as e:
        current_app.logger.error(f"Get server metrics error: {e}")
        return jsonify({'error': 'Failed to fetch metrics'}), 500

@api_bp.route('/servers/<server_id>/costs')
def get_server_costs(server_id: str):
    """Get daily cost data for a server."""
    try:
        # Validate server_id
        valid_servers = ['web-frontend', 'api-backend', 'database', 'cache-server']
        if server_id not in valid_servers:
            return jsonify({'error': 'Invalid server ID'}), 400
        
        # Get query parameters
        days = int(request.args.get('days', 30))
        
        db = get_database()
        costs_dao = DailyCostsDAO(db)
        
        # Get date range
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        # Get daily costs
        costs_df = costs_dao.get_costs_by_date_range(start_date, end_date, server_id)
        
        # Convert to JSON format
        costs_data = []
        for _, row in costs_df.iterrows():
            # Handle date formatting - check if it's already a string or needs conversion
            if isinstance(row['date'], str):
                date_str = row['date']
            else:
                date_str = row['date'].strftime('%Y-%m-%d')
            
            costs_data.append({
                'date': date_str,
                'cpu_cost': round(row['cpu_cost'], 6),
                'ram_cost': round(row['ram_cost'], 6),
                'bandwidth_cost': round(row['bandwidth_cost'], 6),
                'total_cost': round(row['total_cost'], 6),
                'cpu_hours': round(row['cpu_hours'], 4),
                'ram_gb_hours': round(row['ram_gb_hours'], 4),
                'bandwidth_out_gb': round(row['bandwidth_out_gb'], 4)
            })
        
        # Calculate summary statistics
        if costs_data:
            total_costs = [c['total_cost'] for c in costs_data]
            summary = {
                'total_cost': round(sum(total_costs), 4),
                'avg_daily_cost': round(sum(total_costs) / len(total_costs), 6),
                'max_daily_cost': round(max(total_costs), 6),
                'min_daily_cost': round(min(total_costs), 6)
            }
        else:
            summary = {
                'total_cost': 0,
                'avg_daily_cost': 0,
                'max_daily_cost': 0,
                'min_daily_cost': 0
            }
        
        return jsonify({
            'server_id': server_id,
            'costs': costs_data,
            'summary': summary,
            'count': len(costs_data),
            'date_range': {
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d')
            }
        })
    
    except Exception as e:
        current_app.logger.error(f"Get server costs error: {e}")
        return jsonify({'error': 'Failed to fetch costs'}), 500

@api_bp.route('/costs/comparison')
def get_cost_comparison():
    """Get cost comparison data across all servers."""
    try:
        # Get query parameters
        days = int(request.args.get('days', 30))
        
        db = get_database()
        costs_dao = DailyCostsDAO(db)
        
        # Get date range
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        # Get all costs
        costs_df = costs_dao.get_costs_by_date_range(start_date, end_date)
        
        if costs_df.empty:
            return jsonify({
                'servers': [],
                'comparison': {},
                'totals': {'cpu': 0, 'ram': 0, 'bandwidth': 0, 'total': 0}
            })
        
        # Calculate comparison data
        comparison = {}
        servers = costs_df['server_id'].unique()
        
        for server_id in servers:
            server_costs = costs_df[costs_df['server_id'] == server_id]
            
            comparison[server_id] = {
                'total_cost': round(server_costs['total_cost'].sum(), 4),
                'avg_daily_cost': round(server_costs['total_cost'].mean(), 6),
                'cpu_cost': round(server_costs['cpu_cost'].sum(), 4),
                'ram_cost': round(server_costs['ram_cost'].sum(), 4),
                'bandwidth_cost': round(server_costs['bandwidth_cost'].sum(), 4),
                'days_active': len(server_costs),
                'cost_variability': round(server_costs['total_cost'].std(), 6)
            }
        
        # Calculate totals
        totals = {
            'cpu': round(costs_df['cpu_cost'].sum(), 4),
            'ram': round(costs_df['ram_cost'].sum(), 4),
            'bandwidth': round(costs_df['bandwidth_cost'].sum(), 4),
            'total': round(costs_df['total_cost'].sum(), 4)
        }
        
        return jsonify({
            'servers': list(servers),
            'comparison': comparison,
            'totals': totals,
            'date_range': {
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d')
            }
        })
    
    except Exception as e:
        current_app.logger.error(f"Get cost comparison error: {e}")
        return jsonify({'error': 'Failed to fetch comparison data'}), 500

@api_bp.route('/costs/trends')
def get_cost_trends():
    """Get cost trend data for charts."""
    try:
        # Get query parameters
        days = int(request.args.get('days', 30))
        granularity = request.args.get('granularity', 'daily')  # daily, weekly
        
        db = get_database()
        costs_dao = DailyCostsDAO(db)
        
        # Get date range
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        # Get all costs
        costs_df = costs_dao.get_costs_by_date_range(start_date, end_date)
        
        if costs_df.empty:
            return jsonify({
                'dates': [],
                'trends': {'total': [], 'cpu': [], 'ram': [], 'bandwidth': []},
                'server_trends': {}
            })
        
        # Convert date column to datetime for grouping
        costs_df['date'] = pd.to_datetime(costs_df['date'])
        
        # Group by specified granularity
        if granularity == 'weekly':
            costs_df['period'] = costs_df['date'].dt.to_period('W').dt.start_time
            period_format = '%Y-%m-%d'
        else:
            costs_df['period'] = costs_df['date']
            period_format = '%Y-%m-%d'
        
        # Aggregate by period
        daily_totals = costs_df.groupby('period').agg({
            'total_cost': 'sum',
            'cpu_cost': 'sum',
            'ram_cost': 'sum',
            'bandwidth_cost': 'sum'
        }).reset_index()
        
        daily_totals = daily_totals.sort_values('period')
        
        # Prepare trend data
        dates = [p.strftime(period_format) for p in daily_totals['period']]
        trends = {
            'total': [round(x, 4) for x in daily_totals['total_cost'].tolist()],
            'cpu': [round(x, 4) for x in daily_totals['cpu_cost'].tolist()],
            'ram': [round(x, 4) for x in daily_totals['ram_cost'].tolist()],
            'bandwidth': [round(x, 4) for x in daily_totals['bandwidth_cost'].tolist()]
        }
        
        # Server-specific trends
        server_trends = {}
        for server_id in costs_df['server_id'].unique():
            server_data = costs_df[costs_df['server_id'] == server_id]
            server_daily = server_data.groupby('period')['total_cost'].sum().reset_index()
            server_daily = server_daily.sort_values('period')
            
            # Align with main dates
            server_costs_aligned = []
            server_dict = dict(zip(server_daily['period'], server_daily['total_cost']))
            
            for period in daily_totals['period']:
                cost = server_dict.get(period, 0)
                server_costs_aligned.append(round(cost, 4))
            
            server_trends[server_id] = server_costs_aligned
        
        return jsonify({
            'dates': dates,
            'trends': trends,
            'server_trends': server_trends,
            'granularity': granularity
        })
    
    except Exception as e:
        current_app.logger.error(f"Get cost trends error: {e}")
        return jsonify({'error': 'Failed to fetch trend data'}), 500

@api_bp.route('/forecast/<server_id>')
def get_forecast(server_id: str):
    """Get forecast data for a server."""
    try:
        # Validate server_id
        valid_servers = ['web-frontend', 'api-backend', 'database', 'cache-server']
        if server_id not in valid_servers:
            return jsonify({'error': 'Invalid server ID'}), 400
        
        # Get query parameters
        horizon_days = int(request.args.get('horizon', 30))
        
        db = get_database()
        forecast_dao = ForecastDAO(db)
        
        # Get forecasts
        end_date = date.today() + timedelta(days=horizon_days)
        forecasts_df = forecast_dao.get_forecasts(server_id, date.today(), end_date)
        
        # Convert to JSON format
        forecast_data = []
        for _, row in forecasts_df.iterrows():
            forecast_data.append({
                'date': row['forecast_date'].strftime('%Y-%m-%d'),
                'predicted_cost': round(row['predicted_cost'], 6),
                'confidence_lower': round(row['confidence_interval_lower'], 6) if row['confidence_interval_lower'] is not None else None,
                'confidence_upper': round(row['confidence_interval_upper'], 6) if row['confidence_interval_upper'] is not None else None,
                'model_used': row['model_used'],
                'mae': round(row['mae'], 6) if row['mae'] is not None else None,
                'rmse': round(row['rmse'], 6) if row['rmse'] is not None else None,
                'mape': round(row['mape'], 2) if row['mape'] is not None else None
            })
        
        # Calculate forecast summary
        if forecast_data:
            predicted_costs = [f['predicted_cost'] for f in forecast_data]
            forecast_summary = {
                'total_predicted_cost': round(sum(predicted_costs), 4),
                'avg_daily_cost': round(sum(predicted_costs) / len(predicted_costs), 6),
                'forecast_days': len(forecast_data),
                'confidence_available': any(f['confidence_lower'] is not None for f in forecast_data)
            }
        else:
            forecast_summary = {
                'total_predicted_cost': 0,
                'avg_daily_cost': 0,
                'forecast_days': 0,
                'confidence_available': False
            }
        
        return jsonify({
            'server_id': server_id,
            'forecasts': forecast_data,
            'summary': forecast_summary,
            'horizon_days': horizon_days
        })
    
    except Exception as e:
        current_app.logger.error(f"Get forecast error: {e}")
        return jsonify({'error': 'Failed to fetch forecast data'}), 500

@api_bp.route('/analytics/insights')
def get_analytics_insights():
    """Get analytical insights and recommendations."""
    try:
        # Get query parameters
        days = int(request.args.get('days', 30))
        server_id = request.args.get('server_id')
        
        db = get_database()
        costs_dao = DailyCostsDAO(db)
        
        # Get date range
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        # Get costs data
        costs_df = costs_dao.get_costs_by_date_range(start_date, end_date, server_id)
        
        insights = {
            'cost_insights': [],
            'recommendations': [],
            'anomalies': [],
            'efficiency_metrics': {}
        }
        
        if not costs_df.empty:
            # Convert DataFrame rows to DailyCost objects for analysis
            from app.models.cost_models import DailyCost
            
            daily_costs = []
            for _, row in costs_df.iterrows():
                daily_cost = DailyCost(
                    server_id=row['server_id'],
                    date=row['date'],
                    cpu_hours=row['cpu_hours'],
                    cpu_cost=row['cpu_cost'],
                    ram_gb_hours=row['ram_gb_hours'],
                    ram_cost=row['ram_cost'],
                    bandwidth_in_gb=row['bandwidth_in_gb'],
                    bandwidth_out_gb=row['bandwidth_out_gb'],
                    bandwidth_cost=row['bandwidth_cost'],
                    total_cost=row['total_cost']
                )
                daily_costs.append(daily_cost)
            
            # Generate insights
            insights['cost_insights'] = CostAnalyzer.generate_cost_insights(daily_costs)
            
            # Calculate trends
            if len(daily_costs) >= 7:
                trend_data = CostAnalyzer.calculate_cost_trends(daily_costs)
                insights['efficiency_metrics'] = {
                    'cost_trend_percent': round(trend_data['change_percent'], 2),
                    'recent_avg_cost': round(trend_data['recent_avg'], 4),
                    'cost_stability': 'stable' if abs(trend_data['change_percent']) < 10 else 'volatile'
                }
            
            # Identify anomalies
            anomalies = CostAnalyzer.identify_cost_anomalies(daily_costs)
            insights['anomalies'] = [
                {
                    'date': anomaly[0].date.strftime('%Y-%m-%d'),
                    'cost': round(anomaly[0].total_cost, 4),
                    'description': anomaly[1]
                }
                for anomaly in anomalies[:5]  # Limit to 5 most recent
            ]
        
        return jsonify(insights)
    
    except Exception as e:
        current_app.logger.error(f"Get analytics insights error: {e}")
        return jsonify({'error': 'Failed to generate insights'}), 500

@api_bp.route('/health')
def health_check():
    """Health check endpoint."""
    try:
        db = get_database()
        
        # Test database connectivity
        test_query = "SELECT COUNT(*) as count FROM servers"
        result = db.execute_query(test_query)
        server_count = result[0]['count'] if result else 0
        
        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'servers_configured': server_count,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        current_app.logger.error(f"Health check error: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500