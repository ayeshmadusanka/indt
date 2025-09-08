#!/usr/bin/env python3
"""
Sample Data Generation Script for Cost Forecasting Project
Generates realistic resource metrics and calculates costs for analysis and testing.
"""

import os
import sys
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import argparse

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_generation.resource_simulator import ResourceSimulator
from data_generation.cost_calculator import CostCalculator

def insert_data_to_database(db_path: Path, metrics_df: pd.DataFrame, daily_costs_df: pd.DataFrame):
    """Insert generated data into the database."""
    try:
        with sqlite3.connect(db_path) as conn:
            # Prepare metrics data for insertion (remove any extra columns)
            metrics_columns = ['server_id', 'timestamp', 'cpu_usage_percent', 'ram_usage_mb', 
                             'ram_total_mb', 'bandwidth_in_mb', 'bandwidth_out_mb']
            metrics_clean = metrics_df[metrics_columns].copy()
            
            # Insert resource metrics
            metrics_clean.to_sql('resource_metrics', conn, if_exists='append', index=False)
            print(f"Inserted {len(metrics_clean)} resource metric records")
            
            # Prepare costs data for insertion (remove any extra columns)
            cost_columns = ['server_id', 'date', 'cpu_hours', 'cpu_cost', 'ram_gb_hours', 
                          'ram_cost', 'bandwidth_in_gb', 'bandwidth_out_gb', 'bandwidth_cost', 'total_cost']
            costs_clean = daily_costs_df[cost_columns].copy()
            
            # Insert daily costs
            costs_clean.to_sql('daily_costs', conn, if_exists='append', index=False)
            print(f"Inserted {len(costs_clean)} daily cost records")
            
            conn.commit()
            
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        raise
    except Exception as e:
        print(f"Error inserting data: {e}")
        raise

def generate_and_store_data(days: int = 90, interval_minutes: int = 5, seed: int = 42):
    """Generate sample data and store in database."""
    
    # Initialize components
    simulator = ResourceSimulator(seed=seed)
    calculator = CostCalculator()
    
    # Calculate date range
    end_date = datetime.now().replace(hour=23, minute=59, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days)
    
    print(f"Generating {days} days of data ({start_date.date()} to {end_date.date()})")
    print(f"Data points every {interval_minutes} minutes")
    
    # Generate resource metrics
    print("Generating resource usage data...")
    metrics_df = simulator.generate_all_servers_data(
        start_date=start_date,
        end_date=end_date,
        interval_minutes=interval_minutes
    )
    
    print(f"Generated {len(metrics_df)} resource metric data points")
    
    # Calculate daily costs
    print("Calculating daily costs...")
    daily_costs_df = calculator.calculate_daily_costs(metrics_df)
    
    print(f"Calculated costs for {len(daily_costs_df)} server-days")
    
    # Database path
    db_path = project_root / 'database' / 'cost_forecasting.db'
    
    # Check if database exists
    if not db_path.exists():
        print(f"Database not found at {db_path}")
        print("Please run: python scripts/setup_database.py")
        return False
    
    # Clear existing data if requested
    print("Clearing existing generated data...")
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM daily_costs")
        conn.execute("DELETE FROM resource_metrics") 
        conn.commit()
    
    # Insert data
    print("Inserting data into database...")
    insert_data_to_database(db_path, metrics_df, daily_costs_df)
    
    # Display summary statistics
    print("\n=== Data Generation Summary ===")
    
    # Metrics summary
    print(f"Resource Metrics:")
    print(f"  Total records: {len(metrics_df)}")
    print(f"  Date range: {metrics_df['timestamp'].min()} to {metrics_df['timestamp'].max()}")
    print(f"  Servers: {', '.join(metrics_df['server_id'].unique())}")
    
    # Cost summary
    print(f"\nDaily Costs:")
    print(f"  Total records: {len(daily_costs_df)}")
    print(f"  Date range: {daily_costs_df['date'].min()} to {daily_costs_df['date'].max()}")
    total_cost = daily_costs_df['total_cost'].sum()
    avg_daily_cost = daily_costs_df.groupby('date')['total_cost'].sum().mean()
    print(f"  Total cost: ${total_cost:.2f}")
    print(f"  Average daily cost (all servers): ${avg_daily_cost:.2f}")
    
    # Server-wise breakdown
    print(f"\nCost breakdown by server:")
    for server_id in daily_costs_df['server_id'].unique():
        server_costs = daily_costs_df[daily_costs_df['server_id'] == server_id]
        server_total = server_costs['total_cost'].sum()
        server_avg_daily = server_costs['total_cost'].mean()
        breakdown = calculator.get_cost_breakdown(daily_costs_df, server_id)
        
        print(f"  {server_id}:")
        print(f"    Total: ${server_total:.2f} | Avg daily: ${server_avg_daily:.4f}")
        print(f"    CPU: {breakdown['cpu_percentage']:.1f}% | RAM: {breakdown['ram_percentage']:.1f}% | Bandwidth: {breakdown['bandwidth_percentage']:.1f}%")
    
    # Resource utilization summary
    print(f"\nResource Utilization Summary:")
    for server_id in metrics_df['server_id'].unique():
        server_metrics = metrics_df[metrics_df['server_id'] == server_id]
        print(f"  {server_id}:")
        print(f"    CPU: {server_metrics['cpu_usage_percent'].mean():.1f}% ± {server_metrics['cpu_usage_percent'].std():.1f}% (max: {server_metrics['cpu_usage_percent'].max():.1f}%)")
        print(f"    RAM: {server_metrics['ram_usage_mb'].mean()/1024:.1f}GB ± {server_metrics['ram_usage_mb'].std()/1024:.1f}GB (max: {server_metrics['ram_usage_mb'].max()/1024:.1f}GB)")
        print(f"    Bandwidth Out: {server_metrics['bandwidth_out_mb'].mean():.1f}MB/interval ± {server_metrics['bandwidth_out_mb'].std():.1f}MB/interval")
    
    return True

def verify_generated_data():
    """Verify the generated data in the database."""
    db_path = project_root / 'database' / 'cost_forecasting.db'
    
    if not db_path.exists():
        print("Database not found")
        return False
    
    try:
        with sqlite3.connect(db_path) as conn:
            print("\n=== Database Verification ===")
            
            # Check resource metrics
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM resource_metrics")
            metrics_info = cursor.fetchone()
            print(f"Resource metrics: {metrics_info[0]} records from {metrics_info[1]} to {metrics_info[2]}")
            
            # Check daily costs  
            cursor.execute("SELECT COUNT(*), MIN(date), MAX(date), SUM(total_cost) FROM daily_costs")
            costs_info = cursor.fetchone()
            print(f"Daily costs: {costs_info[0]} records from {costs_info[1]} to {costs_info[2]}")
            print(f"Total cost in database: ${costs_info[3]:.2f}")
            
            # Check servers
            cursor.execute("SELECT server_id, COUNT(*) FROM daily_costs GROUP BY server_id ORDER BY server_id")
            server_costs = cursor.fetchall()
            print(f"Server cost records:")
            for server_id, count in server_costs:
                print(f"  {server_id}: {count} days")
            
            # Recent data sample
            cursor.execute("""
                SELECT server_id, timestamp, cpu_usage_percent, ram_usage_mb/1024 as ram_gb, 
                       bandwidth_out_mb 
                FROM resource_metrics 
                ORDER BY timestamp DESC 
                LIMIT 10
            """)
            recent_metrics = cursor.fetchall()
            
            print(f"\nRecent resource metrics sample:")
            print("Server | Time | CPU% | RAM(GB) | Bandwidth(MB)")
            for row in recent_metrics:
                print(f"{row[0]} | {row[1]} | {row[2]:.1f}% | {row[3]:.1f}GB | {row[4]:.1f}MB")
        
        return True
        
    except sqlite3.Error as e:
        print(f"Database verification error: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sample data for cost forecasting project")
    parser.add_argument('--days', type=int, default=90, help='Number of days to generate (default: 90)')
    parser.add_argument('--interval', type=int, default=5, help='Interval in minutes (default: 5)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible data (default: 42)')
    parser.add_argument('--verify', action='store_true', help='Only verify existing data, do not generate')
    
    args = parser.parse_args()
    
    if args.verify:
        success = verify_generated_data()
    else:
        success = generate_and_store_data(args.days, args.interval, args.seed)
        if success:
            verify_generated_data()
    
    sys.exit(0 if success else 1)