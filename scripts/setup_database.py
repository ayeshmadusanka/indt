#!/usr/bin/env python3
"""
Database setup script for the Resource-Based Web Server Cost Forecasting Project
Creates and initializes the SQLite database with schema and default data.
"""

import os
import sys
import sqlite3
from pathlib import Path
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def setup_database():
    """Create and initialize the database with schema and default data."""
    
    # Ensure database directory exists
    db_dir = project_root / 'database'
    db_dir.mkdir(exist_ok=True)
    
    db_path = db_dir / 'cost_forecasting.db'
    schema_path = db_dir / 'schema.sql'
    
    # Read the schema file
    if not schema_path.exists():
        print(f"Error: Schema file not found at {schema_path}")
        return False
    
    with open(schema_path, 'r') as f:
        schema_sql = f.read()
    
    # Create database and execute schema
    try:
        with sqlite3.connect(db_path) as conn:
            conn.executescript(schema_sql)
            conn.commit()
            print(f"Database created successfully at: {db_path}")
            
            # Verify tables were created
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """)
            tables = cursor.fetchall()
            
            print("Created tables:")
            for table in tables:
                print(f"  - {table[0]}")
            
            # Verify views were created
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='view'
                ORDER BY name
            """)
            views = cursor.fetchall()
            
            if views:
                print("Created views:")
                for view in views:
                    print(f"  - {view[0]}")
            
            # Check default data
            cursor.execute("SELECT COUNT(*) FROM servers")
            server_count = cursor.fetchone()[0]
            print(f"Default servers inserted: {server_count}")
            
            cursor.execute("SELECT COUNT(*) FROM pricing_config")
            pricing_count = cursor.fetchone()[0]
            print(f"Default pricing config inserted: {pricing_count}")
            
        return True
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def verify_database():
    """Verify database integrity and show basic statistics."""
    db_path = project_root / 'database' / 'cost_forecasting.db'
    
    if not db_path.exists():
        print("Database does not exist. Run setup first.")
        return False
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            print("\n=== Database Verification ===")
            
            # Check each table
            tables = ['servers', 'resource_metrics', 'daily_costs', 'cost_forecasts', 'model_performance', 'pricing_config']
            
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"{table}: {count} records")
            
            # Show server details
            cursor.execute("SELECT server_id, server_name, server_type FROM servers ORDER BY server_id")
            servers = cursor.fetchall()
            
            print("\nConfigured servers:")
            for server in servers:
                print(f"  - {server[0]} ({server[1]}) - Type: {server[2]}")
            
            # Show pricing config
            cursor.execute("""
                SELECT config_name, cpu_hourly_rate, ram_hourly_rate, 
                       bandwidth_tier_1_price, bandwidth_tier_2_price, bandwidth_tier_3_price
                FROM pricing_config 
                WHERE config_name = 'default'
            """)
            pricing = cursor.fetchone()
            
            if pricing:
                print(f"\nPricing configuration ({pricing[0]}):")
                print(f"  - CPU: ${pricing[1]}/hour")
                print(f"  - RAM: ${pricing[2]}/GB/hour")
                print(f"  - Bandwidth Tier 1: ${pricing[3]}/GB")
                print(f"  - Bandwidth Tier 2: ${pricing[4]}/GB")
                print(f"  - Bandwidth Tier 3: ${pricing[5]}/GB")
        
        return True
        
    except sqlite3.Error as e:
        print(f"Database verification error: {e}")
        return False

def reset_database():
    """Reset the database by removing and recreating it."""
    db_path = project_root / 'database' / 'cost_forecasting.db'
    
    if db_path.exists():
        os.remove(db_path)
        print(f"Removed existing database: {db_path}")
    
    return setup_database()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Database setup for Cost Forecasting Project")
    parser.add_argument('--reset', action='store_true', help='Reset the database (remove existing)')
    parser.add_argument('--verify', action='store_true', help='Verify database integrity')
    
    args = parser.parse_args()
    
    if args.reset:
        success = reset_database()
    elif args.verify:
        success = verify_database()
    else:
        success = setup_database()
    
    sys.exit(0 if success else 1)