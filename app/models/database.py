"""
Database connection and management utilities.
Provides context managers and query helpers for SQLite database operations.
"""

import sqlite3
import pandas as pd
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, date
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, db_path: Union[str, Path]):
        """Initialize database manager with database path."""
        self.db_path = str(db_path)
        self._ensure_database_exists()
    
    def _ensure_database_exists(self):
        """Ensure database file exists and is accessible."""
        db_file = Path(self.db_path)
        if not db_file.exists():
            logger.warning(f"Database file not found at {self.db_path}")
            # Create parent directories if needed
            db_file.parent.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections with automatic cleanup."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row  # Enable dict-like access to rows
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def execute_query(self, query: str, params: Optional[Union[tuple, dict]] = None) -> List[sqlite3.Row]:
        """Execute a SELECT query and return results as list of Row objects."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                return cursor.fetchall()
            except sqlite3.Error as e:
                logger.error(f"Query execution failed: {e}")
                logger.error(f"Query: {query}")
                logger.error(f"Params: {params}")
                raise
    
    def execute_query_df(self, query: str, params: Optional[Union[tuple, dict]] = None) -> pd.DataFrame:
        """Execute a SELECT query and return results as a pandas DataFrame."""
        with self.get_connection() as conn:
            try:
                if params:
                    df = pd.read_sql_query(query, conn, params=params)
                else:
                    df = pd.read_sql_query(query, conn)
                return df
            except (sqlite3.Error, pd.errors.DatabaseError) as e:
                logger.error(f"DataFrame query execution failed: {e}")
                logger.error(f"Query: {query}")
                raise
    
    def execute_update(self, query: str, params: Optional[Union[tuple, dict]] = None) -> int:
        """Execute an INSERT, UPDATE, or DELETE query and return affected rows count."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                conn.commit()
                return cursor.rowcount
            except sqlite3.Error as e:
                logger.error(f"Update execution failed: {e}")
                logger.error(f"Query: {query}")
                logger.error(f"Params: {params}")
                raise
    
    def execute_many(self, query: str, params_list: List[Union[tuple, dict]]) -> int:
        """Execute a query multiple times with different parameters."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.executemany(query, params_list)
                conn.commit()
                return cursor.rowcount
            except sqlite3.Error as e:
                logger.error(f"Batch execution failed: {e}")
                logger.error(f"Query: {query}")
                raise
    
    def get_table_info(self, table_name: str) -> List[Dict[str, Any]]:
        """Get information about table structure."""
        query = f"PRAGMA table_info({table_name})"
        rows = self.execute_query(query)
        return [dict(row) for row in rows]
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        query = """
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        """
        result = self.execute_query(query, (table_name,))
        return len(result) > 0
    
    def get_row_count(self, table_name: str, where_clause: str = "", params: Optional[tuple] = None) -> int:
        """Get count of rows in a table with optional WHERE clause."""
        query = f"SELECT COUNT(*) FROM {table_name}"
        if where_clause:
            query += f" WHERE {where_clause}"
        
        result = self.execute_query(query, params)
        return result[0][0] if result else 0

class ResourceMetricsDAO:
    """Data Access Object for resource_metrics table."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def get_latest_metrics(self, server_id: Optional[str] = None, limit: int = 100) -> pd.DataFrame:
        """Get the most recent resource metrics."""
        query = """
            SELECT * FROM resource_metrics
            WHERE 1=1
        """
        params = []
        
        if server_id:
            query += " AND server_id = ?"
            params.append(server_id)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        return self.db.execute_query_df(query, tuple(params))
    
    def get_metrics_by_date_range(self, start_date: datetime, end_date: datetime, 
                                 server_id: Optional[str] = None) -> pd.DataFrame:
        """Get metrics within a specific date range."""
        query = """
            SELECT * FROM resource_metrics
            WHERE timestamp BETWEEN ? AND ?
        """
        params = [start_date, end_date]
        
        if server_id:
            query += " AND server_id = ?"
            params.append(server_id)
        
        query += " ORDER BY timestamp"
        
        return self.db.execute_query_df(query, tuple(params))
    
    def get_server_summary(self, server_id: str, days: int = 7) -> Dict[str, Any]:
        """Get summary statistics for a server over the last N days."""
        query = """
            SELECT 
                COUNT(*) as data_points,
                MIN(timestamp) as first_metric,
                MAX(timestamp) as last_metric,
                AVG(cpu_usage_percent) as avg_cpu,
                MAX(cpu_usage_percent) as max_cpu,
                AVG(ram_usage_mb / ram_total_mb * 100) as avg_ram_percent,
                MAX(ram_usage_mb / ram_total_mb * 100) as max_ram_percent,
                SUM(bandwidth_out_mb) / 1024.0 as total_bandwidth_gb
            FROM resource_metrics
            WHERE server_id = ? 
            AND timestamp >= datetime('now', '-{} days')
        """.format(days)
        
        result = self.db.execute_query(query, (server_id,))
        return dict(result[0]) if result else {}

class DailyCostsDAO:
    """Data Access Object for daily_costs table."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def get_costs_by_date_range(self, start_date: date, end_date: date, 
                               server_id: Optional[str] = None) -> pd.DataFrame:
        """Get daily costs within a specific date range."""
        query = """
            SELECT * FROM daily_costs
            WHERE date BETWEEN ? AND ?
        """
        params = [start_date, end_date]
        
        if server_id:
            query += " AND server_id = ?"
            params.append(server_id)
        
        query += " ORDER BY date, server_id"
        
        return self.db.execute_query_df(query, tuple(params))
    
    def get_monthly_summary(self, year: int, month: int) -> pd.DataFrame:
        """Get monthly cost summary for all servers."""
        query = """
            SELECT 
                server_id,
                COUNT(*) as days_active,
                SUM(cpu_cost) as total_cpu_cost,
                SUM(ram_cost) as total_ram_cost,
                SUM(bandwidth_cost) as total_bandwidth_cost,
                SUM(total_cost) as total_monthly_cost,
                AVG(total_cost) as avg_daily_cost
            FROM daily_costs
            WHERE strftime('%Y', date) = ? AND strftime('%m', date) = ?
            GROUP BY server_id
            ORDER BY total_monthly_cost DESC
        """
        
        return self.db.execute_query_df(query, (str(year), f"{month:02d}"))
    
    def get_cost_trends(self, server_id: str, days: int = 30) -> pd.DataFrame:
        """Get cost trends for a server over the last N days."""
        query = """
            SELECT 
                date,
                total_cost,
                cpu_cost,
                ram_cost,
                bandwidth_cost
            FROM daily_costs
            WHERE server_id = ? 
            AND date >= date('now', '-{} days')
            ORDER BY date
        """.format(days)
        
        return self.db.execute_query_df(query, (server_id,))

class ForecastDAO:
    """Data Access Object for cost_forecasts table."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def save_forecast(self, server_id: str, forecast_date: date, predicted_cost: float,
                     confidence_lower: Optional[float], confidence_upper: Optional[float],
                     model_used: str, mae: Optional[float] = None, 
                     rmse: Optional[float] = None, mape: Optional[float] = None) -> bool:
        """Save a cost forecast to the database."""
        query = """
            INSERT OR REPLACE INTO cost_forecasts 
            (server_id, forecast_date, predicted_cost, confidence_interval_lower, 
             confidence_interval_upper, model_used, mae, rmse, mape)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (server_id, forecast_date, predicted_cost, confidence_lower,
                 confidence_upper, model_used, mae, rmse, mape)
        
        try:
            rows_affected = self.db.execute_update(query, params)
            return rows_affected > 0
        except sqlite3.Error as e:
            logger.error(f"Failed to save forecast: {e}")
            return False
    
    def get_forecasts(self, server_id: str, start_date: Optional[date] = None,
                     end_date: Optional[date] = None) -> pd.DataFrame:
        """Get forecasts for a server within a date range."""
        query = """
            SELECT * FROM cost_forecasts
            WHERE server_id = ?
        """
        params = [server_id]
        
        if start_date:
            query += " AND forecast_date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND forecast_date <= ?"
            params.append(end_date)
        
        query += " ORDER BY forecast_date"
        
        return self.db.execute_query_df(query, tuple(params))

# Global database manager instance (initialized in app factory)
db_manager = None

def init_database(db_path: str):
    """Initialize the global database manager."""
    global db_manager
    db_manager = DatabaseManager(db_path)
    return db_manager

def get_database() -> DatabaseManager:
    """Get the global database manager instance."""
    if db_manager is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return db_manager