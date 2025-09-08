-- Resource-Based Web Server Cost Forecasting Database Schema
-- SQLite Database for INDT 4216 Project

-- Table for storing server information
CREATE TABLE IF NOT EXISTS servers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    server_id TEXT UNIQUE NOT NULL,
    server_name TEXT NOT NULL,
    server_type TEXT NOT NULL DEFAULT 'web',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Table for storing resource metrics (5-minute intervals)
CREATE TABLE IF NOT EXISTS resource_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    server_id TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    cpu_usage_percent REAL NOT NULL CHECK(cpu_usage_percent >= 0 AND cpu_usage_percent <= 100),
    ram_usage_mb REAL NOT NULL CHECK(ram_usage_mb >= 0),
    ram_total_mb REAL NOT NULL CHECK(ram_total_mb > 0),
    bandwidth_in_mb REAL NOT NULL CHECK(bandwidth_in_mb >= 0),
    bandwidth_out_mb REAL NOT NULL CHECK(bandwidth_out_mb >= 0),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (server_id) REFERENCES servers (server_id) ON DELETE CASCADE
);

-- Table for storing daily aggregated costs
CREATE TABLE IF NOT EXISTS daily_costs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    server_id TEXT NOT NULL,
    date DATE NOT NULL,
    cpu_hours REAL NOT NULL CHECK(cpu_hours >= 0),
    cpu_cost REAL NOT NULL CHECK(cpu_cost >= 0),
    ram_gb_hours REAL NOT NULL CHECK(ram_gb_hours >= 0),
    ram_cost REAL NOT NULL CHECK(ram_cost >= 0),
    bandwidth_in_gb REAL NOT NULL CHECK(bandwidth_in_gb >= 0),
    bandwidth_out_gb REAL NOT NULL CHECK(bandwidth_out_gb >= 0),
    bandwidth_cost REAL NOT NULL CHECK(bandwidth_cost >= 0),
    total_cost REAL NOT NULL CHECK(total_cost >= 0),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (server_id) REFERENCES servers (server_id) ON DELETE CASCADE,
    UNIQUE(server_id, date)
);

-- Table for storing cost forecasts
CREATE TABLE IF NOT EXISTS cost_forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    server_id TEXT NOT NULL,
    forecast_date DATE NOT NULL,
    predicted_cost REAL NOT NULL CHECK(predicted_cost >= 0),
    confidence_interval_lower REAL CHECK(confidence_interval_lower >= 0),
    confidence_interval_upper REAL CHECK(confidence_interval_upper >= confidence_interval_lower),
    model_used TEXT NOT NULL,
    mae REAL CHECK(mae >= 0),
    rmse REAL CHECK(rmse >= 0),
    mape REAL CHECK(mape >= 0),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (server_id) REFERENCES servers (server_id) ON DELETE CASCADE
);

-- Table for storing model performance metrics
CREATE TABLE IF NOT EXISTS model_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    server_id TEXT NOT NULL,
    training_start_date DATE NOT NULL,
    training_end_date DATE NOT NULL,
    test_start_date DATE NOT NULL,
    test_end_date DATE NOT NULL,
    mae REAL NOT NULL CHECK(mae >= 0),
    rmse REAL NOT NULL CHECK(rmse >= 0),
    mape REAL NOT NULL CHECK(mape >= 0),
    r2_score REAL CHECK(r2_score >= -1 AND r2_score <= 1),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (server_id) REFERENCES servers (server_id) ON DELETE CASCADE
);

-- Table for storing pricing configuration
CREATE TABLE IF NOT EXISTS pricing_config (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_name TEXT UNIQUE NOT NULL,
    cpu_hourly_rate REAL NOT NULL CHECK(cpu_hourly_rate > 0),
    ram_hourly_rate REAL NOT NULL CHECK(ram_hourly_rate > 0),
    bandwidth_tier_1_limit INTEGER NOT NULL CHECK(bandwidth_tier_1_limit > 0),
    bandwidth_tier_1_price REAL NOT NULL CHECK(bandwidth_tier_1_price > 0),
    bandwidth_tier_2_limit INTEGER NOT NULL CHECK(bandwidth_tier_2_limit > bandwidth_tier_1_limit),
    bandwidth_tier_2_price REAL NOT NULL CHECK(bandwidth_tier_2_price > 0),
    bandwidth_tier_3_price REAL NOT NULL CHECK(bandwidth_tier_3_price > 0),
    effective_from DATE NOT NULL DEFAULT CURRENT_DATE,
    effective_to DATE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_resource_metrics_server_timestamp ON resource_metrics(server_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_resource_metrics_timestamp ON resource_metrics(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_daily_costs_server_date ON daily_costs(server_id, date DESC);
CREATE INDEX IF NOT EXISTS idx_daily_costs_date ON daily_costs(date DESC);
CREATE INDEX IF NOT EXISTS idx_cost_forecasts_server_date ON cost_forecasts(server_id, forecast_date DESC);
CREATE INDEX IF NOT EXISTS idx_cost_forecasts_date ON cost_forecasts(forecast_date DESC);

-- Views for common queries
CREATE VIEW IF NOT EXISTS latest_metrics AS
SELECT 
    rm.*,
    s.server_name,
    s.server_type,
    (rm.ram_usage_mb / rm.ram_total_mb * 100) as ram_usage_percent
FROM resource_metrics rm
JOIN servers s ON rm.server_id = s.server_id
WHERE rm.timestamp >= datetime('now', '-24 hours')
ORDER BY rm.timestamp DESC;

CREATE VIEW IF NOT EXISTS monthly_costs AS
SELECT 
    dc.server_id,
    s.server_name,
    strftime('%Y-%m', dc.date) as month,
    COUNT(dc.date) as days_active,
    SUM(dc.cpu_cost) as total_cpu_cost,
    SUM(dc.ram_cost) as total_ram_cost,
    SUM(dc.bandwidth_cost) as total_bandwidth_cost,
    SUM(dc.total_cost) as total_monthly_cost,
    AVG(dc.total_cost) as avg_daily_cost,
    MIN(dc.total_cost) as min_daily_cost,
    MAX(dc.total_cost) as max_daily_cost
FROM daily_costs dc
JOIN servers s ON dc.server_id = s.server_id
GROUP BY dc.server_id, strftime('%Y-%m', dc.date)
ORDER BY month DESC, dc.server_id;

CREATE VIEW IF NOT EXISTS server_summary AS
SELECT 
    s.server_id,
    s.server_name,
    s.server_type,
    COUNT(DISTINCT DATE(rm.timestamp)) as days_monitored,
    MIN(rm.timestamp) as first_metric,
    MAX(rm.timestamp) as last_metric,
    AVG(rm.cpu_usage_percent) as avg_cpu_usage,
    AVG(rm.ram_usage_mb / rm.ram_total_mb * 100) as avg_ram_usage,
    SUM(rm.bandwidth_out_mb) / 1024 as total_bandwidth_gb,
    COALESCE(SUM(dc.total_cost), 0) as total_cost_to_date
FROM servers s
LEFT JOIN resource_metrics rm ON s.server_id = rm.server_id
LEFT JOIN daily_costs dc ON s.server_id = dc.server_id
GROUP BY s.server_id, s.server_name, s.server_type;

-- Insert default servers
INSERT OR IGNORE INTO servers (server_id, server_name, server_type) VALUES
('web-frontend', 'Web Frontend Server', 'web'),
('api-backend', 'API Backend Server', 'api'),
('database', 'Database Server', 'database'),
('cache-server', 'Cache Server', 'cache');

-- Insert default pricing configuration
INSERT OR IGNORE INTO pricing_config (
    config_name, cpu_hourly_rate, ram_hourly_rate,
    bandwidth_tier_1_limit, bandwidth_tier_1_price,
    bandwidth_tier_2_limit, bandwidth_tier_2_price,
    bandwidth_tier_3_price
) VALUES (
    'default',
    0.0116,  -- $0.0116 per CPU hour
    0.0058,  -- $0.0058 per GB RAM hour
    10000,   -- First 10GB
    0.09,    -- $0.09 per GB
    40000,   -- Next 30GB (10-40GB)
    0.085,   -- $0.085 per GB
    0.07     -- Over 40GB: $0.07 per GB
);