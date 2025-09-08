# Resource-Based Web Server Cost Forecasting

A comprehensive cost monitoring and forecasting system for web server resources, developed for **INDT 4216 - Industrial Training Programme** at **Wayamba University of Sri Lanka**.

## Project Overview

This system monitors CPU, RAM, and bandwidth usage of web servers and calculates daily operational costs using a tiered pricing model. It includes machine learning-based forecasting to predict monthly expenses and provides interactive dashboards for cost analysis.

## Features

### Core Functionality
- **Real-time Resource Monitoring**: Track CPU, RAM, and bandwidth usage every 5 minutes
- **Tiered Cost Calculation**: Multi-tier pricing for bandwidth with CPU and RAM hourly rates
- **Time Series Forecasting**: ARIMA and Facebook Prophet models for cost prediction
- **Interactive Dashboard**: Server-side rendered templates with Chart.js visualizations
- **Cost Optimization**: Automated recommendations and anomaly detection

### Technical Stack
- **Backend**: Flask (Python 3.12), SQLite database, PyMySQL connector
- **Frontend**: Jinja2 templates, Tailwind CSS, Chart.js for visualizations
- **ML/Analytics**: pandas, scikit-learn, Facebook Prophet, ARIMA, NumPy
- **Development**: Local development environment, no containerization required

## Quick Start

### Prerequisites
- Python 3.12 or higher
- Node.js (for Tailwind CSS compilation)
- Git (optional, for version control)

### Installation

1. **Clone or download the project**:
```bash
git clone <repository-url>
cd cost-forecasting
```

2. **Set up Python virtual environment**:
```bash
# Create virtual environment
python3.12 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

3. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

4. **Install Node.js dependencies** (for Tailwind CSS):
```bash
npm install
```

5. **Set up the database**:
```bash
python scripts/setup_database.py
```

6. **Generate sample data**:
```bash
python scripts/generate_sample_data.py --days 90
```

7. **Build Tailwind CSS**:
```bash
npm run build-css-prod
```

8. **Start the application**:
```bash
python app/app.py
```

The application will be available at `http://127.0.0.1:5000`

## Project Structure

```
cost-forecasting/
├── app/
│   ├── models/              # Database models and data structures
│   │   ├── database.py      # Database connection and DAOs
│   │   └── cost_models.py   # Cost-related data classes
│   ├── routes/              # Flask route handlers
│   │   ├── dashboard.py     # Main dashboard routes
│   │   └── api.py          # JSON API endpoints
│   ├── static/             # Static assets
│   │   ├── css/            # Stylesheets
│   │   └── js/             # JavaScript files
│   ├── templates/          # Jinja2 HTML templates
│   ├── config.py           # Configuration settings
│   └── app.py             # Main Flask application
├── data_generation/        # Data simulation and cost calculation
│   ├── resource_simulator.py  # Realistic usage pattern generator
│   └── cost_calculator.py     # Tiered cost calculation engine
├── ml_models/              # Machine learning forecasting models
│   ├── arima_forecaster.py    # ARIMA time series model
│   └── prophet_forecaster.py  # Facebook Prophet model
├── database/               # Database files and schema
│   ├── schema.sql          # Database schema definition
│   └── cost_forecasting.db # SQLite database (created automatically)
├── scripts/                # Utility scripts
│   ├── setup_database.py  # Database initialization
│   └── generate_sample_data.py # Sample data generation
├── tests/                  # Test files
├── requirements.txt        # Python dependencies
├── package.json           # Node.js dependencies (Tailwind CSS)
├── tailwind.config.js     # Tailwind configuration
└── CLAUDE.md              # Development guidelines
```

## Usage Guide

### Dashboard Navigation

1. **Main Dashboard** (`/`): Overview of all servers with cost metrics and trends
2. **Server Analytics** (`/server/<server_id>`): Detailed analysis for specific servers
3. **Forecasting** (`/forecasting`): Cost prediction and trend analysis
4. **Server Comparison** (`/compare`): Side-by-side server performance comparison

### Key Features

#### Cost Calculation
- **CPU Cost**: `(CPU Usage % / 100) × Hours × $0.0116/hour`
- **RAM Cost**: `RAM GB × Hours × $0.0058/GB/hour`
- **Bandwidth Cost**: Tiered pricing (First 10GB: $0.09/GB, Next 30GB: $0.085/GB, Over 40GB: $0.07/GB)

#### Resource Monitoring
- **Business Hours Factor**: Higher usage during 9 AM - 5 PM on weekdays
- **Seasonal Patterns**: Holiday and end-of-month traffic spikes
- **Memory Leaks**: Gradual RAM usage increase with nightly resets
- **Traffic Spikes**: Random spike events during business hours

#### Forecasting Models
- **ARIMA**: Auto-selected parameters (p,d,q) for time series forecasting
- **Prophet**: Handles seasonality, holidays, and trend changes
- **Validation Metrics**: MAE, RMSE, MAPE with target <10% error

## API Reference

### REST Endpoints

#### Server Information
- `GET /api/servers` - List all configured servers
- `GET /api/servers/<server_id>/metrics?hours=24` - Resource metrics
- `GET /api/servers/<server_id>/costs?days=30` - Daily cost data

#### Analytics
- `GET /api/costs/comparison?days=30` - Server cost comparison
- `GET /api/costs/trends?days=30` - Cost trend analysis
- `GET /api/analytics/insights` - Cost optimization recommendations

#### Forecasting
- `GET /api/forecast/<server_id>?horizon=30` - Cost forecasts

#### Health Check
- `GET /api/health` - Application and database health status

## Development

### Environment Configuration

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Key configuration options:
- `FLASK_ENV=development` - Development mode
- `DATABASE_PATH=./database/cost_forecasting.db` - Database location
- `CPU_HOURLY_RATE=0.0116` - CPU pricing
- `RAM_HOURLY_RATE=0.0058` - RAM pricing

### Development Workflow

1. **Start development server with auto-reload**:
```bash
python app/app.py
```

2. **Watch Tailwind CSS changes**:
```bash
npm run build-css
```

3. **Generate fresh sample data**:
```bash
python scripts/generate_sample_data.py --days 60 --seed 42
```

4. **Reset database**:
```bash
python scripts/setup_database.py --reset
```

### Data Generation

The system includes realistic data simulation:

```python
# Generate 90 days of data for all servers
python scripts/generate_sample_data.py --days 90

# Verify generated data
python scripts/generate_sample_data.py --verify
```

### Database Management

```bash
# Initialize database
python scripts/setup_database.py

# Verify database integrity
python scripts/setup_database.py --verify

# Reset database (removes all data)
python scripts/setup_database.py --reset
```

## Performance Targets

- **Forecast Accuracy**: MAPE < 10% for monthly predictions
- **Web Response Time**: < 500ms for dashboard loads  
- **Data Processing**: Handle 90+ days of 5-minute interval data
- **Chart Rendering**: < 2 seconds for complex visualizations

## Troubleshooting

### Common Issues

1. **Database not found**:
   ```bash
   python scripts/setup_database.py
   ```

2. **No data in dashboard**:
   ```bash
   python scripts/generate_sample_data.py
   ```

3. **Tailwind CSS not loading**:
   ```bash
   npm run build-css-prod
   ```

4. **Prophet import error**:
   ```bash
   pip install prophet
   # On macOS: brew install cmake
   ```

### Logging

Application logs are displayed in the console during development. Key log locations:
- Database operations: SQLite connection status
- Data generation: Progress and statistics
- Forecasting: Model fitting and validation results
- API requests: Request/response logging

## Deployment

### Production Setup

1. **Environment variables**:
```bash
export FLASK_ENV=production
export SECRET_KEY=your-production-secret-key
```

2. **Database optimization**:
```bash
# Consider PostgreSQL for production
# Update DATABASE_URL in config.py
```

3. **Web server**:
```bash
# Use gunicorn or uWSGI
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app.app:app
```

4. **Reverse proxy**: Configure Nginx or Apache for static file serving

## Research Context

This project demonstrates practical application of:
- **Time Series Forecasting**: ARIMA and Prophet for cost prediction
- **Web Application Development**: Full-stack Flask application
- **Data Visualization**: Interactive charts and dashboards
- **Database Design**: Efficient schema for time-series data
- **Cost Optimization**: Automated recommendations and insights

## Academic Information

- **Course**: INDT 4216 - Industrial Training Programme
- **Institution**: Wayamba University of Sri Lanka
- **Focus**: Resource monitoring, cost analysis, and predictive modeling
- **Technologies**: Python, Flask, SQLite, Machine Learning, Web Development

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review application logs for error details
3. Verify database setup and sample data generation
4. Ensure all dependencies are properly installed

## License

This project is developed for academic purposes as part of the INDT 4216 course at Wayamba University of Sri Lanka.