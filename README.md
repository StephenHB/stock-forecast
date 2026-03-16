# Stock Forecast

A comprehensive stock forecasting system with machine learning models for analyzing and predicting stock market trends.

## Features

- **Data Download**: Download S&P 500 stock data from Yahoo Finance API
- **Configurable Stock Selection**: Customize which stocks to analyze via YAML configuration
- **Data Preprocessing**: Clean, validate, and engineer features from raw stock data
- **Technical Indicators**: Calculate various technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **Feature Engineering**: Create machine learning features with lag variables and rolling statistics
- **Flexible Storage**: Save data in multiple formats (CSV, Parquet, Pickle)
- **Incremental Updates**: Efficiently update data by downloading only new records
- **Centralized Storage**: Data stored in `/Users/stephenzhang/Downloads/stock_data`

## Project Structure

```
stock-forecast/
├── src/
│   ├── data_preprocess/          # Data cleaning and feature engineering
│   │   ├── __init__.py
│   │   ├── stock_data_loader.py  # Main data loading functionality
│   │   └── data_preprocess_utils.py  # Utility functions
│   └── model/                    # Machine learning models (to be implemented)
├── data/                         # Local data directory (for structure only)
│   ├── raw/                      # Raw downloaded data (moved to Downloads)
│   └── processed/                # Processed data (moved to Downloads)
├── config/
│   └── stocks_config.yaml        # Stock selection and download configuration
├── notebooks/                    # Jupyter notebooks for development
├── test/                         # Unit tests
├── pyproject.toml               # Dependencies and project configuration
└── example_usage.py             # Example usage script
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd stock-forecast
```

2. Install dependencies:
```bash
pip install -e .
```

### Running the UI (Streamlit)

```bash
streamlit run app.py
```

Interactive UI for stock selection, forecast horizon (n days), and backtesting (default: 2 years).

## Quick Start

### 1. Download Stock Data

```python
from src.data_preprocess import StockDataLoader

# Initialize the data loader (uses default path: /Users/stephenzhang/Downloads/stock_data)
loader = StockDataLoader()

# Or use a custom data directory
loader = StockDataLoader(data_dir="/path/to/your/custom/data")

# Download data for specific stocks
stock_data = loader.download_stock_data(
    stock_symbols=['AAPL', 'MSFT', 'GOOGL'],
    start_date='2023-01-01',
    end_date='2023-12-31'
)
```

### 2. Process and Clean Data

```python
from src.data_preprocess import validate_stock_data, clean_stock_data, calculate_technical_indicators

# Validate data quality
validation_results = validate_stock_data(stock_data['AAPL'])

# Clean the data
cleaned_data = clean_stock_data(stock_data['AAPL'])

# Add technical indicators
data_with_indicators = calculate_technical_indicators(cleaned_data)
```

### 3. Create Features for Machine Learning

```python
from src.data_preprocess import create_features

# Create features for ML models
features_data = create_features(data_with_indicators, lookback_days=5)
```

### 4. Visualize Stock Data

```bash
# Start Jupyter Lab
uv run jupyter lab

# Open the stock visualization notebook
notebooks/stock_visualization.ipynb
```

The notebook provides comprehensive visualizations including:
- Time series plots with moving averages
- Comparative performance analysis
- Volume analysis
- Technical indicators (RSI, MACD, Bollinger Bands)
- Correlation analysis
- Recent performance trends

## Configuration

Edit `config/stocks_config.yaml` to customize:

- **Stock Selection**: Choose which stocks to download (default: top 100 S&P 500 stocks)
- **Download Settings**: Date ranges, intervals, and API parameters
- **Storage Settings**: Data format and compression options

### Example Configuration

```yaml
default_stocks:
  - AAPL
  - MSFT
  - GOOGL

download_settings:
  start_date: "2020-01-01"
  end_date: null  # null means current date
  interval: "1d"
  auto_adjust: true

storage_settings:
  data_format: "csv"
  save_raw_data: true
  save_processed_data: true
```

## Usage Examples

Run the example script to see the system in action:

```bash
python example_usage.py
```

This will:
1. Download data for sample stocks (AAPL, MSFT, GOOGL)
2. Validate and clean the data
3. Calculate technical indicators
4. Create machine learning features
5. Save processed data to `/Users/stephenzhang/Downloads/stock_data`
6. Generate summary statistics

### Advanced Usage

The system also supports:
- **Incremental Updates**: Use `loader.update_stock_data()` to download only new data
- **Data Freshness Checking**: Use `loader.check_data_freshness()` to monitor data age
- **Custom Data Paths**: Specify custom storage locations with `StockDataLoader(data_dir="/path/to/data")`

## Data Features

The system automatically creates the following features:

### Technical Indicators
- Simple Moving Averages (SMA)
- Exponential Moving Averages (EMA)
- Relative Strength Index (RSI)
- Bollinger Bands
- MACD (Moving Average Convergence Divergence)
- Average True Range (ATR)

### Price Features
- Daily returns
- Log returns
- Price changes
- Rolling statistics (mean, std, min, max)
- Price position within rolling windows

### Time Features
- Year, Month, Day
- Day of week, Day of year
- Quarter

### Volume Features
- Volume moving averages
- Volume rate of change

## Dependencies

Key dependencies include:
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `yfinance`: Yahoo Finance API integration
- `scikit-learn`: Machine learning algorithms
- `matplotlib` & `seaborn`: Data visualization
- `pyyaml`: Configuration file handling

## AI Agent Setup

This project uses Cursor agent skills and rules for consistent AI assistance:

- **`.cursor/rules/`** — Core project standards (always applied)
- **`.cursor/skills/stock-forecast-analysis/`** — Data analysis, ML, and visualization guidance (applied when working with notebooks, pandas, scikit-learn, etc.)
- **`.cursor/skills/testing-agent/`** — Testing guidance: verify new features work as designed and integrate with existing workflows
- **`.cursor/skills/git-agent/`** — Git commit, push, and pull request assistance when there are significant changes
- **`.cursor/skills/ui-agent/`** — Interactive UI guidance (Streamlit, dashboards, stock selection)

## Contributing

1. Follow the coding standards in `.cursor/rules/` and `.cursor/skills/stock-forecast-analysis/`
2. Use functional programming where appropriate
3. Implement rigorous statistical validation
4. Add unit tests for new functionality
5. Document all functions and classes

## License

This project is licensed under the MIT License - see the LICENSE file for details.