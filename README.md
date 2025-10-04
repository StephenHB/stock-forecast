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

### Basic Usage
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

### Incremental Updates
Use the incremental update functionality to keep data current:

```bash
python simple_update_example.py
```

This will:
1. Check data freshness
2. Update only stale data
3. Show last update dates

### Custom Data Paths
Use custom data storage locations:

```bash
python custom_data_path_example.py
```

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

## Contributing

1. Follow the coding standards outlined in `AI_README.md`
2. Use functional programming where appropriate
3. Implement rigorous statistical validation
4. Add unit tests for new functionality
5. Document all functions and classes

## License

This project is licensed under the MIT License - see the LICENSE file for details.