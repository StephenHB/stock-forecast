# Stock Forecast

A comprehensive stock forecasting system with machine learning models for analyzing and predicting stock market trends.

## Features

- **Streamlit UI**: Interactive app for stock selection, forecasting, backtesting, and trading simulation
- **S&P 100 + Market Indices**: Select from 100+ stocks and ETFs (SPY, QQQ, DIA, etc.)
- **LGBM 2-Stage**: Prophet or MA trend/seasonality + LightGBM; daily (≤5 days) and weekly horizon with volatility features
- **Trading Simulation**: $100k simulation with forecast-based buy/sell, transaction fees
- **Capital Market Research**: News sentiment (FinBERT or keyword), SEC filings (10-K, 10-Q, 8-K), impact features; optional LGBM input via sidebar checkbox
- **Feature Importance**: Gain-based and SHAP (optional) for directional analysis
- **Data Download**: Yahoo Finance API, configurable via YAML
- **Data Preprocessing**: Validate, clean, technical indicators (RSI, MACD, Bollinger Bands)
- **Incremental Updates**: Efficient data refresh

## Project Structure

```
stock-forecast/
├── app.py                        # Streamlit UI (forecast, backtest, simulation)
├── requirements.txt              # Dependencies for Streamlit Community Cloud
├── src/
│   ├── data_preprocess/          # Data loading, validation, cleaning
│   │   ├── stock_data_loader.py
│   │   └── data_preprocess_utils.py
│   ├── forecasting/              # LightGBM, backtesting, simulation
│   │   ├── feature_factory.py    # Daily/weekly features + trend/seasonality
│   │   ├── trend_seasonality.py  # Prophet or MA for LGBM 2-stage
│   │   ├── standalone_backtester.py
│   │   ├── trading_simulator.py
│   │   ├── feature_importance.py  # SHAP, permutation importance
│   │   └── research_features.py  # News/report feature integration
│   ├── research/                 # Capital market research
│   │   ├── capital_market_researcher.py
│   │   ├── news_report_analyzer.py  # Sentiment, SEC filings
│   │   └── research_agent.py
│   └── feature_engineering/      # Lag, rolling, technical features
├── config/
│   └── stocks_config.yaml        # S&P 100, market indices, download settings
├── tests/                        # Unit and integration tests
├── notebooks/                    # Backtesting, model enhancement research
├── docs/research/                # Research documentation
└── pyproject.toml
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

### Deploy to Streamlit Community Cloud

1. Push your repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub and click **Deploy an app**
4. Select repo `StephenHB/stock-forecast`, branch `main`, main file `app.py`
5. Click **Deploy** — dependencies install from `requirements.txt`

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

- **sp100_stocks**: S&P 100 constituents
- **market_indices**: SPY, QQQ, DIA, IWM, VOO, VTI, OEF
- **default_stocks**: Quick-pick subset
- **download_settings**: Date ranges, intervals, API parameters

### Example Configuration

```yaml
sp100_stocks:
  - AAPL
  - MSFT
  - GOOGL
  # ... full S&P 100 list

market_indices:
  - SPY
  - QQQ
  - DIA

download_settings:
  start_date: "2020-01-01"
  end_date: null
  interval: "1d"
  auto_adjust: true
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

Key dependencies:
- `pandas`, `numpy`: Data manipulation
- `yfinance`: Yahoo Finance API
- `lightgbm`, `scikit-learn`: Forecasting
- `streamlit`: Interactive UI
- `pyyaml`: Configuration

Optional extras:
- `pip install stock-forecast[prophet]` — Prophet trend/seasonality (else MA fallback)
- `pip install stock-forecast[finbert]` — FinBERT news sentiment (else keyword)
- `pip install stock-forecast[research_ml]` — SHAP feature importance
- `pip install stock-forecast[research]` — duckduckgo-search for research agent

## AI Agent Setup

This project uses Cursor agent skills and rules for consistent AI assistance:

- **`.cursor/rules/`** — Core project standards (always applied)
- **`.cursor/skills/stock-forecast-analysis/`** — Data analysis, ML, and visualization guidance (applied when working with notebooks, pandas, scikit-learn, etc.)
- **`.cursor/skills/testing-agent/`** — Testing guidance: verify new features work as designed and integrate with existing workflows
- **`.cursor/skills/git-agent/`** — Git commit, push, and pull request assistance when there are significant changes
- **`.cursor/skills/ui-agent/`** — Interactive UI guidance (Streamlit, dashboards, stock selection)

## Contributing

### Pull Request Workflow

1. Create a feature branch: `git checkout -b feat/your-feature`
2. Make changes, commit with `type(scope): subject` format
3. Push: `git push -u origin feat/your-feature`
4. Open a PR on GitHub: **Pull requests** → **New pull request** → base: `main`, compare: your branch
5. Merge via the PR (do not merge directly to `main`)

### Code Standards

1. Follow the coding standards in `.cursor/rules/` and `.cursor/skills/stock-forecast-analysis/`
2. Use functional programming where appropriate
3. Implement rigorous statistical validation
4. Add unit tests for new functionality
5. Document all functions and classes

## License

This project is licensed under the MIT License - see the LICENSE file for details.