# Stock Forecast

A comprehensive stock forecasting system with machine learning models for analyzing and predicting stock market trends.

## Features

- **Streamlit UI**: Interactive app for stock selection, forecasting, backtesting, and trading simulation
- **S&P 500 + Market Indices**: Select from 470+ stocks and ETFs (SPY, QQQ, DIA, etc.) covering the full S&P 500 universe
- **Horizon-Aware Feature Tiers**: Three distinct daily feature sets keyed on forecast horizon — Short (1–5d), Medium (6–15d), Long (16–30d) — each progressively enriched with technical indicators, momentum, regime signals, and macro context
- **Intraday Price Dynamics**: Open-close diff, overnight gap, candlestick shadows, and volume-spike features specifically tuned for 1-day forecasts
- **FOMC Proximity Features**: Days-to/since Federal Reserve announcement, pre/post-meeting window flags (2022–2027 dates); all horizons
- **Market-Wide Context**: SPY/QQQ return, VIX level/change/regime, and 10-Year Treasury yield merged as features for every stock — automatically skipped when forecasting SPY or QQQ themselves
- **Adaptive Hyperparameter Tuning**: At each backtest window, `HalvingRandomSearchCV` searches 30 candidates using Successive Halving (cheap first screen → promote survivors) on a fixed 1-lag-back validation split; tuning runs every 7 windows and the best structural params are applied to the final model (n_estimators=100). ~20× faster than the original fixed-param design
- **LGBM 2-Stage**: Prophet or MA trend/seasonality + LightGBM; all horizons use daily OHLCV data
- **Trading Simulation**: $100k simulation with 5 research-driven enhancements — implied-return signal, dead-zone threshold, proportional position sizing, transaction cost model, and 200-day MA regime filter; user-adjustable parameters
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
│   │   ├── lgbm_tuner.py         # HalvingRandomSearchCV adaptive tuning (1-lag-back window)
│   │   ├── trading_simulator.py
│   │   ├── feature_importance.py  # SHAP, permutation importance
│   │   └── research_features.py  # News/report feature integration
│   ├── research/                 # Capital market research
│   │   ├── capital_market_researcher.py
│   │   ├── news_report_analyzer.py  # Sentiment, SEC filings
│   │   └── research_agent.py
│   └── feature_engineering/      # Lag, rolling, technical, intraday, FOMC, market features
│       ├── horizon_features.py   # Medium/long-term technical indicators (RSI, MACD, ATR, CCI, regime)
│       ├── intraday_features.py  # Open-close diff, gap, candlestick shadows, volume dynamics
│       ├── macro_features.py     # FOMC meeting proximity (days_to_fomc, fomc_week_ahead/after)
│       └── market_features.py    # SPY/QQQ/VIX/yield download and alignment
├── config/
│   └── stocks_config.yaml        # S&P 500 (sp100_stocks + sp500_additional), market indices, download settings
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

> **Cloud storage**: Streamlit Community Cloud sets `STREAMLIT_SHARING_MODE` automatically.
> The app detects this and switches to in-session-only caching (`@st.cache_data`);
> no files are written to the ephemeral filesystem.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `STOCK_DATA_DIR` | `<project_root>/data/` | Override the local data directory (absolute or relative path). |
| `IS_CLOUD` | unset | Set to `"true"` to force cloud mode on platforms that don't set `STREAMLIT_SHARING_MODE` automatically (e.g. Railway, Render, Heroku). |

On cloud, the sidebar shows a **Storage mode** indicator confirming whether data is saved locally or fetched fresh each session.

## Quick Start

### 1. Download Stock Data

```python
from src.data_preprocess import StockDataLoader

# Initialize the data loader (defaults to <project_root>/data/, or STOCK_DATA_DIR env var)
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

## Trading Simulation Strategy

The simulation applies five research-driven rules on top of the LGBM forecast:

| # | Rule | Description |
|---|---|---|
| 1 | **Implied-return signal** | Signal = (Predicted Close − Close[T]) ÷ Close[T] × 100. Measures expected move magnitude in %, not just raw direction. |
| 2 | **Dead-zone threshold** | Only trade when the implied return exceeds a configurable threshold (default **0.5%**). Signals inside the dead zone keep the current position, eliminating low-conviction whipsaw trades. |
| 3 | **Proportional position sizing** | BUY deploys a fraction of available cash proportional to signal strength (0% at threshold → 100% at 3× threshold). Weak signals get small positions; strong signals go all-in. |
| 4 | **Transaction cost model** | Every execution is adjusted by a configurable cost per side (default **0.1%**, representing commissions + bid-ask spread). |
| 5 | **Regime filter** | SELL signals are suppressed to HOLD when the stock is above its 200-day MA, preventing premature exits during sustained uptrends. |

Threshold and transaction cost are adjustable via the **Simulation Settings** expander in the sidebar.
The **Simulation tab** shows a live "What to Do Now" recommendation for each selected stock using the same logic as the backtest engine.
The **Forecast Summary** sidebar shows each stock's predicted price and expected % change, colour-coded green (up) / red (down).

Additionally, backtesting uses **non-overlapping windows** — the backtest steps forward by the full forecast horizon so consecutive signals don't share target days, producing clean, independent period results.

## Configuration

Edit `config/stocks_config.yaml` to customize:

- **sp100_stocks**: S&P 100 constituents (105 tickers)
- **sp500_additional**: Remaining S&P 500 members organised by GICS sector (~360 tickers); combined with `sp100_stocks` these cover the full S&P 500 universe (~465 names)
- **market_indices**: SPY, QQQ, DIA, IWM, VOO, VTI, OEF, COIN (Coinbase) and other non-S&P tradeable assets
- **default_stocks**: Quick-pick subset shown first in the dropdown
- **download_settings**: Date ranges, intervals, API parameters

### Example Configuration

```yaml
sp100_stocks:
  - AAPL
  - MSFT
  - GOOGL
  # ... full S&P 100 list

sp500_additional:
  # Additional S&P 500 members not in sp100_stocks, organised by GICS sector
  - CMG   # Consumer Discretionary
  - SHW   # Materials
  - ICE   # Financials
  # ... ~360 more tickers

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
5. Save processed data to `data/` (or the path set by `STOCK_DATA_DIR`)
6. Generate summary statistics

### Advanced Usage

The system also supports:
- **Incremental Updates**: Use `loader.update_stock_data()` to download only new data
- **Data Freshness Checking**: Use `loader.check_data_freshness()` to monitor data age
- **Custom Data Paths**: Specify custom storage locations with `StockDataLoader(data_dir="/path/to/data")`

## Data Features

The system automatically creates the following features:

Features are tiered by forecast horizon. All tiers operate on **daily OHLCV data**.

### Short-horizon features (1–5 days)
- Close lags (1–5d), rolling MA/std (5/10/20d), daily volatility (Parkinson, ADR)
- **Intraday dynamics**: open-close diff, overnight gap, upper/lower candlestick shadows, body size, volume-spike ratio (vs 20-day MA), all with 1-2 day lags

### Medium-horizon features (6–15 days)
- All short features, plus: extended lags (10/15d), MA/std(50d)
- RSI(14/21), MACD(12,26,9), Bollinger Band position/width, ATR(14)%, Stochastic %K(14), CCI(20), OBV-normalised
- Multi-day momentum: 5d, 10d, 15d % returns

### Long-horizon features (16–30 days)
- All medium features, plus: extended lags (20/30d), MA/std(100d), MA(200d) regime
- 50/200-day MA ratio (golden/death-cross), distance from 52-week high/low
- Long-horizon momentum: 20d, 30d, 60d; cyclic calendar encoding (month/quarter sin/cos)
- Volatility-regime ratio: 30-day vol ÷ 60-day vol

### Macro & market context (all horizons)
- **FOMC proximity**: days to/since next Fed announcement, pre/post-meeting binary flags (±5 days)
- **SPY**: 1d return, 5d return, distance from 20-day MA
- **QQQ**: 1d return (Nasdaq-100 tech/growth sentiment)
- **VIX**: level, 1d change, distance from 20-day MA, elevated-fear flag (>25)
- **10Y Treasury yield**: level and 1d change (interest-rate environment)

## Dependencies

Key dependencies:
- `pandas`, `numpy`: Data manipulation
- `yfinance`: Yahoo Finance API (with 3-attempt exponential backoff on transient 403 errors)
- `lightgbm`, `scikit-learn`: Forecasting + `HalvingRandomSearchCV` for adaptive tuning
- `streamlit`: Interactive UI
- `pyyaml`: Configuration

Optional extras:
- `pip install stock-forecast[prophet]` — Prophet trend/seasonality (else MA fallback)
- `pip install stock-forecast[finbert]` — FinBERT news sentiment (else keyword)
- `pip install stock-forecast[research_ml]` — SHAP feature importance
- `pip install stock-forecast[research]` — duckduckgo-search for research agent

## AI Agent Setup

This project uses Cursor agent skills and rules for consistent AI assistance.

### Orchestration

For **multi-step or cross-domain** work (e.g. forecasting code, Streamlit UI, tests, and git in one effort), use **`.cursor/skills/agent-manager/`** first: clarify intent when needed, plan phases, then assign tasks to specialist skills. Details are in that skill’s `SKILL.md` and in `.cursor/rules/` (Orchestration section).

### Rules and specialist skills

- **`.cursor/rules/`** — Core project standards (always applied), including orchestration and safety conventions
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

1. Follow the coding standards in `.cursor/rules/` and `.cursor/skills/stock-forecast-analysis/`; for multi-step or cross-domain tasks, follow `.cursor/skills/agent-manager/` (orchestration)
2. Use functional programming where appropriate
3. Implement rigorous statistical validation
4. Add unit tests for new functionality
5. Document all functions and classes

## License

This project is licensed under the MIT License - see the LICENSE file for details.