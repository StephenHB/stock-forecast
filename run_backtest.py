#!/usr/bin/env python
"""Run backtesting for GOOGL, NVDA, AMD with 5 years of daily data."""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocess.stock_data_loader import StockDataLoader
from src.forecasting import WeeklyAggregator, StandaloneBacktester

# Configuration
STOCK_SYMBOLS = ['GOOGL', 'NVDA', 'AMD']
YEARS = 5
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=365 * YEARS)
FORECAST_HORIZON = 4
TARGET_COLUMN = 'Close'

QUARTERLY_PARAMS = {
    'initial_train_size': 13,
    'test_size': 1,
    'step_size': 1,
    'min_train_size': 6,
    'target_column': TARGET_COLUMN,
    'forecast_horizon': FORECAST_HORIZON,
}


def create_simple_features(data):
    """Create simple features for backtesting."""
    features = data.copy()
    for lag in [1, 2, 4]:
        features[f'close_lag_{lag}'] = features['Close'].shift(lag)
    for window in [4, 8]:
        features[f'close_ma_{window}'] = features['Close'].rolling(window=window).mean()
        features[f'close_std_{window}'] = features['Close'].rolling(window=window).std()
    features['price_change_1w'] = features['Close'].pct_change(1)
    features['price_change_4w'] = features['Close'].pct_change(4)
    features['volume_ma_4'] = features['Volume'].rolling(window=4).mean()
    features['volume_ratio'] = features['Volume'] / features['volume_ma_4']
    features['week_of_year'] = features.index.isocalendar().week
    features['month'] = features.index.month
    features['quarter'] = features.index.quarter
    return features


def create_simple_targets(data, horizon=4):
    """Create target variables."""
    targets = data.copy()
    targets[f'target_{horizon}w_pct'] = (
        (targets['Close'].shift(-horizon) - targets['Close']) / targets['Close'] * 100
    )
    return targets


def main():
    print("=" * 70)
    print("📊 BACKTESTING: GOOGL, NVDA, AMD | 5 Years Daily Data")
    print("=" * 70)
    print(f"\n📅 Date range: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
    print(f"📈 Stocks: {', '.join(STOCK_SYMBOLS)}")
    print(f"📐 Forecast horizon: {FORECAST_HORIZON} weeks\n")

    # 1. Download data
    print("📥 Downloading data...")
    loader = StockDataLoader()
    stock_data = loader.download_stock_data(
        stock_symbols=STOCK_SYMBOLS,
        start_date=START_DATE.strftime('%Y-%m-%d'),
        end_date=END_DATE.strftime('%Y-%m-%d'),
        interval='1d',
        save_data=False,
    )

    if not stock_data:
        print("❌ No data downloaded")
        return 1

    print(f"✅ Downloaded {len(stock_data)} stocks\n")

    # 2. Aggregate to weekly and create features
    print("📊 Converting to weekly and creating features...")
    weekly_aggregator = WeeklyAggregator(
        price_columns=['Open', 'High', 'Low', 'Close'],
        volume_columns=['Volume'],
    )
    backtesting_data = {}
    for symbol in STOCK_SYMBOLS:
        if symbol not in stock_data:
            continue
        daily_data = stock_data[symbol].copy()
        if 'Date' in daily_data.columns:
            daily_data['Date'] = pd.to_datetime(daily_data['Date'], utc=True).dt.tz_localize(None)
            daily_data.set_index('Date', inplace=True)
        weekly_data = weekly_aggregator.aggregate(daily_data)
        features_data = create_simple_features(weekly_data)
        targets_data = create_simple_targets(features_data, FORECAST_HORIZON)
        backtesting_data[symbol] = targets_data
        print(f"   ✅ {symbol}: {len(targets_data)} weeks")

    # 3. Run backtesting
    print("\n🔄 Running backtesting...")
    backtesting_results = {}
    overall_performance = {}

    for symbol in STOCK_SYMBOLS:
        if symbol not in backtesting_data:
            continue
        print(f"\n📈 Backtesting {symbol}...")
        try:
            data = backtesting_data[symbol].dropna()
            if len(data) < 20:
                print(f"   ⚠️ Insufficient data ({len(data)} rows)")
                continue

            backtester = StandaloneBacktester(
                initial_train_size=QUARTERLY_PARAMS['initial_train_size'],
                test_size=QUARTERLY_PARAMS['test_size'],
                step_size=QUARTERLY_PARAMS['step_size'],
                min_train_size=QUARTERLY_PARAMS['min_train_size'],
                target_column=QUARTERLY_PARAMS['target_column'],
                forecast_horizon=QUARTERLY_PARAMS['forecast_horizon'],
            )
            results = backtester.backtest(data)

            metrics = results['overall_metrics']
            backtesting_results[symbol] = results
            overall_performance[symbol] = {
                'mape': metrics['mape'] * 100,
                'rmse': metrics['rmse'],
                'r2': metrics['r2'],
                'windows': results['total_windows'],
            }
            print(f"   ✅ MAPE={metrics['mape']*100:.2f}%, RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.3f}, Windows={results['total_windows']}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            import traceback
            traceback.print_exc()

    # 4. Summary
    print("\n" + "=" * 70)
    print("📊 BACKTESTING RESULTS")
    print("=" * 70)
    print(f"\n{'Stock':<8} {'MAPE (%)':<12} {'RMSE':<12} {'R²':<10} {'Windows':<8}")
    print("-" * 55)
    for symbol in STOCK_SYMBOLS:
        if symbol in overall_performance:
            p = overall_performance[symbol]
            print(f"{symbol:<8} {p['mape']:<12.2f} {p['rmse']:<12.2f} {p['r2']:<10.3f} {p['windows']:<8}")
        else:
            print(f"{symbol:<8} {'N/A':<12} {'N/A':<12} {'N/A':<10} {'N/A':<8}")

    if overall_performance:
        mape_avg = np.mean([p['mape'] for p in overall_performance.values()])
        r2_avg = np.mean([p['r2'] for p in overall_performance.values()])
        print(f"\n📊 Average MAPE: {mape_avg:.2f}% | Average R²: {r2_avg:.3f}")
    print("\n✅ Backtesting complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
