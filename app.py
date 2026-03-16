"""
Stock Forecast UI - Streamlit POC

Interactive UI for:
- Selecting stocks to predict (multi-select)
- Setting forecast horizon (n days)
- Viewing backtesting results (sidebar, default 2 years)
"""

import streamlit as st
import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
from pathlib import Path

# Project setup
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.data_preprocess.stock_data_loader import StockDataLoader
from src.forecasting import WeeklyAggregator, StandaloneBacktester, ForecastingPipeline

# Page config
st.set_page_config(page_title="Stock Forecast", page_icon="📈", layout="wide")

# Default stocks for quick selection (top tech + popular)
DEFAULT_STOCKS = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMD", "TSLA", "META", "AMZN"]


@st.cache_data(ttl=3600)
def load_available_stocks():
    """Load stock list from config."""
    config_path = Path(__file__).parent / "config" / "stocks_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config.get("default_stocks", DEFAULT_STOCKS)


@st.cache_data(ttl=300)
def download_stock_data(symbols: tuple, start_date: str, end_date: str):
    """Download stock data (cached)."""
    loader = StockDataLoader()
    return loader.download_stock_data(
        stock_symbols=list(symbols),
        start_date=start_date,
        end_date=end_date,
        interval="1d",
        save_data=False,
    )


def run_backtest(stock_data: dict, forecast_horizon: int = 4) -> dict:
    """Run backtesting for each stock."""
    weekly_aggregator = WeeklyAggregator(
        price_columns=["Open", "High", "Low", "Close"],
        volume_columns=["Volume"],
    )

    def create_features(data):
        f = data.copy()
        for lag in [1, 2, 4]:
            f[f"close_lag_{lag}"] = f["Close"].shift(lag)
        for w in [4, 8]:
            f[f"close_ma_{w}"] = f["Close"].rolling(window=w).mean()
            f[f"close_std_{w}"] = f["Close"].rolling(window=w).std()
        f["price_change_1w"] = f["Close"].pct_change(1)
        f["price_change_4w"] = f["Close"].pct_change(4)
        f["volume_ma_4"] = f["Volume"].rolling(window=4).mean()
        f["volume_ratio"] = f["Volume"] / f["volume_ma_4"]
        f["week_of_year"] = f.index.isocalendar().week
        f["month"] = f.index.month
        f["quarter"] = f.index.quarter
        return f

    def create_targets(data, h=4):
        t = data.copy()
        t[f"target_{h}w_pct"] = (t["Close"].shift(-h) - t["Close"]) / t["Close"] * 100
        return t

    results = {}
    for symbol, daily_df in stock_data.items():
        try:
            daily = daily_df.copy()
            if "Date" in daily.columns:
                daily["Date"] = pd.to_datetime(daily["Date"], utc=True).dt.tz_localize(None)
                daily.set_index("Date", inplace=True)
            weekly = weekly_aggregator.aggregate(daily)
            features = create_features(weekly)
            targets = create_targets(features, forecast_horizon)
            data = targets.dropna()
            if len(data) < 20:
                continue
            backtester = StandaloneBacktester(
                initial_train_size=13,
                test_size=1,
                step_size=1,
                min_train_size=6,
                target_column="Close",
                forecast_horizon=forecast_horizon,
            )
            bt_results = backtester.backtest(data)
            metrics = bt_results["overall_metrics"]
            results[symbol] = {
                "mape": metrics["mape"] * 100,
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "windows": bt_results["total_windows"],
                "predictions": bt_results["predictions"],
                "actuals": bt_results["actuals"],
                "dates": bt_results["dates"],
            }
        except Exception as e:
            results[symbol] = {"error": str(e)}
    return results


def run_forecast(stock_data: dict, forecast_horizon_weeks: int) -> dict:
    """Run forecasting pipeline for each stock."""
    results = {}
    for symbol, daily_df in stock_data.items():
        try:
            daily = daily_df.copy()
            if "Date" in daily.columns:
                daily["Date"] = pd.to_datetime(daily["Date"], utc=True).dt.tz_localize(None)
            # Ensure OHLCV columns exist (yfinance may use different names)
            cols = {c: c.replace(" ", "_") for c in daily.columns if " " in c}
            daily = daily.rename(columns=cols)
            pipeline = ForecastingPipeline(
                forecast_horizon=forecast_horizon_weeks,
                backtest_windows=6,
                hyperparameter_tuning=False,
            )
            out = pipeline.fit_predict(daily, run_backtesting=False)
            results[symbol] = {
                "predictions": out.get("predictions", {}),
                "last_price": daily["Close"].iloc[-1] if "Close" in daily.columns else None,
            }
        except Exception as e:
            results[symbol] = {"error": str(e)}
    return results


def main():
    st.title("📈 Stock Forecast")
    st.markdown("Select stocks, set forecast horizon, and view backtesting results.")

    available_stocks = load_available_stocks()
    # Use default + config, deduplicated
    all_stocks = list(dict.fromkeys(DEFAULT_STOCKS + available_stocks[:20]))

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        selected_stocks = st.multiselect(
            "Select stocks",
            options=all_stocks,
            default=["AAPL", "GOOGL", "NVDA"],
            help="Choose one or more stocks to predict",
        )
        forecast_days = st.slider(
            "Forecast horizon (days)",
            min_value=1,
            max_value=30,
            value=5,
            step=1,
            help="Number of days ahead to predict (model uses weekly data; 5 days ≈ 1 week)",
        )
        backtest_years = st.slider(
            "Backtest data (years)",
            min_value=1,
            max_value=5,
            value=2,
            help="Years of daily data for backtesting",
        )
        run_btn = st.button("🚀 Run Forecast & Backtest", type="primary")

    forecast_weeks = max(1, (forecast_days + 6) // 7)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * backtest_years)

    if not selected_stocks:
        st.warning("Please select at least one stock.")
        return

    if run_btn:
        with st.spinner("Downloading data and running models..."):
            stock_data = download_stock_data(
                tuple(selected_stocks),
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            )
            if not stock_data:
                st.error("Failed to download data. Check your connection.")
                return

            # Run backtest and forecast
            backtest_results = run_backtest(stock_data, forecast_weeks)
            forecast_results = run_forecast(stock_data, forecast_weeks)

        # Main content
        st.success(f"✅ Processed {len(stock_data)} stocks")

        # Backtest results - sidebar summary
        st.sidebar.header("📊 Backtest Summary")
        for sym in selected_stocks:
            if sym in backtest_results and "error" not in backtest_results[sym]:
                r = backtest_results[sym]
                pred_price = None
                if sym in forecast_results and "error" not in forecast_results[sym]:
                    preds = forecast_results[sym].get("predictions", {})
                    pred_price = list(preds.values())[0] if preds else None
                st.sidebar.metric(
                    f"{sym} MAPE",
                    f"{r['mape']:.1f}%",
                    f"Pred: ${pred_price:,.2f}" if pred_price is not None else "—",
                )
            elif sym in backtest_results:
                st.sidebar.error(f"{sym}: {backtest_results[sym]['error']}")

        # ═══════════════════════════════════════════════════════════════
        # 1. TOP: Predicted prices (most prominent)
        # ═══════════════════════════════════════════════════════════════
        st.subheader("🔮 Predicted Prices")
        st.caption(f"Forecast for next {forecast_days} day(s) ahead")
        pred_cols = st.columns(min(len(selected_stocks), 4))
        for i, sym in enumerate(selected_stocks):
            if sym in forecast_results and "error" not in forecast_results[sym]:
                r = forecast_results[sym]
                preds = r.get("predictions", {})
                last = r.get("last_price")
                with pred_cols[i % len(pred_cols)]:
                    # Use first horizon prediction for "N days ahead" (model is weekly)
                    pred_price = list(preds.values())[0] if preds else None
                    if pred_price is not None:
                        st.metric(
                            sym,
                            f"${pred_price:,.2f}",
                            f"In {forecast_days} days (last: ${last:,.2f})" if last else f"In {forecast_days} days",
                        )
                    else:
                        st.metric(sym, "N/A", f"Last: ${last:,.2f}" if last else "N/A")
        # Forecast horizon chart (all horizons)
        horizon_data = {}
        for sym in selected_stocks:
            if sym in forecast_results and "error" not in forecast_results[sym]:
                preds = forecast_results[sym].get("predictions", {})
                if preds:
                    horizon_data[sym] = {f"Day {k*7}": v for k, v in preds.items()}
        if horizon_data:
            horizon_df = pd.DataFrame(horizon_data)
            st.bar_chart(horizon_df, use_container_width=True)
            st.caption("Predicted close price by horizon (days ahead)")

        # ═══════════════════════════════════════════════════════════════
        # 2. MIDDLE: Statistical metrics
        # ═══════════════════════════════════════════════════════════════
        st.subheader("📊 Statistical Metrics")
        bt_rows = []
        for sym in selected_stocks:
            if sym in backtest_results and "error" not in backtest_results[sym]:
                r = backtest_results[sym]
                pred_price = None
                if sym in forecast_results and "error" not in forecast_results[sym]:
                    preds = forecast_results[sym].get("predictions", {})
                    pred_price = list(preds.values())[0] if preds else None
                bt_rows.append({
                    "Stock": sym,
                    "MAPE (%)": f"{r['mape']:.2f}",
                    "RMSE": f"{r['rmse']:.2f}",
                    "Predicted Price": f"${pred_price:,.2f}" if pred_price is not None else "N/A",
                    "Windows": r["windows"],
                })
        if bt_rows:
            st.dataframe(pd.DataFrame(bt_rows), use_container_width=True, hide_index=True)

        # ═══════════════════════════════════════════════════════════════
        # 3. BOTTOM: Backtesting plots (forecast vs true price)
        # ═══════════════════════════════════════════════════════════════
        st.subheader("📉 Backtesting: Forecast vs True Price")
        valid_backtest_stocks = [
            s for s in selected_stocks
            if s in backtest_results
            and "error" not in backtest_results[s]
            and "dates" in backtest_results[s]
        ]
        if valid_backtest_stocks:
            tabs = st.tabs(valid_backtest_stocks)
            for tab, sym in zip(tabs, valid_backtest_stocks):
                r = backtest_results[sym]
                with tab:
                    bt_df = pd.DataFrame({
                        "Actual": r["actuals"],
                        "Forecast": r["predictions"],
                    }, index=pd.DatetimeIndex(r["dates"]))
                    bt_df.index.name = "Date"
                    st.line_chart(bt_df, use_container_width=True)
                    st.caption("Backtest predictions vs actual close price (weekly)")

        # Historical price chart (optional, at very bottom)
        st.subheader("📈 Historical Prices")
        chart_dfs = []
        for sym in selected_stocks:
            if sym in stock_data:
                df = stock_data[sym].copy()
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date")[["Close"]].rename(columns={"Close": sym})
                chart_dfs.append(df)
        if chart_dfs:
            chart_data = pd.concat(chart_dfs, axis=1).dropna(how="all")
            if not chart_data.empty:
                st.line_chart(chart_data)

    else:
        st.info("👈 Select stocks and click **Run Forecast & Backtest** to start.")
        st.markdown("""
        **Default settings:**
        - Backtest: Last 2 years of daily data
        - Forecast: 5 days ahead (1–30 days)
        """)


if __name__ == "__main__":
    main()
