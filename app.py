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

from src.utils.environment import is_cloud_environment, get_data_dir

from src.data_preprocess.stock_data_loader import StockDataLoader
from src.forecasting import (
    WeeklyAggregator,
    StandaloneBacktester,
    run_multi_stock_simulation,
)
from src.research import CapitalMarketResearcher
from src.forecasting.feature_factory import (
    create_daily_features,
    create_weekly_features,
    create_daily_targets,
    create_weekly_targets,
    get_feature_columns,
)
from src.forecasting.research_features import (
    get_research_features_for_symbols,
    append_research_features_to_data,
)

# Page config
st.set_page_config(page_title="Stock Forecast", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# Default stocks for quick selection (top tech + popular)
DEFAULT_STOCKS = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMD", "TSLA", "META", "AMZN"]

# Minimal custom CSS (avoids forcing light backgrounds that clash with dark mode)
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] { padding: 0.75rem 1.25rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_available_stocks():
    """Load full stock universe: S&P 100 + market indices."""
    config_path = Path(__file__).parent / "config" / "stocks_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    sp100 = config.get("sp100_stocks", [])
    indices = config.get("market_indices", [])
    defaults = config.get("default_stocks", DEFAULT_STOCKS)
    # Combine: defaults first (quick picks), then indices, then full S&P 100
    combined = list(dict.fromkeys(defaults + indices + sp100))
    return combined


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


def _days_to_weeks(forecast_days: int) -> int:
    """Convert forecast days to weeks (business days: ~5 per week)."""
    return max(1, (forecast_days + 4) // 5)


def _get_daily_close_series(daily_df: pd.DataFrame) -> pd.Series:
    """
    Extract daily Close series using the SAME logic as the Historical Price chart.
    Single source of truth for chart and simulation table.
    """
    df = daily_df.copy()
    date_col = "Date" if "Date" in df.columns else ("Datetime" if "Datetime" in df.columns else None)
    close_col = "Close" if "Close" in df.columns else next(
        (c for c in df.columns if "close" in str(c).lower()), df.columns[0]
    )
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    closes = df[close_col].dropna()
    # Normalize to timezone-naive for consistent comparison
    if getattr(closes.index, "tz", None) is not None:
        closes = closes.copy()
        closes.index = pd.DatetimeIndex(
            [x.replace(tzinfo=None) if hasattr(x, "tzinfo") and x.tzinfo else x for x in closes.index]
        )
    return closes


def _get_start_end_prices_from_daily(
    daily_df: pd.DataFrame,
    first_test_date: pd.Timestamp,
    last_test_date: pd.Timestamp,
) -> tuple[float, float]:
    """
    Get start and end Close prices from raw daily data to match Historical chart.
    Uses _get_daily_close_series for consistency.
    Start = prior close before first test date; End = close on last test date.
    """
    closes = _get_daily_close_series(daily_df)
    if closes.empty:
        return 0.0, 0.0
    first_ts = pd.Timestamp(first_test_date)
    last_ts = pd.Timestamp(last_test_date)
    if first_ts.tz is not None:
        first_ts = first_ts.replace(tzinfo=None)
    if last_ts.tz is not None:
        last_ts = last_ts.replace(tzinfo=None)
    before_first = closes[closes.index < first_ts]
    start_price = float(before_first.iloc[-1]) if len(before_first) > 0 else float(closes.iloc[0])
    on_or_before_last = closes[closes.index <= last_ts]
    end_price = float(on_or_before_last.iloc[-1]) if len(on_or_before_last) > 0 else float(closes.iloc[-1])
    return start_price, end_price


def run_backtest(
    stock_data: dict,
    forecast_horizon_weeks: int,
    forecast_days: int,
    include_research_features: bool = False,
) -> dict:
    """Run backtesting. Uses daily data + volatility features when horizon <= 5 days."""
    use_daily = forecast_days <= 5
    weekly_aggregator = (
        WeeklyAggregator(
            price_columns=["Open", "High", "Low", "Close"],
            volume_columns=["Volume"],
        )
        if not use_daily
        else None
    )

    results = {}
    for symbol, daily_df in stock_data.items():
        try:
            daily = daily_df.copy()
            if "Date" in daily.columns:
                daily["Date"] = pd.to_datetime(daily["Date"], utc=True).dt.tz_localize(None)
                daily.set_index("Date", inplace=True)
            cols = {c: c.replace(" ", "_") for c in daily.columns if " " in c}
            daily = daily.rename(columns=cols)

            if use_daily:
                features = create_daily_features(daily)
                data = create_daily_targets(features, forecast_days)
                target_col = f"target_{forecast_days}d"
                returns = daily["Close"].pct_change().dropna()
                periods = 252
            else:
                weekly = weekly_aggregator.aggregate(daily)
                features = create_weekly_features(weekly)
                data = create_weekly_targets(features, forecast_horizon_weeks)
                target_col = f"target_{forecast_horizon_weeks}w"
                initial_train = 13
                min_train = 6
                returns = weekly["Close"].pct_change().dropna()
                periods = 52

            data = data.dropna()
            n_rows = len(data)
            min_rows = 60 if use_daily else 20
            if n_rows < min_rows:
                continue

            # Optional: append research features (news sentiment, SEC, financials)
            if include_research_features:
                try:
                    research_feats = get_research_features_for_symbols([symbol])
                    data = append_research_features_to_data(data, symbol, research_feats)
                except Exception:
                    pass  # Fall back to price-only features if research fails

            # Scale daily training size for short backtests (1yr ~252 days)
            if use_daily:
                initial_train = min(260, max(60, n_rows - 30))
                min_train = min(60, max(20, n_rows // 4))

            feature_columns = get_feature_columns(data, target_col)

            backtester = StandaloneBacktester(
                initial_train_size=initial_train,
                test_size=1,
                # Non-overlapping windows: step forward by the full horizon so
                # consecutive signals don't share target days.
                # Daily: step by forecast_days (e.g. 5-day horizon → signal every 5 days).
                # Weekly: step by 1 week (already matches a 1-week-ahead target).
                step_size=forecast_days if use_daily else 1,
                min_train_size=min_train,
                target_column=target_col,
                forecast_horizon=forecast_horizon_weeks if not use_daily else forecast_days,
            )
            bt_results = backtester.backtest(data, feature_columns=feature_columns)
            metrics = bt_results["overall_metrics"]
            volatility_annual = (
                returns.std() * np.sqrt(periods) * 100 if len(returns) > 1 else 0.0
            )

            # Price series for trading simulation (same frequency as backtest)
            price_series = data["Close"] if "Close" in data.columns else data.iloc[:, 0]

            results[symbol] = {
                "mape": metrics["mape"] * 100,
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "windows": bt_results["total_windows"],
                "predictions": bt_results["predictions"],
                "actuals": bt_results["actuals"],
                "dates": bt_results["dates"],
                "volatility_pct": volatility_annual,
                "forecast_days": forecast_days,
                "price_series": price_series,
            }
        except Exception as e:
            results[symbol] = {"error": str(e)}
    return results


def run_forecast(
    stock_data: dict,
    forecast_horizon_weeks: int,
    forecast_days: int,
    backtest_results: dict,
    include_research_features: bool = False,
) -> dict:
    """Run forecast. Uses daily data + volatility features when horizon <= 5 days."""
    import lightgbm as lgb
    from sklearn.preprocessing import StandardScaler

    use_daily = forecast_days <= 5
    weekly_aggregator = (
        WeeklyAggregator(
            price_columns=["Open", "High", "Low", "Close"],
            volume_columns=["Volume"],
        )
        if not use_daily
        else None
    )

    best_params = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "verbose": -1,
    }
    scaler = StandardScaler()

    results = {}
    for symbol, daily_df in stock_data.items():
        try:
            daily = daily_df.copy()
            if "Date" in daily.columns:
                daily["Date"] = pd.to_datetime(daily["Date"], utc=True).dt.tz_localize(None)
                daily.set_index("Date", inplace=True)
            cols = {c: c.replace(" ", "_") for c in daily.columns if " " in c}
            daily = daily.rename(columns=cols)

            if use_daily:
                features = create_daily_features(daily)
                data = create_daily_targets(features, forecast_days)
                target_col = f"target_{forecast_days}d"
            else:
                weekly = weekly_aggregator.aggregate(daily)
                features = create_weekly_features(weekly)
                data = create_weekly_targets(features, forecast_horizon_weeks)
                target_col = f"target_{forecast_horizon_weeks}w"

            data = data.dropna()
            min_rows = 60 if use_daily else 20
            if len(data) < min_rows:
                results[symbol] = {"error": "Insufficient data"}
                continue

            # Optional: append research features (news sentiment, SEC, financials)
            if include_research_features:
                try:
                    research_feats = get_research_features_for_symbols([symbol])
                    data = append_research_features_to_data(data, symbol, research_feats)
                except Exception:
                    pass  # Fall back to price-only features if research fails

            feature_columns = get_feature_columns(data, target_col)

            X = data[feature_columns]
            y = data[target_col]
            X_scaled = scaler.fit_transform(X)
            model = lgb.LGBMRegressor(**best_params)
            model.fit(X_scaled, y)

            last_row = data.iloc[-1:][feature_columns]
            X_pred = scaler.transform(last_row)
            pred_price = float(model.predict(X_pred)[0])
            last_price = float(daily["Close"].iloc[-1])

            mape = backtest_results.get(symbol, {}).get("mape", 10.0)
            volatility = backtest_results.get(symbol, {}).get("volatility_pct", 20.0)

            results[symbol] = {
                "predictions": {1: pred_price},
                "last_price": last_price,
                "risk_rating": _mape_to_risk_rating(mape),
                "volatility_pct": volatility,
            }
        except Exception as e:
            results[symbol] = {"error": str(e)}
    return results


def _mape_to_risk_rating(mape: float) -> str:
    """Convert MAPE to risk/confidence rating."""
    if mape < 3:
        return "Low"
    if mape < 7:
        return "Medium"
    if mape < 12:
        return "Moderate-High"
    return "High"


def main():
    st.title("📈 Stock Forecast")
    st.markdown("Select stocks, set forecast horizon, and view backtesting results.")

    available_stocks = load_available_stocks()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        selected_stocks = st.multiselect(
            "Select stocks",
            options=available_stocks,
            default=["AAPL", "GOOGL", "NVDA"],
            help="S&P 100 stocks + market indices (SPY, QQQ, etc.). Type to search.",
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
        include_research_features = st.checkbox(
            "Include research features (news, SEC, financials)",
            value=False,
            help="Add news sentiment, earnings, and financial metrics as LGBM features. May be slow or unstable on some systems.",
        )

        with st.expander("⚙️ Simulation Settings"):
            sim_threshold_pct = st.slider(
                "Signal threshold (%)",
                min_value=0.0,
                max_value=3.0,
                value=0.5,
                step=0.1,
                help=(
                    "Dead-zone half-width. Only trade when the model's implied "
                    "predicted move exceeds this percentage. Signals inside the "
                    "dead zone keep the current position unchanged, reducing "
                    "whipsaw trades. Set to 0 to disable."
                ),
            )
            sim_cost_pct = st.slider(
                "Transaction cost per side (%)",
                min_value=0.0,
                max_value=0.5,
                value=0.1,
                step=0.01,
                help=(
                    "Fractional cost applied to every buy and sell execution "
                    "(commissions + bid-ask spread). 0.1% is a realistic "
                    "estimate for liquid US equities. Set to 0 to disable."
                ),
            )

        run_btn = st.button("🚀 Run Forecast & Backtest", type="primary")

        st.divider()
        st.caption("**Storage mode**")
        if is_cloud_environment():
            st.info("☁️ Cloud — data is fetched fresh each session (no local saving).", icon="ℹ️")
        else:
            _data_dir = get_data_dir()
            st.success(f"💾 Local — data directory: `{_data_dir}`", icon="✅")

    forecast_weeks = _days_to_weeks(forecast_days)
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

            backtest_results = run_backtest(
                stock_data, forecast_weeks, forecast_days, include_research_features
            )
            forecast_results = run_forecast(
                stock_data, forecast_weeks, forecast_days, backtest_results,
                include_research_features,
            )

        # Trading simulation: 100k total, split equally across stocks
        initial_cash_total = 100_000.0
        n_stocks = len([s for s in selected_stocks if s in backtest_results and "error" not in backtest_results.get(s, {})])
        cash_per_stock = initial_cash_total / n_stocks if n_stocks > 0 else 0.0

        price_series_by_symbol = {}
        for sym in selected_stocks:
            if sym in backtest_results and "error" not in backtest_results[sym]:
                ps = backtest_results[sym].get("price_series")
                if ps is not None:
                    price_series_by_symbol[sym] = ps

        sim_results = {}
        if price_series_by_symbol:
            sim_results = run_multi_stock_simulation(
                backtest_results,
                price_series_by_symbol,
                initial_cash_per_stock=cash_per_stock,
                threshold_pct=sim_threshold_pct,
                cost_per_side=sim_cost_pct / 100.0,
            )

        # Main content
        st.success(f"✅ Processed {len(stock_data)} stocks")

        # Analysis and forecast dates (top of UI)
        analysis_date = None
        for sym in selected_stocks:
            if sym in stock_data:
                df = stock_data[sym]
                date_col = "Date" if "Date" in df.columns else ("Datetime" if "Datetime" in df.columns else None)
                if date_col:
                    analysis_date = pd.to_datetime(df[date_col]).max()
                elif isinstance(df.index, pd.DatetimeIndex):
                    analysis_date = df.index.max()
                if analysis_date is not None:
                    break
        if analysis_date is None:
            analysis_date = pd.Timestamp(end_date)
        analysis_date = pd.Timestamp(analysis_date)
        forecast_target_date = analysis_date + pd.offsets.BDay(forecast_days)
        current_dt = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Summary metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Stocks", len(stock_data))
        with col2:
            st.metric("Forecast horizon", f"{forecast_days} days")
        with col3:
            st.metric("Backtest period", f"{backtest_years} yr")
        with col4:
            st.metric("Target date", forecast_target_date.strftime("%Y-%m-%d"))
        st.caption(f"Analysis date: {analysis_date.strftime('%Y-%m-%d')} · Current: {current_dt}")
        st.divider()

        # Forecast summary - sidebar
        st.sidebar.header("📊 Forecast Summary")
        for sym in selected_stocks:
            has_error = sym in backtest_results and "error" in backtest_results[sym]
            if has_error:
                st.sidebar.error(f"{sym}: {backtest_results[sym]['error']}")
                continue

            pred_price = None
            last_price_fc = None
            if sym in forecast_results and "error" not in forecast_results[sym]:
                preds = forecast_results[sym].get("predictions", {})
                pred_price = list(preds.values())[0] if preds else None
                last_price_fc = forecast_results[sym].get("last_price")

            if pred_price is not None and last_price_fc is not None:
                pct_change = (pred_price - last_price_fc) / last_price_fc * 100
                delta_str = f"{pct_change:+.2f}% in {forecast_days}d"
                delta_color = "normal" if pred_price >= last_price_fc else "inverse"
                st.sidebar.metric(
                    label=sym,
                    value=f"${pred_price:,.2f}",
                    delta=delta_str,
                    delta_color=delta_color,
                )
            elif sym in backtest_results and "error" not in backtest_results[sym]:
                st.sidebar.metric(label=sym, value="—", delta=None)

        # Main content tabs
        tab_overview, tab_backtest, tab_simulation, tab_historical = st.tabs([
            "📋 Overview", "📉 Backtest", "💰 Simulation", "📈 Historical"
        ])

        # ═══════════════════════════════════════════════════════════════
        # TAB 1: Overview (Predictions, Research, Metrics, Volatility)
        # ═══════════════════════════════════════════════════════════════
        with tab_overview:
            st.subheader("🔮 Predicted Prices")
        mode_note = (
            "daily data + volatility features"
            if forecast_days <= 5
            else f"~{forecast_weeks} week(s), business days"
        )
        st.caption(
            f"Forecast for next {forecast_days} day(s) ahead ({mode_note})"
        )
        pred_rows = []
        pred_directions = []  # 1=up, -1=down, 0=neutral/N/A
        for sym in selected_stocks:
            if sym in forecast_results and "error" not in forecast_results[sym]:
                r = forecast_results[sym]
                preds = r.get("predictions", {})
                pred_price = list(preds.values())[0] if preds else None
                last = r.get("last_price")
                risk = r.get("risk_rating", "—")
                vol = r.get("volatility_pct")
                if pred_price is not None and last is not None:
                    pred_directions.append(1 if pred_price > last else (-1 if pred_price < last else 0))
                else:
                    pred_directions.append(0)
                pred_rows.append({
                    "Stock": sym,
                    "Predicted Price": f"${pred_price:,.2f}" if pred_price is not None else "N/A",
                    "Last Price": f"${last:,.2f}" if last is not None else "—",
                    "Risk Rating": risk,
                    "Volatility (ann. %)": f"{vol:.1f}" if vol is not None else "—",
                })
            elif sym in forecast_results:
                pred_directions.append(0)
                pred_rows.append({
                    "Stock": sym,
                    "Predicted Price": "N/A",
                    "Last Price": "—",
                    "Risk Rating": "—",
                    "Volatility (ann. %)": "—",
                })
        if pred_rows:
            pred_df = pd.DataFrame(pred_rows)

            def _color_pred(col):
                if col.name != "Predicted Price":
                    return [""] * len(col)
                return [
                    "color: green" if d == 1 else ("color: red" if d == -1 else "")
                    for d in pred_directions
                ]

            st.dataframe(
                pred_df.style.apply(_color_pred, axis=0),
                use_container_width=True,
                hide_index=True,
            )

        # ═══════════════════════════════════════════════════════════════
        # 1b. Capital Market Research (news sentiment, SEC filings, impact features)
        # ═══════════════════════════════════════════════════════════════
        with st.expander("📰 Capital Market Research (news sentiment, SEC filings, impact features)"):
            from src.research.news_report_analyzer import analyze_news_sentiment, summarize_financial_reports
            cmr = CapitalMarketResearcher()
            for sym in selected_stocks:
                try:
                    result = cmr.research(sym)
                    st.markdown(f"### {sym}")

                    # Impact features
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Short-run impact**")
                        for k, v in result.short_run.items():
                            st.caption(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")
                    with col2:
                        st.markdown("**Long-run impact**")
                        for k, v in result.long_run.items():
                            st.caption(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")

                    # News with sentiment
                    st.markdown("**Top news (with sentiment)**")
                    analyses = analyze_news_sentiment([n.title for n in result.news])
                    for n, a in zip(result.news[:6], analyses[:6]):
                        label = "🟢" if a.sentiment_label == "positive" else ("🔴" if a.sentiment_label == "negative" else "⚪")
                        st.caption(f"{label} {n.title[:85]}{'...' if len(n.title) > 85 else ''}")
                    if analyses:
                        pos = sum(1 for a in analyses if a.sentiment_label == "positive")
                        neg = sum(1 for a in analyses if a.sentiment_label == "negative")
                        neu = len(analyses) - pos - neg
                        st.caption(f"*Sentiment summary: {pos} positive, {neg} negative, {neu} neutral*")

                    # SEC filings
                    if result.reports:
                        report_sum = summarize_financial_reports(result.reports)
                        st.markdown(f"**SEC filings** ({report_sum.total_count} recent)")
                        for r in result.reports[:6]:
                            link = f"[{r.report_type}]({r.url})" if r.url else f"[{r.report_type}]"
                            st.caption(f"• {link}: {r.title[:55]}{'...' if len(r.title) > 55 else ''}")
                except Exception as e:
                    st.caption(f"{sym}: {e}")

        # ═══════════════════════════════════════════════════════════════
        # 2. MIDDLE: Statistical metrics with risk and volatility
        # ═══════════════════════════════════════════════════════════════
        st.subheader("📊 Statistical Metrics")
        bt_rows = []
        for sym in selected_stocks:
            if sym in backtest_results and "error" not in backtest_results[sym]:
                r = backtest_results[sym]
                pred_price = None
                risk = "—"
                vol = None
                if sym in forecast_results and "error" not in forecast_results[sym]:
                    fr = forecast_results[sym]
                    preds = fr.get("predictions", {})
                    pred_price = list(preds.values())[0] if preds else None
                    risk = fr.get("risk_rating", "—")
                    vol = fr.get("volatility_pct")
                bt_rows.append({
                    "Stock": sym,
                    "MAPE (%)": f"{r['mape']:.2f}",
                    "RMSE": f"{r['rmse']:.2f}",
                    "Predicted Price": f"${pred_price:,.2f}" if pred_price is not None else "N/A",
                    "Risk Rating": risk,
                    "Volatility (ann. %)": f"{vol:.1f}" if vol is not None else "—",
                    "Windows": r["windows"],
                })
        if bt_rows:
            st.dataframe(pd.DataFrame(bt_rows), use_container_width=True, hide_index=True)
            st.caption(
                "**Risk Rating** is derived from backtest MAPE: "
                "Low (<3%), Medium (3–7%), Moderate-High (7–12%), High (>12%). "
                "Lower MAPE indicates higher forecast confidence."
            )

        # ═══════════════════════════════════════════════════════════════
        # 2b. Volatility analysis
        # ═══════════════════════════════════════════════════════════════
        st.subheader("📈 Volatility Analysis")
        vol_data = []
        for sym in selected_stocks:
            if sym in backtest_results and "error" not in backtest_results[sym]:
                vol = backtest_results[sym].get("volatility_pct")
                if vol is not None:
                    vol_data.append({"Stock": sym, "Annualized Volatility (%)": f"{vol:.2f}"})
        if vol_data:
            st.caption("Historical volatility (annualized std of weekly returns)")
            st.dataframe(pd.DataFrame(vol_data), use_container_width=True, hide_index=True)

        # ═══════════════════════════════════════════════════════════════
        # TAB 2: Backtesting plots (forecast vs true price)
        # ═══════════════════════════════════════════════════════════════
        with tab_backtest:
            st.subheader("📉 Forecast vs True Price")
            valid_backtest_stocks = [
                s for s in selected_stocks
                if s in backtest_results
                and "error" not in backtest_results[s]
                and "dates" in backtest_results[s]
            ]
            if valid_backtest_stocks:
                bt_tabs = st.tabs(valid_backtest_stocks)
                for tab, sym in zip(bt_tabs, valid_backtest_stocks):
                    r = backtest_results[sym]
                    with tab:
                        bt_df = pd.DataFrame({
                            "Actual": r["actuals"],
                            "Forecast": r["predictions"],
                        }, index=pd.DatetimeIndex(r["dates"]))
                        bt_df.index.name = "Date"
                        st.line_chart(bt_df, use_container_width=True)
                        st.caption("Backtest predictions vs actual close price (weekly)")
            else:
                st.info("No backtest data available.")

        # ═══════════════════════════════════════════════════════════
        # TAB 3: Trading Simulation
        # ═══════════════════════════════════════════════════════════
        with tab_simulation:
            st.subheader("💰 Trading Simulation")

            # ── Strategy explanation ──────────────────────────────
            with st.expander("📖 How This Strategy Works", expanded=False):
                full_conviction_display = sim_threshold_pct * 3.0
                st.markdown(f"""
This simulation applies **5 research-driven rules** on top of the LGBM forecast:

| # | Rule | Detail |
|---|---|---|
| 1 | **Implied-return signal** | Signal = (Predicted Close − Today's Close) ÷ Today's Close × 100. Measures the *magnitude* of the expected move, not just direction. |
| 2 | **Dead-zone threshold** | Only trade when the implied return magnitude exceeds **{sim_threshold_pct:.1f}%**. Signals weaker than this keep the current position unchanged, eliminating low-conviction whipsaw trades. Adjust in *Simulation Settings*. |
| 3 | **Proportional position sizing** | A BUY deploys a fraction of available cash proportional to signal strength: 0% at the threshold → 100% at **{full_conviction_display:.1f}%** move. Weak signals get a small position; strong signals go all-in. |
| 4 | **Transaction costs** | Every execution is adjusted by **{sim_cost_pct:.2f}%** per side (commission + bid-ask spread). Adjust in *Simulation Settings*. |
| 5 | **Regime filter** | If a SELL signal fires while the stock is **above its 200-day moving average** (uptrend), the signal is suppressed to HOLD. This prevents exiting winning long-term positions on short-term noise. |

**Portfolio is marked-to-market at the actual close {forecast_days} day(s) after each signal date.**
Capital of ${initial_cash_total:,.0f} is split equally across selected stocks.
                """)

            # ── Current signal: what to do NOW ───────────────────
            st.markdown("### 🎯 Current Signal — What to Do Now")
            rec_stocks = [
                s for s in selected_stocks
                if s in forecast_results and "error" not in forecast_results.get(s, {})
            ]
            if rec_stocks:
                rec_cols = st.columns(len(rec_stocks))
                for rec_col, sym in zip(rec_cols, rec_stocks):
                    with rec_col:
                        fc = forecast_results[sym]
                        last_price = fc.get("last_price", 0.0)
                        pred_price = fc.get("predictions", {}).get(1, last_price)
                        implied_return_pct = (
                            (pred_price - last_price) / last_price * 100
                            if last_price > 0 else 0.0
                        )

                        # Base signal
                        if implied_return_pct > sim_threshold_pct:
                            raw_signal = "BUY"
                        elif implied_return_pct < -sim_threshold_pct:
                            raw_signal = "SELL"
                        else:
                            raw_signal = "HOLD"

                        # Regime check (200-day MA)
                        in_uptrend = False
                        regime_label = "—"
                        if sym in stock_data:
                            closes = _get_daily_close_series(stock_data[sym])
                            if len(closes) >= 100:
                                ma_200 = closes.rolling(200, min_periods=100).mean()
                                in_uptrend = float(closes.iloc[-1]) > float(ma_200.iloc[-1])
                                regime_label = "Uptrend ↑" if in_uptrend else "Downtrend ↓"

                        # Apply regime filter
                        regime_note = ""
                        final_signal = raw_signal
                        if raw_signal == "SELL" and in_uptrend:
                            final_signal = "HOLD"
                            regime_note = "SELL overridden by uptrend regime."

                        # Confidence / suggested position size
                        full_conviction = max(sim_threshold_pct * 3.0, 0.3)
                        confidence = min(abs(implied_return_pct) / full_conviction, 1.0)

                        # Risk rating from backtest
                        risk = fc.get("risk_rating", "—")

                        # Display
                        st.metric(
                            label=f"**{sym}**",
                            value=f"${last_price:,.2f}",
                            delta=f"{implied_return_pct:+.2f}% implied ({forecast_days}d)",
                        )
                        st.caption(
                            f"Predicted: **${pred_price:,.2f}** &nbsp;|&nbsp; "
                            f"Regime: **{regime_label}** &nbsp;|&nbsp; "
                            f"Model confidence: **{risk}**"
                        )
                        if final_signal == "BUY":
                            position_size = f"{confidence * 100:.0f}% of available cash"
                            st.success(
                                f"**BUY** — deploy {position_size}\n\n"
                                f"Predicted move of {implied_return_pct:+.2f}% exceeds the "
                                f"{sim_threshold_pct:.1f}% threshold."
                            )
                        elif final_signal == "SELL":
                            st.error(
                                f"**SELL** — liquidate full position\n\n"
                                f"Predicted move of {implied_return_pct:+.2f}% is below "
                                f"−{sim_threshold_pct:.1f}% threshold."
                            )
                        else:
                            if regime_note:
                                st.warning(
                                    f"**HOLD** — keep current position\n\n"
                                    f"{regime_note} Raw SELL signal overridden because "
                                    f"stock is in an uptrend (above 200-day MA)."
                                )
                            else:
                                st.warning(
                                    f"**HOLD** — keep current position\n\n"
                                    f"Implied return {implied_return_pct:+.2f}% is within "
                                    f"the ±{sim_threshold_pct:.1f}% dead zone — "
                                    f"signal too weak to act on."
                                )
            else:
                st.info("No forecast results available to generate recommendations.")

            st.divider()
            valid_sim_stocks = [
                s for s in selected_stocks
                if s in sim_results and sim_results[s] is not None
            ]
            if valid_sim_stocks:
                # Summary table with start/end price and buy-and-hold comparison
                # Use raw daily stock_data for start/end prices to match Historical chart
                sim_rows = []
                total_final = 0.0
                total_buy_hold = 0.0
                for sym in valid_sim_stocks:
                    res = sim_results[sym]
                    bt = backtest_results.get(sym, {})
                    dates = bt.get("dates", [])
                    start_price_display = res.start_price
                    end_price_display = res.end_price
                    buy_hold_pct = res.buy_hold_return_pct
                    if sym in stock_data:
                        # Use full backtest data period (e.g. 2 years), not test window
                        closes = _get_daily_close_series(stock_data[sym])
                        if not closes.empty:
                            start_price_display = float(closes.iloc[0])
                            end_price_display = float(closes.iloc[-1])
                            buy_hold_pct = (
                                (end_price_display / start_price_display - 1) * 100
                                if start_price_display > 0
                                else 0.0
                            )
                    total_final += res.final_value
                    total_buy_hold += res.initial_cash * (1 + buy_hold_pct / 100)
                    total_signals = res.n_buys + res.n_sells + res.n_holds
                    sim_rows.append({
                        "Stock": sym,
                        "Start Price": f"${start_price_display:,.2f}",
                        "End Price": f"${end_price_display:,.2f}",
                        "Initial ($)": f"${res.initial_cash:,.0f}",
                        "Final Value ($)": f"${res.final_value:,.0f}",
                        "Forecast Trade (%)": f"{res.total_return_pct:+.1f}%",
                        "Buy & Hold (%)": f"{buy_hold_pct:+.1f}%",
                        "Buys": res.n_buys,
                        "Sells": res.n_sells,
                        "Holds": res.n_holds,
                        "Costs ($)": f"${res.total_cost_paid:,.0f}",
                    })
                sim_rows.append({
                    "Stock": "**Total**",
                    "Start Price": "—",
                    "End Price": "—",
                    "Initial ($)": f"${initial_cash_total:,.0f}",
                    "Final Value ($)": f"${total_final:,.0f}",
                    "Forecast Trade (%)": f"{(total_final - initial_cash_total) / initial_cash_total * 100:+.1f}%",
                    "Buy & Hold (%)": f"{(total_buy_hold - initial_cash_total) / initial_cash_total * 100:+.1f}%",
                    "Buys": "—",
                    "Sells": "—",
                    "Holds": "—",
                    "Costs ($)": f"${sum(sim_results[s].total_cost_paid for s in valid_sim_stocks):,.0f}",
                })
                st.dataframe(
                    pd.DataFrame(sim_rows),
                    use_container_width=True,
                    hide_index=True,
                )
                date_range = ""
                if valid_sim_stocks and valid_sim_stocks[0] in stock_data:
                    closes = _get_daily_close_series(stock_data[valid_sim_stocks[0]])
                    if not closes.empty:
                        date_range = f" Full backtest period: {closes.index[0].strftime('%Y-%m-%d')} to {closes.index[-1].strftime('%Y-%m-%d')}."
                st.caption(
                    f"**Start/End Price:** Daily close at beginning and end of full backtest period (matches Historical chart).{date_range} "
                    "**Buy & Hold:** Gain/loss if you invested at the start and held until the end."
                )
                # Equity curves
                st.markdown("**Portfolio value over test period**")
                sim_tabs = st.tabs(valid_sim_stocks)
                for tab, sym in zip(sim_tabs, valid_sim_stocks):
                    res = sim_results[sym]
                    with tab:
                        eq_df = res.equity_curve.to_frame(name="Portfolio Value ($)")
                        eq_df.index.name = "Date"
                        st.line_chart(eq_df, use_container_width=True)
            else:
                st.info("No simulation results available.")

        # ═══════════════════════════════════════════════════════════
        # TAB 4: Historical price chart
        # ═══════════════════════════════════════════════════════════
        with tab_historical:
            st.subheader("📈 Historical Prices")
            st.caption("Daily close prices (full backtest period)")
            chart_dfs = []
            for sym in selected_stocks:
                if sym in stock_data:
                    close_series = _get_daily_close_series(stock_data[sym])
                    if not close_series.empty:
                        chart_dfs.append(close_series.to_frame(name=sym))
            if chart_dfs:
                chart_data = pd.concat(chart_dfs, axis=1).dropna(how="all")
                if not chart_data.empty:
                    st.line_chart(chart_data)
            else:
                st.info("No historical price data available.")

    else:
        st.info("👈 Select stocks and click **Run Forecast & Backtest** to start.")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**Quick start**")
            st.markdown("""
            1. Pick one or more stocks from the sidebar  
            2. Set forecast horizon (default: 5 days)  
            3. Click **Run Forecast & Backtest**
            """)
        with col2:
            st.markdown("**Default settings**")
            st.markdown("""
            - **Backtest:** Last 2 years of daily data  
            - **Forecast:** 5 days ahead (1–30 days)  
            - **Simulation:** $100k split across selected stocks
            """)


if __name__ == "__main__":
    main()
