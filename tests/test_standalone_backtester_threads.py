"""StandaloneBacktester optional LightGBM num_threads."""

from src.forecasting.standalone_backtester import StandaloneBacktester


def test_lgb_num_threads_merged_into_best_params():
    bt = StandaloneBacktester(
        initial_train_size=10,
        test_size=1,
        step_size=1,
        min_train_size=5,
        target_column="Close",
        forecast_horizon=1,
        lgb_num_threads=3,
    )
    assert bt.best_params["num_threads"] == 3


def test_lgb_num_threads_none_omits_key():
    bt = StandaloneBacktester(
        initial_train_size=10,
        test_size=1,
        step_size=1,
        min_train_size=5,
        target_column="Close",
        forecast_horizon=1,
        lgb_num_threads=None,
    )
    assert "num_threads" not in bt.best_params
