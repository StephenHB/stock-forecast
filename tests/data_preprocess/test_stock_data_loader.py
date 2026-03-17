"""Unit tests for src.data_preprocess.stock_data_loader."""

import pytest
import pandas as pd
from pathlib import Path
import tempfile

from src.data_preprocess.stock_data_loader import StockDataLoader


@pytest.fixture
def temp_config_dir():
    """Create temp dir with minimal config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "stocks_config.yaml"
        config_path.write_text("""
default_stocks:
  - AAPL
  - MSFT
download_settings:
  start_date: "2023-01-01"
  end_date: "2023-06-30"
  interval: "1d"
  auto_adjust: true
""")
        yield tmpdir, str(config_path)


@pytest.fixture
def loader_with_temp_config(temp_config_dir):
    """StockDataLoader with temp config and data dir."""
    tmpdir, config_path = temp_config_dir
    data_dir = Path(tmpdir) / "stock_data"
    return StockDataLoader(config_path=config_path, data_dir=str(data_dir))


class TestStockDataLoader:
    """Tests for StockDataLoader."""

    def test_loads_config(self, loader_with_temp_config):
        """Should load config from YAML."""
        loader = loader_with_temp_config
        assert loader.config is not None
        assert "default_stocks" in loader.config
        assert "download_settings" in loader.config

    def test_get_stock_list_uses_config_default(self, loader_with_temp_config):
        """get_stock_list returns config default when no custom_stocks."""
        loader = loader_with_temp_config
        stocks = loader.get_stock_list()
        assert stocks == ["AAPL", "MSFT"]

    def test_get_stock_list_uses_custom_stocks(self, loader_with_temp_config):
        """get_stock_list returns custom list when provided."""
        loader = loader_with_temp_config
        custom = ["GOOGL", "TSLA"]
        stocks = loader.get_stock_list(custom_stocks=custom)
        assert stocks == custom

    @pytest.mark.skip(reason="Requires network; use mock in CI")
    def test_download_stock_data_returns_dict(self, loader_with_temp_config):
        """download_stock_data returns dict of symbol -> DataFrame."""
        loader = loader_with_temp_config
        result = loader.download_stock_data(
            stock_symbols=["AAPL"],
            start_date="2023-01-01",
            end_date="2023-01-31",
            save_data=False,
        )
        assert isinstance(result, dict)
        if "AAPL" in result:
            assert isinstance(result["AAPL"], pd.DataFrame)
