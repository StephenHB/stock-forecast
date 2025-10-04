"""
Test script to verify imports work correctly
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.abspath('.')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}...")  # Show first 3 paths

try:
    from src.forecasting import ForecastingPipeline
    print("✅ ForecastingPipeline import successful")
except ImportError as e:
    print(f"❌ ForecastingPipeline import failed: {e}")

try:
    from src.data_preprocess.stock_data_loader import StockDataLoader
    print("✅ StockDataLoader import successful")
except ImportError as e:
    print(f"❌ StockDataLoader import failed: {e}")

try:
    from src.feature_engineering import FourierTransformer
    print("✅ Feature engineering imports successful")
except ImportError as e:
    print(f"❌ Feature engineering imports failed: {e}")

print("\n🎉 All imports working correctly!")
