#!/usr/bin/env python3
"""
Launcher script for the stock forecasting notebook

This script ensures the proper environment is set up and launches the Jupyter notebook
with the correct Python path and dependencies.
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    """Launch the forecasting notebook with proper environment setup."""
    
    # Get the project root directory
    project_root = Path(__file__).parent.absolute()
    
    print("🚀 Stock Forecasting Notebook Launcher")
    print("=" * 50)
    print(f"📁 Project root: {project_root}")
    
    # Check if we're in the right directory
    if not (project_root / "src").exists():
        print("❌ Error: src directory not found. Please run this script from the project root.")
        sys.exit(1)
    
    # Check if the notebook exists
    notebook_path = project_root / "notebooks" / "stock_forecasting.ipynb"
    if not notebook_path.exists():
        print(f"❌ Error: Notebook not found at {notebook_path}")
        sys.exit(1)
    
    print(f"📓 Notebook found: {notebook_path}")
    
    # Check if uv is available
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ uv found: {result.stdout.strip()}")
        else:
            print("❌ uv not found. Please install uv first.")
            sys.exit(1)
    except FileNotFoundError:
        print("❌ uv not found. Please install uv first.")
        sys.exit(1)
    
    # Check if dependencies are installed
    print("🔍 Checking dependencies...")
    try:
        result = subprocess.run([
            "uv", "run", "python", "-c", 
            "import yfinance, lightgbm, pandas, numpy, matplotlib, seaborn, sklearn; print('✅ All dependencies available')"
        ], capture_output=True, text=True, cwd=project_root)
        
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print("❌ Some dependencies are missing. Installing...")
            subprocess.run(["uv", "sync"], cwd=project_root, check=True)
            print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error checking dependencies: {e}")
        sys.exit(1)
    
    # Launch the notebook
    print("\n🎯 Launching Jupyter Notebook...")
    print("📝 The notebook will open in your default browser")
    print("💡 Make sure to run all cells in order for the complete forecasting pipeline")
    print("\n" + "=" * 50)
    
    try:
        # Change to project root and launch notebook
        subprocess.run([
            "uv", "run", "jupyter", "notebook", 
            str(notebook_path)
        ], cwd=project_root)
    except KeyboardInterrupt:
        print("\n👋 Notebook launcher stopped by user")
    except Exception as e:
        print(f"❌ Error launching notebook: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
