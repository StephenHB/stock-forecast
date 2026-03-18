"""
Environment detection utilities for cloud-aware behavior.

Detects whether the application is running locally or on a cloud platform
(e.g., Streamlit Community Cloud) to route data I/O accordingly.

Environment variables:
- STREAMLIT_SHARING_MODE: Set automatically by Streamlit Community Cloud.
- IS_CLOUD: Set to "true" to force cloud mode on other hosted platforms
  (e.g., Railway, Render, Heroku).
- STOCK_DATA_DIR: Override the local data directory path. When unset, defaults
  to the project-relative data/ folder.
"""

import os
from pathlib import Path


def is_cloud_environment() -> bool:
    """Return True when running on a cloud platform with no persistent filesystem.

    Detection order:
    1. STREAMLIT_SHARING_MODE env var (set automatically by Streamlit Cloud).
    2. IS_CLOUD=true env var (manual override for other cloud platforms).
    """
    return bool(
        os.getenv("STREAMLIT_SHARING_MODE")
        or os.getenv("IS_CLOUD", "").strip().lower() == "true"
    )


def get_data_dir() -> Path | None:
    """Return the configured local data directory, or None on cloud.

    Resolution order (local only):
    1. STOCK_DATA_DIR env var (absolute or relative path).
    2. Project-relative data/ directory (two levels up from this file).
    """
    if is_cloud_environment():
        return None

    custom = os.getenv("STOCK_DATA_DIR")
    if custom:
        return Path(custom)

    # Resolve to <project_root>/data/
    return Path(__file__).parent.parent.parent / "data"
