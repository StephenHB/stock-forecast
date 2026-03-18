"""
Environment detection utilities for cloud-aware behavior.

Detects whether the application is running locally or on a cloud platform
(e.g., Streamlit Community Cloud) to route data I/O accordingly.

Environment variables:
- STREAMLIT_SHARING_MODE: Set automatically by older Streamlit Community Cloud
  deployments (may not be present in newer ones).
- IS_CLOUD: Set to "true" to force cloud mode on other hosted platforms
  (e.g., Railway, Render, Heroku).
- STOCK_DATA_DIR: Override the local data directory path. When unset, defaults
  to the project-relative data/ folder.

Streamlit Community Cloud also exposes itself via two reliable signals that
require no env var configuration:
- The repo is mounted at /mount/src/<repo-name>/, so __file__ will start with
  /mount/src/ when running on the platform.
- The HOME directory is set to /home/appuser on Streamlit Cloud workers.
"""

import os
from pathlib import Path


def is_cloud_environment() -> bool:
    """Return True when running on a cloud platform with no persistent filesystem.

    Detection order:
    1. STREAMLIT_SHARING_MODE env var (older Streamlit Cloud deployments).
    2. IS_CLOUD=true env var (manual override for other cloud platforms).
    3. Repo path starts with /mount/src/ (Streamlit Community Cloud mount point).
    4. HOME directory is /home/appuser (Streamlit Community Cloud worker default).
    """
    if os.getenv("STREAMLIT_SHARING_MODE"):
        return True
    if os.getenv("IS_CLOUD", "").strip().lower() == "true":
        return True
    # Streamlit Community Cloud mounts repos at /mount/src/<repo-name>/
    if str(Path(__file__).resolve()).startswith("/mount/src/"):
        return True
    # Streamlit Community Cloud worker home directory
    if os.getenv("HOME") == "/home/appuser":
        return True
    return False


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
