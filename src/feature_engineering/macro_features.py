"""
Macroeconomic Calendar Features

Encodes the proximity of Federal Reserve FOMC interest-rate announcements
as numeric features so the model can learn pre/post-meeting price patterns.

Markets typically exhibit elevated volatility in the 1–5 days before and
after each FOMC statement, making proximity a useful signal for all horizons.
"""

import numpy as np
import pandas as pd


# FOMC announcement dates — second (decision) day of each two-day meeting.
# Covers 2022–2027; extend as new dates are confirmed by the Fed.
_FOMC_ANNOUNCEMENT_DATES = [
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
    # 2027 (tentative — typically released ~1 year in advance)
    "2027-01-27", "2027-03-17", "2027-05-05", "2027-06-16",
    "2027-07-28", "2027-09-15", "2027-10-27", "2027-12-08",
]

FOMC_DATES = pd.to_datetime(_FOMC_ANNOUNCEMENT_DATES)


def add_fomc_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add Federal Reserve FOMC meeting proximity features.

    Features added:
      - days_to_fomc:      calendar days until the next FOMC announcement
      - days_since_fomc:   calendar days since the last FOMC announcement
      - fomc_week_ahead:   1 if within 5 calendar days BEFORE an announcement
                           (pre-meeting uncertainty window)
      - fomc_week_after:   1 if within 5 calendar days AFTER an announcement
                           (post-meeting reaction window)

    A value of 999 is used when no past/future date is found in the table;
    models learn to ignore it as a rare out-of-range value.
    """
    f = data.copy()
    idx = pd.DatetimeIndex(f.index).floor("D")

    days_to_next = np.empty(len(idx), dtype=float)
    days_since_last = np.empty(len(idx), dtype=float)

    for i, d in enumerate(idx):
        d_ts = pd.Timestamp(d)
        future = FOMC_DATES[FOMC_DATES >= d_ts]
        past = FOMC_DATES[FOMC_DATES <= d_ts]
        days_to_next[i] = int((future.min() - d_ts).days) if len(future) > 0 else 999
        days_since_last[i] = int((d_ts - past.max()).days) if len(past) > 0 else 999

    f["days_to_fomc"] = days_to_next
    f["days_since_fomc"] = days_since_last
    f["fomc_week_ahead"] = (days_to_next <= 5).astype(float)
    f["fomc_week_after"] = (days_since_last <= 5).astype(float)

    return f
