"""Fixed settings shared by the baselines (Phase 3) and the ML model (Phase 4).

Changing anything here changes the test set, so earlier results are no longer
comparable. Add a new entry to DECISIONS.md if you do.
"""

import pandas as pd

from ml.grid import OSLO

# Train on everything before SPLIT, test on origins in [SPLIT, TEST_END).
# Both are Fridays at midnight Oslo time, so the test set is exactly one week
# with every weekday once. It includes the end of daylight saving time
# (Sunday 25 October), which tests that time features use local time.
SPLIT = pd.Timestamp("2026-10-23 00:00", tz=OSLO)
TEST_END = pd.Timestamp("2026-10-30 00:00", tz=OSLO)

# Forecast horizon in minutes -> number of 15-minute grid steps.
HORIZONS = {15: 1, 30: 2, 60: 4}

# Evaluation subsets, by the local time the prediction is made (the origin).
DAYTIME_HOURS = range(7, 21)  # 07:00-20:59, all days
RUSH_HOURS = (7, 8, 16, 17)  # 07:00-08:59 and 16:00-17:59, Monday-Friday
# No Norwegian public holidays fall inside the data period, so none are handled.

# A station whose bikes and docks have not changed for this long is treated as
# out of service. Only past data is used, so this also works at prediction time.
FROZEN_AFTER = pd.Timedelta("72h")
