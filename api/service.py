"""Keeps the recent history in memory and turns it into forecasts.

No network or database here, so it can be tested with plain DataFrames.
"""

import threading
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from ml.features import HISTORY_NEEDED
from ml.grid import FREQ, to_grid
from ml.predict import Forecaster

# The MAE model does not beat "unchanged" at 15 minutes (validation and test),
# so the expected number at 15 minutes is the current value. Probabilities
# still come from the classifiers, which do beat their baselines there.
UNCHANGED_HORIZONS = (15,)

# Data older than this (no successful live poll) is flagged as stale.
STALE_AFTER = pd.Timedelta("35min")


class ForecastService:
    def __init__(
        self, forecaster: Forecaster, stations: pd.DataFrame, history: pd.DataFrame
    ):
        self.forecaster = forecaster
        self.stations = stations.assign(station_id=stations["station_id"].astype(str))
        self.history = history
        self.lock = threading.Lock()
        self.cache: tuple[pd.Timestamp, pd.DataFrame] | None = None

    def add_status(self, rows: pd.DataFrame) -> None:
        """Add new status rows (e.g. a live snapshot) and drop what is no
        longer needed: rows before the window, except each station's latest."""
        with self.lock:
            status = pd.concat([self.history, rows], ignore_index=True)
            cutoff = status["fetched_at"].max() - HISTORY_NEEDED - pd.Timedelta(FREQ)
            old = status[status["fetched_at"] < cutoff]
            latest_old = old.sort_values("fetched_at").groupby("station_id").tail(1)
            recent = status[status["fetched_at"] >= cutoff]
            self.history = pd.concat([latest_old, recent], ignore_index=True)

    @property
    def data_time(self) -> pd.Timestamp:
        """Time of the latest measurement the forecasts are based on (UTC)."""
        return self.history["fetched_at"].max()

    def is_stale(self, now: datetime | None = None) -> bool:
        now = pd.Timestamp(now or datetime.now(UTC))
        return now - self.data_time > STALE_AFTER

    def forecasts(self) -> pd.DataFrame:
        """One row per station and horizon, recomputed only when new data
        arrives. Includes station name, position and capacity."""
        with self.lock:
            key = self.data_time
            if self.cache is not None and self.cache[0] == key:
                return self.cache[1]
            history = self.history.copy()

        start = key - HISTORY_NEEDED - pd.Timedelta(FREQ)
        grid = to_grid(history, start=start)
        out = self.forecaster.predict(grid, self.stations)
        unchanged = out["horizon"].isin(UNCHANGED_HORIZONS)
        out["bikes"] = np.where(unchanged, out["bikes_now"], out["bikes"])
        out["docks"] = np.where(unchanged, out["docks_now"], out["docks"])
        out["station_id"] = out["station_id"].astype(str)
        out = out.merge(self.stations, on="station_id", how="left")

        with self.lock:
            self.cache = (key, out)
        return out
