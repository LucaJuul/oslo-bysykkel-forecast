"""Naive baseline models. The ML model in Phase 4 has to beat these.

Every model has the same interface:
    model.fit(train_pairs)        -> learns from training pairs only
    model.predict(pairs)          -> predicted bikes at the target, one per pair
Predictions are clipped to [0, capacity]. Docks are derived from bikes with
`derive_docks`, so a model only predicts bikes.
"""

import numpy as np
import pandas as pd

from ml.grid import OSLO


def clip(predicted, pairs: pd.DataFrame, current: str = "bikes") -> np.ndarray:
    """Clip to [0, capacity]. If the current value is already above capacity
    (it happens at some stations), allow up to the current value, so
    "unchanged" is never clipped away from the truth."""
    upper = np.maximum(pairs["capacity"].to_numpy(), pairs[current].to_numpy())
    return np.clip(np.asarray(predicted, dtype="float64"), 0, upper)


def derive_docks(pairs: pd.DataFrame, predicted_bikes) -> np.ndarray:
    """Docks = (bikes + docks now) - predicted bikes.

    Uses the station's current total, not the nominal capacity, so it stays
    right when some docks are out of order."""
    total_now = (pairs["bikes"] + pairs["docks"]).to_numpy()
    return clip(total_now - np.asarray(predicted_bikes), pairs, current="docks")


class Unchanged:
    """The number of bikes stays as it is now."""

    name = "unchanged"

    def fit(self, train: pd.DataFrame) -> "Unchanged":
        return self

    def predict(self, pairs: pd.DataFrame) -> np.ndarray:
        return clip(pairs["bikes"], pairs)


def one_week_before(slots: pd.Series) -> pd.Series:
    """Same local clock time 7 days earlier, in UTC.

    Uses Oslo wall-clock time: after daylight saving ends, Monday 08:00 must
    map to the previous Monday 08:00, which is a different UTC hour. Times that
    do not exist or are ambiguous in local time become NaT.
    """
    local = slots.dt.tz_convert(OSLO).dt.tz_localize(None) - pd.Timedelta(days=7)
    local = local.dt.tz_localize(OSLO, ambiguous="NaT", nonexistent="NaT")
    return local.dt.tz_convert("UTC")


class SameTimeLastWeek:
    """Bikes at the target time = bikes at the same local time one week earlier.

    Needs the grid to look up past values. Falls back to "unchanged" when last
    week's value is missing (gap, or before the data starts), so it can be
    evaluated on exactly the same pairs as the other models.
    """

    name = "same_time_last_week"

    def __init__(self, grid: pd.DataFrame):
        self.values = grid.set_index(["station_id", "slot"])[
            "num_bikes_available"
        ].astype("float64")
        self.fallback_share = float("nan")

    def fit(self, train: pd.DataFrame) -> "SameTimeLastWeek":
        return self

    def predict(self, pairs: pd.DataFrame) -> np.ndarray:
        lookup = one_week_before(pairs["target_slot"])
        # Guard against leakage: we may only use values known at the origin.
        assert (lookup.dropna() < pairs.loc[lookup.notna(), "slot"]).all()
        index = pd.MultiIndex.from_arrays([pairs["station_id"], lookup])
        last_week = self.values.reindex(index).to_numpy()
        missing = np.isnan(last_week)
        self.fallback_share = float(missing.mean()) if len(missing) else 0.0
        return clip(np.where(missing, pairs["bikes"], last_week), pairs)


class MeanChange:
    """Current bikes + the average change seen in training for the same
    station, weekday, time of day (local) and horizon. Unknown keys add 0."""

    name = "mean_change_profile"
    keys = ["station_id", "weekday", "slot_of_day", "horizon"]

    def fit(self, train: pd.DataFrame) -> "MeanChange":
        change = train["target_bikes"] - train["bikes"]
        self.profile = change.groupby([train[k] for k in self.keys]).mean()
        return self

    def predict(self, pairs: pd.DataFrame) -> np.ndarray:
        index = pd.MultiIndex.from_frame(pairs[self.keys])
        change = self.profile.reindex(index).fillna(0).to_numpy()
        return clip(pairs["bikes"].to_numpy() + change, pairs)
