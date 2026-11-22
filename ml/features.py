"""Features for the Phase 4 model, computed for (origin, target) pairs.

Rule: a feature may only use values known at the origin – the current slot or
earlier. Every past value is looked up by (station, time), and the lookup times
are checked to be at or before the origin, so a bug cannot read the future.

The same function serves training (pairs from ml.pairs) and prediction in the
API (a frame with station_id, slot, bikes, docks, capacity and horizon).
"""

import numpy as np
import pandas as pd

from ml.baselines import one_week_before
from ml.grid import FREQ
from ml.pairs import local_time
from ml.weather import WEATHER_COLUMNS, weather_at

LAGS = {15: 1, 30: 2, 60: 4}  # minutes back -> grid steps

FEATURES = [
    "station",
    "bikes",
    "capacity",
    "total_now",
    *[f"change_{m}" for m in LAGS],
    "weekday_sin",
    "weekday_cos",
    "slot_sin",
    "slot_cos",
    "last_week_change",
    *WEATHER_COLUMNS,
]
CATEGORICAL = ["station"]

# How far back the features look: the longest lag, and one week plus the
# longest horizon for last_week_change (the API must keep this much history).
HISTORY_NEEDED = pd.Timedelta(days=7) + pd.Timedelta(minutes=60)


def bike_values(grid: pd.DataFrame) -> pd.Series:
    """Bikes per (station_id, slot) as float, NaN in gaps."""
    return grid.set_index(["station_id", "slot"])["num_bikes_available"].astype(
        "float64"
    )


def lookup(values: pd.Series, station: pd.Series, when: pd.Series, origin: pd.Series):
    """Bikes at `station` at time `when`, refusing any time after the origin."""
    known = when.notna()
    assert (when[known] <= origin[known]).all(), "feature reads the future"
    index = pd.MultiIndex.from_arrays([station, when])
    return values.reindex(index).to_numpy()


def cyclic(value: pd.Series, period: int) -> tuple[np.ndarray, np.ndarray]:
    """Sine and cosine, so the last value of a cycle lies next to the first."""
    angle = 2 * np.pi * value.to_numpy(dtype="float64") / period
    return np.sin(angle), np.cos(angle)


def add_features(
    pairs: pd.DataFrame,
    grid: pd.DataFrame,
    weather: pd.DataFrame | None = None,
    stations: list | None = None,
) -> pd.DataFrame:
    """Return `pairs` with the FEATURES columns added.

    `stations` fixes the category list for station IDs; pass the list saved
    with the model at prediction time so the codes match training. Unknown
    stations become missing. Without `weather`, the weather columns are NaN.
    """
    p = pairs.copy()
    values = bike_values(grid)
    station, origin = p["station_id"], p["slot"]
    step = pd.Timedelta(FREQ)

    p["total_now"] = p["bikes"] + p["docks"]
    for minutes, steps in LAGS.items():
        before = lookup(values, station, origin - steps * step, origin)
        p[f"change_{minutes}"] = p["bikes"].to_numpy() - before

    # How the number changed over the same local clock interval one week ago.
    target = origin + pd.to_timedelta(p["horizon"], unit="min")
    then_target = lookup(values, station, one_week_before(target), origin)
    then_origin = lookup(values, station, one_week_before(origin), origin)
    p["last_week_change"] = then_target - then_origin

    if "weekday" not in p or "slot_of_day" not in p:
        p = p.join(local_time(p["slot"]))
    p["weekday_sin"], p["weekday_cos"] = cyclic(p["weekday"], 7)
    p["slot_sin"], p["slot_cos"] = cyclic(p["slot_of_day"], 96)

    categories = sorted(p["station_id"].unique()) if stations is None else stations
    p["station"] = pd.Categorical(p["station_id"], categories=categories)

    if weather is None:
        for column in WEATHER_COLUMNS:
            p[column] = np.nan
    else:
        p[WEATHER_COLUMNS] = weather_at(p["slot"], weather)
    return p
