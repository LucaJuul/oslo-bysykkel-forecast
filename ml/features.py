"""Features for the Phase 4 model, computed for (origin, target) pairs.

Rule: a feature may only use values known at the origin – the current slot or
earlier. Every past value is looked up by (station, time), and the lookup times
are checked to be at or before the origin, so a bug cannot read the future.

The same function serves training (pairs from ml.pairs) and prediction in the
API (a frame with station_id, slot, bikes, docks, capacity and horizon).

Neighbour and system features are first computed for every (slot, station) in
a wide table (one column per station) using only that slot and earlier slots,
then looked up at the origin.
"""

import numpy as np
import pandas as pd

from ml.baselines import one_week_before
from ml.grid import FREQ
from ml.pairs import local_time
from ml.weather import WEATHER_COLUMNS, weather_at

LAGS = {15: 1, 30: 2, 60: 4}  # minutes back -> grid steps
NEIGHBOURS = 5  # nearest stations used for the neighbour features
EARTH_RADIUS_M = 6_371_000

# Mean change in bikes at the 5 nearest stations over the last 15/30/60 min.
NEIGHBOUR_FEATURES = [f"nb_change_{m}" for m in LAGS]
# The whole city: how many bikes moved in the last 15 and 60 minutes (sum of
# absolute changes over all stations), and the net change in docked bikes over
# the last 60 minutes (negative = more bikes out on trips).
SYSTEM_FEATURES = ["sys_activity_15", "sys_activity_60", "sys_change_60"]

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
    *NEIGHBOUR_FEATURES,
    *SYSTEM_FEATURES,
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


def nearest_neighbours(stations: pd.DataFrame, k: int = NEIGHBOURS) -> dict:
    """The k nearest other stations for each station, by straight-line distance.

    Oslo is small enough for a flat-earth approximation (error well under 1 %).
    Station IDs are returned as str, so the mapping survives a JSON round trip.
    """
    ids = stations["station_id"].astype(str).to_numpy()
    lat = np.radians(stations["lat"].to_numpy(dtype="float64"))
    lon = np.radians(stations["lon"].to_numpy(dtype="float64"))
    x = lon * np.cos(lat.mean()) * EARTH_RADIUS_M
    y = lat * EARTH_RADIUS_M
    distance = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    np.fill_diagonal(distance, np.inf)
    nearest = np.argsort(distance, axis=1)[:, :k]
    return {ids[i]: [ids[j] for j in row] for i, row in enumerate(nearest)}


def wide_bikes(grid: pd.DataFrame) -> pd.DataFrame:
    """Bikes with one row per slot and one column per station (as str),
    every 15-minute slot present."""
    wide = grid.pivot(index="slot", columns="station_id", values="num_bikes_available")
    wide = wide.astype("float64").sort_index()
    wide.columns = wide.columns.astype(str)
    slots = pd.date_range(wide.index.min(), wide.index.max(), freq=FREQ)
    return wide.reindex(slots)


def neighbour_changes(wide: pd.DataFrame, neighbours: dict) -> pd.DataFrame:
    """Mean change over the last 15/30/60 min at each station's neighbours.

    Long format, indexed by (slot, station_id). Neighbours with a missing value
    are skipped; if all are missing the feature is missing.
    """
    columns = list(wide.columns)
    position = {s: i for i, s in enumerate(columns)}
    weights = np.zeros((len(columns), len(columns)))
    for station, near in neighbours.items():
        for other in near:
            if station in position and other in position:
                weights[position[station], position[other]] = 1.0

    out = {}
    for minutes, steps in LAGS.items():
        change = (wide - wide.shift(steps)).to_numpy()  # this slot and earlier only
        known = ~np.isnan(change)
        total = np.where(known, change, 0.0) @ weights.T
        count = known.astype("float64") @ weights.T
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        out[f"nb_change_{minutes}"] = pd.DataFrame(
            mean, index=wide.index, columns=columns
        ).stack()
    return pd.DataFrame(out)


def system_features(wide: pd.DataFrame) -> pd.DataFrame:
    """City-wide activity per slot, from this slot and earlier ones only."""
    moved = (wide - wide.shift(1)).abs().sum(axis=1, min_count=1)
    return pd.DataFrame(
        {
            "sys_activity_15": moved,
            "sys_activity_60": moved.rolling(4, min_periods=4).sum(),
            "sys_change_60": (wide - wide.shift(4)).sum(axis=1, min_count=1),
        }
    )


def cyclic(value: pd.Series, period: int) -> tuple[np.ndarray, np.ndarray]:
    """Sine and cosine, so the last value of a cycle lies next to the first."""
    angle = 2 * np.pi * value.to_numpy(dtype="float64") / period
    return np.sin(angle), np.cos(angle)


def add_features(
    pairs: pd.DataFrame,
    grid: pd.DataFrame,
    weather: pd.DataFrame | None = None,
    stations: list | None = None,
    neighbours: dict | None = None,
) -> pd.DataFrame:
    """Return `pairs` with the FEATURES columns added.

    `stations` fixes the category list for station IDs; pass the list saved
    with the model at prediction time so the codes match training. Unknown
    stations become missing. `neighbours` comes from `nearest_neighbours`
    (saved with the model). Without `weather` or `neighbours`, those columns
    are NaN. System features use every station in `grid`, so pass the grid
    for the whole city, not a subset.
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

    wide = wide_bikes(grid)
    if neighbours is None:
        for column in NEIGHBOUR_FEATURES:
            p[column] = np.nan
    else:
        near = neighbour_changes(wide, neighbours)
        at_origin = pd.MultiIndex.from_arrays([origin, station.astype(str)])
        for column in NEIGHBOUR_FEATURES:
            p[column] = near[column].reindex(at_origin).to_numpy()
    p[SYSTEM_FEATURES] = system_features(wide).reindex(origin).to_numpy()

    if weather is None:
        for column in WEATHER_COLUMNS:
            p[column] = np.nan
    else:
        p[WEATHER_COLUMNS] = weather_at(p["slot"], weather)
    return p
