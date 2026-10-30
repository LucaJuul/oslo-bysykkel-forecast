"""Turn the 15-minute grid into (origin, target) pairs for training and evaluation.

One pair = "at station s and time t (the origin) we know the current status;
what is the number of bikes at t + horizon (the target)?". Every model is
trained and evaluated on the same pairs, so their errors are comparable.
"""

import pandas as pd

from ml.config import FROZEN_AFTER, HORIZONS, SPLIT, TEST_END
from ml.grid import FREQ, OSLO

ACTIVE_FLAGS = ["is_installed", "is_renting", "is_returning"]


def local_time(slots: pd.Series) -> pd.DataFrame:
    """Weekday (0 = Monday), hour and 15-minute slot of day (0-95), Oslo time."""
    local = slots.dt.tz_convert(OSLO)
    return pd.DataFrame(
        {
            "weekday": local.dt.day_of_week,
            "hour": local.dt.hour,
            "slot_of_day": local.dt.hour * 4 + local.dt.minute // 15,
        },
        index=slots.index,
    )


def frozen_mask(grid: pd.DataFrame, frozen_after: pd.Timedelta = FROZEN_AFTER):
    """True where a station's bikes and docks have not changed for frozen_after.

    Looks only backwards in time. Gap slots (missing values) never count as a
    change, so a gap does not reset the clock.
    """
    station = grid["station_id"]
    values = grid[["num_bikes_available", "num_docks_available"]].astype("float64")
    previous = values.groupby(station).ffill().groupby(station).shift()
    changed = values.notna().all(axis=1) & (values != previous).any(axis=1)
    last_change = grid["slot"].where(changed).groupby(station).ffill()
    return (grid["slot"] - last_change) >= frozen_after


def make_pairs(
    grid: pd.DataFrame, stations: pd.DataFrame, horizons: dict = HORIZONS
) -> pd.DataFrame:
    """All pairs for all horizons, with a `valid` column and reasons when not.

    Valid means: origin and target both have values (not in a gap), the station
    is active (installed, renting, returning) at both, it is not frozen at the
    origin, and its capacity is known.
    """
    g = grid.sort_values(["station_id", "slot"]).reset_index(drop=True)
    g = g.merge(stations[["station_id", "capacity"]], on="station_id", how="left")
    g["bikes"] = g["num_bikes_available"].astype("float64")
    g["docks"] = g["num_docks_available"].astype("float64")
    g["active"] = g[ACTIVE_FLAGS].astype("boolean").fillna(False).all(axis=1)
    g["frozen"] = frozen_mask(g).to_numpy()

    by_station = g.groupby("station_id")
    columns = ["station_id", "slot", "bikes", "docks", "capacity", "active", "frozen"]
    frames = []
    for minutes, steps in horizons.items():
        p = g[columns].copy()
        p["horizon"] = minutes
        # The grid has every slot for every station, so shifting by `steps`
        # rows within a station is the same as moving `steps` x 15 minutes.
        p["target_slot"] = p["slot"] + steps * pd.Timedelta(FREQ)
        p["target_bikes"] = by_station["bikes"].shift(-steps)
        p["target_docks"] = by_station["docks"].shift(-steps)
        p["target_active"] = by_station["active"].shift(-steps, fill_value=False)
        frames.append(p)
    pairs = pd.concat(frames, ignore_index=True)

    pairs["missing"] = (
        pairs[["bikes", "docks", "target_bikes", "target_docks"]].isna().any(axis=1)
    )
    # Gap slots have unknown flags; they are already counted as missing.
    pairs["inactive"] = ~pairs["missing"] & ~(
        pairs["active"] & pairs["target_active"].astype(bool)
    )
    pairs["no_capacity"] = pairs["capacity"].isna()
    pairs["valid"] = ~(
        pairs["missing"] | pairs["inactive"] | pairs["frozen"] | pairs["no_capacity"]
    )
    return pairs.join(local_time(pairs["slot"]))


def split(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Valid training and test pairs.

    Training pairs need their *target* before SPLIT: a pair at 23:45 the day
    before SPLIT with a 60-minute horizon would otherwise learn from a test value.
    """
    valid = pairs[pairs["valid"]]
    train = valid[valid["target_slot"] < SPLIT]
    test = valid[(valid["slot"] >= SPLIT) & (valid["slot"] < TEST_END)]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def exclusion_summary(pairs: pd.DataFrame) -> pd.Series:
    """How many pairs each rule removes (a pair can fail several rules)."""
    counts = pairs[["missing", "inactive", "frozen", "no_capacity"]].sum()
    counts["valid"] = pairs["valid"].sum()
    counts["total"] = len(pairs)
    return counts


def capacity_check(grid: pd.DataFrame, stations: pd.DataFrame) -> dict:
    """Is bikes + docks always equal to the station's capacity?"""
    g = grid.dropna(subset=["num_bikes_available", "num_docks_available"])
    g = g.merge(stations[["station_id", "capacity"]], on="station_id")
    g = g[g[ACTIVE_FLAGS].astype("boolean").fillna(False).all(axis=1)]
    total = g["num_bikes_available"] + g["num_docks_available"]
    equal = total == g["capacity"]
    by_station = equal.groupby(g["station_id"]).mean()
    sum_per_station = total.groupby(g["station_id"]).nunique()
    return {
        "rows": len(g),
        "share_equal": float(equal.mean()),
        "share_below": float((total < g["capacity"]).mean()),
        "share_above": float((total > g["capacity"]).mean()),
        "stations": len(by_station),
        "stations_always_equal": int((by_station == 1).sum()),
        "stations_constant_sum": int((sum_per_station == 1).sum()),
        "share_bikes_above_capacity": float(
            (g["num_bikes_available"] > g["capacity"]).mean()
        ),
    }
