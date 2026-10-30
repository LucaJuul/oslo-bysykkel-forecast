import pandas as pd

from ml.config import SPLIT
from ml.pairs import frozen_mask, make_pairs, split

T0 = pd.Timestamp("2026-10-10 12:00", tz="UTC")
STATIONS = pd.DataFrame({"station_id": ["a", "b"], "capacity": [10, 10]})


def make_grid(bikes_by_station: dict, start=T0, renting=None) -> pd.DataFrame:
    """A grid like ml.grid.to_grid returns. None means a gap slot."""
    rows = []
    for station, values in bikes_by_station.items():
        for i, bikes in enumerate(values):
            rows.append(
                {
                    "station_id": station,
                    "slot": start + pd.Timedelta(minutes=15 * i),
                    "num_bikes_available": bikes,
                    "num_docks_available": None if bikes is None else 10 - bikes,
                    "is_installed": True,
                    "is_renting": True if renting is None else renting[i],
                    "is_returning": True,
                    "is_gap": bikes is None,
                }
            )
    grid = pd.DataFrame(rows)
    counts = ["num_bikes_available", "num_docks_available"]
    grid[counts] = grid[counts].astype("Int16")
    return grid


def pairs_for(grid, horizon=15):
    pairs = make_pairs(grid, STATIONS, horizons={horizon: horizon // 15})
    return pairs.sort_values(["station_id", "slot"]).reset_index(drop=True)


def test_target_is_value_horizon_steps_later():
    p = pairs_for(make_grid({"a": [1, 2, 3, 4, 5]}), horizon=30)
    assert p["target_bikes"].tolist()[:3] == [3, 4, 5]
    assert p["valid"].tolist() == [True, True, True, False, False]


def test_targets_do_not_cross_stations():
    p = pairs_for(make_grid({"a": [1, 2], "b": [7, 8]}))
    assert p.loc[p["station_id"] == "a", "target_bikes"].tolist()[0] == 2
    assert pd.isna(p.loc[p["station_id"] == "a", "target_bikes"].tolist()[1])


def test_pairs_touching_a_gap_are_invalid():
    p = pairs_for(make_grid({"a": [1, None, 3, 4]}))
    # origin 1 -> gap, origin gap -> 3, origin 3 -> 4
    assert p["valid"].tolist()[:3] == [False, False, True]


def test_inactive_station_at_origin_or_target_is_invalid():
    grid = make_grid({"a": [1, 2, 3, 4]}, renting=[True, False, True, True])
    p = pairs_for(grid)
    assert p["valid"].tolist()[:3] == [False, False, True]


def test_frozen_after_threshold_only_looks_back():
    grid = make_grid({"a": [5, 5, 5, 5, 6, 6]})
    frozen = frozen_mask(grid, frozen_after=pd.Timedelta("45min"))
    assert frozen.tolist() == [False, False, False, True, False, False]


def test_gap_does_not_count_as_change():
    grid = make_grid({"a": [5, None, None, 5]})
    frozen = frozen_mask(grid, frozen_after=pd.Timedelta("45min"))
    assert frozen.tolist() == [False, False, False, True]


def test_training_pairs_never_have_targets_in_test_period():
    # Four slots ending exactly at SPLIT, then two after it.
    start = SPLIT.tz_convert("UTC") - pd.Timedelta("1h")
    grid = make_grid({"a": [1, 2, 3, 4, 5, 6]}, start=start)
    train, test = split(pairs_for(grid, horizon=30))
    assert (train["target_slot"] < SPLIT).all()
    assert len(train) == 2  # origins at -60 and -45 min
    assert (test["slot"] >= SPLIT).all()
