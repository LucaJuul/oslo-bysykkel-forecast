import numpy as np
import pandas as pd
import pytest

from ml.features import (
    FEATURES,
    NEIGHBOUR_FEATURES,
    SYSTEM_FEATURES,
    add_features,
    lookup,
    nearest_neighbours,
)
from ml.pairs import make_pairs
from ml.weather import tidy, weather_at

T0 = pd.Timestamp("2026-10-10 12:00", tz="UTC")
STATIONS = pd.DataFrame({"station_id": ["a", "b", "c"], "capacity": [20, 20, 20]})
# Each station's nearest neighbour, as nearest_neighbours would return with k=1.
NEIGHBOURS = {"a": ["b"], "b": ["a"], "c": ["b"]}


def make_grid(bikes: dict, start=T0) -> pd.DataFrame:
    """A grid like ml.grid.to_grid returns. None means a gap slot."""
    rows = [
        {
            "station_id": station,
            "slot": start + pd.Timedelta(minutes=15 * i),
            "num_bikes_available": b,
            "num_docks_available": None if b is None else 20 - b,
            "is_installed": True,
            "is_renting": True,
            "is_returning": True,
            "is_gap": b is None,
        }
        for station, values in bikes.items()
        for i, b in enumerate(values)
    ]
    grid = pd.DataFrame(rows)
    counts = ["num_bikes_available", "num_docks_available"]
    grid[counts] = grid[counts].astype("Int16")
    return grid


def features_for(grid, horizons=None, weather=None):
    pairs = make_pairs(grid, STATIONS, horizons=horizons or {15: 1})
    features = add_features(pairs, grid, weather, neighbours=NEIGHBOURS)
    return features.sort_values(["station_id", "horizon", "slot"]).reset_index(
        drop=True
    )


def test_changes_compare_now_with_the_past():
    f = features_for(make_grid({"a": [1, 3, 6, 10, 15]}))
    assert f["change_15"].tolist()[:5] == pytest.approx(
        [np.nan, 2, 3, 4, 5], nan_ok=True
    )
    assert f["change_30"].tolist()[2:5] == [5, 7, 9]
    assert f["change_60"].tolist()[4] == 14


def test_change_across_a_gap_is_missing():
    f = features_for(make_grid({"a": [1, None, 3, 4]}))
    assert np.isnan(f["change_15"][2])
    assert f["change_15"][3] == 1


def test_lags_do_not_cross_stations():
    f = features_for(make_grid({"a": [1, 2], "b": [10, 12]}))
    assert np.isnan(f.loc[f["station_id"] == "b", "change_15"].iloc[0])


def test_no_feature_changes_when_the_future_changes():
    """The core leakage test: rewrite every value after a cut-off and check that
    the features of all pairs with origin at or before it are identical."""
    rng = np.random.default_rng(0)
    n = 7 * 96 + 20  # a week and a bit, so last_week_change has values
    bikes = {s: rng.integers(0, 20, n).tolist() for s in ["a", "b"]}
    grid = make_grid(bikes)
    cut = T0 + pd.Timedelta(minutes=15 * (n - 10))

    future = grid["slot"] > cut
    changed = grid.copy()
    changed.loc[future, "num_bikes_available"] = 0
    changed.loc[future, "num_docks_available"] = 20

    horizons = {15: 1, 30: 2, 60: 4}
    numeric = [c for c in FEATURES if c != "station"]
    before = features_for(grid, horizons)
    after = features_for(changed, horizons)
    keep = before["slot"] <= cut
    for column in ["last_week_change", *NEIGHBOUR_FEATURES, *SYSTEM_FEATURES]:
        assert before.loc[keep, column].notna().any(), column
    pd.testing.assert_frame_equal(before.loc[keep, numeric], after.loc[keep, numeric])
    # Sanity check: the rewrite did change the targets, so the test can fail.
    assert not before["target_bikes"].equals(after["target_bikes"])


def test_lookup_refuses_times_after_the_origin():
    values = pd.Series(
        [1.0],
        index=pd.MultiIndex.from_tuples([("a", T0)], names=["station_id", "slot"]),
    )
    origin = pd.Series([T0 - pd.Timedelta("15min")])
    with pytest.raises(AssertionError):
        lookup(values, pd.Series(["a"]), pd.Series([T0]), origin)


def test_last_week_change_uses_same_interval_one_week_earlier():
    n = 7 * 96 + 4
    values = [5] * n
    values[1] = 9  # one week before slot 7*96+1 (the target of origin 7*96)
    f = features_for(make_grid({"a": values}))
    row = f[f["slot"] == T0 + pd.Timedelta(days=7)].iloc[0]
    assert row["last_week_change"] == 4


def test_time_features_use_local_time_and_are_cyclic():
    # 23:45 and 00:00 Oslo time are neighbours on the circle.
    grid = make_grid({"a": [1, 1]}, start=pd.Timestamp("2026-10-10 21:45", tz="UTC"))
    f = features_for(grid)
    assert f["slot_of_day"].tolist()[:2] == [95, 0]
    distance = np.hypot(
        f["slot_sin"][0] - f["slot_sin"][1], f["slot_cos"][0] - f["slot_cos"][1]
    )
    assert distance < 0.1


def test_station_categories_can_be_fixed_for_prediction():
    grid = make_grid({"b": [1, 2]})
    pairs = make_pairs(grid, STATIONS, horizons={15: 1})
    f = add_features(pairs, grid, stations=["a", "b", "c"])
    assert list(f["station"].cat.categories) == ["a", "b", "c"]
    assert f["station"].cat.codes.tolist()[0] == 1


RAW_WEATHER = pd.DataFrame(
    {
        "reference_time": ["2026-10-10T10:00:00.000Z", "2026-10-10T11:00:00.000Z"] * 2,
        "element": ["air_temperature"] * 2 + ["sum(precipitation_amount PT1H)"] * 2,
        "value": [8.0, 9.0, 0.0, 2.5],
    }
)


def test_weather_is_only_known_an_hour_after_its_reference_time():
    weather = tidy(RAW_WEATHER)
    origins = pd.Series(
        pd.to_datetime(["2026-10-10 11:45", "2026-10-10 12:00"], utc=True)
    )
    w = weather_at(origins, weather)
    # At 11:45 the 11:00 observation is not available yet, so 10:00 is used.
    assert w["temperature"].tolist() == [8.0, 9.0]
    assert w["precipitation"].tolist() == [0.0, 2.5]


def test_old_weather_counts_as_missing():
    weather = tidy(RAW_WEATHER)
    w = weather_at(pd.Series([pd.Timestamp("2026-10-10 20:00", tz="UTC")]), weather)
    assert w.isna().all(axis=None)


def test_weather_features_come_from_the_origin():
    weather = tidy(RAW_WEATHER)
    grid = make_grid(
        {"a": [1, 2, 3, 4, 5]}, start=pd.Timestamp("2026-10-10 11:30", tz="UTC")
    )
    f = features_for(grid, horizons={60: 4}, weather=weather)
    # Origin 11:30: only 10:00 is known, even though the target (12:30) is later.
    assert f["temperature"].iloc[0] == 8.0


def test_nearest_neighbours_by_distance():
    stations = pd.DataFrame(
        {
            "station_id": [1, 2, 3, 4],
            # 2 is ~110 m north of 1, 3 is ~1.1 km north, 4 is ~5.5 km east.
            "lat": [59.910, 59.911, 59.920, 59.910],
            "lon": [10.750, 10.750, 10.750, 10.850],
        }
    )
    near = nearest_neighbours(stations, k=2)
    assert near["1"] == ["2", "3"]
    assert near["3"] == ["2", "1"]
    assert "1" not in near["1"]


def test_neighbour_change_is_the_mean_change_at_the_neighbours():
    grid = make_grid({"a": [5, 5, 5, 5, 5], "b": [1, 3, 6, 10, 15], "c": [9] * 5})
    f = features_for(grid, horizons={15: 1})
    a = f[f["station_id"] == "a"].reset_index(drop=True)
    # a's neighbour is b; a itself never changes.
    assert a["nb_change_15"].tolist()[1:4] == [2, 3, 4]
    assert np.isnan(a["nb_change_60"][3])
    assert a["nb_change_60"][4] == 14


def test_neighbour_in_a_gap_is_skipped():
    neighbours = {"a": ["b", "c"], "b": ["a"], "c": ["a"]}
    grid = make_grid({"a": [5, 5, 5], "b": [1, None, 4], "c": [2, 3, 5]})
    pairs = make_pairs(grid, STATIONS, horizons={15: 1})
    f = add_features(pairs, grid, neighbours=neighbours)
    a = f[f["station_id"] == "a"].sort_values("slot")
    # Slot 1: b is in a gap, so only c counts (+1). Slot 2: b's change crosses
    # the gap and is unknown, so only c counts (+2).
    assert a["nb_change_15"].tolist()[1:] == [1, 2]


def test_system_features_sum_over_all_stations():
    grid = make_grid({"a": [5, 7, 4, 4, 6], "b": [1, 1, 3, 2, 2], "c": [9] * 5})
    f = features_for(grid, horizons={15: 1})
    a = f[f["station_id"] == "a"].reset_index(drop=True)
    # Bikes moved per slot: |2|+0, |-3|+|2|, 0+|-1|, |2|+0 -> 2, 5, 1, 2
    assert a["sys_activity_15"].tolist()[1:5] == [2, 5, 1, 2]
    assert np.isnan(a["sys_activity_60"][3])
    assert a["sys_activity_60"][4] == 10
    # Net change in docked bikes over 60 min: (6-5) + (2-1) + 0 = 2
    assert a["sys_change_60"][4] == 2
    # The same city-wide value at every station.
    assert f.groupby("slot")["sys_activity_15"].nunique().max() == 1
