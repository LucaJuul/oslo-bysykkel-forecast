import json

import numpy as np
import pandas as pd
import pytest

from ml.features import add_features
from ml.model import PARAMS, load, predict, save, train
from ml.pairs import make_pairs
from ml.predict import OUTPUT_COLUMNS, Forecaster

T0 = pd.Timestamp("2026-10-10 00:00", tz="UTC")
STATIONS = pd.DataFrame({"station_id": ["a", "b"], "capacity": [20, 20]})
N = 8 * 96  # eight days of 15-minute slots


def make_grid(n=N, seed=0) -> pd.DataFrame:
    """Random bikes (3-17) at two stations, in the shape of ml.grid.to_grid.
    Not a random walk: a walk can stick at 0 or 20 for days and look frozen."""
    rng = np.random.default_rng(seed)
    rows = []
    for station in ["a", "b"]:
        bikes = rng.integers(3, 18, n)
        for i, b in enumerate(bikes):
            rows.append(
                {
                    "station_id": station,
                    "slot": T0 + pd.Timedelta(minutes=15 * i),
                    "num_bikes_available": int(b),
                    "num_docks_available": int(20 - b),
                    "is_installed": True,
                    "is_renting": True,
                    "is_returning": True,
                    "is_gap": False,
                }
            )
    grid = pd.DataFrame(rows)
    counts = ["num_bikes_available", "num_docks_available"]
    grid[counts] = grid[counts].astype("Int16")
    return grid


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    """A tiny model trained on synthetic data and saved like the real one."""
    grid = make_grid()
    pairs = make_pairs(grid, STATIONS)
    features = add_features(pairs[pairs["valid"]], grid, stations=["a", "b"])
    params = PARAMS | {"num_leaves": 4, "min_data_in_leaf": 5}
    booster, _ = train(features, None, params, rounds=5)
    path = tmp_path_factory.mktemp("model")
    save(booster, ["a", "b"], {"trees": 5}, path)
    return path


def test_one_forecast_per_station_and_horizon_from_the_last_slot(model_dir):
    grid = make_grid()
    out = Forecaster(model_dir).predict(grid, STATIONS)
    assert list(out.columns) == OUTPUT_COLUMNS
    assert len(out) == 2 * 3
    assert (out["slot"] == grid["slot"].max()).all()
    assert sorted(out["horizon"].unique()) == [15, 30, 60]
    assert (
        out["target_slot"] - out["slot"] == pd.to_timedelta(out["horizon"], "min")
    ).all()


def test_forecasts_are_clipped_and_docks_follow_from_bikes(model_dir):
    out = Forecaster(model_dir).predict(make_grid(), STATIONS)
    assert out["bikes"].between(0, 20).all()
    total_now = out["bikes_now"] + out["docks_now"]
    np.testing.assert_allclose(out["docks"], total_now - out["bikes"])


def test_same_forecast_as_the_training_path(model_dir):
    """The API path must give exactly what evaluation scored for that origin."""
    grid = make_grid()
    out = Forecaster(model_dir).predict(grid, STATIONS)

    booster, meta = load(model_dir)
    pairs = make_pairs(grid, STATIONS)
    last = pairs[pairs["slot"] == grid["slot"].max()]
    expected = add_features(last, grid, stations=meta["stations"])
    expected["bikes_pred"] = predict(booster, expected)
    merged = out.merge(expected, on=["station_id", "horizon"])
    np.testing.assert_allclose(merged["bikes_x"], merged["bikes_pred"])


def test_a_short_history_still_gives_forecasts(model_dir, caplog):
    grid = make_grid()
    recent = grid[grid["slot"] > grid["slot"].max() - pd.Timedelta("2h")]
    out = Forecaster(model_dir).predict(recent, STATIONS)
    assert len(out) == 6
    assert "history" in caplog.text


def test_inactive_station_is_marked_unreliable(model_dir):
    grid = make_grid()
    last = grid["slot"] == grid["slot"].max()
    grid.loc[last & (grid["station_id"] == "b"), "is_renting"] = False
    out = Forecaster(model_dir).predict(grid, STATIONS)
    assert out.groupby("station_id")["reliable"].all().to_dict() == {
        "a": True,
        "b": False,
    }


def test_station_in_a_gap_now_is_left_out(model_dir):
    grid = make_grid()
    last = grid["slot"] == grid["slot"].max()
    grid.loc[last & (grid["station_id"] == "b"), "num_bikes_available"] = pd.NA
    out = Forecaster(model_dir).predict(grid, STATIONS)
    assert set(out["station_id"]) == {"a"}


def test_loading_refuses_a_model_with_other_features(model_dir, tmp_path):
    for name in ["lightgbm.txt", "metadata.json"]:
        (tmp_path / name).write_bytes((model_dir / name).read_bytes())
    meta = json.loads((tmp_path / "metadata.json").read_text())
    meta["features"] = meta["features"][:-1]
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        load(tmp_path)
