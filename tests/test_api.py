from datetime import UTC, datetime

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import create_app, distance_m, seconds_to_next_poll
from api.service import ForecastService
from api.sources import live_status
from ml.features import HISTORY_NEEDED, add_features
from ml.model import PARAMS, save, train
from ml.pairs import make_pairs
from ml.predict import Forecaster
from ml.probability import CLS_PARAMS, train_classifier
from tests.test_predict import STATIONS, make_grid
from tests.test_transform import payload, status_entry

# Two stations about 1.1 km apart in central Oslo.
STATION_INFO = pd.DataFrame(
    {
        "station_id": ["a", "b"],
        "name": ["Alpha", "Beta"],
        "lat": [59.910, 59.920],
        "lon": [10.750, 10.750],
        "capacity": [20, 20],
    }
)


def to_status(grid: pd.DataFrame) -> pd.DataFrame:
    """Grid slots as collector rows, measured one minute into each slot."""
    status = grid.drop(columns="is_gap").copy()
    status["fetched_at"] = status["slot"] + pd.Timedelta("1min")
    status["last_reported"] = status["fetched_at"]
    status["num_ebikes_available"] = 0
    return status.drop(columns="slot")


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    grid = make_grid()
    pairs = make_pairs(grid, STATIONS)
    features = add_features(pairs[pairs["valid"]], grid, stations=["a", "b"])
    booster, _ = train(
        features, None, PARAMS | {"num_leaves": 4, "min_data_in_leaf": 5}, rounds=5
    )
    small = CLS_PARAMS | {"num_leaves": 4, "min_data_in_leaf": 5}
    classifiers = {
        t: train_classifier(features, None, t, small, rounds=5)[0]
        for t in ["bike", "dock"]
    }
    path = tmp_path_factory.mktemp("model")
    info = {"trained_on": "test data", "neighbours": {"a": ["b"], "b": ["a"]}}
    save(booster, ["a", "b"], info, path, classifiers=classifiers)
    return path


@pytest.fixture
def service(model_dir):
    return ForecastService(Forecaster(model_dir), STATION_INFO, to_status(make_grid()))


@pytest.fixture
def client(service):
    with TestClient(create_app(lambda: service, poll=False)) as c:
        yield c


def test_health(client, service):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["stations"] == 2
    assert body["model_trained_on"] == "test data"
    assert pd.Timestamp(body["data_time"]) == service.data_time


def test_all_stations_with_three_horizons(client):
    body = client.get("/forecasts").json()
    assert [s["station_id"] for s in body["stations"]] == ["a", "b"]
    for station in body["stations"]:
        assert [f["minutes"] for f in station["forecasts"]] == [15, 30, 60]
        assert station["name"] in {"Alpha", "Beta"}
        for f in station["forecasts"]:
            assert 0 <= f["p_bike"] <= 1 and 0 <= f["p_dock"] <= 1


def test_expected_number_at_15_minutes_is_the_current_value(client):
    for station in client.get("/forecasts").json()["stations"]:
        at_15 = station["forecasts"][0]
        assert at_15["bikes"] == station["bikes_now"]
        assert at_15["docks"] == station["docks_now"]


def test_nearest_stations_first(client):
    body = client.get("/forecasts", params={"lat": 59.921, "lon": 10.75}).json()
    assert [s["station_id"] for s in body["stations"]] == ["b", "a"]
    assert body["stations"][0]["distance_m"] < body["stations"][1]["distance_m"]
    one = client.get("/forecasts", params={"lat": 59.921, "lon": 10.75, "limit": 1})
    assert [s["station_id"] for s in one.json()["stations"]] == ["b"]


def test_position_needs_both_coordinates(client):
    assert client.get("/forecasts", params={"lat": 59.9}).status_code == 422
    assert client.get("/forecasts", params={"lat": 200, "lon": 10}).status_code == 422


def test_one_station(client):
    body = client.get("/forecasts/a").json()
    assert body["station_id"] == "a" and len(body["forecasts"]) == 3
    assert client.get("/forecasts/nope").status_code == 404


def test_new_data_gives_new_forecasts(service):
    first = service.forecasts()
    assert service.forecasts() is first  # cached while no new data arrives
    last = (
        to_status(make_grid()).sort_values("fetched_at").groupby("station_id").tail(1)
    )
    newer = last.assign(
        fetched_at=last["fetched_at"] + pd.Timedelta("15min"),
        num_bikes_available=0,
        num_docks_available=20,
    )
    service.add_status(newer)
    second = service.forecasts()
    assert second is not first
    assert (second["bikes_now"] == 0).all()


def test_old_rows_are_dropped_but_the_latest_per_station_is_kept(service):
    september = pd.DataFrame(
        {
            "station_id": ["a", "c"],
            "fetched_at": pd.to_datetime(["2026-09-01", "2026-09-03"], utc=True),
        }
    )
    service.history = pd.concat([september, service.history], ignore_index=True)
    service.add_status(service.history.iloc[:0])
    h = service.history
    cutoff = h["fetched_at"].max() - HISTORY_NEEDED - pd.Timedelta("15min")
    old = h[h["fetched_at"] < cutoff]
    # One row per station before the window: the latest one. a and b have
    # newer rows than September (the test grid covers 8 days); c has only one.
    assert old["station_id"].value_counts().to_dict() == {"a": 1, "b": 1, "c": 1}
    assert pd.Timestamp("2026-09-01", tz="UTC") not in set(h["fetched_at"])
    assert pd.Timestamp("2026-09-03", tz="UTC") in set(h["fetched_at"])


def test_stale_after_35_minutes(service):
    assert not service.is_stale(service.data_time + pd.Timedelta("30min"))
    assert service.is_stale(service.data_time + pd.Timedelta("40min"))


def test_next_poll_is_one_minute_into_the_next_slot():
    assert seconds_to_next_poll(datetime(2026, 11, 1, 12, 0, 30, tzinfo=UTC)) == 30
    assert seconds_to_next_poll(datetime(2026, 11, 1, 12, 5, tzinfo=UTC)) == 11 * 60


def test_distance_in_metres():
    # 0.01 degrees of latitude is about 1.1 km.
    assert distance_m([59.92], [10.75], 59.91, 10.75)[0] == pytest.approx(
        1112, rel=0.01
    )


def test_live_status_reads_the_gbfs_feed():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/station_status.json")
        return httpx.Response(200, json=payload(status_entry("7", bikes=3, docks=9)))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        rows = live_status(client)
    assert rows.loc[0, "station_id"] == "7"
    assert rows.loc[0, "num_bikes_available"] == 3
    assert str(rows["fetched_at"].dt.tz) == "UTC"
