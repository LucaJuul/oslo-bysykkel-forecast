import numpy as np
import pandas as pd
import pytest

from ml.baselines import (
    MeanChange,
    SameTimeLastWeek,
    Unchanged,
    derive_docks,
    one_week_before,
)
from ml.pairs import local_time


def utc(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def make_pairs(rows: list[dict]) -> pd.DataFrame:
    """Rows with station_id, slot, bikes and optionally the other columns."""
    defaults = {"docks": 5.0, "capacity": 10.0, "horizon": 15, "target_bikes": 0.0}
    pairs = pd.DataFrame([defaults | r for r in rows])
    pairs["target_slot"] = pairs["slot"] + pd.to_timedelta(pairs["horizon"], "min")
    return pairs.join(local_time(pairs["slot"]))


def test_unchanged_predicts_current_value():
    pairs = make_pairs(
        [{"station_id": "a", "slot": utc("2026-10-10 12:00"), "bikes": 3}]
    )
    assert Unchanged().fit(pairs).predict(pairs).tolist() == [3]


def test_one_week_before_keeps_local_clock_across_dst():
    # Monday 26 Oct 08:00 Oslo (winter time) = 07:00 UTC.
    # Monday 19 Oct 08:00 Oslo (summer time) = 06:00 UTC, not 07:00.
    slots = pd.Series([utc("2026-10-26 07:00")])
    assert one_week_before(slots).tolist() == [utc("2026-10-19 06:00")]


def test_same_time_last_week_and_fallback():
    grid = pd.DataFrame(
        {
            "station_id": ["a"],
            "slot": [utc("2026-10-03 12:15")],
            "num_bikes_available": pd.array([8], dtype="Int16"),
        }
    )
    pairs = make_pairs(
        [
            {"station_id": "a", "slot": utc("2026-10-10 12:00"), "bikes": 3},
            {"station_id": "a", "slot": utc("2026-10-10 13:00"), "bikes": 4},
        ]
    )
    model = SameTimeLastWeek(grid)
    # First pair: target 12:15, a week earlier there were 8 bikes.
    # Second pair: no value a week before, so fall back to the current 4.
    assert model.predict(pairs).tolist() == [8, 4]
    assert model.fallback_share == 0.5


def test_mean_change_uses_training_average_per_key():
    monday_8 = utc("2026-10-12 06:00")  # 08:00 Oslo
    train = make_pairs(
        [
            {"station_id": "a", "slot": monday_8, "bikes": 5, "target_bikes": 3},
            {"station_id": "a", "slot": monday_8, "bikes": 5, "target_bikes": 4},
        ]
    )
    test = make_pairs(
        [
            # Same station, weekday and time a week later: 6 - 1.5
            {"station_id": "a", "slot": utc("2026-10-19 06:00"), "bikes": 6},
            # Other time of day: no profile, so unchanged
            {"station_id": "a", "slot": utc("2026-10-19 10:00"), "bikes": 6},
        ]
    )
    assert MeanChange().fit(train).predict(test).tolist() == [4.5, 6]


def test_predictions_are_clipped_to_capacity_and_zero():
    t = utc("2026-10-12 06:00")
    train = make_pairs(
        [
            {"station_id": "a", "slot": t, "bikes": 1, "target_bikes": 9},
            {"station_id": "b", "slot": t, "bikes": 9, "target_bikes": 0},
        ]
    )
    test = make_pairs(
        [
            {"station_id": "a", "slot": t, "bikes": 5},
            {"station_id": "b", "slot": t, "bikes": 5},
        ]
    )
    assert MeanChange().fit(train).predict(test).tolist() == [10, 0]


def test_clip_allows_current_value_above_capacity():
    pairs = make_pairs(
        [{"station_id": "a", "slot": utc("2026-10-10 12:00"), "bikes": 12}]
    )
    assert Unchanged().predict(pairs).tolist() == [12]


def test_docks_follow_from_current_total():
    # 4 bikes + 4 docks now (2 docks out of order at a 10-dock station).
    pairs = make_pairs(
        [{"station_id": "a", "slot": utc("2026-10-10 12:00"), "bikes": 4, "docks": 4}]
    )
    assert derive_docks(pairs, np.array([1.5])).tolist() == pytest.approx([6.5])
