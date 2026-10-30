import pandas as pd

from ml.grid import find_silences, gap_mask, to_grid

T0 = pd.Timestamp("2026-10-10 12:00", tz="UTC")


def minutes(m: float) -> pd.Timestamp:
    return T0 + pd.Timedelta(minutes=m)


def row(station: str, m: float, bikes: int) -> dict:
    return {
        "station_id": station,
        "fetched_at": minutes(m),
        "num_bikes_available": bikes,
        "num_ebikes_available": 0,
        "num_docks_available": 10 - bikes,
        "is_installed": True,
        "is_renting": True,
        "is_returning": True,
    }


def grid(rows, **kwargs) -> pd.DataFrame:
    kwargs = {"start": T0, "known_outages": []} | kwargs
    return to_grid(pd.DataFrame(rows), **kwargs)


def bikes(g: pd.DataFrame, station: str) -> list:
    values = g.loc[g["station_id"] == station, "num_bikes_available"]
    return [None if pd.isna(v) else int(v) for v in values]


def test_forward_fills_unchanged_slots():
    g = grid([row("a", 1, 3), row("a", 46, 5)])
    assert bikes(g, "a") == [3, 3, 3, 5]
    assert not g["is_gap"].any()


def test_keeps_latest_run_within_a_slot():
    g = grid([row("a", 1, 3), row("a", 9, 4)])
    assert bikes(g, "a") == [4]


def test_stations_are_filled_independently():
    g = grid([row("a", 1, 3), row("b", 31, 7)])
    assert bikes(g, "a") == [3, 3, 3]
    assert bikes(g, "b") == [None, None, 7]


def test_value_before_start_carries_into_grid():
    g = grid([row("a", 1, 3), row("a", 31, 4)], start=minutes(15))
    assert bikes(g, "a") == [3, 4]


def test_known_outage_is_masked_and_value_after_it_is_carried():
    # Station "a" did not change during or after the outage, so after the
    # outage it still has 3 bikes. Station "b" keeps the run times visible.
    rows = [row("a", 1, 3), row("b", 1, 1), row("b", 61, 2)]
    g = grid(rows, known_outages=[(minutes(5), minutes(62))])
    assert bikes(g, "a") == [3, None, None, None, 3]
    assert g.loc[g["station_id"] == "a", "is_gap"].tolist() == [
        False,
        True,
        True,
        True,
        False,
    ]


def test_long_silence_becomes_gap_but_short_does_not():
    rows = [row("a", 1, 3), row("a", 61, 4), row("a", 241, 5)]
    g = grid(rows, max_silence=pd.Timedelta("2h"))
    is_gap = g["is_gap"].tolist()
    assert not any(is_gap[:5])  # 1 h silence: unchanged, not a gap
    assert all(is_gap[5:16])  # 3 h silence: outage
    assert not is_gap[16]


def test_find_silences():
    runs = pd.Series([minutes(0), minutes(15), minutes(200), minutes(200)])
    assert find_silences(runs, pd.Timedelta("1h")) == [(minutes(15), minutes(200))]


def test_gap_mask_keeps_boundary_slots():
    slots = pd.date_range(T0, periods=5, freq="15min")
    mask = gap_mask(slots, [(minutes(1), minutes(46))])
    assert mask.tolist() == [False, True, True, False, False]
