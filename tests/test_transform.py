from datetime import UTC, datetime

from collector.transform import (
    StationStatus,
    filter_changed,
    parse_station_information,
    parse_station_status,
)


def status_entry(station_id="1", bikes=5, ebikes=2, docks=10, **overrides):
    entry = {
        "station_id": station_id,
        "is_installed": True,
        "is_renting": True,
        "is_returning": True,
        "last_reported": 1790873731,
        "num_bikes_available": bikes,
        "num_docks_available": docks,
        "vehicle_types_available": [
            {"vehicle_type_id": "bike", "count": bikes - ebikes},
            {"vehicle_type_id": "ebike", "count": ebikes},
        ],
    }
    entry.update(overrides)
    return entry


def payload(*entries):
    return {"data": {"stations": list(entries)}}


def make_status(station_id="1", bikes=5, ebikes=2, docks=10, renting=True):
    return StationStatus(
        station_id=station_id,
        last_reported=datetime(2026, 10, 1, tzinfo=UTC),
        num_bikes_available=bikes,
        num_ebikes_available=ebikes,
        num_docks_available=docks,
        is_installed=True,
        is_renting=renting,
        is_returning=True,
    )


# --- parse_station_status ---


def test_parse_status_reads_fields_and_counts_ebikes():
    [s] = parse_station_status(payload(status_entry(bikes=11, ebikes=4, docks=22)))
    assert s.station_id == "1"
    assert s.num_bikes_available == 11
    assert s.num_ebikes_available == 4
    assert s.num_docks_available == 22
    assert s.last_reported == datetime.fromtimestamp(1790873731, tz=UTC)


def test_parse_status_without_vehicle_types_gives_zero_ebikes():
    entry = status_entry()
    del entry["vehicle_types_available"]
    [s] = parse_station_status(payload(entry))
    assert s.num_ebikes_available == 0


def test_parse_status_skips_broken_entry_but_keeps_others():
    broken = status_entry(station_id="2")
    del broken["num_bikes_available"]
    result = parse_station_status(payload(status_entry(station_id="1"), broken))
    assert [s.station_id for s in result] == ["1"]


# --- parse_station_information ---


def test_parse_information_reads_fields_and_allows_missing_address():
    entry = {
        "station_id": "6026",
        "name": "Østbanehallen",
        "lat": 59.91,
        "lon": 10.75,
        "capacity": 35,
    }
    [s] = parse_station_information(payload(entry))
    assert s.name == "Østbanehallen"
    assert s.capacity == 35
    assert s.address is None
    assert s.is_virtual_station is False


# --- filter_changed ---


def test_new_station_is_kept():
    current = [make_status("1")]
    assert filter_changed(current, previous={}) == current


def test_unchanged_station_is_dropped():
    s = make_status("1")
    assert filter_changed([s], previous={"1": s.key()}) == []


def test_changed_count_is_kept():
    old = make_status("1", bikes=5)
    new = make_status("1", bikes=6)
    assert filter_changed([new], previous={"1": old.key()}) == [new]


def test_changed_renting_flag_is_kept():
    old = make_status("1", renting=True)
    new = make_status("1", renting=False)
    assert filter_changed([new], previous={"1": old.key()}) == [new]
