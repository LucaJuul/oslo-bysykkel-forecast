"""Database helpers for the collector."""

import os
from datetime import UTC, datetime, timedelta

import psycopg

from collector.transform import Station, StationStatus, StatusKey


def connect() -> psycopg.Connection:
    """Open a connection using DATABASE_URL from the environment."""
    return psycopg.connect(os.environ["DATABASE_URL"])


def stations_need_refresh(conn, max_age: timedelta = timedelta(hours=24)) -> bool:
    """True if the stations table is empty or older than max_age."""
    (last_update,) = conn.execute("SELECT max(updated_at) FROM stations").fetchone()
    return last_update is None or datetime.now(UTC) - last_update > max_age


def upsert_stations(conn, stations: list[Station]) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO stations
                (station_id, name, address, lat, lon, capacity, is_virtual_station)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (station_id) DO UPDATE SET
                name = EXCLUDED.name,
                address = EXCLUDED.address,
                lat = EXCLUDED.lat,
                lon = EXCLUDED.lon,
                capacity = EXCLUDED.capacity,
                is_virtual_station = EXCLUDED.is_virtual_station,
                updated_at = now()
            """,
            [
                (s.station_id, s.name, s.address, s.lat, s.lon, s.capacity,
                 s.is_virtual_station)
                for s in stations
            ],
        )


def latest_status_keys(conn) -> dict[str, StatusKey]:
    """The most recent stored measurement per station."""
    rows = conn.execute(
        """
        SELECT DISTINCT ON (station_id)
            station_id, num_bikes_available, num_ebikes_available,
            num_docks_available, is_installed, is_renting, is_returning
        FROM station_status
        ORDER BY station_id, fetched_at DESC
        """
    ).fetchall()
    return {row[0]: tuple(row[1:]) for row in rows}


def insert_statuses(
    conn, statuses: list[StationStatus], fetched_at: datetime
) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO station_status
                (station_id, fetched_at, last_reported, num_bikes_available,
                 num_ebikes_available, num_docks_available,
                 is_installed, is_renting, is_returning)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            [
                (s.station_id, fetched_at, s.last_reported, s.num_bikes_available,
                 s.num_ebikes_available, s.num_docks_available,
                 s.is_installed, s.is_renting, s.is_returning)
                for s in statuses
            ],
        )
