"""Where the API gets its data: the database once at startup, GBFS live after.

The database holds the history the features need (7 days + 1 hour). After
startup the API polls the public GBFS feed itself every 15 minutes, so serving
forecasts does not keep the Neon database awake (its free compute is limited).
"""

from datetime import UTC, datetime

import httpx
import pandas as pd

from collector.main import fetch
from collector.transform import parse_station_status

STATUS_COLUMNS = [
    "station_id",
    "fetched_at",
    "last_reported",
    "num_bikes_available",
    "num_ebikes_available",
    "num_docks_available",
    "is_installed",
    "is_renting",
    "is_returning",
]
STATION_COLUMNS = ["station_id", "name", "lat", "lon", "capacity"]

# The window, plus each station's latest row before it: the collector only
# stores changes, so that row is the starting value for forward fill.
HISTORY_QUERY = f"""
    (SELECT {", ".join(STATUS_COLUMNS)} FROM station_status
     WHERE fetched_at >= %(start)s)
    UNION ALL
    (SELECT DISTINCT ON (station_id) {", ".join(STATUS_COLUMNS)} FROM station_status
     WHERE fetched_at < %(start)s
     ORDER BY station_id, fetched_at DESC)
"""


def to_frame(rows: list, columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=columns)
    for column in ["fetched_at", "last_reported"]:
        if column in df:
            df[column] = pd.to_datetime(df[column], utc=True)
    return df


def history_from_db(conn, start: datetime) -> pd.DataFrame:
    """Status rows from `start` onwards plus the latest row per station before."""
    with conn.cursor() as cur:
        cur.execute(HISTORY_QUERY, {"start": start})
        return to_frame(cur.fetchall(), STATUS_COLUMNS)


def stations_from_db(conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(f"SELECT {', '.join(STATION_COLUMNS)} FROM stations")
        return to_frame(cur.fetchall(), STATION_COLUMNS)


def live_status(client: httpx.Client) -> pd.DataFrame:
    """One full snapshot of every station from GBFS, as status rows."""
    fetched_at = datetime.now(UTC)
    statuses = parse_station_status(fetch(client, "station_status"))
    rows = [
        {
            "station_id": s.station_id,
            "fetched_at": fetched_at,
            "last_reported": s.last_reported,
            "num_bikes_available": s.num_bikes_available,
            "num_ebikes_available": s.num_ebikes_available,
            "num_docks_available": s.num_docks_available,
            "is_installed": s.is_installed,
            "is_renting": s.is_renting,
            "is_returning": s.is_returning,
        }
        for s in statuses
    ]
    return to_frame(rows, STATUS_COLUMNS)
