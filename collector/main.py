"""Collector entry point: python -m collector.main"""

import logging
import os
import sys
from datetime import UTC, datetime

import httpx
import psycopg

from collector import db
from collector.transform import (
    filter_changed,
    parse_station_information,
    parse_station_status,
)

BASE_URL = "https://gbfs.urbansharing.com/oslobysykkel.no"
log = logging.getLogger("collector")


def fetch(client: httpx.Client, feed: str) -> dict:
    response = client.get(f"{BASE_URL}/{feed}.json")
    response.raise_for_status()
    data = response.json()
    if not data.get("data", {}).get("stations"):
        raise ValueError(f"{feed}: response contained no stations")
    return data


def run() -> None:
    headers = {"Client-Identifier": os.environ["BYSYKKEL_CLIENT_ID"]}
    fetched_at = datetime.now(UTC)

    with httpx.Client(headers=headers, timeout=10) as client, db.connect() as conn:
        if db.stations_need_refresh(conn):
            stations = parse_station_information(fetch(client, "station_information"))
            db.upsert_stations(conn, stations)
            log.info("Updated %d stations", len(stations))

        statuses = parse_station_status(fetch(client, "station_status"))
        changed = filter_changed(statuses, db.latest_status_keys(conn))
        db.insert_statuses(conn, changed, fetched_at)
        log.info("Fetched %d statuses, saved %d changed", len(statuses), len(changed))


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        run()
    except (httpx.HTTPError, psycopg.Error, ValueError, KeyError) as e:
        log.error("Collector failed: %s: %s", type(e).__name__, e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
