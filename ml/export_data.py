"""Export the database to local Parquet files in data/.

Run:  uv run --env-file .env python -m ml.export_data

Each run reads both tables once and overwrites the files, so re-running it later
simply gives fresher data. All analysis then reads the files, not the database.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

from collector.db import connect

DATA_DIR = Path("data")
QUERIES = {
    "stations": "SELECT * FROM stations",
    "station_status": "SELECT * FROM station_status ORDER BY station_id, fetched_at",
}
TIMESTAMP_COLUMNS = ["fetched_at", "last_reported", "updated_at"]

log = logging.getLogger("export")


def export_table(conn, query: str, path: Path) -> int:
    with conn.cursor() as cur:
        cur.execute(query)
        columns = [d.name for d in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=columns)
    for col in TIMESTAMP_COLUMNS:
        if col in df:
            df[col] = pd.to_datetime(df[col], utc=True)
    df.to_parquet(path, index=False)
    return len(df)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    DATA_DIR.mkdir(exist_ok=True)
    with connect() as conn:
        for table, query in QUERIES.items():
            path = DATA_DIR / f"{table}.parquet"
            rows = export_table(conn, query, path)
            log.info("%s: %d rows -> %s", table, rows, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
