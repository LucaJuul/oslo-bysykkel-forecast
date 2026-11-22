"""Hourly weather at Oslo Blindern from MET Norway's Frost API, saved to data/.

Run:  uv run --env-file .env python -m ml.weather

Needs FROST_CLIENT_ID in .env (free, register at
https://frost.met.no/auth/requestCredentials.html). Like export_data, each run
fetches the whole period again and overwrites data/weather.parquet.

Leakage: an observation is only treated as known one hour after its reference
time (`available_at`). Frost's hourly precipitation is a sum over an hour, and
this way it is never used before the hour is over, whichever end of the hour
the reference time marks. Features look up the latest observation with
`available_at <= origin`, never anything at the target time.
"""

import logging
import os
import sys
from pathlib import Path

import httpx
import pandas as pd

URL = "https://frost.met.no/observations/v0.jsonld"
SOURCE = "SN18700"  # Oslo - Blindern
ELEMENTS = {
    "air_temperature": "temperature",
    "sum(precipitation_amount PT1H)": "precipitation",
}
DELAY = pd.Timedelta("1h")
START = pd.Timestamp("2026-10-01", tz="UTC")
PATH = Path("data") / "weather.parquet"
WEATHER_COLUMNS = list(ELEMENTS.values())

log = logging.getLogger("weather")


def fetch(client_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Raw hourly observations: reference_time (UTC), element, value."""
    params = {
        "sources": SOURCE,
        "elements": ",".join(ELEMENTS),
        "referencetime": f"{start:%Y-%m-%dT%H:%M}/{end:%Y-%m-%dT%H:%M}",
        "timeresolutions": "PT1H",
    }
    response = httpx.get(URL, params=params, auth=(client_id, ""), timeout=60)
    response.raise_for_status()
    rows = [
        {
            "reference_time": item["referenceTime"],
            "element": obs["elementId"],
            "value": obs["value"],
        }
        for item in response.json()["data"]
        for obs in item["observations"]
    ]
    return pd.DataFrame(rows)


def tidy(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per hour: available_at, temperature, precipitation.

    Frost can return several series per element (e.g. sensor levels); the
    first one per hour is kept.
    """
    df = raw.copy()
    df["reference_time"] = pd.to_datetime(df["reference_time"], utc=True)
    df["element"] = df["element"].map(ELEMENTS)
    wide = df.pivot_table(
        index="reference_time", columns="element", values="value", aggfunc="first"
    )
    wide = wide.reindex(columns=WEATHER_COLUMNS).reset_index()
    wide["available_at"] = wide["reference_time"] + DELAY
    return wide.sort_values("available_at").reset_index(drop=True)


def load_weather(data_dir: Path = Path("data")) -> pd.DataFrame:
    return pd.read_parquet(data_dir / PATH.name)


def weather_at(origins: pd.Series, weather: pd.DataFrame) -> pd.DataFrame:
    """Latest weather known at each origin (UTC). Same index as `origins`.

    Values older than 3 hours count as missing, so a stopped feed does not
    silently repeat the same weather for days.
    """
    left = pd.DataFrame({"origin": origins.to_numpy(), "row": range(len(origins))})
    left = left.sort_values("origin")
    right = weather[["available_at", *WEATHER_COLUMNS]].dropna(
        how="all", subset=WEATHER_COLUMNS
    )
    merged = pd.merge_asof(
        left,
        right,
        left_on="origin",
        right_on="available_at",
        direction="backward",
        tolerance=pd.Timedelta("3h"),
    )
    merged = merged.sort_values("row")
    return pd.DataFrame(
        merged[WEATHER_COLUMNS].to_numpy(dtype="float64"),
        columns=WEATHER_COLUMNS,
        index=origins.index,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    client_id = os.environ.get("FROST_CLIENT_ID")
    if not client_id:
        log.error("FROST_CLIENT_ID is not set. See the module docstring.")
        return 1
    end = pd.Timestamp.now(tz="UTC").ceil("h")
    # One request per 30 days keeps each response small.
    chunks = pd.date_range(START, end, freq="30D").append(pd.DatetimeIndex([end]))
    raw = pd.concat(
        [fetch(client_id, a, b) for a, b in zip(chunks[:-1], chunks[1:], strict=True)],
        ignore_index=True,
    )
    weather = tidy(raw).drop_duplicates("reference_time")
    PATH.parent.mkdir(exist_ok=True)
    weather.to_parquet(PATH, index=False)
    log.info(
        "%d hours, %s to %s -> %s",
        len(weather),
        weather["reference_time"].min(),
        weather["reference_time"].max(),
        PATH,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
