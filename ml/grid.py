"""Turn change-only measurements into a regular 15-minute grid per station.

The collector only stores a row when a station's numbers change. A missing row
therefore usually means "unchanged", and forward fill recovers the true value.
That is only wrong when the collector itself was not running, so those periods
are marked as gaps and their values set to missing.
"""

from pathlib import Path

import pandas as pd

FREQ = "15min"
OSLO = "Europe/Oslo"

# Before this, GitHub's throttled schedule left gaps of several hours.
ANALYSIS_START = pd.Timestamp("2026-10-04 18:00", tz=OSLO)

# Outages we know about. GitHub Actions incident, 5-6 October.
KNOWN_OUTAGES = [
    (
        pd.Timestamp("2026-10-05 21:00", tz=OSLO),
        pd.Timestamp("2026-10-06 07:15", tz=OSLO),
    ),
]

# Quiet nights produce up to ~2.5 h with no changed rows even though the
# collector ran. Silences longer than this are treated as outages.
MAX_SILENCE = pd.Timedelta("3h")

COUNT_COLUMNS = ["num_bikes_available", "num_ebikes_available", "num_docks_available"]
FLAG_COLUMNS = ["is_installed", "is_renting", "is_returning"]
VALUE_COLUMNS = COUNT_COLUMNS + FLAG_COLUMNS
NULLABLE_TYPES = {c: "Int16" for c in COUNT_COLUMNS} | {
    c: "boolean" for c in FLAG_COLUMNS
}


def load_status(data_dir: Path = Path("data")) -> pd.DataFrame:
    return pd.read_parquet(data_dir / "station_status.parquet")


def load_stations(data_dir: Path = Path("data")) -> pd.DataFrame:
    return pd.read_parquet(data_dir / "stations.parquet")


def find_silences(
    fetched_at: pd.Series, max_silence: pd.Timedelta = MAX_SILENCE
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Periods between consecutive runs that are longer than max_silence."""
    runs = pd.Series(fetched_at.unique()).sort_values()
    previous = runs.shift()
    long = (runs - previous) > max_silence
    return list(zip(previous[long], runs[long], strict=True))


def gap_mask(
    slots: pd.DatetimeIndex, gaps: list[tuple[pd.Timestamp, pd.Timestamp]]
) -> pd.Series:
    """True for slots that fall strictly between the last run before a gap and
    the first run after it. Those two slots hold real measurements."""
    mask = pd.Series(False, index=slots)
    for start, end in gaps:
        first = pd.Timestamp(start).tz_convert("UTC").floor(FREQ)
        last = pd.Timestamp(end).tz_convert("UTC").floor(FREQ)
        mask |= (slots > first) & (slots < last)
    return mask


def to_grid(
    status: pd.DataFrame,
    start: pd.Timestamp = ANALYSIS_START,
    known_outages: list[tuple[pd.Timestamp, pd.Timestamp]] = KNOWN_OUTAGES,
    max_silence: pd.Timedelta = MAX_SILENCE,
) -> pd.DataFrame:
    """One row per station and 15-minute slot, from start to the last run.

    Columns: station_id, slot (UTC), the value columns, and is_gap.
    """
    df = status.copy()
    df["slot"] = df["fetched_at"].dt.floor(FREQ)
    # Several runs in one slot (e.g. a manual run): keep the latest.
    df = df.sort_values("fetched_at").groupby(["station_id", "slot"]).last()

    slots = pd.date_range(
        df.index.get_level_values("slot").min(),
        df.index.get_level_values("slot").max(),
        freq=FREQ,
    )
    stations = df.index.get_level_values("station_id").unique()
    full = pd.MultiIndex.from_product([stations, slots], names=["station_id", "slot"])

    # Fill from the very first slot, so the value just before `start` carries in.
    # Nullable dtypes, so integers and booleans can hold missing values.
    grid = df[VALUE_COLUMNS].astype(NULLABLE_TYPES).reindex(full)
    grid = grid.groupby(level="station_id").ffill()

    gaps = list(known_outages) + find_silences(status["fetched_at"], max_silence)
    is_gap = gap_mask(slots, gaps)
    grid["is_gap"] = is_gap.reindex(grid.index, level="slot").to_numpy()
    grid.loc[grid["is_gap"], VALUE_COLUMNS] = pd.NA

    grid = grid.reset_index()
    grid = grid[grid["slot"] >= pd.Timestamp(start).tz_convert("UTC")]
    return grid.reset_index(drop=True)
