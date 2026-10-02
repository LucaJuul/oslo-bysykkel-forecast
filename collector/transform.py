"""Pure functions that turn GBFS responses into rows. No network, no database."""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

log = logging.getLogger(__name__)

StatusKey = tuple[int, int, int, bool, bool, bool]


@dataclass(frozen=True)
class Station:
    station_id: str
    name: str
    address: str | None
    lat: float
    lon: float
    capacity: int
    is_virtual_station: bool


@dataclass(frozen=True)
class StationStatus:
    station_id: str
    last_reported: datetime
    num_bikes_available: int
    num_ebikes_available: int
    num_docks_available: int
    is_installed: bool
    is_renting: bool
    is_returning: bool

    def key(self) -> StatusKey:
        """The values that decide whether a measurement has changed."""
        return (
            self.num_bikes_available,
            self.num_ebikes_available,
            self.num_docks_available,
            self.is_installed,
            self.is_renting,
            self.is_returning,
        )


def parse_station_information(payload: dict) -> list[Station]:
    stations = []
    for s in payload["data"]["stations"]:
        try:
            stations.append(
                Station(
                    station_id=str(s["station_id"]),
                    name=s["name"],
                    address=s.get("address"),
                    lat=float(s["lat"]),
                    lon=float(s["lon"]),
                    capacity=int(s["capacity"]),
                    is_virtual_station=bool(s.get("is_virtual_station", False)),
                )
            )
        except (KeyError, TypeError, ValueError):
            log.warning("Skipping station_information entry: %s", s.get("station_id"))
    return stations


def parse_station_status(payload: dict) -> list[StationStatus]:
    statuses = []
    for s in payload["data"]["stations"]:
        try:
            ebikes = sum(
                v["count"]
                for v in s.get("vehicle_types_available", [])
                if v.get("vehicle_type_id") == "ebike"
            )
            statuses.append(
                StationStatus(
                    station_id=str(s["station_id"]),
                    last_reported=datetime.fromtimestamp(s["last_reported"], tz=UTC),
                    num_bikes_available=int(s["num_bikes_available"]),
                    num_ebikes_available=int(ebikes),
                    num_docks_available=int(s["num_docks_available"]),
                    is_installed=bool(s["is_installed"]),
                    is_renting=bool(s["is_renting"]),
                    is_returning=bool(s["is_returning"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            log.warning("Skipping station_status entry: %s", s.get("station_id"))
    return statuses


def filter_changed(
    current: list[StationStatus], previous: dict[str, StatusKey]
) -> list[StationStatus]:
    """Keep only statuses that are new or differ from the previous measurement."""
    return [s for s in current if previous.get(s.station_id) != s.key()]
