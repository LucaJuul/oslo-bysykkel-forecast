"""Response models. FastAPI uses them to validate output and document the API."""

from datetime import datetime

from pydantic import BaseModel, Field


class HorizonForecast(BaseModel):
    minutes: int = Field(description="How far ahead: 15, 30 or 60")
    bikes: float = Field(description="Expected number of bikes")
    docks: float = Field(description="Expected number of free docks")
    p_bike: float | None = Field(description="Probability of at least one bike")
    p_dock: float | None = Field(description="Probability of at least one free dock")


class StationForecast(BaseModel):
    station_id: str
    name: str | None
    lat: float | None
    lon: float | None
    capacity: int | None
    bikes_now: int
    docks_now: int
    reliable: bool = Field(
        description="False if the station is closed or has not changed for 72 "
        "hours; the models were not trained on such stations"
    )
    distance_m: float | None = Field(
        default=None, description="Distance from the given position, if any"
    )
    forecasts: list[HorizonForecast]


class ForecastResponse(BaseModel):
    data_time: datetime = Field(description="When the latest data was measured (UTC)")
    stale: bool = Field(description="True if the data is more than 35 minutes old")
    stations: list[StationForecast]


class Health(BaseModel):
    status: str
    data_time: datetime
    stale: bool
    stations: int
    model_trained_on: str
