"""FastAPI app: forecasts of bikes and free docks per Oslo Bysykkel station.

Run locally:  uv run --env-file .env uvicorn api.main:app --reload
Docs:         http://127.0.0.1:8000/docs

At startup it loads the models from models/ and the last 7 days + 1 hour of
history from the database (one query). After that it polls the public GBFS
feed right after every 15-minute slot starts, like the collector, so the data
matches what the models were trained on and the database is not touched again.
"""

import asyncio
import logging
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import httpx
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import ForecastResponse, Health, HorizonForecast, StationForecast
from api.service import ForecastService
from api.sources import history_from_db, live_status, stations_from_db
from collector import db
from ml.features import EARTH_RADIUS_M, HISTORY_NEEDED
from ml.grid import FREQ
from ml.predict import Forecaster

log = logging.getLogger("api")
POLL_DELAY = 60  # seconds after each slot starts, so GBFS has fresh data
DEFAULT_LIMIT = 10  # nearest stations returned when a position is given


def load_service() -> ForecastService:
    """Models from disk, history from the database, then one live snapshot."""
    forecaster = Forecaster()
    start = datetime.now(UTC) - HISTORY_NEEDED - pd.Timedelta(FREQ)
    with db.connect() as conn:
        stations = stations_from_db(conn)
        history = history_from_db(conn, start)
    service = ForecastService(forecaster, stations, history)
    log.info("Loaded %d status rows for %d stations", len(history), len(stations))
    try:
        service.add_status(fetch_live())
    except (httpx.HTTPError, ValueError, KeyError) as e:
        log.warning("Live poll at startup failed: %s", e)
    return service


def fetch_live() -> pd.DataFrame:
    headers = {"Client-Identifier": os.environ["BYSYKKEL_CLIENT_ID"]}
    with httpx.Client(headers=headers, timeout=10) as client:
        return live_status(client)


def seconds_to_next_poll(now: datetime) -> float:
    slot = pd.Timestamp(now).floor(FREQ)
    next_poll = slot + pd.Timedelta(FREQ) + pd.Timedelta(seconds=POLL_DELAY)
    if pd.Timestamp(now) < slot + pd.Timedelta(seconds=POLL_DELAY):
        next_poll = slot + pd.Timedelta(seconds=POLL_DELAY)
    return (next_poll - pd.Timestamp(now)).total_seconds()


async def poll_live(service: ForecastService) -> None:
    """Add a GBFS snapshot after every slot start. Errors are logged and the
    next slot is tried; meanwhile responses are marked stale after 35 min."""
    while True:
        await asyncio.sleep(seconds_to_next_poll(datetime.now(UTC)))
        try:
            rows = await asyncio.to_thread(fetch_live)
            service.add_status(rows)
            log.info("Live snapshot: %d stations", len(rows))
        except (httpx.HTTPError, ValueError, KeyError) as e:
            log.warning("Live poll failed: %s", e)


def distance_m(lat, lon, lat0: float, lon0: float) -> np.ndarray:
    """Straight-line distance in metres (flat-earth approximation, fine in Oslo)."""
    x = np.radians(np.asarray(lon) - lon0) * np.cos(np.radians(lat0))
    y = np.radians(np.asarray(lat) - lat0)
    return EARTH_RADIUS_M * np.hypot(x, y)


def to_station(rows: pd.DataFrame) -> StationForecast:
    """One station's rows (one per horizon) as a response object."""
    first = rows.iloc[0]

    def optional(value, cast):
        return None if pd.isna(value) else cast(value)

    return StationForecast(
        station_id=first["station_id"],
        name=optional(first["name"], str),
        lat=optional(first["lat"], float),
        lon=optional(first["lon"], float),
        capacity=optional(first["capacity"], int),
        bikes_now=int(first["bikes_now"]),
        docks_now=int(first["docks_now"]),
        reliable=bool(first["reliable"]),
        distance_m=optional(first.get("distance_m"), lambda d: round(float(d))),
        forecasts=[
            HorizonForecast(
                minutes=int(r["horizon"]),
                bikes=round(float(r["bikes"]), 1),
                docks=round(float(r["docks"]), 1),
                p_bike=optional(r["p_bike"], lambda p: round(float(p), 2)),
                p_dock=optional(r["p_dock"], lambda p: round(float(p), 2)),
            )
            for _, r in rows.sort_values("horizon").iterrows()
        ],
    )


def create_app(
    service_factory: Callable[[], ForecastService] = load_service, poll: bool = True
) -> FastAPI:
    """The app. Tests pass their own `service_factory` and `poll=False`."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service_factory()
        task = asyncio.create_task(poll_live(app.state.service)) if poll else None
        yield
        if task:
            task.cancel()

    app = FastAPI(
        title="Finner jeg en bysykkel?",
        description="Forecasts of bikes and free docks at Oslo Bysykkel stations "
        "15, 30 and 60 minutes ahead. Data: Oslo Bysykkel (NLOD 2.0).",
        version="1.0.0",
        lifespan=lifespan,
    )
    # Public, read-only data: any website may call the API.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])

    def service(request: Request) -> ForecastService:
        return request.app.state.service

    @app.get("/health")
    def health(request: Request) -> Health:
        s = service(request)
        return Health(
            status="ok",
            data_time=s.data_time.to_pydatetime(),
            stale=s.is_stale(),
            stations=len(s.stations),
            model_trained_on=s.forecaster.meta.get("trained_on", "unknown"),
        )

    @app.get("/forecasts")
    def forecasts(
        request: Request,
        lat: float | None = Query(None, ge=-90, le=90, description="Your latitude"),
        lon: float | None = Query(None, ge=-180, le=180, description="Your longitude"),
        limit: int | None = Query(None, ge=1, le=300, description="Max stations"),
    ) -> ForecastResponse:
        """All stations, or the nearest ones when `lat` and `lon` are given."""
        if (lat is None) != (lon is None):
            raise HTTPException(422, "Give both lat and lon, or neither.")
        s = service(request)
        df = s.forecasts()
        order = df["station_id"].drop_duplicates().tolist()
        if lat is not None:
            df = df.assign(distance_m=distance_m(df["lat"], df["lon"], lat, lon))
            order = (
                df.drop_duplicates("station_id")
                .sort_values("distance_m")["station_id"]
                .tolist()
            )
            limit = limit or DEFAULT_LIMIT
        order = order[:limit] if limit else order
        groups = dict(tuple(df.groupby("station_id", sort=False)))
        return ForecastResponse(
            data_time=s.data_time.to_pydatetime(),
            stale=s.is_stale(),
            stations=[to_station(groups[station]) for station in order],
        )

    @app.get("/forecasts/{station_id}")
    def station_forecast(request: Request, station_id: str) -> StationForecast:
        df = service(request).forecasts()
        rows = df[df["station_id"] == station_id]
        if rows.empty:
            raise HTTPException(404, f"No forecast for station {station_id}.")
        return to_station(rows)

    return app


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
app = create_app()
