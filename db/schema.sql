-- Database schema for oslo-bysykkel-forecast.
-- Safe to run multiple times (IF NOT EXISTS).

CREATE TABLE IF NOT EXISTS stations (
    station_id          TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    address             TEXT,
    lat                 DOUBLE PRECISION NOT NULL,
    lon                 DOUBLE PRECISION NOT NULL,
    capacity            SMALLINT NOT NULL,
    is_virtual_station  BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS station_status (
    station_id            TEXT NOT NULL,
    fetched_at            TIMESTAMPTZ NOT NULL,
    last_reported         TIMESTAMPTZ NOT NULL,
    num_bikes_available   SMALLINT NOT NULL,
    num_ebikes_available  SMALLINT NOT NULL,
    num_docks_available   SMALLINT NOT NULL,
    is_installed          BOOLEAN NOT NULL,
    is_renting            BOOLEAN NOT NULL,
    is_returning          BOOLEAN NOT NULL,
    PRIMARY KEY (station_id, fetched_at)
);
