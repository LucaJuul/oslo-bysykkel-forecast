# api

FastAPI backend that serves forecasts of bikes and free docks per Oslo Bysykkel
station, 15, 30 and 60 minutes ahead.

```
uv run python -m ml.run_model                          # creates models/ (once)
uv run --env-file .env uvicorn api.main:app --reload   # http://127.0.0.1:8000/docs
```

Needs `DATABASE_URL` and `BYSYKKEL_CLIENT_ID` in `.env`, and the model files in
`models/`.

## Endpoints

| Endpoint | Returns |
|----------|---------|
| `GET /health` | Status, time of the latest data, whether it is stale, number of stations |
| `GET /forecasts` | Every station with its current bikes and docks and a forecast per horizon |
| `GET /forecasts?lat=59.91&lon=10.75&limit=5` | The nearest stations first, with `distance_m` |
| `GET /forecasts/{station_id}` | One station (404 if unknown) |

Each forecast has the expected number of bikes and docks and the probability of
at least one bike (`p_bike`) and at least one free dock (`p_dock`). `reliable`
is false for stations that are closed or have not changed for 72 hours; the
models were not trained on those. `stale` is true if no new data has arrived
for 35 minutes.

## How it gets data

```
startup:  models/  ──►  Forecaster
          Neon     ──►  last 7 days + 1 h of status rows (one query)
          GBFS     ──►  one live snapshot
every 15 min (1 min into each slot):  GBFS  ──►  new snapshot, old rows dropped
request:  history ──► 15-minute grid ──► features ──► models  (cached per data time)
```

The database is read only at startup. Serving forecasts would otherwise keep
the free Neon compute awake, which the collector already uses most of.

| File | Purpose |
|------|---------|
| `main.py` | App, endpoints, startup and the 15-minute live poll |
| `service.py` | History in memory, forecasts with a cache (no network, easy to test) |
| `sources.py` | Database query at startup and the live GBFS snapshot |
| `schemas.py` | Response models (also the documentation at `/docs`) |
