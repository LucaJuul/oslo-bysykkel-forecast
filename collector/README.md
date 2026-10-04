# collector

Fetches station data from the Oslo Bysykkel GBFS API and stores it in PostgreSQL.

- `station_status` is fetched on every run. Only stations whose numbers or
  operating flags changed since the previous run are saved.
- `station_information` is refreshed when the stored copy is older than 24 hours.

## How it runs

- Workflow: `.github/workflows/collect.yml`
- Triggered every 15 minutes by cron-job.org through the GitHub API
  (`workflow_dispatch`). GitHub's own `schedule` is kept as a fallback, since it
  was throttled to runs every 2–6 hours.
- Repository secrets: `DATABASE_URL` and `BYSYKKEL_CLIENT_ID`

## Run locally

    uv run --env-file .env python -m collector.main

## Files

- `transform.py`: parses GBFS responses and filters unchanged rows (pure functions, covered by tests)
- `db.py`: database reads and writes
- `main.py`: entry point with HTTP requests, logging and error handling

## Monitoring

See `db/monitoring.sql` for queries showing time since the last run, rows per
day, database size and gaps between runs.
