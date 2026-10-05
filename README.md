# Finner jeg en bysykkel?

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)

*"Will I find a city bike?"* – a web app that predicts the number of available bikes and free docks at Oslo Bysykkel stations 15, 30 and 60 minutes ahead.

**Status:** Data collection running every 15 minutes since 4 October 2026. Runs are triggered by cron-job.org because GitHub Actions' scheduled runs were too unreliable.

## Architecture

- **Data source:** Oslo Bysykkel's open real-time API (GBFS 2.3)
- **Collector:** Python script run every 15 minutes in GitHub Actions, triggered by cron-job.org
- **Storage:** PostgreSQL on Neon (free plan)
- **Machine learning (planned):** pandas, scikit-learn and LightGBM
- **API (planned):** FastAPI serving predictions per station
- **Frontend (planned):** Interactive map built with Leaflet
- **Quality:** pytest for tests, ruff for linting and formatting

## Data collection

The collector fetches station status from Oslo Bysykkel every 15 minutes and
stores it in Postgres. Only measurements that changed since the previous run are
saved. Station information (name, position, capacity) is refreshed once a day.
See [collector/README.md](collector/README.md) for details.

### Data source and licence

Real-time data from [Oslo Bysykkel](https://oslobysykkel.no/en/open-data/realtime),
licensed under the
[Norwegian Licence for Open Government Data (NLOD) 2.0](https://data.norge.no/nlod/en/2.0).
The MIT licence below covers the code in this repository, not the data.

## Project structure

```
collector/   Data collection from the Oslo Bysykkel API
db/          Database schema and monitoring queries
api/         FastAPI backend
ml/          Feature engineering and model training
frontend/    Leaflet web map
tests/       Automated tests
notebooks/   Exploratory data analysis
```

## Getting started

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/LucaJuul/oslo-bysykkel-forecast.git
cd oslo-bysykkel-forecast
uv sync
cp .env.example .env   # then fill in real values
uv run pytest
```

To run the collector locally, create the tables once by running `db/schema.sql`
against your database, then:

```bash
uv run --env-file .env python -m collector.main
```

## Results

*To be added.*

## Limitations

*To be added.*

## Future work

*To be added.*

## Design decisions

Key choices and their reasoning are documented in [DECISIONS.md](DECISIONS.md).

## License

[MIT](LICENSE)
