# Finner jeg en bysykkel?

*"Will I find a city bike?"* – a web app that predicts the number of available bikes and free docks at Oslo Bysykkel stations 15, 30 and 60 minutes ahead.

**Status:** Under development – currently in phase 0 (project setup).

## Planned architecture

- **Data source:** Oslo Bysykkel's open real-time API (GBFS)
- **Link to data souce:** https://oslobysykkel.no/en/open-data/realtime
- **Collector:** Python script run on a schedule by GitHub Actions
- **Storage:** PostgreSQL (Neon or Supabase)
- **Machine learning:** pandas, scikit-learn and LightGBM
- **API:** FastAPI serving predictions per station
- **Frontend:** Interactive map built with Leaflet
- **Quality:** pytest for tests, ruff for linting and formatting

## Project structure

```
collector/   Data collection from the Oslo Bysykkel API
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
