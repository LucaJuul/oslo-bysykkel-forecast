# ml

Data preparation, models and evaluation.

| File | Purpose |
|------|---------|
| `export_data.py` | One-off export of the database to `data/*.parquet` |
| `grid.py` | 15-minute grid per station with gap marking – the only entry point to the data |
| `config.py` | Fixed train/test split, horizons and evaluation subsets |
| `pairs.py` | (origin, target) pairs with validity rules and the train/test split |
| `baselines.py` | Naive models: unchanged, same time last week, mean change profile |
| `evaluate.py` | MAE per horizon: all hours, daytime, weekday rush |
| `run_baselines.py` | Runs the baselines and writes `results/baselines.*` |
| `weather.py` | Hourly weather at Oslo Blindern from MET Frost to `data/weather.parquet` |
| `features.py` | Model features, using only values known at the origin |
| `model.py` | LightGBM training, prediction, save and load |
| `run_model.py` | Trains the final model, saves `models/`, writes `results/model*.*` |
| `predict.py` | `Forecaster`: forecasts from current data, for the API |

Run everything:

```
uv run --env-file .env python -m ml.export_data   # fresh data
uv run --env-file .env python -m ml.weather       # weather (needs FROST_CLIENT_ID)
uv run python -m ml.run_baselines
uv run python -m ml.run_model                      # final model + test results
```
