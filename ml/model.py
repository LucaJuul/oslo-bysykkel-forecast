"""LightGBM model: predicts the change in bikes, then adds it to the current value.

One model for all horizons, with the horizon (minutes) as a feature, so it
learns from all pairs and shares patterns between 15, 30 and 60 minutes.

Target:      target_bikes - bikes            (0 means "unchanged")
Prediction:  clip(bikes + predicted change)  (same clip as the baselines)
Loss:        L1 (mean absolute error), the metric we report.
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ml.baselines import clip
from ml.features import CATEGORICAL, FEATURES
from ml.weather import WEATHER_COLUMNS

# Weather is computed in ml.features but left out of the model: on the
# validation week it made MAE slightly worse (see DECISIONS.md).
MODEL_FEATURES = [f for f in FEATURES if f not in WEATHER_COLUMNS] + ["horizon"]

# Tuned step by step on the validation week (see DECISIONS.md).
PARAMS = {
    "objective": "l1",  # optimise MAE directly
    "metric": "l1",  # early stopping and learning curve always use MAE
    "learning_rate": 0.05,  # how much of each new tree is added
    # Tuned on the validation week. With 31 leaves (the default) L1 stalls:
    # each leaf's value is the median change in it, which is 0 when most pairs
    # in the leaf do not change. More, smaller leaves find groups that do.
    "num_leaves": 255,  # max leaves per tree
    "min_data_in_leaf": 100,  # each leaf must cover at least this many pairs
    "seed": 0,
    "verbose": -1,
}
MAX_ROUNDS = 3000
EARLY_STOPPING = 100  # stop when validation MAE has not improved for this many trees

# The final model trains on the whole training period, with no validation week
# left for early stopping. Tuning stopped at 57 trees on 832k pairs; the full
# period has 1.64 times as many, so the tree count is scaled to match.
FINAL_ROUNDS = 94


def dataset(features: pd.DataFrame, reference=None) -> lgb.Dataset:
    """Features and target (change in bikes). A validation set passes the
    training set as `reference`, so both use the same histogram bins."""
    change = features["target_bikes"] - features["bikes"]
    return lgb.Dataset(
        features[MODEL_FEATURES],
        label=change.to_numpy(dtype="float64"),
        categorical_feature=CATEGORICAL,
        reference=reference,
        free_raw_data=False,
    )


def train(
    fit: pd.DataFrame,
    val: pd.DataFrame | None = None,
    params: dict = PARAMS,
    rounds: int = MAX_ROUNDS,
) -> tuple[lgb.Booster, dict]:
    """Train on `fit`. With `val`, use early stopping on its MAE.

    Returns the model and the learning curve: {"fit": {"l1": [...]},
    "val": {"l1": [...]}}, one value per tree.
    """
    fit_set = dataset(fit)
    sets, names = [fit_set], ["fit"]
    callbacks = []
    history: dict = {}
    if val is not None:
        sets.append(dataset(val, reference=fit_set))
        names.append("val")
        callbacks.append(lgb.early_stopping(EARLY_STOPPING, verbose=False))
    callbacks.append(lgb.record_evaluation(history))
    booster = lgb.train(
        params,
        fit_set,
        num_boost_round=rounds,
        valid_sets=sets,
        valid_names=names,
        callbacks=callbacks,
    )
    return booster, history


def predict(booster: lgb.Booster, features: pd.DataFrame) -> np.ndarray:
    """Predicted bikes at the target, clipped to [0, capacity]."""
    change = booster.predict(
        features[MODEL_FEATURES], num_iteration=booster.best_iteration or None
    )
    return clip(features["bikes"].to_numpy() + change, features)


MODEL_DIR = Path("models")


def save(booster: lgb.Booster, stations: list, info: dict, path: Path = MODEL_DIR):
    """Write the model (LightGBM text format) and what is needed to use it."""
    path.mkdir(parents=True, exist_ok=True)
    booster.save_model(path / "lightgbm.txt")
    meta = {"features": MODEL_FEATURES, "stations": stations, **info}
    (path / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load(path: Path = MODEL_DIR) -> tuple[lgb.Booster, dict]:
    booster = lgb.Booster(model_file=str(path / "lightgbm.txt"))
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if meta["features"] != MODEL_FEATURES:
        raise ValueError("Saved model was trained with other features; retrain it.")
    return booster, meta
