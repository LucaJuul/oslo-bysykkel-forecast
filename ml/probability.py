"""Probability forecasts: "will there be at least one bike (or free dock)?"

Two LightGBM classifiers on the same features as the MAE model:
    P(bikes at target >= 1)   target "bike"
    P(docks at target >= 1)   target "dock"
Loss: log loss (binary cross-entropy), which rewards honest probabilities.
Scored with the Brier score and calibration, against two baselines.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from ml.evaluate import subset_masks
from ml.features import CATEGORICAL
from ml.model import (
    EARLY_STOPPING,
    MAX_ROUNDS,
    MODEL_FEATURES,
    PARAMS,
    TRAIN_SCALE,
)

# What "available" means for each target: the column now and at the target.
TARGETS = {"bike": "bikes", "dock": "docks"}
FEW = 3  # "few now": the subset where the answer is genuinely uncertain

# The MAE model's features plus docks now. Without it the dock classifier has
# to rebuild "docks now" from total_now - bikes and was worse than the count
# table on the validation week (see DECISIONS.md).
CLS_FEATURES = [*MODEL_FEATURES, "docks"]

# Tuned on the validation week. Unlike L1, log loss learns from every pair, so
# small trees are enough: 31 leaves (LightGBM's default) was as good as 63 or
# 255 within noise, and 255 leaves overfit (see DECISIONS.md).
CLS_PARAMS = PARAMS | {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 31,
    "min_data_in_leaf": 100,
}
# Final tree counts: best tree on the validation week (bike 89, dock 97),
# scaled to the whole training period like the MAE model.
CLS_FINAL_ROUNDS = {"bike": round(89 * TRAIN_SCALE), "dock": round(97 * TRAIN_SCALE)}


def outcome(pairs: pd.DataFrame, target: str) -> np.ndarray:
    """1 if at least one bike (dock) at the target time, else 0."""
    return (pairs[f"target_{TARGETS[target]}"] >= 1).to_numpy(dtype="float64")


def train_classifier(
    fit: pd.DataFrame,
    val: pd.DataFrame | None,
    target: str,
    params: dict = CLS_PARAMS,
    rounds: int = MAX_ROUNDS,
    columns: list = CLS_FEATURES,
) -> tuple[lgb.Booster, dict]:
    """Like ml.model.train, but for the 0/1 outcome of `target`."""
    categorical = [c for c in CATEGORICAL if c in columns]
    fit_set = lgb.Dataset(
        fit[columns], label=outcome(fit, target), categorical_feature=categorical
    )
    sets, names, callbacks, history = [fit_set], ["fit"], [], {}
    if val is not None:
        sets.append(
            lgb.Dataset(val[columns], label=outcome(val, target), reference=fit_set)
        )
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


def predict_proba(booster: lgb.Booster, features: pd.DataFrame) -> np.ndarray:
    return booster.predict(
        features[booster.feature_name()], num_iteration=booster.best_iteration or None
    )


class UnchangedProbability:
    """100 % if there is one now, else 0 %: `unchanged` read as a probability."""

    name = "unchanged"

    def __init__(self, target: str):
        self.column = TARGETS[target]

    def fit(self, train: pd.DataFrame) -> "UnchangedProbability":
        return self

    def predict(self, pairs: pd.DataFrame) -> np.ndarray:
        return (pairs[self.column] >= 1).to_numpy(dtype="float64")


class CountTable:
    """Share of training pairs with at least one at the target, for the same
    number now (capped at `cap`) and horizon. A strong, simple baseline: it is
    calibrated by construction and knows that 1 bike now is risky."""

    name = "count_table"

    def __init__(self, target: str, cap: int = 15):
        self.target, self.column, self.cap = target, TARGETS[target], cap

    def keys(self, pairs: pd.DataFrame) -> pd.MultiIndex:
        now = pairs[self.column].clip(upper=self.cap).astype(int)
        return pd.MultiIndex.from_arrays([now, pairs["horizon"]])

    def fit(self, train: pd.DataFrame) -> "CountTable":
        hit = pd.Series(outcome(train, self.target), index=self.keys(train))
        self.table = hit.groupby(level=[0, 1]).mean()
        return self

    def predict(self, pairs: pd.DataFrame) -> np.ndarray:
        return self.table.reindex(self.keys(pairs)).fillna(1.0).to_numpy()


def probability_masks(pairs: pd.DataFrame, target: str) -> dict[str, np.ndarray]:
    """The usual subsets plus "few_now": at most FEW available at the origin."""
    masks = subset_masks(pairs)
    masks["few_now"] = (pairs[TARGETS[target]] <= FEW).to_numpy()
    return masks


def mean(values: np.ndarray) -> float:
    """Mean, or NaN for an empty subset (without a warning)."""
    return float(values.mean()) if len(values) else float("nan")


def brier_table(
    pairs: pd.DataFrame, predictions: dict[str, np.ndarray], target: str
) -> pd.DataFrame:
    """Long table: model, target, subset, horizon, brier, log_loss, n."""
    truth = outcome(pairs, target)
    horizon = pairs["horizon"].to_numpy()
    rows = []
    for model, p in predictions.items():
        p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
        brier = (p - truth) ** 2
        log_loss = -(truth * np.log(p) + (1 - truth) * np.log(1 - p))
        for subset, mask in probability_masks(pairs, target).items():
            for h in sorted(np.unique(horizon)):
                selected = mask & (horizon == h)
                rows.append(
                    {
                        "model": model,
                        "target": target,
                        "subset": subset,
                        "horizon": int(h),
                        "brier": mean(brier[selected]),
                        "log_loss": mean(log_loss[selected]),
                        "n": int(selected.sum()),
                    }
                )
    return pd.DataFrame(rows)


def calibration(p: np.ndarray, truth: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Mean predicted probability vs observed share, in equal-width bins."""
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    df = pd.DataFrame({"bin": which, "predicted": p, "observed": truth})
    return df.groupby("bin").agg(
        predicted=("predicted", "mean"),
        observed=("observed", "mean"),
        n=("observed", "size"),
    )
