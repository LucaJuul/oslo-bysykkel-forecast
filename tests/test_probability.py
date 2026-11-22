import numpy as np
import pandas as pd
import pytest

from ml.features import add_features
from ml.pairs import make_pairs
from ml.probability import (
    CLS_PARAMS,
    CountTable,
    UnchangedProbability,
    brier_table,
    calibration,
    outcome,
    predict_proba,
    train_classifier,
)
from tests.test_predict import STATIONS, make_grid


def pairs_frame(rows: list[dict]) -> pd.DataFrame:
    """Pairs with bikes/docks now and at the target, plus subset columns."""
    defaults = {"horizon": 15, "weekday": 0, "hour": 12}
    return pd.DataFrame([defaults | r for r in rows])


def test_outcome_is_at_least_one():
    p = pairs_frame(
        [
            {"target_bikes": 0, "target_docks": 3},
            {"target_bikes": 1, "target_docks": 0},
        ]
    )
    assert outcome(p, "bike").tolist() == [0, 1]
    assert outcome(p, "dock").tolist() == [1, 0]


def test_unchanged_probability_is_zero_or_one_from_now():
    p = pairs_frame([{"bikes": 0, "docks": 5}, {"bikes": 2, "docks": 0}])
    assert UnchangedProbability("bike").predict(p).tolist() == [0, 1]
    assert UnchangedProbability("dock").predict(p).tolist() == [1, 0]


def test_count_table_is_the_share_seen_in_training():
    train = pairs_frame(
        [
            {"bikes": 1, "target_bikes": 0},
            {"bikes": 1, "target_bikes": 1},
            {"bikes": 1, "target_bikes": 2},
            {"bikes": 1, "target_bikes": 0},
            {"bikes": 30, "target_bikes": 30},
        ]
    )
    table = CountTable("bike", cap=15).fit(train)
    test = pairs_frame([{"bikes": 1}, {"bikes": 20}, {"bikes": 7}])
    # 1 bike now: 2 of 4 had a bike later. 20 is capped to 15, like 30 in
    # training. 7 was never seen and falls back to 100 %.
    assert table.predict(test).tolist() == [0.5, 1.0, 1.0]


def test_brier_and_log_loss_by_subset():
    p = pairs_frame(
        [
            {"bikes": 2, "target_bikes": 1},
            {"bikes": 9, "target_bikes": 0},
        ]
    )
    t = brier_table(p, {"m": np.array([0.8, 0.5])}, "bike").set_index("subset")
    assert t.loc["all", "brier"] == pytest.approx((0.2**2 + 0.5**2) / 2)
    assert t.loc["all", "log_loss"] == pytest.approx(-(np.log(0.8) + np.log(0.5)) / 2)
    # "few_now" is only the first pair (2 bikes now).
    assert t.loc["few_now", "n"] == 1
    assert t.loc["few_now", "brier"] == pytest.approx(0.04)


def test_calibration_bins():
    p = np.array([0.05, 0.08, 0.92, 0.95])
    truth = np.array([0.0, 1.0, 1.0, 1.0])
    c = calibration(p, truth, bins=10)
    assert c.loc[0, "observed"] == 0.5
    assert c.loc[9, "n"] == 2
    assert c.loc[9, "predicted"] == pytest.approx(0.935)


def test_classifier_gives_probabilities():
    grid = make_grid()
    pairs = make_pairs(grid, STATIONS)
    features = add_features(pairs[pairs["valid"]], grid, stations=["a", "b"])
    params = CLS_PARAMS | {"num_leaves": 4, "min_data_in_leaf": 5}
    booster, history = train_classifier(features, None, "bike", params, rounds=5)
    p = predict_proba(booster, features)
    assert ((p > 0) & (p < 1)).all()
    assert len(history["fit"]["binary_logloss"]) == 5
    assert "docks" in booster.feature_name()
