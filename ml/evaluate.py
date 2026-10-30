"""Mean absolute error per horizon, overall and for the hours that matter most."""

import numpy as np
import pandas as pd

from ml.config import DAYTIME_HOURS, RUSH_HOURS


def subset_masks(pairs: pd.DataFrame) -> dict[str, np.ndarray]:
    """Subsets by local time at the origin (needs `weekday` and `hour`)."""
    weekday = pairs["weekday"] < 5
    return {
        "all": np.ones(len(pairs), dtype=bool),
        "daytime": pairs["hour"].isin(DAYTIME_HOURS).to_numpy(),
        "weekday_rush": (weekday & pairs["hour"].isin(RUSH_HOURS)).to_numpy(),
    }


def mae_table(
    pairs: pd.DataFrame, predictions: dict[str, np.ndarray], target: str = "bikes"
) -> pd.DataFrame:
    """Long table: model, target, subset, horizon, mae, n."""
    truth = pairs[f"target_{target}"].to_numpy(dtype="float64")
    horizon = pairs["horizon"].to_numpy()
    rows = []
    for model, predicted in predictions.items():
        error = np.abs(np.asarray(predicted, dtype="float64") - truth)
        for subset, mask in subset_masks(pairs).items():
            for h in sorted(np.unique(horizon)):
                selected = mask & (horizon == h)
                rows.append(
                    {
                        "model": model,
                        "target": target,
                        "subset": subset,
                        "horizon": int(h),
                        "mae": float(error[selected].mean()),
                        "n": int(selected.sum()),
                    }
                )
    return pd.DataFrame(rows)


def to_wide(table: pd.DataFrame) -> pd.DataFrame:
    """One row per model and subset, one column per horizon."""
    wide = table.pivot_table(
        index=["target", "subset", "model"], columns="horizon", values="mae"
    )
    wide.columns = [f"{h} min" for h in wide.columns]
    return wide.reset_index()
