"""Forecasts from current data, for the API (Phase 5).

    forecaster = Forecaster()                    # loads models/
    forecaster.predict(grid, stations)           # one row per station and horizon

`grid` is ml.grid.to_grid output that ends at the current slot and reaches at
least HISTORY_NEEDED (7 days + 1 hour) back; with less, the lag features are
missing and the forecasts get closer to "unchanged". Because the collector only
stores changes, build it from the rows in that window plus each station's
latest row before it, so forward fill has a starting value.
"""

import logging
from pathlib import Path

import pandas as pd

from ml.baselines import derive_docks
from ml.config import HORIZONS
from ml.features import HISTORY_NEEDED, add_features
from ml.grid import FREQ
from ml.model import MODEL_DIR, load, predict
from ml.pairs import ACTIVE_FLAGS, frozen_mask, local_time

log = logging.getLogger("predict")

OUTPUT_COLUMNS = [
    "station_id",
    "slot",
    "target_slot",
    "horizon",
    "bikes_now",
    "docks_now",
    "bikes",
    "docks",
    "reliable",
]


class Forecaster:
    def __init__(self, path: Path = MODEL_DIR):
        self.booster, self.meta = load(path)

    def predict(
        self, grid: pd.DataFrame, stations: pd.DataFrame, horizons: dict = HORIZONS
    ) -> pd.DataFrame:
        """Forecast bikes and docks at every station for each horizon.

        Origin is the last slot in `grid`. Stations without a current value
        (gap) or without a known capacity are left out. `reliable` is False
        for stations that are not installed, renting and returning, or have
        been frozen for 72 h; the model was neither trained nor evaluated on
        those, so the API should show them as unavailable.
        """
        g = grid.sort_values(["station_id", "slot"]).reset_index(drop=True)
        now = g["slot"].max()
        if g["slot"].min() > now - HISTORY_NEEDED:
            log.warning(
                "Less than %s of history; lag features are missing.", HISTORY_NEEDED
            )
        g["frozen"] = frozen_mask(g).to_numpy()

        current = g[g["slot"] == now].merge(
            stations[["station_id", "capacity"]], on="station_id", how="inner"
        )
        current["bikes"] = current["num_bikes_available"].astype("float64")
        current["docks"] = current["num_docks_available"].astype("float64")
        current = current.dropna(subset=["bikes", "docks", "capacity"])
        active = current[ACTIVE_FLAGS].astype("boolean").fillna(False).all(axis=1)
        current["reliable"] = active & ~current["frozen"]

        columns = ["station_id", "slot", "bikes", "docks", "capacity", "reliable"]
        rows = []
        for minutes, steps in horizons.items():
            r = current[columns].copy()
            r["horizon"] = minutes
            r["target_slot"] = r["slot"] + steps * pd.Timedelta(FREQ)
            rows.append(r)
        pairs = pd.concat(rows, ignore_index=True)
        pairs = pairs.join(local_time(pairs["slot"]))

        features = add_features(pairs, g, stations=self.meta["stations"])
        bikes = predict(self.booster, features)
        out = features.rename(columns={"bikes": "bikes_now", "docks": "docks_now"})
        out["bikes"] = bikes
        out["docks"] = derive_docks(features, bikes)
        return out[OUTPUT_COLUMNS].reset_index(drop=True)
