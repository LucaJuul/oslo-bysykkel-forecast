import numpy as np
import pandas as pd
import pytest

from ml.evaluate import mae_table, subset_masks
from ml.pairs import local_time


def make_pairs(slots: list[str], targets: list[float], horizon=15) -> pd.DataFrame:
    pairs = pd.DataFrame(
        {
            "slot": pd.to_datetime(slots).tz_localize("Europe/Oslo").tz_convert("UTC"),
            "target_bikes": targets,
            "horizon": horizon,
        }
    )
    return pairs.join(local_time(pairs["slot"]))


def test_subsets_use_local_time_and_weekdays():
    pairs = make_pairs(
        [
            "2026-10-26 07:30",  # Monday morning rush (winter time)
            "2026-10-26 12:00",  # Monday midday: daytime, not rush
            "2026-10-24 08:00",  # Saturday: daytime, not weekday rush
            "2026-10-26 03:00",  # night
        ],
        [0, 0, 0, 0],
    )
    masks = subset_masks(pairs)
    assert masks["all"].tolist() == [True, True, True, True]
    assert masks["daytime"].tolist() == [True, True, True, False]
    assert masks["weekday_rush"].tolist() == [True, False, False, False]


def test_mae_per_model_subset_and_horizon():
    pairs = pd.concat(
        [
            make_pairs(["2026-10-26 07:30", "2026-10-26 03:00"], [4, 2], horizon=15),
            make_pairs(["2026-10-26 07:30"], [6], horizon=60),
        ],
        ignore_index=True,
    )
    table = mae_table(pairs, {"m": np.array([3, 2, 2])})
    mae = table.set_index(["subset", "horizon"])["mae"]
    assert mae[("all", 15)] == pytest.approx(0.5)  # errors 1 and 0
    assert mae[("weekday_rush", 15)] == pytest.approx(1.0)
    assert mae[("all", 60)] == pytest.approx(4.0)
    assert table.set_index(["subset", "horizon"])["n"][("all", 15)] == 2
