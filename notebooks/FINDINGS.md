# Findings from exploration (Phase 2)

Source: [01_exploration.ipynb](01_exploration.ipynb), data from 4 to 30 October 2026,
267 stations, 15-minute grid.

## Data quality

- **Gaps are small: 1.3 % of 15-minute slots.** The main one is a GitHub Actions
  outage on 5–6 October (21:05–23:32 and 00:45–07:14). Data before 4 October 18:00 is
  excluded, because GitHub's throttled schedule left gaps of 3–6 hours.
- **Quiet nights look like gaps but are not.** Between about 02:00 and 05:00 no station
  changes for up to 2.5 hours, so nothing is stored. The Actions run history confirms that
  the collector ran. Silence alone therefore cannot mark a gap; only silences over 3 hours
  and known outages do.
- **Some runs are duplicated** (the external trigger plus GitHub's fallback schedule, and
  manual runs): 3151 runs where 2585 were expected. The grid keeps the latest run per slot.
- **2 stations have not changed for over 3 days**, probably out of service.
- **E-bikes are a subset of `num_bikes_available`**, so regular bikes = all bikes − e-bikes.

## Patterns

- **Weekdays show a clear commuter pattern.** The number of parked bikes drops sharply
  around 07:00–09:00 (lowest around 08:15) and again around 16:00–17:00. Weekends are
  flatter, with one dip in the afternoon (lowest around 15:00).
- **The system total varies little, about 6 % (roughly 1770–1900 bikes).** At any time
  only around a hundred bikes are on trips. What matters for a single station is where
  bikes move, not how many are in use.
- **The fleet shrinks during the period:** the daily average fell from 1952 to 1841
  bikes, probably as the season winds down. Later test data has fewer bikes than earlier
  training data.
- **E-bikes are 33 % of parked bikes.**

## Differences between stations

- **Empty is a bigger problem than full.** Stations are empty 13.6 % of the time and
  full 4.4 % of the time.
- **76 stations are empty more than 20 % of the time, and 16 are full more than 20 % of
  the time.** Behaviour varies a lot between stations, so the station itself is an
  important feature.

## How much the numbers change (baseline for Phase 3)

| Horizon | Mean abs. change (bikes) | Median | 90th percentile | Unchanged |
|---------|--------------------------|--------|-----------------|-----------|
| 15 min  | 0.25 | 0 | 1 | 83 % |
| 30 min  | 0.44 | 0 | 1 | 73 % |
| 60 min  | 0.75 | 0 | 2 | 62 % |

- **The naive "unchanged from now" model is strong.** Its MAE is exactly the mean
  absolute change: 0.25, 0.44 and 0.75 bikes. Most slots do not change at all.
- **Docks change as much as bikes (identical figures at every horizon).** Bikes + docks
  is close to constant per station, so it should be enough to predict bikes and derive
  docks. Check this per station in Phase 3 before relying on it.
- **The error is concentrated in the rush hours.** At night almost nothing changes, which
  pulls the overall MAE down. A model has the most to gain on weekday mornings and
  afternoons.

## Limitations

- Only about 3.5 weeks of autumn data: around 19 weekdays and 8 weekend days. The weekend
  curve in particular is noisy.
- No summer data, and no weather yet.
- The changing fleet size means older data is not fully representative of newer data.
