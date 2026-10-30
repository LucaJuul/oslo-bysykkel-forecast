-- Monitoring queries. Run in the Neon SQL Editor.

-- 1. How long since the last successful run? Should be under ~30 minutes.
SELECT now() - max(fetched_at) AS since_last_run FROM station_status;

-- 2. Runs and saved rows per day (Oslo time).
SELECT date_trunc('day', fetched_at AT TIME ZONE 'Europe/Oslo') AS day,
       count(DISTINCT fetched_at) AS runs,
       count(*) AS rows_saved
FROM station_status
GROUP BY day
ORDER BY day DESC;

-- 3. Database and table sizes.
SELECT pg_size_pretty(pg_database_size(current_database())) AS database,
       pg_size_pretty(pg_total_relation_size('station_status')) AS station_status,
       pg_size_pretty(pg_total_relation_size('stations')) AS stations;

-- 4. Time between runs. Gaps should be around 15 minutes.
SELECT fetched_at AT TIME ZONE 'Europe/Oslo' AS oslo_time,
       fetched_at - lag(fetched_at) OVER (ORDER BY fetched_at) AS gap
FROM (SELECT DISTINCT fetched_at FROM station_status) AS runs
ORDER BY fetched_at DESC
LIMIT 20;

-- 5. All gaps longer than 25 minutes between visible runs (Oslo time), longest first.
-- Note: a run that saved no changed rows is not visible, so a gap here means
-- "no run, or runs where no station changed".
WITH runs AS (
    SELECT DISTINCT fetched_at FROM station_status
), gaps AS (
    SELECT lag(fetched_at) OVER (ORDER BY fetched_at) AS gap_start,
           fetched_at AS gap_end
    FROM runs
)
SELECT gap_start AT TIME ZONE 'Europe/Oslo' AS gap_start_oslo,
       gap_end   AT TIME ZONE 'Europe/Oslo' AS gap_end_oslo,
       gap_end - gap_start AS duration,
       round(extract(epoch FROM gap_end - gap_start) / 900) - 1 AS missed_runs
FROM gaps
WHERE gap_end - gap_start > interval '25 minutes'
ORDER BY duration DESC;

-- 6. Coverage summary: visible runs vs. expected runs since collection started.
SELECT min(fetched_at) AT TIME ZONE 'Europe/Oslo' AS first_run,
       max(fetched_at) AT TIME ZONE 'Europe/Oslo' AS last_run,
       count(DISTINCT fetched_at) AS visible_runs,
       floor(extract(epoch FROM max(fetched_at) - min(fetched_at)) / 900) + 1
           AS expected_runs
FROM station_status;
