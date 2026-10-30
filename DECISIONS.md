# Design decisions

A log of significant choices made in this project, with a short reason for each.
New entries are added at the bottom. Earlier decisions are not edited – if a decision changes, a new entry explains why.

| Date | Decision | Reason |
|------|----------|--------|
| 2026-10-01 | Public GitHub repository | Portfolio visibility, and GitHub Actions minutes are free and unlimited for public repositories. |
| 2026-10-01 | MIT license | Short, permissive and the most widely used license on GitHub. Apache 2.0's explicit patent grant is not needed for this project. |
| 2026-10-01 | uv with pyproject.toml and uv.lock | The lockfile makes environments reproducible locally and in CI, installs are fast, and pyproject.toml is the Python packaging standard. |
| 2026-10-01 | Python 3.13, pinned in .python-version | Stable and supported by all planned libraries. Pinning the minor version allows patch updates but prevents accidental upgrades. |
| 2026-10-01 | README.md in each folder instead of .gitkeep | Git does not track empty directories. A README solves this while documenting what each folder is for. |
| 2026-10-01 | Hand-written .gitignore instead of GitHub's template | Shorter and fully explainable. Includes `*.env` as an extra safeguard against committing secrets. |
| 2026-10-01 | LF line endings enforced via .gitattributes | Consistent line endings across Windows development and Linux-based CI, regardless of each developer's Git settings. |
| 2026-10-01 | pytest and ruff set up from the start | Cheap to adopt before any code exists, and ensures consistent formatting and a working test setup for CI. |
| 2026-10-01 | English for code, file names, commits and docs | Industry standard and accessible to international colleagues and reviewers. |
| 2026-10-01 | Deferred: docs/ folder, pre-commit hooks, Conventional Commits | Not needed yet (YAGNI). Will be added if and when the project requires them. |
| 2026-10-01 | GBFS Client-Identifier: `lucajuul-bysykkelforecast`, stored as env var BYSYKKEL_CLIENT_ID | Oslo Bysykkel requires a Client-Identifier header in the format name-app. Not a secret, but kept in config so it can change without code changes. Data is licensed under NLOD 2.0 (attribution required in README). Feed URLs are hardcoded instead of read from gbfs.json, for simplicity. |
| 2026-10-04 | Trigger the collector from cron-job.org via the GitHub API (workflow_dispatch) every 15 min; keep `schedule` as a fallback | GitHub's scheduled runs were heavily throttled: measured gaps of 2–6 hours instead of 15 minutes, too sparse for 15-minute predictions. An external cron calling workflow_dispatch is free and needs no code changes. It uses a fine-grained token scoped to this repo with Actions read/write only, expiring 2027-01-02. Dispatch-triggered runs are also not disabled after 60 days of repo inactivity. Rejected: accepting the gaps (poor data), running locally (laptop must stay on), moving to another platform (rewrite). |
| 2026-10-30 | Export the database once to local Parquet files in data/ (`ml/export_data.py`); all analysis reads the files | Neon's free compute is close to its monthly limit. One full export costs a few seconds of compute; repeated notebook queries would cost far more. Parquet keeps column types (timestamps with time zone) and is fast to read. Full re-export instead of incremental, for simplicity – the data is small. |
| 2026-10-30 | Analysis starts 2026-10-04 18:00 Oslo time | Before cron-job.org took over, GitHub's throttled schedule left gaps of 3–6 hours. |
| 2026-10-30 | Regular 15-minute grid per station with forward fill (`ml/grid.py`) | Only changed rows are stored, so a missing row means "unchanged" and forward fill recovers the true value. Runs are floored to their 15-minute slot; if several runs share a slot, the latest wins. Times are kept in UTC to avoid daylight saving issues. |
| 2026-10-30 | Gaps: known outages listed in code, plus any silence over 3 hours; gap slots get `is_gap = True` and missing values | Quiet nights produce 1.5–2.5 hours with no changed rows although the collector ran (verified in the Actions run history), so silence alone is not a gap. Values inside gaps are set to missing so they cannot leak into training or evaluation. Values after a gap are still carried forward, because the collector compares against the last stored row. |
| 2026-10-30 | Analysis libraries (pandas, pyarrow, matplotlib, ipykernel) in the dev dependency group | The collector workflow installs with `--no-dev`, so it stays small and fast. |
| 2026-10-30 | Deferred: `collector_runs` table logging every run | Would make gap detection exact, but the Actions run history and the 3-hour rule are good enough for this project. |
