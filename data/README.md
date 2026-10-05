# Data

`dji_gaf_analysis.py` reads daily closing values of the Dow Jones Industrial Average (DJI).

Default path: `data/dji_sp_1900_2026.csv`. To use another path, pass `--data path/to/file.csv`.

## Required columns

| Column | Format | Notes |
|---|---|---|
| `Date` | `MM/DD/YYYY` | Trading days only |
| `DJI` | float | Daily closing value |

Other columns (for example `SPX`) are ignored. Rows may be in any order, since the script sorts them chronologically. Spaces around column names are stripped.

Log-returns are computed by the script as `log(DJI_t / DJI_{t-1})`.

## Notes

- The figures in the report use data from 1900 to 2026. The 1929 figures need at least 1920–1940.
- The Fig. 6 window start dates (`FIG6_START_DATES` in `dji_gaf_analysis.py`) depend on the exact trading-day calendar of the file. If your file has a different calendar, edit that list to start dates that exist in your windows.
