# How the results were produced: pipeline summary

Plain-language map of the analysis behind the thesis. Every step is a script in `analysis/`; the CSV it writes is the input to the next step.

## The one-sentence version
Daily MODIS snow maps are turned into "snow cover duration" (SCD, days of snow per hydrological year) for 3 mountain ranges x 3 elevation bands x forest/open, tested for trends, related to climate and streamflow, and checked against ground stations and Sentinel-2.

## Step by step

| # | What happens | Script | Output |
|---|---|---|---|
| 0 | Build the study zones: park boundaries (WDPA), elevation bands (Copernicus DEM, <800 / 800-1,200 / >1,200 m), forest mask (Hansen, year 2000). Combined into one zone-ID raster on the 500 m MODIS grid. | `gee/phase0_static_layers.py` | GEE asset (zone raster) |
| 1 | For every day, classify each MOD10A1 pixel as snow (NDSI > 0.4), no snow, or cloud (no valid retrieval). Sum snow days per pixel per hydrological year (Oct-Sep, 2001-2024). | `gee/phase1_scd_calculation.py` | per-pixel SCD, cloud %, onset, melt-out |
| 2 | Repeat under two more cloud treatments: **forward-fill** (carry last clear observation up to 5 days) and **Terra+Aqua** (either satellite sees snow). Original = **baseline** (clear-sky only). | `gee/phase2_cloud_variants.py` | three variants |
| 3 | Average pixels within each zone, per year, per variant. This is the hand-off from Earth Engine to Python (exported to Google Drive, downloaded as CSV). | `gee/phase3_zonal_export.py` | `data/exports/scd_zonal_<variant>.csv` |
| 4 | Mann-Kendall trend test and Theil-Sen slope per zone, per variant. | `python/phase4_trend_detection.py` | `data/trend_results.csv` |
| 5 | ERA5-Land winter temperature and precipitation per zone; correlate with SCD to see which dominates at each elevation. | `gee/phase5_era5_export.py`, `python/phase5_climate_drivers.py` | `data/climate_correlations.csv` |
| 6 | Correlate satellite melt-out timing with spring streamflow peak timing at CHMU gauges. | `python/chmu_download.py`, `python/phase6_hydrology_correlation.py` | `data/hydrology_correlations.csv` |
| 7a | **Station check:** CHMU snow depth -> station SCD (days > 1 cm) vs the MODIS zone value, for all three variants. | `python/phase7_validation.py` | `data/validation_results.csv`, `validation_summary_by_variant.csv` (superseded for the station comparison -- see 7a' below; `summarize_canopy_validation()` in this script is still authoritative) |
| 7a' | **Station check, pixel-level (supervisor follow-up, 2026-10-19):** same comparison as 7a, but each station is matched to its own MODIS pixel's daily series instead of its zone's mean SCD. **This is what thesis Section 6.6 / Figure 6.3 report.** | `gee/phase7_station_daily_pixels.py`, `gee/phase7_station_daily_pixels_aqua.py`, `python/phase7_validation_pixel.py` | `data/validation_results_pixel.csv`, `validation_summary_by_variant_pixel.csv`, `figures/validation_scatter.png` |
| 7b | **Cloud-matched check:** sample the MODIS pixel over each station every day; compare with the station only on days MODIS could see the ground. | `gee/phase7_station_daily_pixels.py`, `python/phase7_cloud_matched.py` | `data/cloud_matched_*.csv` |
| 7c | **Canopy check:** Sentinel-2 (20 m) vs MODIS (500 m) snow on selected clear winter dates in Sumava, forest vs open. | `gee/phase7_sentinel_validation.py` | `data/canopy_validation_summary.csv` |
| 8 | **Robustness and inference checks:** Benjamini-Hochberg and Bonferroni counts, Sen-slope 95% confidence intervals, climate correlations on detrended series, a permutation field-significance test across strata, trend tests on the three range-mean series, sensitivity to trimming start and end years, and station bias as the forward-fill limit varies from 0 to 30 days. | `python/phase8_robustness_checks.py` | `data/robustness_*.csv` |
| 9 | Figures and tables for the thesis. | `python/make_*.py`, `insert_figures_tables.py` | `analysis/figures/` |

Raw inputs: MODIS MOD10A1 / MYD10A1, Copernicus DEM, Hansen forest, ERA5-Land, Sentinel-2 (all from Google Earth Engine); CHMU snow depth and streamflow from opendata.chmi.cz (CC BY 4.0).

## Choices that are judgment calls (defend these)
- NDSI threshold 0.4; snow-depth threshold 1 cm for the station proxy.
- Baseline treats a cloudy day as "no snow", so baseline SCD is a lower bound.
- Forward-fill capped at 5 days.
- Elevation bands at 800 and 1,200 m.
- One station and one gauge per range, chosen for data availability. Two originally chosen stations were replaced because they were discontinued (1995, 1997); this should be stated in the Methods.
- Station validation: matched to each station's own MODIS pixel (`phase7_validation_pixel.py`), not the mean SCD of its containing zone (`phase7_validation.py`'s original method) -- changed 2026-10-19 following a supervisor comment on Section 6.6. The zone-mean output is kept for provenance but is not what the thesis reports.
- Cloud is identified by a missing NDSI value, so any other retrieval gap is counted as cloud.

## Known limits of the outputs
- Per-pixel significance on the trend map is not shown: GEE's per-pixel Kendall p-value was unusable, so significance is assessed at zone level.
- Station validation is three stations; only one is in a forested zone.
- Clear days are not a random sample (snowfall days are cloudier).

## How to check it yourself
1. Pick one pixel and one hydrological year in the Earth Engine Code Editor, count NDSI > 40 days by hand, compare with the pipeline value.
2. Download one CHMU station file, count days with depth > 1 cm for one year in Excel, compare with `validation_results.csv`.
3. Recompute Mann-Kendall for one zone in another tool (for example R `trend`) and compare with `trend_results.csv`.
4. Rerun the scripts from an empty `data/` folder and confirm the same tables come out.
