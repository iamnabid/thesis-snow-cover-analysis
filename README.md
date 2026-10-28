# Code and data: snow cover duration in the Krkonoše, Šumava and Hrubý Jeseník mountains (2000-2024)

This package contains the code, intermediate data and figures behind the thesis. See `PIPELINE_SUMMARY.md` for a plain-language walkthrough of every step.

## Layout

```
gee/        Google Earth Engine scripts (Python API)
python/     Local analysis scripts (statistics, validation, figures)
data/       Result tables (CSV) and the intermediate data they are computed from
  exports/  Data handed between Earth Engine and Python, plus CHMU station and streamflow data
  raw_gee_exports_before_reshape/   Original Earth Engine CSV exports, before python/reshape_gee_csv.py
figures/    Figures used in the thesis
requirements.txt
```

## Two ways to use it

**A. Check the results without Earth Engine (about a minute).**
All Earth Engine outputs are already in `data/exports/`, so the Python side reruns from them:

```
pip install -r requirements.txt
cd python
python phase4_trend_detection.py          # -> data/trend_results.csv
python phase5_climate_drivers.py          # -> data/climate_correlations.csv
python phase6_hydrology_correlation.py    # -> data/hydrology_correlations.csv
python phase7_validation.py               # -> data/canopy_validation_summary.csv, figures/canopy_underdetection.png
                                          #    (also writes data/validation_results.csv, validation_summary_by_variant.csv
                                          #    and an earlier figures/validation_scatter.png -- SUPERSEDED, see below)
python phase7_validation_pixel.py         # -> data/validation_results_pixel.csv, validation_summary_by_variant_pixel.csv,
                                          #    figures/validation_scatter.png -- THIS is the station comparison and
                                          #    figure actually reported in thesis Section 6.6 / Figure 6.3
python phase7_cloud_matched.py            # -> data/cloud_matched_summary.csv, cloud_matched_by_year.csv
python phase8_robustness_checks.py        # -> data/robustness_*.csv (multiplicity, Sen-slope CIs, detrended
                                          #    climate correlations, field significance, range means,
                                          #    endpoint sensitivity, forward-fill limit); about a minute
```

These reproduced the tables in `data/` exactly when rerun from a clean checkout of this package (September 2026; the pixel-level station comparison added October 2026, see below).

**Two station-comparison methods exist; the pixel-level one is authoritative.** `phase7_validation.py`'s `run_station_comparison()` joins each station to the *mean SCD of its containing zone*. Following a supervisor comment on Section 6.6 (2026-10-19) asking for the comparison to be recalculated at *the MODIS pixel containing each station* instead, `phase7_validation_pixel.py` was written to do exactly that, using each station's own daily pixel series (`station_modis_daily.csv`, `station_modis_daily_aqua.csv`) rather than the zone mean. **`phase7_validation_pixel.py`'s output is what Section 6.6 and Figure 6.3 report.** `phase7_validation.py`'s zone-mean station-comparison output (`validation_results.csv`, `validation_summary_by_variant.csv`) is kept for comparison/provenance but is superseded for that purpose; its `summarize_canopy_validation()` function (canopy/Sentinel-2 check) is unaffected and still authoritative. Both scripts write to `figures/validation_scatter.png` — run `phase7_validation_pixel.py` *after* `phase7_validation.py` (as ordered above) so the correct, pixel-level figure is the one left on disk.

**B. Regenerate everything from raw sources.**
Needs a Google Earth Engine account and your own Cloud project (the scripts contain `ee.Initialize(project="thesis101-492500")`; replace it with your project ID in each `gee/*.py` file).

1. `python/chmu_download.py` downloads CHMU station snow depth and streamflow to `data/exports/`.
2. `gee/phase0_static_layers.py` builds the zone raster (park boundaries, elevation bands, forest mask) and exports it as an Earth Engine asset. Set `GEE_ASSET_ID` in that file to an asset path in your own project.
3. `gee/phase1_scd_calculation.py`, `phase2_cloud_variants.py` and `phase3_zonal_export.py` compute snow cover duration for the three cloud treatments and export one CSV per variant to Google Drive. Download them to `data/exports/` (names: `scd_zonal_baseline.csv`, `scd_zonal_forward_fill.csv`, `scd_zonal_terra_aqua.csv`).
4. `gee/phase5_era5_export.py` exports winter ERA5-Land temperature and precipitation (`era5_land_winter.csv`).
5. `gee/phase7_station_zones.py` assigns each station to a zone; `gee/phase7_station_daily_pixels.py` writes `station_modis_daily.csv` (Terra) and `gee/phase7_station_daily_pixels_aqua.py` writes `station_modis_daily_aqua.csv` (Aqua) -- both feed `phase7_validation_pixel.py`; `gee/phase7_sentinel_validation.py` exports the Sentinel-2 canopy comparison.
6. Run the Python scripts in option A.
7. `python/make_location_map.py` and `python/make_trend_map.py` draw the study-area map and the trend map.

Earth Engine exports are asynchronous: start each export, wait for it to finish in the Tasks tab, and download the result from Drive. Earth Engine datasets are occasionally reprocessed, so a fresh run may differ slightly from the archived tables.

## Data sources

| Data | Source |
|---|---|
| Daily snow cover (NDSI) | MODIS MOD10A1 and MYD10A1 v6.1, via Earth Engine (`MODIS/061/MOD10A1`, `MYD10A1`) |
| Elevation | Copernicus GLO-30 DEM |
| Forest cover | Hansen Global Forest Change, year-2000 baseline |
| Protected-area boundaries | WDPA (`WCMC/WDPA/current/polygons`) |
| Winter temperature and precipitation | ERA5-Land daily aggregates (`ECMWF/ERA5_LAND/DAILY_AGGR`) |
| Snow depth and streamflow | Czech Hydrometeorological Institute (ČHMÚ) open data, opendata.chmi.cz, CC BY 4.0 |
| Canopy check | Sentinel-2 surface reflectance (`COPERNICUS/S2_SR_HARMONIZED`) |

## Key definitions

- Hydrological year: 1 October to 30 September, named by its ending year (2001 = Oct 2000 - Sep 2001).
- Snow: NDSI > 0.4 on a day with a valid retrieval. Cloud: no valid NDSI retrieval. Baseline SCD counts only snow seen on clear days.
- Station SCD: number of days in the hydrological year with snow depth above 1 cm.
- Zones: 3 ranges x 3 elevation bands (<800, 800-1,200, >1,200 m) x forest/open. Zone IDs are decoded in `gee/phase3_zonal_export.py`.

## Not included

The scripts that edit the thesis document, backups and the Python virtual environment are not part of the analysis and are omitted.

## Environment

Python 3.14, packages in `requirements.txt` (versions used are pinned there).
