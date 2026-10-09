"""
Phase 5 (GEE side) -- ERA5-Land winter climate zonal export

Produces the era5_land_winter.csv that python/phase5_climate_drivers.py
needs: winter (DJF) mean 2m air temperature and total precipitation per zone
per hydro-year, using the same zone raster and zonal-aggregation pattern as
Phase 3.

Band names (temperature_2m in Kelvin, total_precipitation_sum in metres)
verified directly against the live ECMWF/ERA5_LAND/DAILY_AGGR asset
2026-09-18, not assumed from docs -- prior phases in this pipeline have
repeatedly found GEE-hosted assets not matching their own catalog pages.

Thesis reference: Objective 4 (2.1), Section 3.8 (ERA5-Land as supporting data).

Output schema (one row per zone x hydro_year):
    zone_id, range_id, elevation_band, forest, hydro_year,
    winter_temp (degC), winter_precip (mm)
"""
import ee

from phase0_static_layers import GEE_ASSET_ID, load_aois
from phase1_scd_calculation import HYDRO_YEARS
from phase3_zonal_export import prep_zone_bands, zonal_means_for_image

ee.Initialize(project="thesis101-492500")

ERA5_COLLECTION = "ECMWF/ERA5_LAND/DAILY_AGGR"


def djf_bounds(end_year: int):
    """DJF winter window for hydro-year end_year: Dec of end_year-1 through
    end of Feb of end_year (exclusive upper bound)."""
    start = ee.Date.fromYMD(end_year - 1, 12, 1)
    end = ee.Date.fromYMD(end_year, 3, 1)
    return start, end


def winter_climate_for_year(era5: ee.ImageCollection, end_year: int) -> ee.Image:
    """Mean DJF temperature (degC) and total DJF precipitation (mm) for one
    hydro-year, per pixel."""
    start, end = djf_bounds(end_year)
    winter = era5.filterDate(start, end)

    winter_temp = (
        winter.select("temperature_2m").mean().subtract(273.15).rename("winter_temp")
    )
    winter_precip = (
        winter.select("total_precipitation_sum").sum().multiply(1000).rename("winter_precip")
    )
    return winter_temp.addBands(winter_precip).set("hydro_year", end_year)


def build_climate_feature_collection(
    zone_bands: ee.Image, aoi: ee.Geometry, era5: ee.ImageCollection
) -> ee.FeatureCollection:
    features = []
    for end_year in HYDRO_YEARS:
        img = winter_climate_for_year(era5, end_year)
        zonal_fc = zonal_means_for_image(zone_bands, img, aoi, n_data_bands=2)
        tagged = zonal_fc.map(lambda f, yr=end_year: f.set("hydro_year", yr))
        features.append(tagged)

    merged = features[0]
    for fc in features[1:]:
        merged = merged.merge(fc)
    return merged


if __name__ == "__main__":
    aoi_fc = load_aois()
    aoi = aoi_fc.geometry()
    zone_raster = ee.Image(GEE_ASSET_ID)
    zone_bands = prep_zone_bands(zone_raster)

    era5 = ee.ImageCollection(ERA5_COLLECTION).filterBounds(aoi)

    fc = build_climate_feature_collection(zone_bands, aoi, era5)
    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description="era5_land_winter",
        folder="thesis_scd_exports",
        fileFormat="CSV",
    )
    task.start()
    print("Export task started -- check https://code.earthengine.google.com/tasks")
    print("Once complete, download era5_land_winter.csv from Drive into data/exports/ for Phase 5.")
