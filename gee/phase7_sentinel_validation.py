"""
Phase 7 (GEE side) -- Sentinel-2 canopy-underdetection spot-check

Compares MOD10A1's 500 m NDSI>0.4 snow classification against a Sentinel-2
NDSI>0.4 classification at native ~20 m resolution, aggregated up to the
MODIS grid, for a curated set of clear-sky winter dates over Sumava --
specifically to test whether MODIS under-detects snow under forest canopy
relative to Sentinel-2's finer resolution (thesis Methodology 5.7,
Objective 6).

Same NDSI formula and >0.4 threshold as Phase 1's classify_daily, using
Sentinel-2's B3 (green, 10 m) and B11 (SWIR1, 20 m) -- deliberately not
Sentinel-2's own SCL/MSK_SNWPRB classification, so the comparison isolates
the effect of resolution, not a difference in classification algorithm.

Verified 2026-09-18 directly against the live COPERNICUS/S2_SR_HARMONIZED
asset: B3/B11/QA60 band names and their 10 m/20 m scales are real, and 90
winter (Dec-Feb) scenes with <10% cloud exist over Sumava since 2015 (this
thesis scopes the check to 2017+ per Methodology 5.7, even though earlier
scenes exist in the catalog).

Output schema (one row per date x forest-class):
    date, forest (0=open, 1=forested), s2_snow_fraction, modis_snow, modis_miss
"""
import ee

from phase0_static_layers import MODIS_SCALE, build_forest_mask, load_aois
from phase1_scd_calculation import NDSI_THRESHOLD, classify_daily

ee.Initialize(project="thesis101-492500")

SENTINEL2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
VALIDATION_START_YEAR = 2017  # thesis Methodology 5.7: Sentinel-2 spot-checks "from 2017"
MODIS_MISS_THRESHOLD = 0.3  # S2 sub-pixel snow fraction above which a MODIS "no snow" pixel counts as a miss

# Picked 2026-09-18 from a real find_candidate_dates() run (93 candidates,
# Nov-May, <10% cloud, 2017-2024) -- deliberately mixes deep-midwinter (full
# cover, canopy effect expected to be small per the earlier 2019-01-19 vs.
# 2018-01-06 spot check) with spring/autumn shoulder-season dates (thin/
# patchy snow, where Parajka et al. 2012 predicts canopy underdetection
# should actually show up), spread across 7 different hydro-years for some
# interannual spread rather than clustering in one or two winters.
CURATED_DATES = [
    "2018-01-06",  # midwinter
    "2019-01-19",  # midwinter
    "2020-01-06",  # midwinter
    "2024-01-10",  # midwinter
    "2017-04-09",  # spring shoulder (melt-out)
    "2019-04-16",  # spring shoulder
    "2021-04-23",  # spring shoulder
    "2022-05-15",  # late spring shoulder
    "2019-11-07",  # autumn shoulder (onset)
    "2021-11-06",  # autumn shoulder
]


def find_candidate_dates(aoi: ee.Geometry, max_cloud_pct: float = 10) -> list:
    """Diagnostic: lists Nov-May dates from VALIDATION_START_YEAR onward with
    low cloud cover over the AOI. A single date can correspond to multiple
    UTM-tile scenes (Sumava straddles a tile boundary) --
    sentinel_snow_fraction_on_modis_grid() mosaics same-date tiles together,
    so the caller only needs one date string per candidate, not one per tile.

    Widened from an earlier Dec-Feb-only version: canopy underdetection is
    expected to show up most under thin/patchy snow (Parajka et al. 2012),
    which is a shoulder-season (Nov, Mar-May) condition, not deep midwinter
    full cover -- a Dec-Feb-only candidate list structurally couldn't include
    the dates most likely to actually show the effect being tested for.
    """
    s2 = (
        ee.ImageCollection(SENTINEL2_COLLECTION)
        .filterBounds(aoi)
        .filterDate(f"{VALIDATION_START_YEAR}-01-01", "2024-10-01")
        .filter(ee.Filter.calendarRange(11, 5, "month"))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
        .map(lambda img: img.set("date_str", img.date().format("YYYY-MM-dd")))
    )
    date_list = s2.aggregate_array("date_str").getInfo()
    return sorted(set(date_list))


def classify_sentinel(image: ee.Image) -> ee.Image:
    """Same NDSI formula/threshold as Phase 1's MODIS classify_daily, applied
    to Sentinel-2's B3 (10 m) and B11 (20 m). B3 is explicitly resampled onto
    B11's 20 m grid before the arithmetic -- combining bands of different
    native scale without an explicit reproject/resample is exactly the kind
    of ambiguity that produced silent bugs earlier in this pipeline (see
    Phase 0's WDPA/DEM fixes), so it's made explicit here rather than relying
    on GEE's implicit default-projection behavior. QA60 bits 10/11
    (opaque/cirrus cloud) are the standard, well-documented S2 cloud flags.
    """
    swir = image.select("B11")
    green_20m = image.select("B3").resample("bilinear").reproject(swir.projection())
    ndsi = green_20m.subtract(swir).divide(green_20m.add(swir))

    qa60 = image.select("QA60").toInt()
    is_cloud = qa60.bitwiseAnd(1 << 10).neq(0).Or(qa60.bitwiseAnd(1 << 11).neq(0))
    is_snow = ndsi.gt(NDSI_THRESHOLD).And(is_cloud.Not())
    return is_snow.rename("snow_s2").unmask(0).set("system:time_start", image.get("system:time_start"))


def sentinel_snow_fraction_on_modis_grid(date_str: str, aoi: ee.Geometry) -> ee.Image:
    """Mosaics same-date Sentinel-2 tiles over aoi, classifies snow at native
    20 m resolution, then aggregates up to the MODIS 500 m grid as a
    fraction (mean of the binary mask) -- directly comparable to MODIS's own
    binary classification for the same day."""
    start = ee.Date(date_str)
    end = start.advance(1, "day")
    s2_day = ee.ImageCollection(SENTINEL2_COLLECTION).filterBounds(aoi).filterDate(start, end)
    classified_mosaic = s2_day.map(classify_sentinel).mosaic()

    snow_fraction = (
        classified_mosaic.reproject(crs="SR-ORG:6974", scale=20)
        .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)
        .reproject(crs="SR-ORG:6974", scale=MODIS_SCALE)
    )
    return snow_fraction.rename("s2_snow_fraction").clip(aoi)


def modis_snow_on_date(date_str: str, aoi: ee.Geometry) -> ee.Image:
    """MODIS's own binary snow classification (Phase 1's classify_daily,
    reused directly) for the same date, at native 500 m."""
    start = ee.Date(date_str)
    end = start.advance(1, "day")
    mod = ee.ImageCollection("MODIS/061/MOD10A1").filterBounds(aoi).filterDate(start, end)
    classified = mod.map(classify_daily).select("snow").mosaic()
    return classified.rename("modis_snow").clip(aoi)


def zonal_stats_for_date(date_str: str, aoi: ee.Geometry, forest_mask: ee.Image) -> ee.FeatureCollection:
    """Forest-vs-open comparison for one date: mean S2 snow fraction, mean
    MODIS binary snow rate, and the canopy-underdetection signature itself
    (modis_miss: MODIS says no-snow but S2 shows well over
    MODIS_MISS_THRESHOLD sub-pixel snow), grouped by Hansen forest mask.

    Uses the same Reducer.mean().repeat(n).group() pattern worked out in
    Phase 3 (group-key band must come after the value bands, and
    Reducer.mean().group() needs .repeat(n) to average more than one band) --
    including the same unpacking fix: repeat(n).group() returns each group as
    {'forest': v, 'mean': [v0, v1, v2]}, a single 'mean' list in band order,
    not one named key per band. Re-expanded against the value bands' own
    names before building each Feature so the exported CSV has real
    s2_snow_fraction/modis_snow/modis_miss columns instead of one 'mean'
    column holding a stringified list (see phase3_zonal_export.
    zonal_means_for_image()'s docstring for how this was found).
    """
    s2_frac = sentinel_snow_fraction_on_modis_grid(date_str, aoi)
    modis_snow = modis_snow_on_date(date_str, aoi)
    modis_miss = modis_snow.eq(0).And(s2_frac.gt(MODIS_MISS_THRESHOLD)).rename("modis_miss")

    value_bands = ee.Image.cat([s2_frac, modis_snow, modis_miss]).toFloat()
    value_band_names = value_bands.bandNames()
    combined = value_bands.addBands(forest_mask.rename("forest"))
    result = combined.reduceRegion(
        reducer=ee.Reducer.mean().repeat(3).group(groupField=3, groupName="forest"),
        geometry=aoi,
        scale=MODIS_SCALE,
        maxPixels=1e9,
    )
    groups = ee.List(result.get("groups"))

    def unpack(g):
        g = ee.Dictionary(g)
        means = ee.List(g.get("mean"))
        props = (
            ee.Dictionary.fromLists(value_band_names, means)
            .set("forest", g.get("forest"))
            .set("date", date_str)
        )
        return ee.Feature(None, props)

    return ee.FeatureCollection(groups.map(unpack))


def build_validation_feature_collection(
    aoi: ee.Geometry, forest_mask: ee.Image, dates: list
) -> ee.FeatureCollection:
    features = [zonal_stats_for_date(d, aoi, forest_mask) for d in dates]
    merged = features[0]
    for fc in features[1:]:
        merged = merged.merge(fc)
    return merged


if __name__ == "__main__":
    aoi_fc = load_aois()
    sumava = aoi_fc.filter(ee.Filter.eq("range", "sumava")).geometry()
    forest_mask = build_forest_mask(sumava)

    if not CURATED_DATES:
        dates = find_candidate_dates(sumava)
        print(f"Found {len(dates)} candidate low-cloud winter dates from {VALIDATION_START_YEAR}:")
        for d in dates:
            print(" ", d)
        print(
            "\nCURATED_DATES is empty -- pick a representative subset (e.g. one "
            "clear midwinter date per hydro-year) from the list above, fill it "
            "in, then re-run this script to export the comparison."
        )
    else:
        fc = build_validation_feature_collection(sumava, forest_mask, CURATED_DATES)
        task = ee.batch.Export.table.toDrive(
            collection=fc,
            description="sentinel_canopy_validation",
            folder="thesis_scd_exports",
            fileFormat="CSV",
        )
        task.start()
        print("Export task started -- check https://code.earthengine.google.com/tasks")
        print(
            "Once complete, download sentinel_canopy_validation.csv from Drive "
            "into data/exports/ for python/phase7_validation.py's canopy summary."
        )
