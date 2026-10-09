"""
Phase 3 -- Zonal aggregation and CSV export

Reduces the per-pixel SCD/cloud%/onset/melt-out stacks (Phases 1-2) to
per-zone, per-hydro-year means, using the zone-ID raster from Phase 0.
Exports one tidy CSV per variant -- the single handoff point between GEE and
the Python side of the pipeline (Phases 4-7).

Output schema (one row per zone x hydro_year, one CSV per variant):
    zone_id, range_id, elevation_band, forest, hydro_year, variant,
    mean_scd, cloud_pct, mean_onset_day_idx, mean_meltout_day_idx

mean_onset_day_idx / mean_meltout_day_idx are NOT calendar day-of-year --
see phase1_scd_calculation.doy_to_hydro_day_index()'s docstring for why: a
run spanning Dec-Jan wraps calendar DOY (onset_doy > meltout_doy for that
pixel) purely from the Jan-1 reset, which would silently distort a zonal
mean that mixes wrapping and non-wrapping pixels. Each pixel's onset/melt-out
is converted to a day count since Oct 1 of that hydro-year (0 = Oct 1,
monotonically increasing through Sep 30) BEFORE the zonal mean is taken, so
the averaging itself never crosses the wrap point.

Thesis reference: Methodology 5.1 (zones), 5.4 (the 9 primary time series).
"""
import sys
from pathlib import Path

import ee

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zone_ids import ZONE_ID_DECODE  # noqa: E402

from phase0_static_layers import GEE_ASSET_ID, load_aois
from phase1_scd_calculation import HYDRO_YEARS, doy_to_hydro_day_index
from phase2_cloud_variants import (
    AQUA_START_YEAR,
    variant_a_baseline,
    variant_b_forward_fill,
    variant_c_terra_aqua,
)

ee.Initialize(project="thesis101-492500")

# zone_id = range_id * 100 + elevation_band * 10 + forest (see Phase 0's
# build_zone_raster). Decoded back into separate columns in the exported CSV
# itself (see prep_zone_bands); ZONE_ID_DECODE itself now lives in
# zone_ids.py so the Python-side phases (6, 7) can read the same range/band/
# forest names without importing `ee`.


def prep_data_bands(data_img: ee.Image, end_year: int) -> ee.Image:
    """Renames Phase 1/2's raw band names to the exported CSV's final column
    names, converting onset/meltout from calendar DOY to hydro-year day index
    in the process (see module docstring)."""
    mean_scd = data_img.select("scd").rename("mean_scd")
    cloud_pct = data_img.select("cloud_pct")
    onset_idx = doy_to_hydro_day_index(data_img.select("onset_doy"), end_year).rename(
        "mean_onset_day_idx"
    )
    meltout_idx = doy_to_hydro_day_index(data_img.select("meltout_doy"), end_year).rename(
        "mean_meltout_day_idx"
    )
    return mean_scd.addBands([cloud_pct, onset_idx, meltout_idx])


def prep_zone_bands(zone_raster: ee.Image) -> ee.Image:
    """Decodes the single zone_id band into range_id/elevation_band/forest as
    their own bands, so the grouped zonal mean (which is a no-op on a value
    that's already constant within a zone) surfaces them as plain columns in
    the output CSV instead of requiring a separate decode step in Python."""
    zone_id = zone_raster.rename("zone_id")
    range_id = zone_id.divide(100).floor().rename("range_id")
    elevation_band = zone_id.mod(100).divide(10).floor().rename("elevation_band")
    forest = zone_id.mod(10).rename("forest")
    return zone_id.addBands([range_id, elevation_band, forest])


def zonal_means_for_image(
    zone_bands: ee.Image, data_img: ee.Image, aoi: ee.Geometry, n_data_bands: int
) -> ee.FeatureCollection:
    """Groups data_img's bands by zone_id and takes the mean within each
    zone, in a single reduceRegion call -- the direct fit for "zonal stats
    over a categorical raster" that Image.reduceRegions on a one-polygon
    FeatureCollection was an awkward substitute for.

    Generic across any data_img (Phase 3's 4 SCD/cloud/onset/meltout bands,
    Phase 5's 2 climate bands, ...) -- n_data_bands is the caller's own known
    band count for data_img, since deriving it via bandNames().size()
    .getInfo() would cost a network round trip on every call (this runs once
    per hydro-year per variant during export construction).

    GEE requires the group-key band to come AFTER the value bands it's
    reduced with ("Group input must come after weighted inputs" if it
    doesn't) -- zone_id is moved to the end here rather than kept first as
    prep_zone_bands() naturally produces it. Reducer.mean().group() on its
    own also only accepts exactly one non-group value band -- averaging
    multiple bands (the 3 decode bands plus data_img's own) simultaneously
    needs .repeat(n) first so the reducer expects that many inputs.

    IMPORTANT (found 2026-09-18, after already exporting real CSVs with the
    bug): repeat(n).group() does NOT come back with one named key per band --
    every group is {'zone_id': v, 'mean': [v0, v1, ..., vn]}, a single 'mean'
    key holding a plain list in band order, not e.g. {'mean_scd': ..,
    'cloud_pct': ..}. Left unpacked, the exported CSV has one 'mean' column
    containing a stringified list instead of the named columns every
    downstream Python script expects. Fixed here by re-expanding that list
    against data_img's own band names (plus the 3 decode bands, in the same
    order they were added) before building each Feature, so this function's
    OUTPUT schema is exactly what its docstring always claimed -- the
    already-exported CSVs from before this fix still need a one-time reshape
    (see python/reshape_gee_csv.py).
    """
    decode_bands = zone_bands.select(["range_id", "elevation_band", "forest"])
    zone_id = zone_bands.select("zone_id")
    value_band_names = ee.List(["range_id", "elevation_band", "forest"]).cat(data_img.bandNames())
    combined = decode_bands.addBands(data_img).addBands(zone_id)
    n_value_bands = 3 + n_data_bands  # decode_bands (3) + data_img, then zone_id last
    result = combined.reduceRegion(
        reducer=ee.Reducer.mean()
        .repeat(n_value_bands)
        .group(groupField=n_value_bands, groupName="zone_id"),
        geometry=aoi,
        scale=500,
        maxPixels=1e9,
    )
    groups = ee.List(result.get("groups"))

    def unpack(g):
        g = ee.Dictionary(g)
        means = ee.List(g.get("mean"))
        props = ee.Dictionary.fromLists(value_band_names, means).set("zone_id", g.get("zone_id"))
        return ee.Feature(None, props)

    return ee.FeatureCollection(groups.map(unpack))


def build_variant_feature_collection(
    zone_bands: ee.Image, aoi: ee.Geometry, variant_name: str, year_to_image: dict
) -> ee.FeatureCollection:
    """Builds one variant's full FeatureCollection across all its hydro-years.

    Deliberately a plain Python loop over year_to_image (a real dict, built
    client-side in __main__) rather than ee.ImageCollection.map() -- the day-
    index conversion in prep_data_bands() needs end_year as an actual Python
    int to build ee.Date.fromYMD(end_year - 1, ...), which isn't available
    inside a .map() callback (there, "hydro_year" is only a server-side
    ee.Number attached to each image, not a client-side value).
    """
    features = []
    for end_year, img in year_to_image.items():
        if img is None:
            continue
        prepped = prep_data_bands(img, end_year)
        zonal_fc = zonal_means_for_image(zone_bands, prepped, aoi, n_data_bands=4)
        tagged = zonal_fc.map(
            lambda f, yr=end_year, vn=variant_name: f.set("hydro_year", yr, "variant", vn)
        )
        features.append(tagged)

    merged = features[0]
    for fc in features[1:]:
        merged = merged.merge(fc)
    return merged


def export_variant_table(zone_bands, aoi, variant_name, year_to_image):
    fc = build_variant_feature_collection(zone_bands, aoi, variant_name, year_to_image)
    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=f"scd_zonal_{variant_name}",
        folder="thesis_scd_exports",
        fileFormat="CSV",
    )
    task.start()
    return task


if __name__ == "__main__":
    aoi_fc = load_aois()
    aoi = aoi_fc.geometry()
    # Loads Phase 0's already-exported asset rather than recomputing
    # build_zone_raster() here -- the DEM/forest reprojection over the full
    # three-range bounding extent is exactly the kind of large computation
    # Phase 0's asset export exists to do once, ahead of time.
    zone_raster = ee.Image(GEE_ASSET_ID)
    zone_bands = prep_zone_bands(zone_raster)

    terra = ee.ImageCollection("MODIS/061/MOD10A1").filterBounds(aoi)
    aqua = ee.ImageCollection("MODIS/061/MYD10A1").filterBounds(aoi)

    variant_years = {
        "baseline": {y: variant_a_baseline(terra, y) for y in HYDRO_YEARS},
        "forward_fill": {y: variant_b_forward_fill(terra, y) for y in HYDRO_YEARS},
        "terra_aqua": {
            y: variant_c_terra_aqua(terra, aqua, y) for y in HYDRO_YEARS if y >= AQUA_START_YEAR
        },
    }

    tasks = [
        export_variant_table(zone_bands, aoi, variant_name, year_to_image)
        for variant_name, year_to_image in variant_years.items()
    ]

    print(f"Started {len(tasks)} export tasks -- check https://code.earthengine.google.com/tasks")
    print("Once complete, download the CSVs from Drive into data/exports/ for Phase 4.")
