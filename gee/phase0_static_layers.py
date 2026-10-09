"""
Phase 0 -- Static layers (one-time setup)

Builds the layers reused by every later phase:
  1. AOI boundaries for the three ranges (Krkonose NP, Sumava NP, Hruby Jesenik PLA)
  2. Elevation-band raster (Copernicus GLO-30 -> 500 m MODIS grid, 3 bands)
  3. Forest mask (Hansen Global Forest Change, year-2000 baseline)
  4. Combined zone-ID raster: range x elevation-band (x forest/open)

Thesis reference: Methodology 5.1 (elevation stratification), 5.6 (forest canopy).

Run once; exports the zone-ID raster as a GEE asset so Phases 1-3 can load it
directly instead of recomputing it every time.
"""
import ee

ee.Initialize(project="thesis101-492500")

# WDPA (World Database on Protected Areas) polygon layer -- public GEE dataset,
# no upload/asset access needed. Schema verified directly against the live
# asset on 2026-09-17 (not the Data Catalog page, which lists a different
# field set than what's actually served): properties include NAME, NAME_ENG,
# DESIG_ENG, ISO3, SITE_ID, REP_AREA (km^2) -- there is NO "WDPAID" or
# "ORIG_NAME" field in this asset despite the catalog docs implying otherwise.
WDPA_ASSET_ID = "WCMC/WDPA/current/polygons"

# Substring(s) to match against NAME/NAME_ENG, scoped to ISO3 == 'CZE' to
# avoid cross-border false positives (e.g. a Polish-side Karkonosze entry, or
# the transboundary biosphere reserve, which is a separate WDPA record from
# the national park itself).
#
# ee.Filter.stringContains is CASE-SENSITIVE, and WDPA names are typically
# Title Case ("Krkonoše"), so each pattern is listed in both its capitalized
# and lowercase form to be robust to whichever convention this data actually
# uses (not independently verified -- confirm against the printed report).
RANGE_NAME_PATTERNS = {
    "krkonose": ["Krkono", "krkono"],           # Krkonoše / Krkonose
    "sumava": ["umava"],                        # Šumava / Sumava -- "umava" dodges the accented/capital first letter either way
    "hruby_jesenik": ["Jesen", "jesen"],        # Jeseníky / Jeseniky (CHKO Jeseníky covers Hruby Jesenik)
}

MODIS_SCALE = 500  # metres, matches MOD10A1 sinusoidal grid
ELEVATION_BREAKS = [800, 1200]  # metres a.s.l. -> 3 bands: <800, 800-1200, >1200
FOREST_COVER_THRESHOLD = 30  # percent canopy cover, Hansen treecover2000


def find_candidate_aois() -> dict:
    """Diagnostic step -- run this FIRST and inspect the printed output before
    trusting load_aois(). Returns, per range, every WDPA feature whose NAME or
    ORIG_NAME matches the range's pattern within Czechia (ISO3 == 'CZE').

    WDPA can carry multiple records per protected area (e.g. a national park
    plus an overlapping biosphere reserve, or several parcels under one
    WDPAID) -- this surfaces all of them so you pick the right WDPAID(s)
    explicitly, rather than the script silently guessing.
    """
    wdpa = ee.FeatureCollection(WDPA_ASSET_ID).filter(ee.Filter.eq("ISO3", "CZE"))
    candidates = {}
    for range_name, patterns in RANGE_NAME_PATTERNS.items():
        name_filter = ee.Filter.Or(*[
            ee.Filter.Or(
                ee.Filter.stringContains("NAME", p),
                ee.Filter.stringContains("NAME_ENG", p),
            )
            for p in patterns
        ])
        matched = wdpa.filter(name_filter)
        info = matched.reduceColumns(
            ee.Reducer.toList(5),
            ["SITE_ID", "NAME", "DESIG_ENG", "STATUS", "REP_AREA"],
        ).getInfo()
        candidates[range_name] = info.get("list", [])
    return candidates


def print_candidate_report(candidates: dict):
    print("=== WDPA candidate features per range (verify before trusting) ===")
    for range_name, rows in candidates.items():
        print(f"\n{range_name}:")
        if not rows:
            print("  NO MATCHES -- widen RANGE_NAME_PATTERNS or check ISO3 filter.")
        for site_id, name, desig, status, area in rows:
            print(f"  SITE_ID={site_id}  NAME={name!r}  DESIG={desig!r}  "
                  f"STATUS={status!r}  AREA={area:.1f} km^2")
    print(
        "\nExpected ballpark areas: Krkonose NP ~363 km^2, Sumava NP ~680 km^2, "
        "Jeseniky PLA ~740 km^2 (Wikipedia, unverified against WDPA's own REP_AREA "
        "-- cross-check before committing to CONFIRMED_SITE_IDS below)."
    )


# Confirmed 2026-09-17 by running find_candidate_aois() against the live WDPA
# asset and matching each range to its single official National Park /
# Protected Landscape Area designation (excluding the overlapping Natura 2000,
# UNESCO-MAB biosphere reserve, and Ramsar records WDPA also carries for the
# same physical area -- those are separate legal designations, not boundary
# alternatives, and would double-count area if merged in).
#   krkonose:      SITE_ID 344524, "Krkonossky narodni park", National Park, 363.5 km^2
#   sumava:        SITE_ID 344527, "Sumava", National Park, 684.6 km^2
#   hruby_jesenik: SITE_ID 344537, "Jeseniky", Protected Landscape Area, 743.7 km^2
# All three match the expected ballpark areas above within a few percent.
CONFIRMED_SITE_IDS = {
    "krkonose": [344524],
    "sumava": [344527],
    "hruby_jesenik": [344537],
}


def load_aois() -> ee.FeatureCollection:
    """Merge the three range boundaries into one FeatureCollection with a
    'range' property (krkonose / sumava / hruby_jesenik) for later grouping.

    Requires CONFIRMED_SITE_IDS to be filled in first -- see
    find_candidate_aois() and print_candidate_report().
    """
    if any(len(ids) == 0 for ids in CONFIRMED_SITE_IDS.values()):
        raise ValueError(
            "CONFIRMED_SITE_IDS is incomplete. Run find_candidate_aois() and "
            "print_candidate_report() first, verify the matches, then fill in "
            "the SITE_ID(s) for every range before calling load_aois()."
        )

    wdpa = ee.FeatureCollection(WDPA_ASSET_ID)
    fcs = []
    for range_name, site_ids in CONFIRMED_SITE_IDS.items():
        fc = (
            wdpa.filter(ee.Filter.inList("SITE_ID", site_ids))
            .map(_polygon_only)
            .map(lambda f, n=range_name: f.set("range", n))
        )
        fcs.append(fc)
    merged = fcs[0]
    for fc in fcs[1:]:
        merged = merged.merge(fc)
    return merged


def _polygon_only(feature: ee.Feature) -> ee.Feature:
    """Strips non-polygonal parts from a feature's geometry, keeping only its
    Polygon/MultiPolygon components.

    Found 2026-09-17 while debugging a zonal-stats leak in Phase 3: Krkonose's
    WDPA record (SITE_ID 344524) is a GeometryCollection containing a stray
    LineString alongside its real Polygon boundary -- almost certainly a
    digitizing artifact in the source data, not a meaningful second feature.
    A mixed-type GeometryCollection doesn't clip/mask reliably in GEE (clip()
    and reduceRegion() against it let background pixels outside the real
    polygon leak through as if unmasked), so this drops anything that isn't a
    Polygon before the geometry is ever used downstream. A no-op for a
    feature that's already a clean Polygon (confirmed true for Sumava and
    Hruby Jesenik's records) -- .geometries() returns a single-element list
    containing itself in that case, so this is safe to apply unconditionally
    rather than only where the bug happens to bite.
    """
    geom = feature.geometry()
    parts = ee.FeatureCollection(
        geom.geometries().map(
            lambda g: ee.Feature(ee.Geometry(g), {"geom_type": ee.Geometry(g).type()})
        )
    )
    polygon_parts = parts.filter(ee.Filter.inList("geom_type", ["Polygon", "MultiPolygon"]))
    return feature.setGeometry(polygon_parts.geometry())


def build_elevation_band_raster(aoi: ee.Geometry) -> ee.Image:
    """Copernicus GLO-30 DEM, resampled to the 500 m MODIS grid, classified
    into 3 elevation bands. Band values: 1 = <800m, 2 = 800-1200m, 3 = >1200m."""
    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").select("DEM").mosaic()
    dem_500m = dem.reproject(crs="SR-ORG:6974", scale=30).reduceResolution(
        reducer=ee.Reducer.mean(), maxPixels=1024
    ).reproject(crs="SR-ORG:6974", scale=MODIS_SCALE)

    band = (
        ee.Image(1)
        .where(dem_500m.gte(ELEVATION_BREAKS[0]), 2)
        .where(dem_500m.gte(ELEVATION_BREAKS[1]), 3)
        .rename("elevation_band")
        .clip(aoi)
    )
    return band


def build_forest_mask(aoi: ee.Geometry) -> ee.Image:
    """Hansen Global Forest Change, year-2000 tree cover, resampled/thresholded
    to the same 500 m grid. 1 = forested, 0 = open."""
    hansen = ee.Image("UMD/hansen/global_forest_change_2025_v1_13")
    treecover = hansen.select("treecover2000")
    forest_500m = treecover.reproject(crs="SR-ORG:6974", scale=30).reduceResolution(
        reducer=ee.Reducer.mean(), maxPixels=1024
    ).reproject(crs="SR-ORG:6974", scale=MODIS_SCALE)
    mask = forest_500m.gte(FOREST_COVER_THRESHOLD).rename("forest_mask").clip(aoi)
    return mask


def build_zone_raster(aoi_fc: ee.FeatureCollection) -> ee.Image:
    """Combine range ID (from AOI FeatureCollection, rasterized), elevation
    band, and forest mask into one zone-ID raster.

    Zone ID encoding: range(1-3) * 100 + elevation_band(1-3) * 10 + forest(0/1)
    e.g. 131 = Krkonose, >1200m, forested.
    """
    range_names = ["krkonose", "sumava", "hruby_jesenik"]
    range_img = ee.Image(0)
    for i, name in enumerate(range_names, start=1):
        sub_fc = aoi_fc.filter(ee.Filter.eq("range", name))
        range_img = range_img.where(
            ee.Image().paint(sub_fc, 1).selfMask().unmask(0), i
        )

    elevation_img = build_elevation_band_raster(aoi_fc.geometry())
    forest_img = build_forest_mask(aoi_fc.geometry())

    zone_id = (
        range_img.multiply(100)
        .add(elevation_img.multiply(10))
        .add(forest_img)
        .rename("zone_id")
    )
    # Mask out any pixel range_img never painted into one of the 3 ranges.
    # Found 2026-09-17: reduceRegion's area-weighted pixel counting gives
    # partial credit to a boundary-straddling pixel (its center just outside
    # the polygon, but part of its area still inside), while paint() is a
    # binary center-in-polygon test -- so a park with a highly convoluted
    # boundary (Krkonose follows a mountain ridge line, unlike Sumava/Hruby
    # Jesenik's much more compact shapes) can leave a real fraction of its
    # nominal area sitting in "unpainted" edge pixels. Confirmed empirically:
    # a direct sum/count of the painted band within Krkonose's own polygon
    # came out ~85% painted, not ~100%, and the gap reduceRegion(scale=500)
    # still counts (fractionally) doesn't belong in any zone. Without this
    # mask those pixels showed up as a spurious "range=0" zone downstream in
    # Phase 3's zonal aggregation, with real-looking (but meaningless)
    # per-zone statistics.
    return zone_id.updateMask(range_img.gt(0))


def export_zone_raster(zone_img: ee.Image, aoi: ee.Geometry, asset_id: str):
    task = ee.batch.Export.image.toAsset(
        image=zone_img,
        description="scd_zone_raster_phase0",
        assetId=asset_id,
        region=aoi,
        scale=MODIS_SCALE,
        maxPixels=1e9,
    )
    task.start()
    return task


GEE_ASSET_ID = "projects/thesis101-492500/assets/scd_zone_raster"

if __name__ == "__main__":
    # Step 1: diagnose. Run this script once, inspect the printed candidates,
    # fill in CONFIRMED_SITE_IDS above, then re-run to actually export.
    if any(len(ids) == 0 for ids in CONFIRMED_SITE_IDS.values()):
        print_candidate_report(find_candidate_aois())
        print(
            "\nCONFIRMED_SITE_IDS is still empty for at least one range -- "
            "fill it in above using the SITE_ID values printed, then re-run "
            "this script to export the zone raster."
        )
    else:
        aois = load_aois()
        zone_raster = build_zone_raster(aois)
        export_zone_raster(zone_raster, aois.geometry(), GEE_ASSET_ID)
        print("Export task started -- check https://code.earthengine.google.com/tasks")
