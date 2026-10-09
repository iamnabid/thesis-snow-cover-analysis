"""
Phase 1 -- Per-pixel, per-hydro-year snow classification

For each hydrological year (1 Oct - 30 Sep, 24 years: 2000/01-2023/24), computes
per pixel:
  - SCD: count of clear-sky snow-covered days
  - cloud_pct: proportion of days obscured by cloud
  - onset_doy: day-of-year snow cover begins (first day of the longest
    continuous snow run)
  - meltout_doy: day-of-year snow cover ends (last day of that run)

Thesis reference: Methodology 5.3 (satellite data, NDSI > 0.4 threshold),
5.4 (SCD calculation, onset/melt-out dates).

Also demonstrates GEE's built-in per-pixel trend reducers (sensSlope,
kendallsCorrelation), which can produce the spatial trend maps for Objective 1
directly, without a full Python round-trip.
"""
import ee

ee.Initialize(project="thesis101-492500")

NDSI_THRESHOLD = 0.4
HYDRO_YEARS = list(range(2001, 2025))  # named by ending calendar year, e.g. 2001 = Oct 2000-Sep 2001


def hydro_year_bounds(end_year: int):
    start = ee.Date.fromYMD(end_year - 1, 10, 1)
    end = ee.Date.fromYMD(end_year, 10, 1)  # exclusive
    return start, end


def classify_daily(image: ee.Image) -> ee.Image:
    """Classify one MOD10A1 daily image into snow / no-snow / cloud, using the
    NDSI_Snow_Cover band's own raster mask.

    Output bands:
      snow: 1 where NDSI > threshold and a valid retrieval exists, else 0
      cloud: 1 where the pixel has no valid NDSI retrieval that day, else 0

    Empirically verified 2026-09-17 by pulling per-pixel histograms across 12
    dates spanning a full hydro-year (not just reading the product docs, which
    describe a >100 in-band sentinel scheme that this GEE-hosted asset does
    NOT actually use): NDSI_Snow_Cover_Basic_QA reads 0/1/2 ("best"/"good"/
    "ok") even on dates where every single pixel in the AOI has no NDSI value
    at all -- so QA cannot distinguish cloud-obscured from clear-sky here.
    What the mask shows instead: on those same all-missing dates,
    NDSI_Snow_Cover_Class reads 250 ("cloud" per the MODIS Class legend)
    across the whole AOI, and NDSI_Snow_Cover itself is genuinely masked
    (absent from a pixel value histogram, not a numeric sentinel) -- i.e. the
    asset already encodes "no valid retrieval" as a real raster mask, and
    that mask lines up with the Class band's cloud flag. So instead of
    depending on QA/Class code tables whose exact meaning in this specific
    GEE asset version isn't otherwise confirmed, this uses the band's own
    mask directly as the clear/cloud signal -- simpler and empirically
    grounded. One caveat worth flagging in the thesis text: this necessarily
    lumps any other retrieval gap (e.g. a genuine sensor gap) in with cloud,
    since both present the same way (a missing NDSI value); cloud is the
    overwhelmingly dominant cause in practice, but it's not a purely
    atmospheric-cloud-only signal.
    """
    ndsi = image.select("NDSI_Snow_Cover")
    is_snow = ndsi.gt(NDSI_THRESHOLD * 100).rename("snow").unmask(0)  # NDSI band is scaled 0-100
    is_cloud = ndsi.mask().Not().rename("cloud")

    return image.addBands([is_snow, is_cloud]).set(
        "system:time_start", image.get("system:time_start")
    )


def add_doy_band(image: ee.Image) -> ee.Image:
    """Adds a 'doy' band: the calendar day-of-year (1-indexed) of this image's
    acquisition date, constant across all pixels. Used by run_length_bands()
    to label the start/end of the longest snow run with a real calendar date
    rather than a same-hydro-year-relative index."""
    doy = ee.Date(image.get("system:time_start")).getRelative("day", "year").add(1)
    return image.addBands(ee.Image.constant(doy).toInt().rename("doy"))


def run_length_bands(daily_sorted: ee.ImageCollection) -> ee.Image:
    """Longest continuous run of clear-sky snow-covered days in one hydro-year,
    per pixel -- returns onset_doy (calendar DOY the run started) and
    meltout_doy (calendar DOY the run ended).

    GEE has no built-in run-length reducer, so this carries a small per-pixel
    state image (current run length/start, best run length/start/end) through
    an iterate() over the date-sorted daily images. A cloud-obscured day has
    snow=0 (set in classify_daily) and so breaks a run, same as a genuine
    no-snow day -- this is the strict clear-sky-only treatment matching Phase
    2's 'baseline' variant, not the forward-filled one.

    Pixels with zero snow days all year (best_len stays 0) get onset/meltout
    masked out rather than left at a sentinel 0 -- a real calendar DOY of 0
    doesn't exist, and leaving it in would silently pull down the zonal mean
    for any zone/year with even one snow-free pixel (e.g. the lowest
    elevation band in a mild year).

    KNOWN CAVEAT (empirically confirmed 2026-09-17, sampling individual
    pixels, not just the zonal mean): calendar DOY resets to 1 on Jan 1, so a
    best-run that starts in December and continues into January comes back
    as onset_doy > meltout_doy for that pixel (e.g. onset_doy=359,
    meltout_doy=11 was an observed real pixel) even though the run is
    perfectly correct and continuous -- onset genuinely happened before
    meltout, the DOY numbering just wrapped. In a spot check of 20 random
    Krkonose pixels for hydro-year 2021 (forward-fill variant), 2/20 showed
    this. A straight arithmetic mean of onset_doy or meltout_doy across
    pixels/years that mixes wrapping and non-wrapping cases will be
    distorted by whichever direction the wrap pulls it. Needs a decision
    before Phase 4/6 report these means: either convert to a hydro-year-
    relative day index (days since Oct 1, monotonic, no wrap) for any
    cross-pixel/cross-year averaging, or use circular statistics on the
    calendar DOY. This module still returns calendar DOY as-is since it's
    the more directly interpretable per-pixel value; the conversion (if
    wanted) belongs in Phase 3's zonal aggregation or Phase 4/6's reading of
    the exported CSV, not here.
    """
    img_list = daily_sorted.select(["snow", "doy"]).toList(daily_sorted.size())

    init = (
        ee.Image.constant(0).toInt().rename("cur_len")
        .addBands(ee.Image.constant(0).toInt().rename("cur_start"))
        .addBands(ee.Image.constant(0).toInt().rename("best_len"))
        .addBands(ee.Image.constant(0).toInt().rename("best_start"))
        .addBands(ee.Image.constant(0).toInt().rename("best_end"))
    )

    def accumulate(img, state):
        img = ee.Image(img)
        state = ee.Image(state)
        snow = img.select("snow")
        doy = img.select("doy")
        is_snow_day = snow.eq(1)

        cur_len = state.select("cur_len")
        cur_start = state.select("cur_start")
        best_len = state.select("best_len")
        best_start = state.select("best_start")
        best_end = state.select("best_end")

        new_cur_start = cur_start.where(cur_len.eq(0), doy).where(is_snow_day.Not(), 0)
        new_cur_len = cur_len.add(1).where(is_snow_day.Not(), 0)

        is_new_best = new_cur_len.gt(best_len)
        new_best_len = best_len.where(is_new_best, new_cur_len)
        new_best_start = best_start.where(is_new_best, new_cur_start)
        new_best_end = best_end.where(is_new_best, doy)

        return (
            new_cur_len.rename("cur_len")
            .addBands(new_cur_start.rename("cur_start"))
            .addBands(new_best_len.rename("best_len"))
            .addBands(new_best_start.rename("best_start"))
            .addBands(new_best_end.rename("best_end"))
        )

    final = ee.Image(img_list.iterate(accumulate, init))
    had_snow = final.select("best_len").gt(0)
    onset_doy = final.select("best_start").updateMask(had_snow).rename("onset_doy")
    meltout_doy = final.select("best_end").updateMask(had_snow).rename("meltout_doy")
    return onset_doy.addBands(meltout_doy)


def doy_to_hydro_day_index(doy_image: ee.Image, end_year: int) -> ee.Image:
    """Converts a calendar-DOY band (1-366) into a hydro-year-relative day
    index (0 = Oct 1 of the hydro-year's start, monotonically increasing
    through Sep 30 of end_year) -- fixes the wraparound documented in
    run_length_bands()'s docstring, where a run spanning Dec-Jan comes back
    as onset_doy > meltout_doy purely because DOY resets to 1 at New Year's.

    The two calendar-year halves of a hydro-year never overlap in DOY space
    (Oct-Dec of the start year is always DOY >= oct1_doy; Jan-Sep of the end
    year is always DOY < oct1_doy, since Oct 1 is the latest a DOY can be
    within that starting year's portion), so which half a given DOY value
    belongs to is unambiguous from the value alone -- no need to track which
    calendar year it came from separately.

    Preserves masking: a masked (snow-free-pixel) input stays masked.
    """
    start = ee.Date.fromYMD(end_year - 1, 10, 1)
    oct1_doy = start.getRelative("day", "year").add(1)
    days_in_start_year = ee.Date.fromYMD(end_year - 1, 12, 31).getRelative("day", "year").add(1)

    belongs_to_end_year = doy_image.lt(oct1_doy)
    idx_from_start_year = doy_image.subtract(oct1_doy)
    idx_from_end_year = doy_image.add(ee.Image.constant(days_in_start_year).subtract(oct1_doy))
    return idx_from_start_year.where(belongs_to_end_year, idx_from_end_year)


def aggregate_hydro_year(yearly: ee.ImageCollection, start: ee.Date, end: ee.Date) -> ee.Image:
    """Reduce one hydro-year's daily 'snow'/'cloud' band pairs into SCD,
    cloud%, onset DOY, and melt-out DOY bands.

    Factored out of scd_for_hydro_year() so Phase 2's forward-fill and
    Terra+Aqua variants -- which produce their own per-day 'snow'/'cloud'
    bands under different rules -- can reuse the same aggregation instead of
    duplicating it.
    """
    n_days = end.difference(start, "day")
    scd = yearly.select("snow").sum().rename("scd")
    cloud_days = yearly.select("cloud").sum()
    cloud_pct = cloud_days.divide(n_days).rename("cloud_pct")

    yearly_sorted = yearly.map(add_doy_band).sort("system:time_start")
    run_bands = run_length_bands(yearly_sorted)
    onset_doy = run_bands.select("onset_doy")
    meltout_doy = run_bands.select("meltout_doy")

    return scd.addBands(cloud_pct).addBands(onset_doy).addBands(meltout_doy)


def scd_for_hydro_year(collection: ee.ImageCollection, end_year: int) -> ee.Image:
    """Aggregate one hydro-year of classified daily images into SCD, cloud%,
    onset DOY, and melt-out DOY bands."""
    start, end = hydro_year_bounds(end_year)
    yearly = collection.filterDate(start, end).map(classify_daily)
    return aggregate_hydro_year(yearly, start, end).set("hydro_year", end_year)


def build_scd_stack(collection: ee.ImageCollection, aoi: ee.Geometry) -> ee.ImageCollection:
    images = [scd_for_hydro_year(collection, y).clip(aoi) for y in HYDRO_YEARS]
    return ee.ImageCollection(images)


def per_pixel_trend_maps(scd_stack: ee.ImageCollection) -> ee.Image:
    """Objective 1 spatial trend maps, computed directly in GEE.

    Builds a (time, scd) band pair per image and runs Sen's slope +
    Kendall's tau across the stack, pixel by pixel.
    """
    def add_time_band(img):
        year = ee.Number(img.get("hydro_year"))
        scd = img.select("scd")
        # ee.Image.constant() has no inherent projection -- reproject it onto
        # scd's own grid explicitly before combining, rather than relying on
        # GEE's implicit default-projection handling when addBands() combines
        # two images of different/undefined projection. Skipping this is what
        # caused the real "Can't transform (0.0,0.0)" export failure below.
        year_band = ee.Image.constant(year).rename("year").float().reproject(scd.projection())
        return scd.addBands(year_band)

    joined = scd_stack.map(add_time_band).select(["year", "scd"])
    sen_slope = joined.reduce(ee.Reducer.sensSlope())  # bands: slope, offset
    kendall = joined.reduce(ee.Reducer.kendallsCorrelation())  # bands: tau, p-value
    return sen_slope.addBands(kendall)


if __name__ == "__main__":
    from phase0_static_layers import load_aois

    aoi = load_aois().geometry()
    mod10a1 = ee.ImageCollection("MODIS/061/MOD10A1").filterBounds(aoi)

    scd_stack = build_scd_stack(mod10a1, aoi)
    trend_map = per_pixel_trend_maps(scd_stack)

    task = ee.batch.Export.image.toDrive(
        image=trend_map,
        description="scd_trend_map_phase1",
        folder="thesis_scd_exports",
        region=aoi,
        crs="SR-ORG:6974",
        scale=500,
        maxPixels=1e9,
    )
    task.start()
    print("Trend map export started -- check https://code.earthengine.google.com/tasks")
