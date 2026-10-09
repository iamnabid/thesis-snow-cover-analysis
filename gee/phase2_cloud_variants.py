"""
Phase 2 -- Cloud-treatment sensitivity variants

Produces three parallel versions of the SCD stack from Phase 1, so the
sensitivity of the trend to cloud-gap handling can be checked directly
(thesis Methodology 5.5):

  A. baseline    -- clear-sky only (Phase 1's output, reused as-is)
  B. forward_fill -- max 5-day forward fill of the last clear-sky classification,
                      mirroring MOD10A1F logic
  C. terra_aqua  -- Terra (MOD10A1) + Aqua (MYD10A1) combined daily composite,
                      restricted to years where Aqua exists (2002/03 onward)

Each variant produces the same SCD/cloud_pct band structure as Phase 1's
scd_for_hydro_year(), so Phase 3 can aggregate all three the same way.
"""
import ee

from phase1_scd_calculation import (
    aggregate_hydro_year,
    classify_daily,
    hydro_year_bounds,
    scd_for_hydro_year,
    HYDRO_YEARS,
)

ee.Initialize(project="thesis101-492500")

FORWARD_FILL_MAX_DAYS = 5
AQUA_START_YEAR = 2003  # first hydro-year with a full Aqua record (Aqua data begins July 2002)


def variant_a_baseline(collection: ee.ImageCollection, end_year: int) -> ee.Image:
    """Same as Phase 1's scd_for_hydro_year -- clear-sky days only. Delegates
    rather than recomputing, so baseline can never silently drift out of sync
    with Phase 1's own onset/melt-out/cloud_pct logic."""
    return scd_for_hydro_year(collection, end_year).set("variant", "baseline")


def forward_fill_stack(daily_sorted: ee.ImageCollection) -> ee.ImageCollection:
    """Per-pixel forward fill: on a cloud day, reuse the most recent
    clear-sky snow classification if it is at most FORWARD_FILL_MAX_DAYS old;
    beyond that the day is left unresolved (snow=0, cloud=1 still), mirroring
    NASA's MOD10A1F max-fill-window logic without requiring that product
    itself (confirmed unavailable in the GEE catalog -- Methodology 5.5).

    Implemented as an ee.List.iterate() carrying a small per-pixel state
    (last clear-sky snow value, days since that observation) forward day by
    day, since GEE has no built-in forward-fill reducer. The per-step output
    (filled snow/cloud bands) is accumulated into a list alongside the state,
    the standard GEE idiom for a scan that needs both a running state and a
    per-step output.
    """
    img_list = daily_sorted.select(["snow", "cloud"]).toList(daily_sorted.size())

    init_state = (
        ee.Image.constant(0).toInt().rename("last_valid_snow")
        .addBands(ee.Image.constant(FORWARD_FILL_MAX_DAYS + 1).toInt().rename("days_since_valid"))
    )
    init = ee.Dictionary({"state": init_state, "outputs": ee.List([])})

    def step(img, acc):
        img = ee.Image(img)
        acc = ee.Dictionary(acc)
        state = ee.Image(acc.get("state"))
        outputs = ee.List(acc.get("outputs"))

        cloud = img.select("cloud")
        snow = img.select("snow")
        is_clear = cloud.eq(0)

        last_valid_snow = state.select("last_valid_snow")
        days_since_valid = state.select("days_since_valid")

        # On a clear day the state resets to today's real reading. On a
        # cloudy day the state is untouched, so it keeps carrying the last
        # real reading forward (and days_since_valid keeps climbing) across
        # however many consecutive cloudy days follow.
        new_last_valid_snow = last_valid_snow.where(is_clear, snow)
        new_days_since_valid = days_since_valid.add(1).where(is_clear, 0)
        new_state = new_last_valid_snow.rename("last_valid_snow").addBands(
            new_days_since_valid.rename("days_since_valid")
        )

        can_fill = new_days_since_valid.lte(FORWARD_FILL_MAX_DAYS)
        filled_snow = snow.where(is_clear.Not(), new_last_valid_snow.multiply(can_fill)).rename("snow")
        residual_cloud = cloud.And(can_fill.Not()).rename("cloud")

        filled_img = filled_snow.addBands(residual_cloud).set(
            "system:time_start", img.get("system:time_start")
        )

        return ee.Dictionary({"state": new_state, "outputs": outputs.add(filled_img)})

    final = ee.Dictionary(img_list.iterate(step, init))
    return ee.ImageCollection(ee.List(final.get("outputs")))


def variant_b_forward_fill(collection: ee.ImageCollection, end_year: int) -> ee.Image:
    start, end = hydro_year_bounds(end_year)
    daily = collection.filterDate(start, end).map(classify_daily).sort("system:time_start")
    filled = forward_fill_stack(daily)
    return aggregate_hydro_year(filled, start, end).set(
        "hydro_year", end_year, "variant", "forward_fill"
    )


def _with_date_str(image: ee.Image) -> ee.Image:
    return image.set("date_str", ee.Date(image.get("system:time_start")).format("yyyy-MM-dd"))


def combine_terra_aqua_stack(
    terra_daily: ee.ImageCollection, aqua_daily: ee.ImageCollection
) -> ee.ImageCollection:
    """Joins Terra and Aqua daily images by calendar date and takes the
    pixelwise OR of 'snow' (snow-covered if either platform sees it) and AND
    of 'cloud' (cloud-obscured only if BOTH platforms are cloud-obscured that
    day) -- a Terra-cloudy/Aqua-clear-no-snow day is correctly resolved as
    "clear, no snow", not left ambiguous.

    Uses a left join keyed on date_str so a day with a Terra image but no
    matching Aqua image (a real, occasional gap in the record) still comes
    through as Terra-only rather than being dropped.
    """
    terra_dated = terra_daily.map(_with_date_str)
    aqua_dated = aqua_daily.map(_with_date_str)

    join = ee.Join.saveFirst(matchKey="aqua_match")
    joined = join.apply(
        primary=terra_dated,
        secondary=aqua_dated,
        condition=ee.Filter.equals(leftField="date_str", rightField="date_str"),
    )

    def combine(feature):
        terra_img = ee.Image(feature)
        aqua_match = feature.get("aqua_match")
        no_match = ee.Algorithms.IsEqual(aqua_match, None)
        aqua_img = ee.Image(ee.Algorithms.If(no_match, terra_img, aqua_match))

        combined_snow = terra_img.select("snow").Or(aqua_img.select("snow")).rename("snow")
        combined_cloud = terra_img.select("cloud").And(aqua_img.select("cloud")).rename("cloud")
        return combined_snow.addBands(combined_cloud).set(
            "system:time_start", terra_img.get("system:time_start")
        )

    return ee.ImageCollection(joined.map(combine)).sort("system:time_start")


def variant_c_terra_aqua(
    terra: ee.ImageCollection, aqua: ee.ImageCollection, end_year: int
) -> ee.Image:
    """Combine Terra + Aqua daily: a pixel counts as snow-covered for the day
    if either platform classifies it as snow (and neither is required to be
    clear on the other pass)."""
    if end_year < AQUA_START_YEAR:
        return None  # not enough Aqua coverage for this hydro-year

    start, end = hydro_year_bounds(end_year)
    terra_y = terra.filterDate(start, end).map(classify_daily)
    aqua_y = aqua.filterDate(start, end).map(classify_daily)

    combined = combine_terra_aqua_stack(terra_y, aqua_y)
    return aggregate_hydro_year(combined, start, end).set(
        "hydro_year", end_year, "variant", "terra_aqua"
    )


def build_all_variants(terra: ee.ImageCollection, aqua: ee.ImageCollection, aoi: ee.Geometry):
    variants = {"baseline": [], "forward_fill": [], "terra_aqua": []}
    for y in HYDRO_YEARS:
        variants["baseline"].append(variant_a_baseline(terra, y).clip(aoi))
        variants["forward_fill"].append(variant_b_forward_fill(terra, y).clip(aoi))
        c = variant_c_terra_aqua(terra, aqua, y)
        if c is not None:
            variants["terra_aqua"].append(c.clip(aoi))
    return {k: ee.ImageCollection(v) for k, v in variants.items()}


if __name__ == "__main__":
    from phase0_static_layers import load_aois

    aoi = load_aois().geometry()
    terra = ee.ImageCollection("MODIS/061/MOD10A1").filterBounds(aoi)
    aqua = ee.ImageCollection("MODIS/061/MYD10A1").filterBounds(aoi)

    variants = build_all_variants(terra, aqua, aoi)
    print(f"Built {sum(v.size().getInfo() for v in variants.values())} variant-years "
          f"across {len(variants)} cloud-treatment variants.")
