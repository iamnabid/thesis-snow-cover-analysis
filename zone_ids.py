"""Zone-ID encoding -- single source of truth, shared by every phase.

zone_id = range_id * 100 + elevation_band * 10 + forest (see
gee/phase0_static_layers.build_zone_raster() for how the raster itself is
built). Kept dependency-free (no `ee` import) so Python-side phases that
only read already-exported CSVs (6, 7's station/hydrology comparisons) can
use these names without pulling in the GEE side's ee.Initialize() call.
"""
ZONE_ID_DECODE = {
    "range_id": {1: "krkonose", 2: "sumava", 3: "hruby_jesenik"},
    "elevation_band": {1: "<800m", 2: "800-1200m", 3: ">1200m"},
    "forest": {0: "open", 1: "forested"},
}
