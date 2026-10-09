"""
Phase 7 (GEE side) -- CHMU station -> zone_id assignment

Given a CSV of CHMU station coordinates, samples Phase 0's exported zone
raster at each station's location to assign it a zone_id -- the join key
python/phase7_validation.py's station-vs-pixel comparison needs.

Input:  data/exports/chmu_stations.csv (columns: station_id, lat, lon;
        WGS84 decimal degrees, from CHMU's station metadata)
Output: data/exports/chmu_station_zones.csv
        (station_id, zone_id, range_id, elevation_band, forest)
"""
from pathlib import Path

import ee
import pandas as pd

from phase0_static_layers import GEE_ASSET_ID
from phase3_zonal_export import prep_zone_bands

ee.Initialize(project="thesis101-492500")

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "exports"


def assign_station_zones(stations_df: pd.DataFrame) -> pd.DataFrame:
    """Samples the zone raster at each station's point location via
    Image.reduceRegions -- the correct, idiomatic use of reduceRegions
    (one value looked up per point feature), unlike the single-polygon
    grouped-reduction case Phase 3 needed a different pattern for."""
    features = [
        ee.Feature(ee.Geometry.Point([row.lon, row.lat]), {"station_id": row.station_id})
        for row in stations_df.itertuples()
    ]
    fc = ee.FeatureCollection(features)

    zone_bands = prep_zone_bands(ee.Image(GEE_ASSET_ID))
    sampled = zone_bands.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=500)

    rows = sampled.getInfo()["features"]
    return pd.DataFrame(
        [
            {
                "station_id": f["properties"].get("station_id"),
                "zone_id": f["properties"].get("zone_id"),
                "range_id": f["properties"].get("range_id"),
                "elevation_band": f["properties"].get("elevation_band"),
                "forest": f["properties"].get("forest"),
            }
            for f in rows
        ]
    )


if __name__ == "__main__":
    stations_path = DATA_DIR / "chmu_stations.csv"
    if not stations_path.exists():
        print(
            f"SKIP: {stations_path} not found -- source CHMU station metadata "
            f"(station_id, lat, lon) first."
        )
    else:
        result_df = assign_station_zones(pd.read_csv(stations_path))
        unresolved = result_df[result_df["zone_id"].isna()]
        if len(unresolved) > 0:
            print(
                f"WARNING: {len(unresolved)} station(s) fell outside the zone "
                f"raster's painted area (see Phase 0's masking fix -- this "
                f"happens for a point right at a park boundary, or a genuine "
                f"coordinate error) and got no zone_id: "
                f"{unresolved['station_id'].tolist()}. These will silently drop "
                f"out of python/phase7_validation.py's merge -- check their "
                f"coordinates before trusting the final comparison."
            )
        out_path = DATA_DIR / "chmu_station_zones.csv"
        result_df.to_csv(out_path, index=False)
        print(f"Wrote {len(result_df)} station-zone assignments to {out_path}")
