"""
Phase 7 (GEE side, round-3 supervisor follow-up) -- daily MYD10A1 (Aqua)
pixel state at each CHMU station point.

Supervisor comment (2026-10-19, Sec 6.6 inline): "recalculate the annual
validation using the MODIS pixel containing each station rather than the
mean SCD of the corresponding zone." The existing Terra pixel series
(station_modis_daily.csv, from phase7_station_daily_pixels.py) already
supports a pixel-level baseline recalculation; this script supplies the
matching Aqua series so the terra_aqua variant can also be recalculated at
the station pixel instead of the zone mean.

Mirrors phase7_station_daily_pixels.py exactly, swapping MOD10A1 for
MYD10A1. Aqua coverage starts mid-2002, so early hydro-years return no rows;
downstream code must handle that (matches AQUA_START_YEAR=2003 in
gee/phase2_cloud_variants.py).

Input:  data/exports/chmu_stations.csv (station_id, lat, lon)
Output: data/exports/station_modis_daily_aqua.csv
        (station_id, date, hydro_year, snow, cloud)
"""
from pathlib import Path

import ee
import pandas as pd

from phase1_scd_calculation import HYDRO_YEARS, classify_daily, hydro_year_bounds
from phase7_station_daily_pixels import station_fc

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "exports"


def daily_for_year(fc: ee.FeatureCollection, end_year: int) -> list:
    start, end = hydro_year_bounds(end_year)
    col = ee.ImageCollection("MODIS/061/MYD10A1").filterDate(start, end).map(classify_daily)

    def per_image(img):
        date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
        sampled = img.select(["snow", "cloud"]).reduceRegions(
            collection=fc, reducer=ee.Reducer.first(), scale=500
        )
        return sampled.map(lambda f: f.set("date", date))

    size = col.size().getInfo()
    if size == 0:
        return []
    rows = col.map(per_image).flatten().getInfo()["features"]
    return [
        {
            "station_id": f["properties"]["station_id"],
            "date": f["properties"]["date"],
            "hydro_year": end_year,
            "snow": f["properties"].get("snow"),
            "cloud": f["properties"].get("cloud"),
        }
        for f in rows
    ]


if __name__ == "__main__":
    ee.Initialize(project="thesis101-492500")
    fc = station_fc(pd.read_csv(DATA_DIR / "chmu_stations.csv"))
    all_rows = []
    for y in HYDRO_YEARS:
        rows = daily_for_year(fc, y)
        print(f"hydro_year {y}: {len(rows)} station-days (Aqua)")
        all_rows.extend(rows)
    out = DATA_DIR / "station_modis_daily_aqua.csv"
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"Wrote {len(all_rows)} rows to {out}")
