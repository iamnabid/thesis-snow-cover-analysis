"""
Phase 7 (GEE side) -- daily MOD10A1 pixel state at each CHMU station point

For the cloud-matched validation: python/phase7_cloud_matched.py needs to
know, for every day, whether the MODIS pixel covering a station was clear
and (if clear) snow-covered, so station snow days can be counted only on
days MODIS could actually see the ground.

Uses the same snow/cloud definition as phase1_scd_calculation.classify_daily
(NDSI > 40, cloud = no valid NDSI retrieval).

Input:  data/exports/chmu_stations.csv (station_id, lat, lon)
Output: data/exports/station_modis_daily.csv
        (station_id, date, hydro_year, snow, cloud)
"""
from pathlib import Path

import ee
import pandas as pd

from phase1_scd_calculation import HYDRO_YEARS, classify_daily, hydro_year_bounds

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "exports"


def station_fc(stations_df: pd.DataFrame) -> ee.FeatureCollection:
    return ee.FeatureCollection(
        [
            ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"station_id": r.station_id})
            for r in stations_df.itertuples()
        ]
    )


def daily_for_year(fc: ee.FeatureCollection, end_year: int) -> list:
    start, end = hydro_year_bounds(end_year)
    col = ee.ImageCollection("MODIS/061/MOD10A1").filterDate(start, end).map(classify_daily)

    def per_image(img):
        date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
        sampled = img.select(["snow", "cloud"]).reduceRegions(
            collection=fc, reducer=ee.Reducer.first(), scale=500
        )
        return sampled.map(lambda f: f.set("date", date))

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
        print(f"hydro_year {y}: {len(rows)} station-days")
        all_rows.extend(rows)
    out = DATA_DIR / "station_modis_daily.csv"
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"Wrote {len(all_rows)} rows to {out}")
