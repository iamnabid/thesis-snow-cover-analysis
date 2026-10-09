"""
Phase 7 -- cloud-matched station vs. MODIS comparison

Removes the cloud-availability mismatch from the station validation: on days
when the MODIS pixel at the station was clear (a valid NDSI retrieval), the
station and MODIS snow states are directly comparable, so we compare them
day by day instead of comparing annual counts.

Input:  data/exports/station_modis_daily.csv  (gee/phase7_station_daily_pixels.py)
        data/exports/chmu_station_snow_depth.csv
Output: data/cloud_matched_summary.csv (per station and overall)
        data/cloud_matched_by_year.csv (per station-year)
"""
from pathlib import Path

import pandas as pd

from phase7_validation import SNOW_DEPTH_THRESHOLD_CM

DATA = Path(__file__).resolve().parents[1] / "data"
EXP = DATA / "exports"


def load_matched() -> pd.DataFrame:
    modis = pd.read_csv(EXP / "station_modis_daily.csv")
    depth = pd.read_csv(EXP / "chmu_station_snow_depth.csv")
    depth["station_snow"] = (depth["depth_cm"] > SNOW_DEPTH_THRESHOLD_CM).astype(int)
    m = modis.merge(
        depth[["station_id", "date", "station_snow"]], on=["station_id", "date"], how="inner"
    )
    m = m.dropna(subset=["snow", "cloud", "station_snow"])
    return m


def summarize(g: pd.DataFrame) -> dict:
    clear = g[g["cloud"] == 0]
    tp = ((clear["snow"] == 1) & (clear["station_snow"] == 1)).sum()
    fn = ((clear["snow"] == 0) & (clear["station_snow"] == 1)).sum()  # MODIS misses snow
    fp = ((clear["snow"] == 1) & (clear["station_snow"] == 0)).sum()  # MODIS false snow
    tn = ((clear["snow"] == 0) & (clear["station_snow"] == 0)).sum()
    return {
        "days_total": len(g),
        "days_clear": len(clear),
        "clear_fraction": len(clear) / len(g),
        "station_snow_days_all": int(g["station_snow"].sum()),
        "station_snow_days_clear": int(clear["station_snow"].sum()),
        "modis_snow_days_clear": int(clear["snow"].sum()),
        "agreement": (tp + tn) / len(clear),
        "omission_rate": fn / (tp + fn) if tp + fn else float("nan"),   # station snow missed by MODIS
        "commission_rate": fp / (fp + tn) if fp + tn else float("nan"),  # MODIS snow where station has none
        "station_snow_frac_all": g["station_snow"].mean(),
        "station_snow_frac_clear": clear["station_snow"].mean(),
        "modis_snow_frac_clear": clear["snow"].mean(),
    }


def main():
    m = load_matched()
    rows = [{"station_id": "ALL", **summarize(m)}]
    for sid, g in m.groupby("station_id"):
        rows.append({"station_id": sid, **summarize(g)})
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "cloud_matched_summary.csv", index=False)

    by_year = []
    for (sid, hy), g in m.groupby(["station_id", "hydro_year"]):
        clear = g[g["cloud"] == 0]
        by_year.append(
            {
                "station_id": sid,
                "hydro_year": hy,
                "station_snow_days_clear": int(clear["station_snow"].sum()),
                "modis_snow_days_clear": int(clear["snow"].sum()),
                "days_clear": len(clear),
            }
        )
    by_year = pd.DataFrame(by_year)
    by_year["bias_clear"] = by_year["modis_snow_days_clear"] - by_year["station_snow_days_clear"]
    by_year.to_csv(DATA / "cloud_matched_by_year.csv", index=False)

    pd.set_option("display.width", 200)
    print(out.round(3).T.to_string())
    print(
        f"\nClear-day snow-day bias per station-year: mean {by_year['bias_clear'].mean():.1f} d, "
        f"RMSE {(by_year['bias_clear'] ** 2).mean() ** 0.5:.1f} d, "
        f"r = {by_year['modis_snow_days_clear'].corr(by_year['station_snow_days_clear']):.2f} "
        f"(n = {len(by_year)})"
    )


if __name__ == "__main__":
    main()
