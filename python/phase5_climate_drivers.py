"""
Phase 5 -- Climate driver relationship

Tests whether winter temperature or precipitation is the dominant driver of
SCD change at each elevation band -- the direct test of the Nedelcev &
Jenicek (2021) elevation-dependent hypothesis (temperature governs below
~900m, precipitation above ~1200m, as reported in that study) against this thesis's independent satellite record.

Thesis reference: Objective 4 (2.1), Section 3.8 (ERA5-Land as supporting data).

Input:
  data/exports/scd_zonal_baseline.csv  (Phase 3 output, baseline variant)
  data/exports/era5_land_winter.csv    (gee/phase5_era5_export.py's output --
                                         run that script, then download its
                                         CSV from Drive into data/exports/)
Output: data/climate_correlations.csv
"""
from pathlib import Path

import pandas as pd
from scipy import stats

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "exports"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "climate_correlations.csv"


def load_scd() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "scd_zonal_baseline.csv")


def load_era5() -> pd.DataFrame:
    path = DATA_DIR / "era5_land_winter.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- see the module docstring for the GEE export "
            f"needed to produce it (winter DJF temp + precip per zone/year)."
        )
    return pd.read_csv(path)


def correlate_zone(scd_df: pd.DataFrame, era5_df: pd.DataFrame, zone_id) -> dict:
    scd_z = scd_df[scd_df.zone_id == zone_id].sort_values("hydro_year")
    era5_z = era5_df[era5_df.zone_id == zone_id].sort_values("hydro_year")
    merged = scd_z.merge(era5_z, on="hydro_year", suffixes=("", "_era5"))

    if len(merged) < 4:
        return {"zone_id": zone_id, "n": len(merged), "error": "too few overlapping years"}

    temp_r, temp_p = stats.pearsonr(merged["mean_scd"], merged["winter_temp"])
    precip_r, precip_p = stats.pearsonr(merged["mean_scd"], merged["winter_precip"])

    return {
        "zone_id": zone_id,
        "n": len(merged),
        "temp_r": temp_r,
        "temp_p": temp_p,
        "precip_r": precip_r,
        "precip_p": precip_p,
        "dominant_driver": "temperature" if abs(temp_r) > abs(precip_r) else "precipitation",
    }


def main():
    scd_df = load_scd()
    era5_df = load_era5()

    results = [correlate_zone(scd_df, era5_df, z) for z in scd_df.zone_id.unique()]
    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(out)} zone-level climate correlations to {OUTPUT_PATH}")
    print(
        "Compare 'dominant_driver' per elevation band with the ~900 m / ~1200 m "
        "thresholds of Nedelcev & Jenicek (2021); this thesis's bands are "
        "<800 m, 800-1200 m and >1200 m."
    )


if __name__ == "__main__":
    main()
