"""
Phase 4 -- Trend detection (Mann-Kendall / Theil-Sen, with and without
trend-free prewhitening)

Reads the tidy CSVs exported from GEE (Phase 3) and, for each zone and each
cloud-treatment variant, runs the trend tests specified in the thesis
methodology: Mann-Kendall significance, Theil-Sen slope (point estimate),
both with and without trend-free prewhitening.

Thesis reference: Objective 2 (2.1), Methodology 5.4 (primary analysis without
prewhitening; prewhitened test reported as a robustness check in Results 6.1).

Input: data/exports/scd_zonal_<variant>.csv (one file per variant, from Phase 3)
Output: data/trend_results.csv (one row per zone x variant x metric)

mean_onset_day_idx / mean_meltout_day_idx are NOT calendar day-of-year -- they're
a day count since Oct 1 of that hydro-year (0 = Oct 1), so a trend in "days
earlier/later per decade" reads directly off the Sen slope without needing to
correct for the Jan-1 wraparound a calendar-DOY column would have. See
phase1_scd_calculation.doy_to_hydro_day_index()'s docstring for why.
"""
from pathlib import Path

import pandas as pd
import pymannkendall as mk

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "exports"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "trend_results.csv"

VARIANTS = ["baseline", "forward_fill", "terra_aqua"]
METRICS = ["mean_scd", "cloud_pct", "mean_onset_day_idx", "mean_meltout_day_idx"]


def load_variant(variant: str) -> pd.DataFrame:
    path = DATA_DIR / f"scd_zonal_{variant}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run Phase 3 and download its export from "
            f"Google Drive into {DATA_DIR} first."
        )
    return pd.read_csv(path)


def run_trend_tests(series: pd.Series) -> dict:
    """Runs both the plain Mann-Kendall/Theil-Sen test and the trend-free
    prewhitened version on one time series, returning both sets of results."""
    values = series.dropna().values
    if len(values) < 4:
        return {"n": len(values), "error": "series too short for trend test"}

    plain = mk.original_test(values)
    prewhitened = mk.trend_free_pre_whitening_modification_test(values)

    return {
        "n": len(values),
        "tau": plain.Tau,
        "p_value": plain.p,
        "sen_slope": plain.slope,
        "trend": plain.trend,
        "tau_prewhitened": prewhitened.Tau,
        "p_value_prewhitened": prewhitened.p,
        "sen_slope_prewhitened": prewhitened.slope,
        "trend_prewhitened": prewhitened.trend,
    }


def main():
    results = []
    for variant in VARIANTS:
        try:
            df = load_variant(variant)
        except FileNotFoundError as e:
            print(f"SKIP {variant}: {e}")
            continue

        for zone_id, zone_df in df.groupby("zone_id"):
            zone_df = zone_df.sort_values("hydro_year")
            for metric in METRICS:
                if metric not in zone_df.columns:
                    continue
                stats = run_trend_tests(zone_df[metric])
                results.append(
                    {"zone_id": zone_id, "variant": variant, "metric": metric, **stats}
                )

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(out)} trend-test rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
