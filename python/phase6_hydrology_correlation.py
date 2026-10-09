"""
Phase 6 -- Hydrological correlation

Correlates satellite-derived snowmelt-out timing (Phase 1/3 output) against
observed spring streamflow peak timing at gauges downstream of each range.

Thesis reference: Objective 5 (2.1).

Input:
  data/exports/scd_zonal_baseline.csv     (Phase 3 output -- mean_meltout_day_idx
                                            is a day count since Oct 1 of that
                                            hydro-year, NOT calendar day-of-year;
                                            see phase1_scd_calculation.
                                            doy_to_hydro_day_index()'s docstring)
  data/exports/chmu_streamflow.csv        (chmu_download.py's output -- run
                                            that script first)

Output: data/hydrology_correlations.csv
"""
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

from chmu_download import GAUGE_IDS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zone_ids import ZONE_ID_DECODE  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "exports"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "hydrology_correlations.csv"

MELT_SEASON_MONTHS = (3, 4, 5, 6)  # March-June, to avoid autumn rain-driven peaks

# range_id -> name comes from zone_ids.ZONE_ID_DECODE (Phase 3's single
# source of truth) -> the same real gauge IDs chmu_download.py fetched, so
# there's one place (chmu_download.GAUGE_IDS) that decides which gauge
# represents each range rather than two lists that could drift apart.
RANGE_ID_TO_NAME = ZONE_ID_DECODE["range_id"]
RANGE_TO_GAUGE = {rid: GAUGE_IDS[name] for rid, name in RANGE_ID_TO_NAME.items()}


def extract_peak_flow_doy(discharge_df: pd.DataFrame, gauge_id: str, hydro_year: int) -> int:
    """Day count since Oct 1 of hydro_year (matching mean_meltout_day_idx's
    convention -- NOT calendar day-of-year) of maximum discharge, restricted
    to the snowmelt season (March-June) to avoid picking up autumn
    rain-driven peaks.

    March-June always falls within hydro_year's own ending calendar year (no
    Jan-1 wraparound to worry about, unlike onset/meltout's Oct-Sep window --
    see doy_to_hydro_day_index()'s docstring for why that one needs care),
    so the day-index conversion here is just "days since Oct 1", computed
    directly from real dates.
    """
    sub = discharge_df[discharge_df["gauge_id"] == gauge_id].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    melt_window = sub[
        (sub["date"].dt.year == hydro_year) & (sub["date"].dt.month.isin(MELT_SEASON_MONTHS))
    ]
    if melt_window.empty:
        raise ValueError(
            f"No discharge data for gauge {gauge_id}, hydro_year {hydro_year}, "
            f"in the Mar-Jun window."
        )

    peak_date = melt_window.loc[melt_window["discharge_m3s"].idxmax(), "date"]
    oct1 = pd.Timestamp(year=hydro_year - 1, month=10, day=1)
    return (peak_date - oct1).days


def main():
    scd_df = pd.read_csv(DATA_DIR / "scd_zonal_baseline.csv")

    chmu_path = DATA_DIR / "chmu_streamflow.csv"
    if not chmu_path.exists():
        print(f"SKIP: {chmu_path} not found -- see module docstring for how to source it.")
        return

    discharge_df = pd.read_csv(chmu_path)

    results = []
    for range_id, gauge_id in RANGE_TO_GAUGE.items():
        range_scd = scd_df[scd_df["range_id"] == range_id]
        for hydro_year, year_df in range_scd.groupby("hydro_year"):
            meltout_idx = year_df["mean_meltout_day_idx"].mean()
            try:
                peak_idx = extract_peak_flow_doy(discharge_df, gauge_id, hydro_year)
            except ValueError as e:
                print(f"  SKIP range_id={range_id} hydro_year={hydro_year}: {e}")
                continue
            results.append(
                {"range_id": range_id, "hydro_year": hydro_year,
                 "meltout_day_idx": meltout_idx, "peak_flow_day_idx": peak_idx}
            )

    if not results:
        print("No overlapping range/hydro_year rows had both SCD and discharge data -- nothing to correlate.")
        return

    df = pd.DataFrame(results)
    for range_id, range_df in df.groupby("range_id"):
        if len(range_df) < 4:
            print(f"range_id={range_id}: only {len(range_df)} year(s) with data -- too few for a "
                  f"meaningful correlation (need >= 4).")
            continue
        r, p = stats.pearsonr(range_df["meltout_day_idx"], range_df["peak_flow_day_idx"])
        print(f"range_id={range_id}: r={r:.3f}, p={p:.4f}, n={len(range_df)}")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
