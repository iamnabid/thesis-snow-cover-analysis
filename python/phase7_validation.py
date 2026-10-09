"""
Phase 7 -- Validation against ground and higher-resolution data

Two independent checks:
  1. CHMU station snow-depth (converted to a station-based SCD proxy) vs.
     the MODIS pixel covering that station -- bias/RMSE, stratified by
     elevation and forest cover. Framed as a bias indicator, not an
     absolute-accuracy benchmark, since a point station != a 500m pixel
     average (thesis Methodology 5.7's own framing -- keep this caveat in
     the write-up).
  2. Sentinel-2 (2017+) spot-checks for canopy underdetection in Sumava
     specifically (gee/phase7_sentinel_validation.py's output).

Thesis reference: Objective 6 (2.1), Methodology 5.7.

Input:
  data/exports/scd_zonal_<variant>.csv       (baseline, forward_fill, terra_aqua)
  data/exports/chmu_station_snow_depth.csv   (python/chmu_download.py's output:
                                               station_id, date, hydro_year, depth_cm)
  data/exports/chmu_station_zones.csv        (gee/phase7_station_zones.py's
                                               output -- run that script
                                               against sourced station
                                               coordinates first)
  data/exports/sentinel_canopy_validation.csv (gee/phase7_sentinel_validation.py's
                                               output)
Output: data/validation_results.csv (all variants), data/validation_summary_by_variant.csv,
        data/canopy_validation_summary.csv,
        figures/validation_scatter.png, figures/canopy_underdetection.png
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zone_ids import ZONE_ID_DECODE  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "exports"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"
STATION_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "validation_results.csv"
CANOPY_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "canopy_validation_summary.csv"

SNOW_DEPTH_THRESHOLD_CM = 1.0  # station snow-day threshold (cm); see thesis Section 5.7
RANGE_ID_TO_NAME = ZONE_ID_DECODE["range_id"]  # single source of truth, see zone_ids.py


def station_scd_from_depth(depth_series: pd.Series, threshold_cm: float = SNOW_DEPTH_THRESHOLD_CM) -> int:
    """Counts days where recorded snow depth exceeds the threshold."""
    return int((depth_series > threshold_cm).sum())


def build_station_scd(depth_df: pd.DataFrame) -> pd.DataFrame:
    """Reduces a daily station snow-depth table to one row per
    station/hydro-year SCD proxy.

    Expects depth_df to have station_id, hydro_year, and depth_cm columns,
    as written by chmu_download.py.
    """
    return (
        depth_df.groupby(["station_id", "hydro_year"])["depth_cm"]
        .apply(station_scd_from_depth)
        .reset_index(name="scd_station")
    )


def compare_station_to_pixel(
    station_scd_df: pd.DataFrame, station_zones_df: pd.DataFrame, modis_df: pd.DataFrame
) -> pd.DataFrame:
    """Joins per-station-year SCD proxy against the MODIS zone SCD at that
    station's location, computing bias per pair.

    modis_df is one scd_zonal_<variant>.csv. The station proxy counts every
    calendar day, so the clear-sky-only baseline variant (~71% cloud) is not
    like-for-like with it -- run this for every variant and read the
    baseline bias as a cloud-treatment effect, not a detection-accuracy one.
    """
    with_zone = station_scd_df.merge(station_zones_df, on="station_id", how="left")
    unresolved = with_zone[with_zone["zone_id"].isna()]
    if len(unresolved) > 0:
        print(
            f"WARNING: {unresolved['station_id'].nunique()} station(s) have no "
            f"zone_id (see gee/phase7_station_zones.py's own warning for why) "
            f"and will be dropped from this comparison: "
            f"{sorted(unresolved['station_id'].unique().tolist())}"
        )
    with_zone = with_zone.dropna(subset=["zone_id"])

    merged = with_zone.merge(modis_df, on=["zone_id", "hydro_year"], suffixes=("", "_modis"))
    merged["bias"] = merged["mean_scd"] - merged["scd_station"]
    return merged


def plot_validation_scatter(merged: pd.DataFrame, out_path: Path):
    """Forward-filled MODIS SCD (filled markers, coloured by range) against the
    station proxy, with the clear-sky-only baseline shown as hollow grey
    markers to make the cloud-treatment effect visible."""
    range_names = {1: "Krkonoše", 2: "Šumava", 3: "Hrubý Jeseník"}
    colors = {1: "#1f77b4", 2: "#2ca02c", 3: "#d62728"}
    base = merged[merged["variant"] == "baseline"]
    ff = merged[merged["variant"] == "forward_fill"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(base["scd_station"], base["mean_scd"], facecolors="none", edgecolors="grey",
               alpha=0.8, label="Baseline (clear-sky only)")
    for rid, name in range_names.items():
        sub = ff[ff["range_id"] == rid]
        if len(sub):
            ax.scatter(sub["scd_station"], sub["mean_scd"], alpha=0.8,
                       label=f"Forward-fill: {name}", color=colors[rid])
    lims = [0, max(merged["scd_station"].max(), merged["mean_scd"].max()) + 10]
    ax.plot(lims, lims, "k--", linewidth=1, label="1:1 line")
    ax.set_xlabel("Station-derived SCD (days)")
    ax.set_ylabel("MODIS pixel SCD (days)")
    ax.set_title("MODIS vs. station SCD (bias indicator, not accuracy benchmark)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def run_station_comparison():
    depth_path = DATA_DIR / "chmu_station_snow_depth.csv"
    zones_path = DATA_DIR / "chmu_station_zones.csv"
    if not depth_path.exists():
        print(f"SKIP station comparison: {depth_path} not found -- source CHMU station data first.")
        return
    if not zones_path.exists():
        print(
            f"SKIP station comparison: {zones_path} not found -- run "
            f"gee/phase7_station_zones.py against sourced station coordinates first."
        )
        return

    depth_df = pd.read_csv(depth_path)
    zones_df = pd.read_csv(zones_path)
    station_scd_df = build_station_scd(depth_df)

    parts = []
    for variant in ["baseline", "terra_aqua", "forward_fill"]:
        modis_df = pd.read_csv(DATA_DIR / f"scd_zonal_{variant}.csv")
        m = compare_station_to_pixel(station_scd_df, zones_df, modis_df)
        m["variant"] = variant
        parts.append(m)
    merged = pd.concat(parts, ignore_index=True)

    if len(merged) == 0:
        print("No overlapping station/zone/hydro_year rows -- nothing to compare.")
        return

    summary_rows = []
    for variant, g in merged.groupby("variant"):
        row = {
            "variant": variant,
            "n": len(g),
            "mean_station_scd": g["scd_station"].mean(),
            "mean_modis_scd": g["mean_scd"].mean(),
            "mean_bias": g["bias"].mean(),
            "rmse": (g["bias"] ** 2).mean() ** 0.5,
            "pearson_r": g["mean_scd"].corr(g["scd_station"]),
        }
        for rid, name in RANGE_ID_TO_NAME.items():
            row[f"bias_{name}"] = g.loc[g["range_id"] == rid, "bias"].mean()
        summary_rows.append(row)
        print(f"{variant}: n={row['n']}, mean bias={row['mean_bias']:.1f} d, "
              f"RMSE={row['rmse']:.1f} d, r={row['pearson_r']:.2f}")
    pd.DataFrame(summary_rows).to_csv(STATION_OUTPUT_PATH.with_name("validation_summary_by_variant.csv"), index=False)

    merged.to_csv(STATION_OUTPUT_PATH, index=False)
    print(f"Wrote {len(merged)} rows to {STATION_OUTPUT_PATH}")
    plot_validation_scatter(merged, FIGURES_DIR / "validation_scatter.png")


def summarize_canopy_validation():
    """Aggregates gee/phase7_sentinel_validation.py's per-date, per-forest-class
    output across all curated dates: mean canopy-underdetection (modis_miss)
    rate in forest vs. open, plus a paired comparison across dates."""
    path = DATA_DIR / "sentinel_canopy_validation.csv"
    if not path.exists():
        print(f"SKIP canopy validation summary: {path} not found -- run gee/phase7_sentinel_validation.py first.")
        return

    df = pd.read_csv(path)
    summary = df.groupby("forest")[["s2_snow_fraction", "modis_snow", "modis_miss"]].mean()
    print("Canopy validation summary (0=open, 1=forested), averaged across curated dates:")
    print(summary)

    forest_dates = df[df["forest"] == 1].set_index("date")["modis_miss"]
    open_dates = df[df["forest"] == 0].set_index("date")["modis_miss"]
    paired = pd.concat([forest_dates, open_dates], axis=1, keys=["forest", "open"]).dropna()
    if len(paired) >= 4:
        t_stat, p = stats.ttest_rel(paired["forest"], paired["open"])
        print(f"\nPaired t-test (forest vs. open modis_miss rate across {len(paired)} dates): "
              f"t={t_stat:.3f}, p={p:.4f}")
    else:
        print(f"\nOnly {len(paired)} dates with both forest and open rows -- too few for a paired t-test "
              f"(need >= 4). Add more CURATED_DATES in gee/phase7_sentinel_validation.py.")

    summary.to_csv(CANOPY_OUTPUT_PATH)
    print(f"\nWrote canopy validation summary to {CANOPY_OUTPUT_PATH}")

    fig, ax = plt.subplots(figsize=(5, 4))
    summary["modis_miss"].plot.bar(ax=ax)
    ax.set_xticklabels(["Open", "Forested"], rotation=0)
    ax.set_ylabel("Mean MODIS miss rate\n(MODIS no-snow, S2 shows sub-pixel snow)")
    ax.set_title("Canopy underdetection: MODIS vs. Sentinel-2, Sumava")
    fig.tight_layout()
    out_fig = FIGURES_DIR / "canopy_underdetection.png"
    fig.savefig(out_fig, dpi=150)
    print(f"Saved {out_fig}")


def main():
    run_station_comparison()
    print()
    summarize_canopy_validation()


if __name__ == "__main__":
    main()
