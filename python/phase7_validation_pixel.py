"""
Phase 7 follow-up -- station validation recalculated at the MODIS pixel
containing each station, instead of the mean SCD of the station's
containing zone.

Supervisor comment (2026-10-19, inline on Sec 6.6): "double-check your
script against the interpretation, i would consider recalculating the
annual validation using the MODIS pixel containing each station rather
than the mean SCD of the corresponding zone."

phase7_validation.py's compare_station_to_pixel() joins the station SCD
proxy to scd_zonal_<variant>.csv's zone-mean SCD (station -> zone_id ->
zone mean, a many-to-one lookup: every pixel in that station's zone,
averaged). This script instead builds annual SCD directly from the daily
snow/cloud series sampled AT the station's own MODIS pixel
(gee/phase7_station_daily_pixels.py for Terra,
gee/phase7_station_daily_pixels_aqua.py for Aqua), for all three
cloud-treatment variants, and reruns the same bias/RMSE/r comparison.

Input:
  data/exports/station_modis_daily.csv       (Terra, pixel-level, baseline source)
  data/exports/station_modis_daily_aqua.csv  (Aqua, pixel-level)
  data/exports/chmu_station_snow_depth.csv
  data/exports/chmu_station_zones.csv        (for range_id/range-name labelling only)
Output: data/validation_results_pixel.csv, data/validation_summary_by_variant_pixel.csv
"""
from pathlib import Path

import pandas as pd

from phase7_validation import build_station_scd, SNOW_DEPTH_THRESHOLD_CM
from zone_ids import ZONE_ID_DECODE

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "exports"
OUT_RESULTS = Path(__file__).resolve().parents[1] / "data" / "validation_results_pixel.csv"
OUT_SUMMARY = Path(__file__).resolve().parents[1] / "data" / "validation_summary_by_variant_pixel.csv"

AQUA_START_YEAR = 2003
FORWARD_FILL_MAX_DAYS = 5
RANGE_ID_TO_NAME = ZONE_ID_DECODE["range_id"]


def baseline_annual(terra: pd.DataFrame) -> pd.DataFrame:
    """Pixel-level baseline SCD: clear-sky snow days per station-year,
    straight from the Terra daily series at the station's own pixel --
    exactly what variant_a_baseline() computes per pixel in GEE, just
    evaluated at one point instead of a full raster."""
    g = terra.dropna(subset=["snow", "cloud"])
    scd = (
        g[g["cloud"] == 0]
        .groupby(["station_id", "hydro_year"])["snow"]
        .sum()
        .reset_index(name="scd_pixel")
    )
    return scd


def forward_fill_annual(terra: pd.DataFrame) -> pd.DataFrame:
    """Pixel-level forward-fill SCD: per station, per hydro-year, carries the
    last clear-sky classification forward up to FORWARD_FILL_MAX_DAYS on
    cloudy days -- the same rule as gee/phase2_cloud_variants.py's
    forward_fill_stack(), applied to one pixel's daily series instead of a
    raster, with the state reset at the start of each hydro-year (matching
    variant_b_forward_fill(), which filters to one hydro-year before
    running the fill)."""
    rows = []
    for (sid, hy), g in terra.dropna(subset=["snow", "cloud"]).groupby(["station_id", "hydro_year"]):
        g = g.sort_values("date")
        last_valid = 0
        days_since = FORWARD_FILL_MAX_DAYS + 1
        filled_snow_sum = 0
        for _, row in g.iterrows():
            if row["cloud"] == 0:
                last_valid = int(row["snow"])
                days_since = 0
                filled_snow_sum += last_valid
            else:
                days_since += 1
                if days_since <= FORWARD_FILL_MAX_DAYS:
                    filled_snow_sum += last_valid
                # else: stays unresolved (counts as 0), same as GEE's can_fill=False branch
        rows.append({"station_id": sid, "hydro_year": hy, "scd_pixel": filled_snow_sum})
    return pd.DataFrame(rows)


def terra_aqua_annual(terra: pd.DataFrame, aqua: pd.DataFrame) -> pd.DataFrame:
    """Pixel-level Terra+Aqua SCD: snow = Terra OR Aqua, cloud = Terra AND
    Aqua, matching combine_terra_aqua_stack()'s pixelwise logic and its
    left-join behaviour (a Terra day with no Aqua match falls back to
    Terra-only, i.e. aqua treated as equal to terra for that day).
    Restricted to hydro years >= AQUA_START_YEAR, matching Phase 2."""
    t = terra.dropna(subset=["snow", "cloud"])[["station_id", "date", "hydro_year", "snow", "cloud"]]
    a = aqua.dropna(subset=["snow", "cloud"])[["station_id", "date", "snow", "cloud"]]
    m = t.merge(a, on=["station_id", "date"], how="left", suffixes=("_t", "_a"))
    no_match = m["snow_a"].isna()
    m.loc[no_match, "snow_a"] = m.loc[no_match, "snow_t"]
    m.loc[no_match, "cloud_a"] = m.loc[no_match, "cloud_t"]
    m["snow_c"] = ((m["snow_t"] == 1) | (m["snow_a"] == 1)).astype(int)
    m["cloud_c"] = ((m["cloud_t"] == 1) & (m["cloud_a"] == 1)).astype(int)
    m = m[m["hydro_year"] >= AQUA_START_YEAR]
    scd = (
        m[m["cloud_c"] == 0]
        .groupby(["station_id", "hydro_year"])["snow_c"]
        .sum()
        .reset_index(name="scd_pixel")
    )
    return scd


def run():
    terra = pd.read_csv(DATA_DIR / "station_modis_daily.csv")
    aqua = pd.read_csv(DATA_DIR / "station_modis_daily_aqua.csv")
    depth = pd.read_csv(DATA_DIR / "chmu_station_snow_depth.csv")
    zones = pd.read_csv(DATA_DIR / "chmu_station_zones.csv")

    station_scd_df = build_station_scd(depth)

    variants = {
        "baseline": baseline_annual(terra),
        "forward_fill": forward_fill_annual(terra),
        "terra_aqua": terra_aqua_annual(terra, aqua),
    }

    parts = []
    for name, pix in variants.items():
        m = station_scd_df.merge(pix, on=["station_id", "hydro_year"], how="inner")
        m = m.merge(zones[["station_id", "range_id"]], on="station_id", how="left")
        m["bias"] = m["scd_pixel"] - m["scd_station"]
        m["variant"] = name
        parts.append(m)
    merged = pd.concat(parts, ignore_index=True)
    merged.to_csv(OUT_RESULTS, index=False)

    summary_rows = []
    for variant, g in merged.groupby("variant"):
        row = {
            "variant": variant,
            "n": len(g),
            "mean_station_scd": g["scd_station"].mean(),
            "mean_pixel_scd": g["scd_pixel"].mean(),
            "mean_bias": g["bias"].mean(),
            "rmse": (g["bias"] ** 2).mean() ** 0.5,
            "pearson_r": g["scd_pixel"].corr(g["scd_station"]),
        }
        for rid, name in RANGE_ID_TO_NAME.items():
            sub = g.loc[g["range_id"] == rid, "bias"]
            row[f"bias_{name}"] = sub.mean() if len(sub) else float("nan")
        summary_rows.append(row)
        print(
            f"{variant}: n={row['n']}, mean bias={row['mean_bias']:.1f} d, "
            f"RMSE={row['rmse']:.1f} d, r={row['pearson_r']:.2f}"
        )
        for rid, name in RANGE_ID_TO_NAME.items():
            print(f"    {name}: bias={row[f'bias_{name}']:.1f} d")
    pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False)
    print(f"\nWrote {len(merged)} rows to {OUT_RESULTS}")
    print(f"Wrote variant summary to {OUT_SUMMARY}")


def plot_pixel_validation_scatter(out_path):
    """Figure 6.2 replacement: station-pixel SCD (not zone-mean) against the
    station-derived proxy, same visual convention as the original
    phase7_validation.plot_validation_scatter()."""
    import matplotlib.pyplot as plt
    merged = pd.read_csv(OUT_RESULTS)
    range_names = {1: "Krkonose", 2: "Sumava", 3: "Hruby Jesenik"}
    colors = {1: "#1f77b4", 2: "#2ca02c", 3: "#d62728"}
    base = merged[merged["variant"] == "baseline"]
    ff = merged[merged["variant"] == "forward_fill"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(base["scd_station"], base["scd_pixel"], facecolors="none", edgecolors="grey",
               alpha=0.8, label="Baseline (clear-sky only)")
    for rid, name in range_names.items():
        sub = ff[ff["range_id"] == rid]
        if len(sub):
            ax.scatter(sub["scd_station"], sub["scd_pixel"], alpha=0.8,
                       label=f"Forward-fill: {name}", color=colors[rid])
    lims = [0, max(merged["scd_station"].max(), merged["scd_pixel"].max()) + 10]
    ax.plot(lims, lims, "k--", linewidth=1, label="1:1 line")
    ax.set_xlabel("Station-derived SCD (days)")
    ax.set_ylabel("MODIS pixel SCD (days)")
    ax.set_title("MODIS vs. station SCD (pixel-level; bias indicator, not accuracy benchmark)",
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    run()
    plot_pixel_validation_scatter(Path(__file__).resolve().parents[1] / "figures" / "validation_scatter.png")
