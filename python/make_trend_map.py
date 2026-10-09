# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "data" / "exports" / "scd_trend_map_phase1.tif"
OUT = Path(__file__).resolve().parents[1] / "figures" / "spatial_trend_map.png"


def main():
    # NOTE 2026-09-19: this raster's 'year_p-value' and 'scd_p-value' bands
    # (4 and 6) are unusable -- confirmed by direct inspection: 'year_tau' is
    # a degenerate self-correlation of the time band against itself (always
    # exactly 1.0, with GEE's ee.Reducer.kendallsCorrelation() returning NaN
    # p-value for that zero-variance case), and the real 'scd_p-value' band
    # is NaN for every valid pixel too -- GEE's per-pixel Kendall p-value
    # does not come out usable at n=24 with ties. 'slope' and 'scd_tau'
    # (bands 1 and 5) ARE real, valid numbers and are what this map plots.
    # Formal significance testing is therefore reported at the zone level
    # only (Table 6.1's Mann-Kendall p-values, computed in pandas/pymannkendall
    # on the zonal-mean series), not per pixel.
    with rasterio.open(SRC) as src:
        slope = src.read(1)  # band 'slope' -- Sen's slope, days/year
        bounds = src.bounds
        nodata = src.nodata

    valid = ~np.isnan(slope)
    if nodata is not None:
        valid &= slope != nodata

    slope_masked = np.ma.masked_where(~valid, slope)

    vmax = np.nanpercentile(np.abs(slope[valid]), 98)
    extent = [bounds.left / 1000, bounds.right / 1000, bounds.bottom / 1000, bounds.top / 1000]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        slope_masked,
        extent=extent,
        origin="upper",
        cmap="RdBu",
        vmin=-vmax,
        vmax=vmax,
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Sen's slope, snow cover duration (days/year)")
    ax.set_xlabel("Easting (km, MODIS sinusoidal)")
    ax.set_ylabel("Northing (km, MODIS sinusoidal)")
    ax.set_title("Pixel-level SCD trend, 2000/01–2023/24 (baseline variant)")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
