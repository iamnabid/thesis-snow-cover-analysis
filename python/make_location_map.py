# -*- coding: utf-8 -*-
"""Downloads the three range boundaries and the Czech national border from
GEE and plots a simple location map for Section 4 (para: 'insert the map of
Czechia with the three study areas marked')."""
import json
import sys
from pathlib import Path

import ee
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gee"))
from phase0_static_layers import load_aois  # noqa: E402

ee.Initialize(project="thesis101-492500")

OUT_DIR = Path(__file__).resolve().parents[1] / "data"
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

RANGE_LABELS = {
    "krkonose": "Krkonoše",
    "sumava": "Šumava",
    "hruby_jesenik": "Hrubý Jeseník",
}
RANGE_COLORS = {
    "krkonose": "#1f77b4",
    "sumava": "#2ca02c",
    "hruby_jesenik": "#d62728",
}


def fetch_aois():
    aois = load_aois()
    info = aois.getInfo()
    out_path = OUT_DIR / "aois.geojson"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(info, f)
    print(f"Saved {out_path} ({len(info['features'])} features)")
    return info


def fetch_czech_border():
    # USDOS LSIB: simplified national boundaries, stable/public GEE dataset
    countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    czechia = countries.filter(ee.Filter.eq("country_na", "Czechia"))
    info = czechia.getInfo()
    out_path = OUT_DIR / "czechia_border.geojson"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(info, f)
    print(f"Saved {out_path} ({len(info['features'])} features)")
    return info


def add_geom_patches(ax, geom, **kwargs):
    gtype = geom["type"]
    if gtype == "Polygon":
        rings = geom["coordinates"]
        polys = [rings[0]]  # exterior only, ignore holes for a location map
    elif gtype == "MultiPolygon":
        polys = [poly[0] for poly in geom["coordinates"]]
    else:
        return
    patches = [MplPolygon(p, closed=True) for p in polys]
    coll = PatchCollection(patches, **kwargs)
    ax.add_collection(coll)


def main():
    aois_info = fetch_aois()
    czechia_info = fetch_czech_border()

    fig, ax = plt.subplots(figsize=(7, 5))

    for feat in czechia_info["features"]:
        add_geom_patches(ax, feat["geometry"], facecolor="#eeeeee", edgecolor="black", linewidth=1, zorder=1)

    centroids = {}
    for feat in aois_info["features"]:
        range_name = feat["properties"]["range"]
        color = RANGE_COLORS[range_name]
        add_geom_patches(ax, feat["geometry"], facecolor=color, edgecolor=color, alpha=0.9, zorder=2)
        # crude centroid from exterior ring for label placement
        geom = feat["geometry"]
        ring = geom["coordinates"][0] if geom["type"] == "Polygon" else geom["coordinates"][0][0]
        lons = [c[0] for c in ring]
        lats = [c[1] for c in ring]
        centroids.setdefault(range_name, []).append((sum(lons) / len(lons), sum(lats) / len(lats)))

    for range_name, pts in centroids.items():
        lon = sum(p[0] for p in pts) / len(pts)
        lat = sum(p[1] for p in pts) / len(pts)
        label = RANGE_LABELS[range_name]
        offset_x, offset_y = {
            "krkonose": (0.15, 0.15),
            "sumava": (-0.9, -0.1),
            "hruby_jesenik": (0.2, 0.05),
        }[range_name]
        ax.annotate(
            label,
            xy=(lon, lat),
            xytext=(lon + offset_x, lat + offset_y),
            fontsize=10,
            fontweight="bold",
            color=RANGE_COLORS[range_name],
            arrowprops=dict(arrowstyle="-", color=RANGE_COLORS[range_name], lw=0.8),
        )

    ax.set_xlim(11.8, 19.0)
    ax.set_ylim(48.5, 51.1)
    ax.set_aspect(1.5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Study areas: three Czech mountain ranges")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / "study_area_map.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
