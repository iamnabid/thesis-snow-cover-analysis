"""
One-time reshape for GEE zonal exports produced before the
zonal_means_for_image() / zonal_stats_for_date() unpacking fix.

Reducer.mean().repeat(n).group() returns each group as
{'zone_id': v, 'mean': [v0, v1, ..., vn]} -- a single 'mean' column holding a
list in band order, not one named key per band. Every CSV exported by
gee/phase3_zonal_export.py and gee/phase5_era5_export.py before 2026-09-18
has this shape (verified against real downloaded files: scd_zonal_baseline.csv,
scd_zonal_forward_fill.csv, scd_zonal_terra_aqua.csv, era5_land_winter.csv all
have a 'mean' column containing a stringified Python list instead of
mean_scd/cloud_pct/etc. as separate columns). The GEE-side functions are
fixed now (see their docstrings), so this script is only needed for files
exported before the fix -- re-running the GEE export scripts today would
produce clean CSVs directly and this wouldn't be needed.

Usage: python reshape_gee_csv.py <path.csv> <col1> <col2> ...
  where col1, col2, ... are the 'mean' list's values in order (always
  range_id, elevation_band, forest, then whatever the data image's bands
  were, in the order they were added -- check the corresponding
  prep_data_bands()/winter_climate_for_year() call for the exact order).
Overwrites the file in place after backing up the original as <path>.bak.
"""
import ast
import shutil
import sys
from pathlib import Path

import pandas as pd


def reshape(path: Path, column_names: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "mean" not in df.columns:
        raise ValueError(f"{path} has no 'mean' column -- already reshaped, or a different export shape.")

    parsed = df["mean"].apply(ast.literal_eval)
    n_values = len(parsed.iloc[0])
    if n_values != len(column_names):
        raise ValueError(
            f"{path}: 'mean' has {n_values} values per row but {len(column_names)} "
            f"column names were given -- check the band order."
        )

    expanded = pd.DataFrame(parsed.tolist(), columns=column_names, index=df.index)
    expanded["range_id"] = expanded["range_id"].astype(int)
    expanded["elevation_band"] = expanded["elevation_band"].astype(int)
    expanded["forest"] = expanded["forest"].astype(int)

    keep_cols = [c for c in df.columns if c not in ("mean", "system:index", ".geo")]
    result = pd.concat([df[keep_cols].reset_index(drop=True), expanded.reset_index(drop=True)], axis=1)
    return result


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    columns = sys.argv[2:]

    backup_path = csv_path.with_suffix(csv_path.suffix + ".bak")
    shutil.copy(csv_path, backup_path)

    result_df = reshape(csv_path, columns)
    result_df.to_csv(csv_path, index=False)
    print(f"Reshaped {csv_path} in place ({len(result_df)} rows). Original backed up to {backup_path}.")
    print(result_df.head(3))
