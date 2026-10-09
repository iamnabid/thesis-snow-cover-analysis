"""
Phase 8 -- Robustness and inference checks (multiplicity, Sen-slope confidence
intervals, detrended climate correlations, field significance, endpoint
sensitivity, forward-fill limit).

Reads the same exports as Phases 4, 5 and 7; does not modify any existing file.

Input:
  data/exports/scd_zonal_<variant>.csv
  data/exports/era5_land_winter.csv
  data/exports/station_modis_daily.csv
  data/validation_results.csv           (station SCD per station-year, Phase 7)
Output (all new files under data/):
  robustness_multiplicity.csv     BH / Bonferroni counts per variant x test type
  robustness_sen_ci.csv           Sen slope + 95% CI per stratum x variant
  robustness_detrended_climate.csv  raw and detrended SCD-climate correlations
  robustness_field_significance.csv permutation field-significance test
  robustness_range_means.csv      MK on the three range-mean series
  robustness_endpoints.csv        significant-stratum counts after trimming years
  robustness_fill_limit.csv       station-pixel bias vs forward-fill limit
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pymannkendall as mk
from scipy import stats

DATA = Path(__file__).resolve().parents[1] / "data"
EXP = DATA / "exports"
VARIANTS = ["baseline", "forward_fill", "terra_aqua"]
RNG = np.random.default_rng(20260921)


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, p[i] * n / k)
        q[i] = prev
    return q


def load(variant):
    return pd.read_csv(EXP / f"scd_zonal_{variant}.csv").sort_values(["zone_id", "hydro_year"])


def series_by_zone(df, col="mean_scd"):
    return {z: g[col].values for z, g in df.groupby("zone_id")}


# ---------------------------------------------------------------- 1. multiplicity
def multiplicity():
    tr = pd.read_csv(DATA / "trend_results.csv")
    rows = []
    for v in VARIANTS:
        t = tr[(tr.variant == v) & (tr.metric == "mean_scd")]
        for label, col in [("plain", "p_value"), ("prewhitened", "p_value_prewhitened")]:
            p = t[col].values
            q = bh(p)
            rows.append({
                "variant": v, "test": label, "n_tests": len(p),
                "raw_p_lt_0.05": int((p < 0.05).sum()),
                "bh_q_lt_0.05": int((q < 0.05).sum()),
                "bh_q_lt_0.10": int((q < 0.10).sum()),
                "bonferroni_p_lt_0.05": int((p * len(p) < 0.05).sum()),
                "all_slopes_negative": bool((t.sen_slope < 0).all()),
            })
    # whole output file family size
    rows.append({"variant": "ALL", "test": "family_size_rows_x2",
                 "n_tests": len(tr) * 2})
    pd.DataFrame(rows).to_csv(DATA / "robustness_multiplicity.csv", index=False)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2. Sen CIs
def sen_ci():
    rows = []
    for v in VARIANTS:
        df = load(v)
        for z, g in df.groupby("zone_id"):
            y = g["mean_scd"].values
            x = g["hydro_year"].values.astype(float)
            slope, intercept, lo, hi = stats.theilslopes(y, x, alpha=0.95)
            mkres = mk.original_test(y)
            rows.append({
                "variant": v, "zone_id": z,
                "band": int(g["elevation_band"].iloc[0]), "n": len(y),
                "sen_slope": slope, "ci95_low": lo, "ci95_high": hi,
                "ci_excludes_zero": bool(hi < 0 or lo > 0),
                "mk_p": mkres.p, "mean_scd": y.mean(),
                "slope_pct_of_mean_per_yr": 100 * slope / y.mean(),
            })
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "robustness_sen_ci.csv", index=False)
    return out


# ---------------------------------------------------------------- 3. detrended climate
def detrended_climate():
    scd = load("baseline")
    era = pd.read_csv(EXP / "era5_land_winter.csv")
    rows = []
    for z, g in scd.groupby("zone_id"):
        m = g.merge(era[era.zone_id == z], on="hydro_year", suffixes=("", "_e")).sort_values("hydro_year")
        yr = m["hydro_year"].values.astype(float)

        def detr(a):
            b = np.polyfit(yr, a, 1)
            return a - np.polyval(b, yr)

        s, t, p_ = m["mean_scd"].values, m["winter_temp"].values, m["winter_precip"].values
        rt, pt = stats.pearsonr(s, t)
        rp, pp = stats.pearsonr(s, p_)
        dt, dpt = stats.pearsonr(detr(s), detr(t))
        dp, dpp = stats.pearsonr(detr(s), detr(p_))
        rows.append({"zone_id": z, "band": int(m["elevation_band"].iloc[0]), "n": len(m),
                     "temp_r": rt, "temp_p": pt, "precip_r": rp, "precip_p": pp,
                     "temp_r_detr": dt, "temp_p_detr": dpt,
                     "precip_r_detr": dp, "precip_p_detr": dpp})
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "robustness_detrended_climate.csv", index=False)
    return out


# ---------------------------------------------------------------- 4. field significance
def count_sig(mat, alpha=0.05):
    """mat: years x strata. Count strata whose Mann-Kendall p < alpha (same test as Phase 4)."""
    c = 0
    for j in range(mat.shape[1]):
        c += mk.original_test(mat[:, j]).p < alpha
    return c


def field_significance(n_perm=2000):
    df = load("baseline")
    zones = sorted(df.zone_id.unique())
    mat = np.column_stack([df[df.zone_id == z].sort_values("hydro_year")["mean_scd"].values for z in zones])
    obs = count_sig(mat)
    null = np.empty(n_perm, int)
    for i in range(n_perm):
        perm = RNG.permutation(mat.shape[0])  # same year shuffle for all strata keeps spatial correlation
        null[i] = count_sig(mat[perm, :])
    p_field = (1 + (null >= obs).sum()) / (1 + n_perm)
    out = pd.DataFrame([{
        "observed_significant_strata": int(obs), "n_strata": mat.shape[1],
        "n_perm": n_perm, "null_mean": null.mean(), "null_95th": np.percentile(null, 95),
        "null_max": null.max(), "field_p": p_field,
        "note": "years permuted jointly across strata (preserves cross-strata correlation; ignores autocorrelation)",
    }])
    out.to_csv(DATA / "robustness_field_significance.csv", index=False)
    return out


def range_means():
    rows = []
    for v in VARIANTS:
        df = load(v)
        for rid, g in df.groupby("range_id"):
            s = g.groupby("hydro_year")["mean_scd"].mean().values
            x = np.sort(g.hydro_year.unique()).astype(float)
            r = mk.original_test(s)
            slope, _, lo, hi = stats.theilslopes(s, x, alpha=0.95)
            rows.append({"variant": v, "range_id": rid, "n": len(s), "tau": r.Tau, "p": r.p,
                         "sen_slope": slope, "ci95_low": lo, "ci95_high": hi})
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "robustness_range_means.csv", index=False)
    return out


# ---------------------------------------------------------------- 5. endpoints
def endpoints():
    df = load("baseline")
    rows = []
    zones = sorted(df.zone_id.unique())
    for label, sl in [("full", slice(0, None)), ("drop_first_3", slice(3, None)),
                      ("drop_last_3", slice(0, -3)), ("drop_first_2_last_2", slice(2, -2))]:
        sig = neg = 0
        for z in zones:
            y = df[df.zone_id == z].sort_values("hydro_year")["mean_scd"].values[sl]
            r = mk.original_test(y)
            sig += r.p < 0.05
            neg += r.slope < 0
        rows.append({"window": label, "n_years": len(y), "significant_strata": int(sig),
                     "negative_slope_strata": int(neg), "n_strata": len(zones)})
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "robustness_endpoints.csv", index=False)
    return out


# ---------------------------------------------------------------- 6. fill-limit
def ffill_limit(snow, cloud, limit):
    """Carry the last clear-sky class forward across cloudy days for at most `limit` days."""
    out = np.zeros(len(snow), float)
    last, gap = 0, 10**9
    for i in range(len(snow)):
        if cloud[i] == 0:
            last, gap = snow[i], 0
            out[i] = snow[i]
        else:
            gap += 1
            out[i] = last if gap <= limit else 0
    return out


def fill_limit():
    d = pd.read_csv(EXP / "station_modis_daily.csv").sort_values(["station_id", "date"])
    val = pd.read_csv(DATA / "validation_results.csv")
    val = val[val.variant == "baseline"][["station_id", "hydro_year", "scd_station", "range_id"]]
    rows = []
    for L in [0, 1, 3, 5, 7, 10, 15, 30]:
        recs = []
        for sid, g in d.groupby("station_id"):
            g = g.sort_values("date")
            filled = ffill_limit(g["snow"].values, g["cloud"].values, L)
            t = g.assign(filled=filled).groupby("hydro_year")["filled"].sum().reset_index()
            t["station_id"] = sid
            recs.append(t)
        m = pd.concat(recs).merge(val, on=["station_id", "hydro_year"])
        m["bias"] = m["filled"] - m["scd_station"]
        for rid, gg in list(m.groupby("range_id")) + [("ALL", m)]:
            r = stats.pearsonr(gg["filled"], gg["scd_station"])[0] if gg["filled"].std() > 0 else np.nan
            rows.append({"limit_days": L, "range_id": rid, "n": len(gg),
                         "mean_modis_pixel_scd": gg["filled"].mean(),
                         "mean_station_scd": gg["scd_station"].mean(),
                         "mean_bias": gg["bias"].mean(),
                         "rmse": float(np.sqrt((gg["bias"] ** 2).mean())), "pearson_r": r})
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "robustness_fill_limit.csv", index=False)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(multiplicity())
    ci = sen_ci()
    b = ci[ci.variant == "baseline"]
    print("Sen CI baseline: CI excludes zero in", int(b.ci_excludes_zero.sum()), "of", len(b))
    print(b.groupby("band")[["sen_slope", "ci95_low", "ci95_high"]].agg(["min", "max"]).round(2))
    dc = detrended_climate()
    print("raw temp sig", int((dc.temp_p < .05).sum()), "raw precip sig", int((dc.precip_p < .05).sum()))
    print("DETRENDED temp sig", int((dc.temp_p_detr < .05).sum()), "precip sig", int((dc.precip_p_detr < .05).sum()))
    print(dc.groupby("band")[["temp_p_detr", "precip_p_detr"]].apply(lambda g: (g < 0.05).sum()))
    print(field_significance())
    print(range_means().round(4))
    print(endpoints())
    print(fill_limit().round(2).to_string())
