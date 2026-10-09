"""
CHMU open-data downloader for Phase 6 (streamflow) and Phase 7 (station snow depth)

Downloads directly from opendata.chmi.cz -- free, no login, CC BY 4.0, thanks
to the National Database of Hydrometeorological Data and Products (law
262/2024 Sb., effective 2025-01-01). Every URL pattern and field name below
was verified against the live portal on 2026-09-18, not assumed from
documentation:

  Discharge:   hydrology/historical/data/daily/H_<gauge_id>_DQ_<year>.json
               One file per gauge per year. Each file's "tsList" holds two
               series -- "HD" (water level, cm) and "QD" (discharge, m3/s).
               We want QD only.

  Snow depth:  meteorology/climate/historical/data/daily/dly-0-203-0-<station_id>.json
               One file per station covering its ENTIRE history (can be
               tens of MB -- there's no per-year endpoint for this one).
               Each row is [station, element, vtype, date, value, flag,
               quality]; SCE is "Vyska snehu" (snow depth, cm), confirmed
               against meteorology/climate/historical_csv/metadata/meta2.csv.
               (SNO, easy to confuse with it, is new/fresh snowfall, not
               ground snow depth -- its values fluctuate day to day instead
               of tracking a slowly-changing snowpack.)

GAUGE_IDS / STATION_IDS below are REAL station IDs, not made-up placeholders
-- pulled from CHMU's own metadata catalogs (hydrology/historical/metadata/
meta1.json; meteorology/climate/historical_csv/metadata/meta1.csv) and picked
for being small headwater catchments / high-elevation stations near each
range. Still worth confirming against the interactive station map
(https://storymaps.arcgis.com/stories/9296c701a3494ed69d0150dd9eeb295f)
before treating them as final -- there are several reasonable candidates per
range (see the comments), and picking the single best one is a judgment call
about which catchment/station best represents each range, not something to
auto-decide.

Output:
  data/exports/chmu_streamflow.csv          (gauge_id, range, date, hydro_year, discharge_m3s)
  data/exports/chmu_stations.csv            (station_id, range, lat, lon)
  data/exports/chmu_station_snow_depth.csv  (station_id, range, date, hydro_year, depth_cm)
"""
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "exports"
BASE_URL = "https://opendata.chmi.cz"

# Real candidate gauges near each range, area in km^2 (small = close to the
# mountain source, not a big downstream river already fed by lowland tributaries):
#   krkonose:      Labsky dul (Labe, 4.4 km^2, right at Labe's source spring) [chosen]
#                  alternatives: Obri dul (0-203-1-009000, Upa, 9.0 km^2),
#                  Spindleruv Mlyn (0-203-1-001000, Labe, 53.1 km^2)
#   sumava:        Modrava (Vydra, 89.8 km^2) [chosen]
#                  alternatives: Cerny Kriz (0-203-1-108000, Studena Vltava,
#                  103.1 km^2), Rejstejn (0-203-1-137000, Otava, 333.5 km^2)
#   hruby_jesenik: Kouty nad Desnou (Desna, 43.3 km^2) [chosen 2026-09-18 --
#                  the original pick, Ludvikov (0-203-1-260000), turned out to
#                  have discharge records only through 1997, no recent data at
#                  all; confirmed Kouty nad Desnou has real files through 2025
#                  before switching]
#                  alternative: Karlov (0-203-1-267000, Moravice, 20.7 km^2 --
#                  also stops in 1997, don't use)
GAUGE_IDS = {
    "krkonose": "0-203-1-000400",
    "sumava": "0-203-1-135000",
    "hruby_jesenik": "0-203-1-346000",
}

# Real candidate climate stations with SCE (snow depth) records:
#   krkonose:      Pomezni boudy / Horni Mala Upa, 1050m [chosen 2026-09-19 --
#                  the original pick, Svaty Petr / Spindleruv Mlyn
#                  (0-203-0-10101004001), turned out to have SCE data only
#                  through 1995-01, a 30-year gap before it resumed in 2025 --
#                  confirmed via CHMU's meteorology/climate/historical_csv/
#                  metadata/meta1.csv that Pomezni boudy has continuous SCE
#                  coverage for all 24 study hydro-years (min 365 days/year,
#                  2001-2024) before switching]
#   sumava:        Kvilda, ~1060m [chosen -- confirmed continuous 1911-2026,
#                  no gap in the study period]
#                  alternative: Horska Kvilda (0-203-0-11494, ~1050m)
#   hruby_jesenik: Karlova Studanka, 795m [chosen 2026-09-19 -- the original
#                  pick, Praded summit (0-203-0-11735), turned out to have SCE
#                  data only through 1997-09, a 27-year gap before it resumed
#                  in 2025 -- confirmed Karlova Studanka has continuous SCE
#                  coverage for all 24 study hydro-years (min 365 days/year,
#                  2001-2024) before switching]
STATION_IDS = {
    "krkonose": "0-203-0-10102006101",
    "sumava": "0-203-0-10601002002",
    "hruby_jesenik": "0-203-0-11733",
}

STATION_COORDS = {
    # lat, lon -- from meteorology/climate/historical_csv/metadata/meta1.csv
    "0-203-0-10102006101": (50.745472, 15.82027),
    "0-203-0-10601002002": (49.0197, 13.5781),
    "0-203-0-11733": (50.074053, 17.306832),
}

YEARS = range(2001, 2026)  # covers the thesis's 2000/01-2023/24 hydro-years plus margin


def hydro_year(date: pd.Timestamp) -> int:
    """Matches gee/phase1_scd_calculation.HYDRO_YEARS's convention: named by
    the ending calendar year (Oct-Dec belongs to the FOLLOWING year's hydro-year)."""
    return date.year + 1 if date.month >= 10 else date.year


def fetch_json(url: str) -> dict | None:
    resp = requests.get(url, timeout=60)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def download_discharge() -> pd.DataFrame:
    rows = []
    for range_name, gauge_id in GAUGE_IDS.items():
        for year in YEARS:
            url = f"{BASE_URL}/hydrology/historical/data/daily/H_{gauge_id}_DQ_{year}.json"
            data = fetch_json(url)
            if data is None:
                continue
            qd_series = next((ts for ts in data["tsList"] if ts["tsConID"] == "QD"), None)
            if qd_series is None:
                continue
            for date_str, value in qd_series["tsData"]["data"]["values"]:
                date = pd.Timestamp(date_str)
                rows.append(
                    {
                        "gauge_id": gauge_id,
                        "range": range_name,
                        "date": date.date().isoformat(),
                        "hydro_year": hydro_year(date),
                        "discharge_m3s": value,
                    }
                )
        n_rows = sum(1 for r in rows if r["range"] == range_name)
        print(f"  {range_name} ({gauge_id}): {n_rows} daily rows so far")
        if n_rows == 0:
            print(
                f"    WARNING: 0 rows for {gauge_id} across all of {YEARS} -- this gauge's "
                f"record may not overlap this range at all (e.g. it stopped reporting "
                f"years ago). Check its real year coverage before trusting this gauge."
            )
    return pd.DataFrame(rows)


def download_station_snow_depth() -> pd.DataFrame:
    rows = []
    for range_name, station_id in STATION_IDS.items():
        url = f"{BASE_URL}/meteorology/climate/historical/data/daily/dly-{station_id}.json"
        print(f"  Downloading {range_name} ({station_id}) -- this file covers the station's full history, can be tens of MB...")
        data = fetch_json(url)
        if data is None:
            print(f"    WARNING: {url} returned 404 -- station ID may be wrong, check the station map.")
            continue
        values = data["data"]["data"]["values"]
        for station, element, _vtype, date_str, value, _flag, _quality in values:
            if element != "SCE" or value == "":
                continue
            date = pd.Timestamp(date_str)
            rows.append(
                {
                    "station_id": station,
                    "range": range_name,
                    "date": date.date().isoformat(),
                    "hydro_year": hydro_year(date),
                    "depth_cm": value,
                }
            )
    return pd.DataFrame(rows)


def build_stations_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"station_id": sid, "range": range_name, "lat": STATION_COORDS[sid][0], "lon": STATION_COORDS[sid][1]}
            for range_name, sid in STATION_IDS.items()
        ]
    )


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading discharge data (Phase 6)...")
    discharge_df = download_discharge()
    out = DATA_DIR / "chmu_streamflow.csv"
    discharge_df.to_csv(out, index=False)
    print(f"Wrote {len(discharge_df)} rows to {out}\n")

    print("Downloading station snow depth data (Phase 7)...")
    depth_df = download_station_snow_depth()
    out = DATA_DIR / "chmu_station_snow_depth.csv"
    depth_df.to_csv(out, index=False)
    print(f"Wrote {len(depth_df)} rows to {out}\n")

    stations_df = build_stations_table()
    out = DATA_DIR / "chmu_stations.csv"
    stations_df.to_csv(out, index=False)
    print(f"Wrote {len(stations_df)} rows to {out}")
    print(
        "\nNext: run gee/phase7_station_zones.py against chmu_stations.csv to "
        "assign each station a zone_id before python/phase7_validation.py can "
        "use chmu_station_snow_depth.csv."
    )


if __name__ == "__main__":
    main()
