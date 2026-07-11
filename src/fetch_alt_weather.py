"""Fetch historical weather for alternative models and build parallel training caches.

Usage:
    python3 src/fetch_alt_weather.py                         # default: IFS cache start → yesterday
    python3 src/fetch_alt_weather.py --start 2025-05-10 --end 2026-07-10
"""
import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from weather import get_historical_forecast

BUILDERS = Path(__file__).parent.parent
MAIN_CACHE = BUILDERS / "data" / "weather_cache.json"
NBM_CACHE  = BUILDERS / "data" / "weather_cache_nbm.json"
ENS_CACHE  = BUILDERS / "data" / "weather_cache_ensemble.json"

# The 6 models for ensemble (best_match is already in main cache)
ENSEMBLE_MODELS = [
    "best_match",
    "ecmwf_ifs025",
    "ncep_nbm_conus",
    "ukmo_seamless",
    "meteofrance_seamless",
    "jma_seamless",
]


def load_cache(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"hf_daily": {}, "hf_hourly": {}}


def save_cache(path, cache):
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def overlay_soil(daily_list, main_hf_daily):
    """Replace None soil moisture values with values from main (best_match) cache."""
    for entry in daily_list:
        d = entry["date"]
        if d in main_hf_daily:
            main = main_hf_daily[d]
            if entry.get("soil_moisture") is None:
                entry["soil_moisture"] = main.get("soil_moisture")
            if entry.get("soil_moisture_deep") is None:
                entry["soil_moisture_deep"] = main.get("soil_moisture_deep")


def _avg(entries, field):
    vals = [e[field] for e in entries if e.get(field) is not None]
    return sum(vals) / len(vals) if vals else None


def _avg_hourly(model_hourlies, date_str):
    """Average hourly precip across models for a single date."""
    per_model = [
        model_hourlies[m][date_str]
        for m in ENSEMBLE_MODELS
        if date_str in model_hourlies.get(m, {})
    ]
    if not per_model:
        return []
    max_len = max(len(h) for h in per_model)
    averaged = []
    for i in range(max_len):
        vals = [h[i]["precip_mm"] for h in per_model if i < len(h)]
        avg_precip = sum(vals) / len(vals) if vals else 0.0
        averaged.append({"hour": i, "precip_mm": avg_precip})
    return averaged


def main():
    parser = argparse.ArgumentParser(description="Extend NBM and ensemble weather caches.")
    parser.add_argument("--start", type=date.fromisoformat, default=None,
                        help="Start date YYYY-MM-DD (default: earliest date in IFS cache)")
    parser.add_argument("--end", type=date.fromisoformat, default=None,
                        help="End date YYYY-MM-DD (default: yesterday)")
    args = parser.parse_args()

    # Load main IFS cache
    if not MAIN_CACHE.exists():
        print("Main IFS cache not found — run train.py first.")
        return
    with open(MAIN_CACHE) as f:
        main_cache = json.load(f)
    main_hf_daily  = main_cache.get("hf_daily",  {})
    main_hf_hourly = main_cache.get("hf_hourly", {})

    ifs_dates = sorted(main_hf_daily.keys())
    if not ifs_dates:
        print("Main cache is empty, nothing to fetch.")
        return

    # Determine date range
    start = args.start or date.fromisoformat(ifs_dates[0])
    end   = args.end   or (date.today() - timedelta(days=1))

    if start > end:
        print(f"Start {start} is after end {end}, nothing to do.")
        return

    # Build list of all dates in the requested range
    all_dates = []
    d = start
    while d <= end:
        all_dates.append(d.isoformat())
        d += timedelta(days=1)

    print(f"Date range: {start} -> {end} ({len(all_dates)} days)")

    # ── NBM Cache ──────────────────────────────────────────────────────────
    print("\n=== Fetching NBM (ncep_nbm_conus) ===")
    nbm_cache     = load_cache(NBM_CACHE)
    nbm_hf_daily  = nbm_cache["hf_daily"]
    nbm_hf_hourly = nbm_cache["hf_hourly"]

    missing_nbm = [d for d in all_dates if d not in nbm_hf_daily]

    if missing_nbm:
        fetch_start = date.fromisoformat(missing_nbm[0])
        fetch_end   = date.fromisoformat(missing_nbm[-1])
        print(f"  Fetching {len(missing_nbm)} missing days: {fetch_start} -> {fetch_end}")
        try:
            daily_list, hourly_by_date = get_historical_forecast(
                fetch_start, fetch_end, model="ncep_nbm_conus"
            )
            # NBM may not provide soil moisture — overlay from main cache
            overlay_soil(daily_list, main_hf_daily)
            for r in daily_list:
                nbm_hf_daily[r["date"]] = r
            for d, records in hourly_by_date.items():
                if d not in nbm_hf_hourly:
                    nbm_hf_hourly[d] = records
            print(f"  Fetched {len(daily_list)} days for NBM.")
        except Exception as e:
            print(f"  ERROR fetching NBM: {e}")
    else:
        print(f"  NBM cache already up to date ({len(nbm_hf_daily)} days cached, 0 missing)")

    # Fill any remaining gaps from main cache
    for d in all_dates:
        if d not in nbm_hf_daily and d in main_hf_daily:
            print(f"  Gap fill from main cache: {d}")
            nbm_hf_daily[d] = dict(main_hf_daily[d])
        if d not in nbm_hf_hourly and d in main_hf_hourly:
            nbm_hf_hourly[d] = main_hf_hourly[d]

    nbm_cache["hf_daily"]  = nbm_hf_daily
    nbm_cache["hf_hourly"] = nbm_hf_hourly
    save_cache(NBM_CACHE, nbm_cache)
    print(f"  NBM cache saved: {len(nbm_hf_daily)} daily, {len(nbm_hf_hourly)} hourly")

    # ── Ensemble Cache ─────────────────────────────────────────────────────
    print("\n=== Fetching Ensemble (6 models) ===")
    ens_cache     = load_cache(ENS_CACHE)
    ens_hf_daily  = ens_cache["hf_daily"]
    ens_hf_hourly = ens_cache["hf_hourly"]

    missing_ens = [d for d in all_dates if d not in ens_hf_daily]

    if not missing_ens:
        print(f"  Ensemble cache already up to date ({len(ens_hf_daily)} days cached, 0 missing)")
    else:
        # Only fetch the range that covers missing dates — not the full cache range
        ens_fetch_start = date.fromisoformat(missing_ens[0])
        ens_fetch_end   = date.fromisoformat(missing_ens[-1])
        print(f"  Fetching {len(missing_ens)} missing days: {ens_fetch_start} -> {ens_fetch_end}")

        model_dailies  = {}  # model -> {date -> entry}
        model_hourlies = {}  # model -> {date -> [hourly]}

        for mdl in ENSEMBLE_MODELS:
            print(f"  Fetching {mdl}...")
            try:
                daily_list, hourly_by_date = get_historical_forecast(
                    ens_fetch_start, ens_fetch_end, model=mdl
                )
                overlay_soil(daily_list, main_hf_daily)
                model_dailies[mdl]  = {r["date"]: r for r in daily_list}
                model_hourlies[mdl] = hourly_by_date
                print(f"    Got {len(daily_list)} days")
            except Exception as e:
                print(f"    ERROR for {mdl}: {e}")
                model_dailies[mdl]  = {}
                model_hourlies[mdl] = {}

        # Average across models for each missing date only
        for d in missing_ens:
            entries = [
                model_dailies[m][d]
                for m in ENSEMBLE_MODELS
                if d in model_dailies.get(m, {})
            ]
            if not entries:
                # Fall back to main cache
                if d in main_hf_daily:
                    ens_hf_daily[d] = dict(main_hf_daily[d])
                continue

            ecmwf_entry = model_dailies.get("ecmwf_ifs025", {}).get(d, {})
            main_entry  = main_hf_daily.get(d, {})
            ens_hf_daily[d] = {
                "date":               d,
                "precip_mm":          _avg(entries, "precip_mm"),
                "temp_max_c":         _avg(entries, "temp_max_c"),
                "temp_min_c":         _avg(entries, "temp_min_c"),
                # Use ecmwf_ifs025 soil (available for both historical + forecast);
                # fall back to best_match if ecmwf doesn't have a value
                "soil_moisture":      ecmwf_entry.get("soil_moisture") or main_entry.get("soil_moisture"),
                "soil_moisture_deep": ecmwf_entry.get("soil_moisture_deep") or main_entry.get("soil_moisture_deep"),
                "precip_prob_pct":    _avg(entries, "precip_prob_pct"),
            }

        # Ensemble hourly: average precip — only for missing dates
        for d in missing_ens:
            if d not in ens_hf_hourly:
                averaged = _avg_hourly(model_hourlies, d)
                if averaged:
                    ens_hf_hourly[d] = averaged
                elif d in main_hf_hourly:
                    ens_hf_hourly[d] = main_hf_hourly[d]

    # Update soil in ALL existing ensemble entries using ecmwf_ifs025
    # (covers entries built before this change that used best_match soil)
    print("\n=== Refreshing ensemble soil from ecmwf_ifs025 ===")
    all_ens_dates = sorted(ens_hf_daily.keys())
    if all_ens_dates:
        ecmwf_start = date.fromisoformat(all_ens_dates[0])
        ecmwf_end   = date.fromisoformat(all_ens_dates[-1])
        try:
            ecmwf_daily, _ = get_historical_forecast(ecmwf_start, ecmwf_end, model="ecmwf_ifs025")
            ecmwf_soil_map  = {r["date"]: r for r in ecmwf_daily}
            updated = 0
            for d, entry in ens_hf_daily.items():
                ecmwf = ecmwf_soil_map.get(d, {})
                sm   = ecmwf.get("soil_moisture")
                smd  = ecmwf.get("soil_moisture_deep")
                if sm is not None:
                    entry["soil_moisture"]      = sm
                    updated += 1
                if smd is not None:
                    entry["soil_moisture_deep"] = smd
            print(f"  Updated soil for {updated} ensemble entries")
        except Exception as e:
            print(f"  ERROR refreshing ecmwf soil: {e}")

    ens_cache["hf_daily"]  = ens_hf_daily
    ens_cache["hf_hourly"] = ens_hf_hourly
    save_cache(ENS_CACHE, ens_cache)
    print(f"\n  Ensemble cache saved: {len(ens_hf_daily)} daily, {len(ens_hf_hourly)} hourly")


if __name__ == "__main__":
    main()
