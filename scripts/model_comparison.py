"""
Fetch historical precipitation from all 6 NWP models and compare against
user-labeled rideable/not-rideable days.

Usage:
  python scripts/model_comparison.py           # analysis only (uses cached fetch)
  python scripts/model_comparison.py --cv      # + CV per model + filtered ensemble CV
  python scripts/model_comparison.py --refetch # re-fetch from API (ignores cache)
"""

import sys, json, argparse, re, subprocess
from datetime import date, timedelta
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from weather import get_historical_forecast

MODELS = [
    "best_match",
    "ecmwf_ifs025",
    "ncep_nbm_conus",
    "ukmo_seamless",
    "meteofrance_seamless",
    "jma_seamless",
]

HIGH_SIGNAL_MODELS = [m for m in MODELS if m in {
    "ncep_nbm_conus", "meteofrance_seamless", "jma_seamless"
}]

BUILDERS      = Path(__file__).parent.parent
DATA_DIR      = BUILDERS / "data"
IFS_CACHE     = DATA_DIR / "weather_cache.json"
ENS_CACHE     = DATA_DIR / "weather_cache_ensemble.json"
FETCH_CACHE   = DATA_DIR / "_model_comparison_fetch.json"

# Full training range start (must match earliest hf_daily in IFS cache)
HISTORY_START = date(2025, 5, 10)


def load_labels():
    fb = json.loads((DATA_DIR / "user_feedback.json").read_text())
    daily = defaultdict(list)
    for r in fb:
        daily[(r["date"], r["trail"])].append(r["vote"])
    return {k: (1 if sum(v) / len(v) >= 0.5 else 0) for k, v in daily.items()}


# ---------------------------------------------------------------------------
# History availability check
# ---------------------------------------------------------------------------

def check_history_availability():
    """Test each model against a small window at the start of our training range."""
    probe_start = HISTORY_START
    probe_end   = HISTORY_START + timedelta(days=4)
    print(f"\n=== History availability check ({probe_start} to {probe_end}) ===")
    print(f"{'Model':<22} {'Available':>12} {'Note'}")
    print("-" * 60)
    available = {}
    for model in MODELS:
        try:
            daily, _ = get_historical_forecast(probe_start, probe_end, model=model)
            count = len([r for r in daily if r["precip_mm"] is not None])
            available[model] = True
            print(f"{model:<22} {'YES':>12}  ({count} days returned)")
        except Exception as e:
            available[model] = False
            print(f"{model:<22} {'NO':>12}  ({e})")
    return available


# ---------------------------------------------------------------------------
# Fetch (with local cache to avoid re-hitting API)
# ---------------------------------------------------------------------------

def fetch_all_models(start: date, end: date, refetch: bool = False) -> dict:
    """
    Fetch daily precip_mm for each model for the date range.
    Returns {model: {date_str: precip_mm}}.
    Saves results to FETCH_CACHE so re-runs are instant.
    """
    cache_key = f"{start}:{end}"
    if not refetch and FETCH_CACHE.exists():
        saved = json.loads(FETCH_CACHE.read_text())
        if saved.get("_key") == cache_key:
            print("  (using cached fetch results — pass --refetch to re-query API)")
            return saved["data"]

    results = {}
    for model in MODELS:
        print(f"  Fetching {model}...", flush=True)
        try:
            daily, _ = get_historical_forecast(start, end, model=model)
            results[model] = {r["date"]: r["precip_mm"] for r in daily}
            print(f"    -> {len(daily)} days")
        except Exception as e:
            print(f"    ERROR: {e}")
            results[model] = {}

    FETCH_CACHE.write_text(json.dumps({"_key": cache_key, "data": results}))
    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(model_data: dict, labels: dict):
    print("\n=== Avg daily precip_mm on labeled days ===")
    print(f"{'Model':<22} {'Rideable':>10} {'Not-ride':>10} {'Ratio':>8} {'Coverage':>10}")
    print("-" * 65)

    ratios = {}
    for model in MODELS:
        data = model_data.get(model, {})
        rideable_vals, notride_vals = [], []
        for (d, _), label in labels.items():
            if d not in data:
                continue
            (rideable_vals if label == 1 else notride_vals).append(data[d])

        if not rideable_vals or not notride_vals:
            print(f"{model:<22} {'NO DATA':>10}")
            continue

        r_avg = sum(rideable_vals) / len(rideable_vals)
        n_avg = sum(notride_vals)  / len(notride_vals)
        ratio = n_avg / r_avg if r_avg > 0 else float("inf")
        ratios[model] = ratio
        coverage = len(rideable_vals) + len(notride_vals)
        marker = " *" if ratio >= 7.0 else ""
        print(f"{model:<22} {r_avg:>10.2f} {n_avg:>10.2f} {ratio:>8.1f}x {coverage:>8}/{len(labels)}{marker}")

    print("  * = selected for high-signal ensemble (ratio >= 7.0)")
    print()

    print("=== Model rain values on rideable days (p50 / p90 / max) ===")
    print(f"{'Model':<22} {'p50':>8} {'p90':>8} {'max':>8}")
    print("-" * 50)
    for model in MODELS:
        data = model_data.get(model, {})
        vals = sorted(data[d] for (d, _), lbl in labels.items() if lbl == 1 and d in data)
        if not vals:
            continue
        p50 = vals[len(vals) // 2]
        p90 = vals[int(len(vals) * 0.9)]
        print(f"{model:<22} {p50:>8.2f} {p90:>8.2f} {max(vals):>8.2f}")


# ---------------------------------------------------------------------------
# CV testing
# ---------------------------------------------------------------------------

def _run_train(source: str) -> str:
    """Run train.py for a source, return CV string like '0.882±0.043'."""
    result = subprocess.run(
        [sys.executable, "src/train.py", "--source", source],
        capture_output=True, text=True, cwd=BUILDERS
    )
    m = re.search(r"Cross-val accuracy:\s*([0-9.]+)\s*±\s*([0-9.]+)", result.stdout)
    if m:
        return f"{m.group(1)}±{m.group(2)}"
    return f"? stdout={result.stdout[-200:]}"


def run_cv_per_model(model_data: dict, hist_available: dict):
    """
    For each model that has full training-range history, patch the IFS cache
    and run train.py --source ifs. Also test the high-signal ensemble.
    """
    print("\n=== CV per individual model (patching IFS cache precip) ===")
    orig_ifs = json.loads(IFS_CACHE.read_text())
    ifs_hf   = orig_ifs.get("hf_daily", {})

    print(f"{'Model':<22} {'Hist avail':>12} {'CV (daily)':>16}")
    print("-" * 55)

    for model in MODELS:
        if not hist_available.get(model):
            print(f"{model:<22} {'NO':>12}  {'(skipped)':>16}")
            continue

        data = model_data.get(model, {})
        if not data:
            print(f"{model:<22} {'YES':>12}  {'NO FETCH DATA':>16}")
            continue

        patched_hf = {d: dict(e) for d, e in ifs_hf.items()}
        for d, precip in data.items():
            if d in patched_hf:
                patched_hf[d]["precip_mm"] = precip
                patched_hf[d]["rain_mm"]   = precip

        IFS_CACHE.write_text(json.dumps({**orig_ifs, "hf_daily": patched_hf}))
        cv = _run_train("ifs")
        print(f"{model:<22} {'YES':>12}  {cv:>16}")

    IFS_CACHE.write_text(json.dumps(orig_ifs))
    print("(IFS cache restored)")

    # --- High-signal ensemble ---
    print(f"\n=== High-signal ensemble CV (models: {', '.join(HIGH_SIGNAL_MODELS)}) ===")
    orig_ens = json.loads(ENS_CACHE.read_text())
    ens_hf   = orig_ens.get("hf_daily", {})

    patched_ens_hf = {d: dict(e) for d, e in ens_hf.items()}
    for d in list(patched_ens_hf.keys()):
        precip_vals = [
            model_data[m][d]
            for m in HIGH_SIGNAL_MODELS
            if m in model_data and d in model_data[m]
        ]
        if precip_vals:
            avg = sum(precip_vals) / len(precip_vals)
            patched_ens_hf[d]["precip_mm"] = avg
            patched_ens_hf[d]["rain_mm"]   = avg

    ENS_CACHE.write_text(json.dumps({**orig_ens, "hf_daily": patched_ens_hf}))
    cv_ens = _run_train("ensemble")
    print(f"High-signal ensemble CV: {cv_ens}")

    ENS_CACHE.write_text(json.dumps(orig_ens))
    print("(Ensemble cache restored)")

    print("\n=== Baseline (current caches) ===")
    for src in ["ifs", "nbm", "ensemble"]:
        cv = _run_train(src)
        print(f"  {src:<10} {cv}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv",      action="store_true", help="Run CV per model + filtered ensemble")
    parser.add_argument("--refetch", action="store_true", help="Re-query API instead of using cached fetch")
    args = parser.parse_args()

    labels = load_labels()
    unique_dates = sorted(set(d for d, _ in labels.keys()))
    fetch_start  = date.fromisoformat(unique_dates[0]) - timedelta(days=14)
    fetch_end    = date.fromisoformat(unique_dates[-1])

    print(f"Labels: {len(labels)} date/trail combos over {len(unique_dates)} unique dates")
    print(f"Fetching {fetch_start} to {fetch_end} for analysis...\n")

    model_data = fetch_all_models(fetch_start, fetch_end, refetch=args.refetch)
    analyze(model_data, labels)

    if args.cv:
        hist_available = check_history_availability()
        run_cv_per_model(model_data, hist_available)


if __name__ == "__main__":
    main()
