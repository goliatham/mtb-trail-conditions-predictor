"""Join scraped condition labels with historical weather, train intraday model."""

import argparse
import csv
import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from features import (
    INTRADAY_FEATURE_COLUMNS,
    TIME_SLOTS,
    build_intraday_features,
)
from weather import get_historical_forecast

DATA_PATH    = Path(__file__).parent.parent / "data" / "mtb_scrape_raw.csv"
DOCS_PATH    = Path(__file__).parent.parent / "docs"
FEEDBACK_PATH = Path(__file__).parent.parent / "data" / "user_feedback.json"
SNAPSHOTS_PATH = Path(__file__).parent.parent / "data" / "feature_snapshots.json"
WEATHER_CACHE_PATH = Path(__file__).parent.parent / "data" / "weather_cache.json"
INTRADAY_MODEL_PATH = Path(__file__).parent.parent / "model" / "model_intraday.joblib"
CREEK_GAUGE_PATH = Path(__file__).parent.parent / "data" / "creek_gauge.json"
HISTORY_DAYS = 14

# Reports are almost always posted in the afternoon. Same-day labels are
# therefore less reliable for morning slots where conditions may differ.
MORNING_SLOT_WEIGHT_MULT = 0.2   # applied to MTBProject labels for 7am/11am slots
MORNING_SLOTS = {7, 11}


def load_labels():
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_feedback():
    if not FEEDBACK_PATH.exists():
        return []
    with open(FEEDBACK_PATH) as f:
        return json.load(f)


def load_snapshots():
    if not SNAPSHOTS_PATH.exists():
        return {"intraday": {}}
    with open(SNAPSHOTS_PATH) as f:
        return json.load(f)


def load_weather_cache():
    if not WEATHER_CACHE_PATH.exists():
        return {"daily": {}, "hourly": {}}
    with open(WEATHER_CACHE_PATH) as f:
        return json.load(f)


def save_weather_cache(cache):
    with open(WEATHER_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def _get_history(label_date, weather_by_date):
    rows = []
    for i in range(HISTORY_DAYS, 0, -1):
        d = (label_date - timedelta(days=i)).isoformat()
        if d in weather_by_date:
            rows.append(weather_by_date[d])
    return rows


def _get_day_weather(label_date, weather_by_date):
    return weather_by_date.get(label_date.isoformat())


def _get_hourly(label_date, hourly_by_date):
    return hourly_by_date.get(label_date.isoformat(), [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", default="ifs", choices=["ifs", "nbm", "ensemble"],
        help="Weather source: ifs (default), nbm, or ensemble",
    )
    args = parser.parse_args()

    _data_dir = WEATHER_CACHE_PATH.parent
    _model_dir = INTRADAY_MODEL_PATH.parent
    if args.source == "nbm":
        cache_path = _data_dir / "weather_cache_nbm.json"
        model_out  = _model_dir / "model_intraday_nbm.joblib"
        use_snapshots = False
    elif args.source == "ensemble":
        cache_path = _data_dir / "weather_cache_ensemble.json"
        model_out  = _model_dir / "model_intraday_ensemble.joblib"
        use_snapshots = False
    else:
        cache_path = WEATHER_CACHE_PATH
        model_out  = INTRADAY_MODEL_PATH
        use_snapshots = True

    print(f"Source: {args.source}  |  cache: {cache_path.name}  |  out: {model_out.name}")
    _model_dir.mkdir(exist_ok=True)

    labels = load_labels()
    snapshots = load_snapshots() if use_snapshots else {"intraday": {}}
    print(f"Loaded {len(labels)} labeled records, {len(snapshots.get('intraday', {}))} intraday snapshots")

    # Pre-compute the most recent prior report for each (trail_id, date)
    by_trail = defaultdict(list)
    for rec in labels:
        by_trail[rec["trail_id"]].append(rec)

    prior_report_map = {}  # (trail_id, date_str) -> {"label": int, "days_ago": int}
    for tid, recs in by_trail.items():
        recs_sorted = sorted(recs, key=lambda r: r["date"])
        for i, rec in enumerate(recs_sorted):
            label_date = date.fromisoformat(rec["date"])
            prior = None
            for j in range(i - 1, -1, -1):
                prior_date = date.fromisoformat(recs_sorted[j]["date"])
                if prior_date < label_date:
                    prior = {"label": int(recs_sorted[j]["label"]),
                             "days_ago": min((label_date - prior_date).days, 30)}
                    break
            prior_report_map[(tid, rec["date"])] = prior  # None if no prior

    valid_dates = sorted(
        date.fromisoformat(r["date"]) for r in labels if r["date"]
    )
    bulk_start = valid_dates[0] - timedelta(days=HISTORY_DAYS)
    bulk_end = valid_dates[-1]

    if cache_path.exists():
        with open(cache_path) as f:
            weather_cache = json.load(f)
    else:
        weather_cache = {"hf_daily": {}, "hf_hourly": {}}
    hf_daily  = weather_cache.get("hf_daily",  {})
    hf_hourly = weather_cache.get("hf_hourly", {})

    # Find date ranges not yet in cache and fetch only those
    all_dates = [bulk_start + timedelta(days=i) for i in range((bulk_end - bulk_start).days + 1)]
    missing = [d for d in all_dates if d.isoformat() not in hf_daily]

    if missing and args.source == "ifs":
        fetch_start, fetch_end = missing[0], missing[-1]
        print(f"Fetching historical forecast {fetch_start} → {fetch_end} ({len(missing)} new days)...")
        daily_list, new_hourly = get_historical_forecast(fetch_start, fetch_end)
        for r in daily_list:
            hf_daily[r["date"]] = r
        for d, records in new_hourly.items():
            hf_hourly[d] = records
        weather_cache["hf_daily"]  = hf_daily
        weather_cache["hf_hourly"] = hf_hourly
        with open(cache_path, "w") as f:
            json.dump(weather_cache, f, indent=2)
        print("Weather cache updated.")
    elif missing:
        print(f"Warning: {len(missing)} dates missing from {cache_path.name} — run fetch_alt_weather.py first")
    else:
        print(f"Weather cache hit — {len(all_dates)} days already cached.")

    # One-shot backfill: patch new fields (rain_mm, snow_cm, soil_temp_*, temp_c in hourly)
    # into existing IFS cache entries that pre-date this feature addition.
    if args.source == "ifs" and hf_daily and any("rain_mm" not in v for v in hf_daily.values()):
        cached_dates = sorted(hf_daily.keys())
        bf_start = date.fromisoformat(cached_dates[0])
        bf_end   = date.fromisoformat(cached_dates[-1])
        print(f"Backfilling new weather fields ({bf_start} → {bf_end})...")
        bf_daily, bf_hourly = get_historical_forecast(bf_start, bf_end)
        new_fields = ("rain_mm", "snow_cm", "soil_temp_0cm", "soil_temp_6cm", "soil_temp_18cm")
        for r in bf_daily:
            if r["date"] in hf_daily:
                for field in new_fields:
                    if field in r:
                        hf_daily[r["date"]][field] = r[field]
        for d, records in bf_hourly.items():
            if d in hf_hourly:
                for i, rec in enumerate(hf_hourly[d]):
                    if i < len(records) and records[i].get("temp_c") is not None:
                        rec["temp_c"] = records[i]["temp_c"]
        weather_cache["hf_daily"]  = hf_daily
        weather_cache["hf_hourly"] = hf_hourly
        with open(cache_path, "w") as f:
            json.dump(weather_cache, f, indent=2)
        print(f"  Backfilled {len(bf_daily)} daily entries.")

    # Augment daily entries with creek gauge peak
    if CREEK_GAUGE_PATH.exists():
        gauge_daily = json.load(open(CREEK_GAUGE_PATH)).get("daily_peak", {})
        for d, entry in hf_daily.items():
            entry["creek_peak_ft"] = gauge_daily.get(d)

    weather_by_date = hf_daily
    hourly_by_date  = hf_hourly

    intraday_rows, intraday_targets, intraday_weights = [], [], []
    skipped = 0

    for i, rec in enumerate(labels):
        try:
            label_date = date.fromisoformat(rec["date"])
        except ValueError:
            skipped += 1
            continue

        trail_id = int(rec["trail_id"])
        day_label = int(rec["label"])
        if day_label == 1:
            day_label = 2  # minor issues treated as rideable
        weight = 1.0 if rec["trusted"] == "True" else 0.4

        history = _get_history(label_date, weather_by_date)
        day_weather = _get_day_weather(label_date, weather_by_date)
        if day_weather is None:
            skipped += 1
            continue

        prior = prior_report_map.get((rec["trail_id"], rec["date"]))
        snap_key = f"{rec['trail_key']}:{rec['date']}"

        # Intraday rows — 4 slots per date using historical hourly precip
        hourly = _get_hourly(label_date, hourly_by_date)
        if hourly:
            prev_hourly = [h for i in range(1, 8)
                           if (h := hourly_by_date.get((label_date - timedelta(days=i)).isoformat()))]
            for hour in TIME_SLOTS:
                intra_snap = snapshots["intraday"].get(f"{snap_key}:{hour}")
                if intra_snap and all(c in intra_snap for c in INTRADAY_FEATURE_COLUMNS):
                    ifeats = intra_snap
                else:
                    ifeats = build_intraday_features(history, day_weather, hourly, hour, trail_id, prior, prev_hourly)
                slot_label = day_label
                intraday_rows.append([ifeats[col] for col in INTRADAY_FEATURE_COLUMNS])
                intraday_targets.append(slot_label)
                # morning slots: assume report posted afternoon → less reliable for 7am/11am
                slot_mult = MORNING_SLOT_WEIGHT_MULT if hour in MORNING_SLOTS else 1.0
                intraday_weights.append(weight * slot_mult)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(labels)}...")

    print(f"Skipped {skipped} records (missing date or weather)")

    # Load user feedback — first-person slot observations, highest weight
    feedback = load_feedback()
    if feedback:
        print(f"\nLoading {len(feedback)} user feedback entries...")
        fb_skipped = 0
        for fb in feedback:
            try:
                fb_date = date.fromisoformat(fb["date"])
            except (ValueError, KeyError):
                fb_skipped += 1
                continue

            vote = fb.get("vote")
            if vote not in (1, -1):
                fb_skipped += 1
                continue

            # vote 1 = good conditions (green=2), -1 = bad (red=0)
            slot_label = 2 if vote == 1 else 0
            trail_id = int(fb.get("trail_id", 0))
            hour = int(fb["hour"])

            history = _get_history(fb_date, weather_by_date)
            day_weather = _get_day_weather(fb_date, weather_by_date)
            if day_weather is None:
                fb_skipped += 1
                continue

            hourly = _get_hourly(fb_date, hourly_by_date)
            if not hourly:
                fb_skipped += 1
                continue

            fb_prior = prior_report_map.get((str(trail_id), fb["date"]))
            fb_trail_key = "phase1" if trail_id == 0 else "phase2"
            intra_snap = snapshots["intraday"].get(f"{fb_trail_key}:{fb['date']}:{hour}")
            if intra_snap and all(c in intra_snap for c in INTRADAY_FEATURE_COLUMNS):
                ifeats = intra_snap
            else:
                fb_prev_hourly = [h for i in range(1, 8)
                                  if (h := hourly_by_date.get((fb_date - timedelta(days=i)).isoformat()))]
                ifeats = build_intraday_features(history, day_weather, hourly, hour, trail_id, fb_prior, fb_prev_hourly)
            intraday_rows.append([ifeats[col] for col in INTRADAY_FEATURE_COLUMNS])
            intraday_targets.append(slot_label)
            intraday_weights.append(3.0)  # direct observation — no morning discount

        print(f"  Added {len(feedback) - fb_skipped} feedback rows, skipped {fb_skipped}")

    def train_model(X, y, w, columns, name, path):
        trusted = sum(1 for wt in w if wt == 1.0)
        print(f"\n{name}: {len(X)} samples ({trusted} trusted)")
        print("Label distribution:", y.value_counts().to_dict())

        m = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
        )
        scores = cross_val_score(m, X, y, cv=5, scoring="accuracy")
        print(f"Cross-val accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
        m.fit(X, y, sample_weight=w)

        top = sorted(zip(columns, m.feature_importances_), key=lambda x: x[1], reverse=True)
        print("Top features:")
        for feat, imp in top[:6]:
            print(f"  {feat}: {imp:.3f}")

        joblib.dump(m, path)
        print(f"Saved -> {path}")

    if intraday_rows:
        X_i = pd.DataFrame(intraday_rows, columns=INTRADAY_FEATURE_COLUMNS)
        train_model(X_i, pd.Series(intraday_targets), pd.Series(intraday_weights),
                    INTRADAY_FEATURE_COLUMNS, f"Intraday model ({args.source})", model_out)
    else:
        print("\nNo hourly data available — intraday model not trained")


def write_feature_importances():
    """Write docs/feature_importances.html with per-model intraday feature importance table."""
    model_dir = INTRADAY_MODEL_PATH.parent
    model_files = {
        "IFS":      INTRADAY_MODEL_PATH,
        "NBM":      model_dir / "model_intraday_nbm.joblib",
        "Ensemble": model_dir / "model_intraday_ensemble.joblib",
    }
    loaded = {name: joblib.load(p) for name, p in model_files.items() if p.exists()}
    if not loaded:
        return

    cols = INTRADAY_FEATURE_COLUMNS
    imps = {name: dict(zip(cols, m.feature_importances_)) for name, m in loaded.items()}
    sorted_cols = sorted(cols, key=lambda c: imps.get("IFS", imps[next(iter(imps))]).get(c, 0), reverse=True)

    col_names = list(loaded.keys())
    header_cells = "".join(f'<th onclick="sortBy({i+1})" data-col="{i+1}">{n} <span class="arrow"></span></th>' for i, n in enumerate(col_names))
    rows = ""
    for c in sorted_cols:
        cells = "".join(f'<td>{imps[n].get(c, 0):.3f}</td>' for n in col_names)
        rows += f"<tr><td>{c}</td>{cells}</tr>\n"

    updated = date.today().isoformat()
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Feature Importances — MTCP</title>
<style>
  body {{ font-family: monospace; padding: 1.5rem; background: #111; color: #eee; }}
  h2 {{ margin-bottom: 0.25rem; }}
  p.sub {{ color: #888; margin-top: 0; font-size: 0.85rem; }}
  table {{ border-collapse: collapse; margin-top: 1rem; }}
  th, td {{ padding: 4px 14px; text-align: left; border-bottom: 1px solid #333; }}
  th {{ color: #aaa; font-size: 0.8rem; cursor: pointer; user-select: none; }}
  th:hover {{ color: #eee; }}
  th .arrow {{ font-size: 0.7rem; margin-left: 3px; color: #555; }}
  th.sorted .arrow {{ color: #c9d1e8; }}
  td:first-child {{ color: #ccc; min-width: 240px; }}
  td:not(:first-child) {{ text-align: right; }}
  tr:hover td {{ background: #1e1e1e; }}
</style>
</head>
<body>
<h2>Intraday model — feature importances</h2>
<p class="sub">Click a column to sort &middot; updated {updated}</p>
<table id="t">
<thead><tr><th onclick="sortBy(0)" data-col="0">Feature <span class="arrow"></span></th>{header_cells}</tr></thead>
<tbody>
{rows}</tbody>
</table>
<script>
  var asc = {{}};
  function sortBy(col) {{
    var tb = document.querySelector('#t tbody');
    var rows = Array.from(tb.rows);
    var dir = asc[col] = !asc[col];
    rows.sort(function(a, b) {{
      var av = a.cells[col].textContent.trim();
      var bv = b.cells[col].textContent.trim();
      var an = parseFloat(av), bn = parseFloat(bv);
      var cmp = isNaN(an) ? av.localeCompare(bv) : an - bn;
      return dir ? cmp : -cmp;
    }});
    rows.forEach(function(r) {{ tb.appendChild(r); }});
    document.querySelectorAll('th').forEach(function(th) {{
      th.classList.remove('sorted');
      th.querySelector('.arrow').textContent = '';
    }});
    var th = document.querySelectorAll('th')[col];
    th.classList.add('sorted');
    th.querySelector('.arrow').textContent = dir ? '▲' : '▼';
  }}
  // default: sort by IFS desc
  sortBy(1);
</script>
</body>
</html>"""
    (DOCS_PATH / "feature_importances.html").write_text(html)
    print(f"Wrote feature importances -> docs/feature_importances.html")


if __name__ == "__main__":
    main()
    write_feature_importances()
