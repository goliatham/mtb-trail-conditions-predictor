"""Daily inference: fetch weather + recent reports, run model, write predictions.json."""

import csv
import json
import os
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd
import requests
from bs4 import BeautifulSoup

from features import (
    FEATURE_COLUMNS,
    INTRADAY_FEATURE_COLUMNS,
    TIME_SLOTS,
    build_features,
    build_intraday_features,
)
from weather import get_forecast, get_historical_forecast

INTRADAY_MODEL_PATH     = Path(__file__).parent.parent / "model" / "model_intraday.joblib"
INTRADAY_MODEL_PATH_NBM = Path(__file__).parent.parent / "model" / "model_intraday_nbm.joblib"
INTRADAY_MODEL_PATH_ENS = Path(__file__).parent.parent / "model" / "model_intraday_ensemble.joblib"
OUTPUT_PATH        = Path(__file__).parent.parent / "docs" / "predictions.json"
DATA_PATH          = Path(__file__).parent.parent / "data" / "mtb_scrape_raw.csv"
SNAPSHOTS_PATH     = Path(__file__).parent.parent / "data" / "feature_snapshots.json"
WEATHER_CACHE_PATH = Path(__file__).parent.parent / "data" / "weather_cache.json"
ENS_CACHE_PATH     = Path(__file__).parent.parent / "data" / "weather_cache_ensemble.json"
LAST_FORECAST_PATH = Path(__file__).parent.parent / "data" / "last_forecast.json"
_TRUSTED_PATH      = Path(__file__).parent.parent / "config" / "trusted_users.txt"

ENSEMBLE_MODELS = [
    "best_match", "ecmwf_ifs025", "ncep_nbm_conus",
    "ukmo_seamless", "meteofrance_seamless", "jma_seamless",
]

TRAINING_FIELDNAMES = ["date", "trail_key", "trail_id", "label", "color",
                       "username", "trusted", "comment"]

TRAILS = {
    "phase1": {"id": "4080717", "name": "Alum Creek Phase 1", "trail_id": 0},
    "phase2": {"id": "4081038", "name": "Alum Creek Phase 2", "trail_id": 1},
}


def _load_trusted():
    lines = _TRUSTED_PATH.read_text().splitlines()
    return {l.strip() for l in lines if l.strip() and not l.startswith("#")}


TRUSTED_USERS = _load_trusted()

MTBPROJECT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "X-Requested-With": "XMLHttpRequest",
}

LABEL_NAMES = {0: "Bad/Closed", 1: "Minor Issues", 2: "All Clear"}

SLOT_LABELS = {7: "7am", 11: "11am", 15: "3pm", 19: "7pm"}


def fetch_recent_report(trail_id: str):
    """Return the most recent condition report, or None."""
    try:
        resp = requests.get(
            f"https://www.mtbproject.com/ajax/public/trail/conditions/{trail_id}",
            headers=MTBPROJECT_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
    except Exception:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    div = soup.find("div", class_="mb-1")
    if not div:
        return None

    img = div.find("img", class_="condition")
    if not img:
        return None

    color = img["src"].split("/")[-1].replace(".svg", "")
    color_to_label = {"green": 2, "yellow": 1, "red": 0}
    label = color_to_label.get(color, 1)

    text = div.get_text(separator=" ", strip=True)

    date_str = None
    m = re.search(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),?\s+(\d{4})",
        text,
    )
    if m:
        try:
            dt = datetime.strptime(m.group(0).replace(",", ""), "%b %d %Y")
            date_str = dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    hours_ago = None
    hm = re.search(r"(\d+)\s+hour", text)
    dm = re.search(r"(\d+)\s+day", text)
    if hm:
        hours_ago = int(hm.group(1))
        date_str = date.today().isoformat()
    elif re.search(r"(\d+)\s+minute", text):
        hours_ago = 0
        date_str = date.today().isoformat()
    elif dm:
        days = int(dm.group(1))
        hours_ago = days * 24
        from datetime import timedelta
        date_str = (date.today() - timedelta(days=days)).isoformat()

    user_link = div.find("a", href=re.compile(r"/user/"))
    username = ""
    if user_link:
        um = re.search(r"/user/\d+/(.+)$", user_link["href"])
        if um:
            username = um.group(1)

    comment = re.sub(r"(All Clear|Minor Issues|Bad / Closed)\s*", "", text)
    comment = re.sub(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}:\s*",
        "", comment,
    )
    comment = re.sub(r"\s*—\s*.+$", "", comment).strip()

    return {
        "date": date_str,
        "hours_ago": hours_ago,
        "color": color,
        "label": label,
        "username": username,
        "trusted": username in TRUSTED_USERS,
        "comment": comment,
    }


def prior_report_for_day(report, forecast_day_offset: int):
    """Build the prior_report dict for build_features, adjusted for forecast horizon."""
    if not report:
        return None
    if report.get("hours_ago") is not None:
        base_days = max(1, report["hours_ago"] // 24)
    elif report.get("date"):
        base_days = max(1, (date.today() - date.fromisoformat(report["date"])).days)
    else:
        return None
    return {"label": report["label"], "days_ago": min(base_days + forecast_day_offset, 30)}


def good_score(proba, classes):
    """Good-conditions score derived from class probabilities."""
    cl = list(classes)
    score = 0.0
    if 2 in cl: score += proba[cl.index(2)]
    if 1 in cl: score += 0.5 * proba[cl.index(1)]
    return round(score, 3)


def weather_signal(fday, feats):
    if fday["precip_mm"] > 5:
        return f'{fday["precip_mm"]:.1f}mm rain forecast'
    elif feats["precip_1d_mm"] > 5:
        return f'{feats["precip_1d_mm"]:.1f}mm rain yesterday'
    elif feats["consecutive_dry_days"] >= 3:
        return f'{feats["consecutive_dry_days"]} dry days'
    return f'Soil {feats["soil_moisture"]:.2f}'


def predict_slots(intraday_model, history, fday, hourly, trail_id, prior_report=None,
                  history_hourly=None):
    """Return 4 time-slot predictions for a single day."""
    slots = []
    for hour in TIME_SLOTS:
        ifeats = build_intraday_features(history, fday, hourly, hour, trail_id, prior_report, history_hourly)
        X = pd.DataFrame([[ifeats[col] for col in INTRADAY_FEATURE_COLUMNS]],
                         columns=INTRADAY_FEATURE_COLUMNS)
        proba = intraday_model.predict_proba(X)[0].tolist()
        slots.append({
            "slot": SLOT_LABELS[hour],
            "hour": hour,
            "score": good_score(proba, intraday_model.classes_),
            "proba": [round(p, 3) for p in proba],
            "precip_midnight_to_slot_mm": round(ifeats["precip_midnight_to_slot_mm"], 1),
            "precip_3h_before_slot_mm": round(ifeats["precip_3h_before_slot_mm"], 1),
            "features": {k: round(v, 4) if isinstance(v, float) else v
                         for k, v in ifeats.items() if k in INTRADAY_FEATURE_COLUMNS},
        })
    return slots


def load_snapshots() -> dict:
    if SNAPSHOTS_PATH.exists():
        with open(SNAPSHOTS_PATH) as f:
            return json.load(f)
    return {"daily": {}, "intraday": {}}


def save_snapshots(snapshots: dict):
    with open(SNAPSHOTS_PATH, "w") as f:
        json.dump(snapshots, f, indent=2)


def append_to_training(trail_key: str, trail_id: int, report: dict):
    """Append a live-fetched report to mtb_scrape_raw.csv if not already present."""
    if not report or not report.get("date"):
        return
    existing_keys = set()
    if DATA_PATH.exists():
        with open(DATA_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing_keys.add((row["date"], row["trail_key"]))
    if (report["date"], trail_key) in existing_keys:
        return
    with open(DATA_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRAINING_FIELDNAMES)
        writer.writerow({
            "date": report["date"],
            "trail_key": trail_key,
            "trail_id": trail_id,
            "label": report["label"],
            "color": report["color"],
            "username": report.get("username", ""),
            "trusted": report.get("trusted", False),
            "comment": report.get("comment", ""),
        })
    print(f"  New training record: {trail_key} {report['date']} {report['color']}")


def _persist_forecast_probs(forecast: list) -> None:
    """Write precip_prob_pct for each forecast day into the weather cache.

    Write-once: first predict run of the day wins.
    """
    if WEATHER_CACHE_PATH.exists():
        with open(WEATHER_CACHE_PATH) as f:
            cache = json.load(f)
    else:
        cache = {"daily": {}, "hourly": {}}
    daily = cache["daily"]
    changed = False
    for fday in forecast:
        d    = fday["date"]
        prob = fday.get("precip_prob_pct")
        if prob is None:
            continue
        if d not in daily:
            daily[d] = {k: v for k, v in fday.items() if k not in ("sunrise", "sunset")}
            changed = True
        elif daily[d].get("precip_prob_pct") is None:
            daily[d]["precip_prob_pct"] = prob
            changed = True
    if changed:
        with open(WEATHER_CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=2)


def _average_forecasts(
    forecast_list: list,
    hourly_list: list,
) -> tuple:
    """Average daily forecast and hourly precip across multiple models.

    Uses the first entry in forecast_list (best_match) for soil_moisture,
    soil_moisture_deep, sunrise, and sunset — other models don't reliably
    provide soil data.
    """
    if not forecast_list:
        return [], {}

    n_days = max(len(fl) for fl in forecast_list)
    averaged_daily = []

    for i in range(n_days):
        entries = [fl[i] for fl in forecast_list if i < len(fl)]
        if not entries:
            continue

        def _avg_f(field, _entries=entries):
            vals = [e[field] for e in _entries if e.get(field) is not None]
            return sum(vals) / len(vals) if vals else None

        best = forecast_list[0][i] if i < len(forecast_list[0]) else entries[0]
        averaged_daily.append({
            "date":               entries[0]["date"],
            "precip_mm":          _avg_f("precip_mm") or 0.0,
            "temp_max_c":         _avg_f("temp_max_c"),
            "temp_min_c":         _avg_f("temp_min_c"),
            "soil_moisture":      best.get("soil_moisture"),
            "soil_moisture_deep": best.get("soil_moisture_deep"),
            "precip_prob_pct":    _avg_f("precip_prob_pct"),
            "sunrise":            best.get("sunrise"),
            "sunset":             best.get("sunset"),
        })

    # Average hourly precip across models
    all_dates: set = set()
    for hd in hourly_list:
        all_dates.update(hd.keys())

    averaged_hourly: dict = {}
    for d in sorted(all_dates):
        per_model = [hd[d] for hd in hourly_list if d in hd]
        if not per_model:
            continue
        max_len = max(len(h) for h in per_model)
        averaged_hourly[d] = [
            {
                "hour": j,
                "precip_mm": (
                    sum(h[j]["precip_mm"] for h in per_model if j < len(h))
                    / len([h for h in per_model if j < len(h)])
                ),
            }
            for j in range(max_len)
        ]

    return averaged_daily, averaged_hourly


def _load_ens_history(
    today: date,
    fallback_daily: list,
    fallback_hourly: dict,
) -> tuple:
    """Load ensemble weather history from the pre-built cache.

    Falls back to IFS entries for any dates missing from the ensemble cache,
    so predict.py always has a 14-day history without extra API calls.
    """
    ens_data: dict = {}
    if ENS_CACHE_PATH.exists():
        with open(ENS_CACHE_PATH) as f:
            ens_data = json.load(f)
    hf_daily  = ens_data.get("hf_daily",  {})
    hf_hourly = ens_data.get("hf_hourly", {})

    ifs_daily_map = {r["date"]: r for r in fallback_daily}

    history_ens: list = []
    hist_hourly_ens: dict = {}
    for i in range(14, 0, -1):
        d = (today - timedelta(days=i)).isoformat()
        if d in hf_daily:
            history_ens.append(hf_daily[d])
        elif d in ifs_daily_map:
            history_ens.append(ifs_daily_map[d])
        hourly = hf_hourly.get(d) or fallback_hourly.get(d)
        if hourly:
            hist_hourly_ens[d] = hourly

    return history_ens, hist_hourly_ens


def _prev_hourly(target: date, all_h: dict) -> list:
    """Return hourly records for the 7 days before target from all_h."""
    result = []
    for i in range(1, 8):
        h = all_h.get((target - timedelta(days=i)).isoformat())
        if h:
            result.append(h)
    return result


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    intraday_model     = joblib.load(INTRADAY_MODEL_PATH)     if INTRADAY_MODEL_PATH.exists()     else None
    intraday_model_nbm = joblib.load(INTRADAY_MODEL_PATH_NBM) if INTRADAY_MODEL_PATH_NBM.exists() else None
    intraday_model_ens = joblib.load(INTRADAY_MODEL_PATH_ENS) if INTRADAY_MODEL_PATH_ENS.exists() else None

    today    = date.today()
    tomorrow = today + timedelta(days=1)

    # ── IFS (best_match) ──────────────────────────────────────────────────
    print("Fetching IFS forecast/history...")
    forecast, hourly_by_date = get_forecast()
    _persist_forecast_probs(forecast)
    history, hist_hourly = get_historical_forecast(
        today - timedelta(days=14), today - timedelta(days=1)
    )

    # ── NBM ───────────────────────────────────────────────────────────────
    print("Fetching NBM forecast/history...")
    try:
        forecast_nbm, hourly_by_date_nbm = get_forecast(model="ncep_nbm_conus")
        history_nbm, hist_hourly_nbm = get_historical_forecast(
            today - timedelta(days=14), today - timedelta(days=1),
            model="ncep_nbm_conus",
        )
    except Exception as e:
        print(f"  Warning: NBM fetch failed ({e}), falling back to IFS")
        forecast_nbm, hourly_by_date_nbm = forecast, hourly_by_date
        history_nbm, hist_hourly_nbm     = history,  hist_hourly

    # ── Ensemble: average 6 models' forecasts ────────────────────────────
    print("Fetching ensemble forecasts (6 models)...")
    _ens_forecasts: list = []
    _ens_hourlies:  list = []
    for mdl in ENSEMBLE_MODELS:
        try:
            _f, _h = get_forecast(model=mdl)
            _ens_forecasts.append(_f)
            _ens_hourlies.append(_h)
            print(f"  {mdl}: ok")
        except Exception as e:
            print(f"  Warning: {mdl} forecast failed: {e}")

    if _ens_forecasts:
        forecast_ens, hourly_by_date_ens = _average_forecasts(_ens_forecasts, _ens_hourlies)
    else:
        print("  All ensemble models failed — falling back to IFS")
        forecast_ens, hourly_by_date_ens = forecast, hourly_by_date

    # Ensemble history from pre-built cache (no extra API calls)
    history_ens, hist_hourly_ens = _load_ens_history(today, history, hist_hourly)

    with open(LAST_FORECAST_PATH, "w") as f:
        json.dump({
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "daily": forecast,
            "hourly_today": hourly_by_date.get(today.isoformat(), []),
            "hourly_tomorrow": hourly_by_date.get(tomorrow.isoformat(), []),
            "hourly_history": {d: h for d, h in hist_hourly.items()},
        }, f, indent=2)

    # Combined hourly lookups (historical + forecast) for each source
    all_hourly     = {**hist_hourly,     **hourly_by_date}
    all_hourly_nbm = {**hist_hourly_nbm, **hourly_by_date_nbm}
    all_hourly_ens = {**hist_hourly_ens, **hourly_by_date_ens}

    results = {}
    snapshots = load_snapshots()

    for key, trail in TRAILS.items():
        recent_report = fetch_recent_report(trail["id"])
        append_to_training(key, trail["trail_id"], recent_report)
        time.sleep(0.5)

        # Fallback soil values if Open-Meteo hourly soil is missing
        last_surface = next((r["soil_moisture"] for r in reversed(history)
                             if r.get("soil_moisture") is not None), 0.20)
        last_deep = next((r.get("soil_moisture_deep") for r in reversed(history)
                          if r.get("soil_moisture_deep") is not None), 0.25)

        days = []
        for i in range(len(forecast)):
            fday     = forecast[i]
            fday_nbm = forecast_nbm[i] if i < len(forecast_nbm) else fday
            fday_ens = forecast_ens[i] if i < len(forecast_ens) else fday

            if i == 0:
                hist   = history
                hist_n = history_nbm
                hist_e = history_ens
            else:
                def _make_confirmed(src_forecast, src_history, _i=i):
                    return src_history + [
                        {
                            "date":               src_forecast[j]["date"],
                            "precip_mm":          src_forecast[j]["precip_mm"],
                            "temp_max_c":         src_forecast[j]["temp_max_c"],
                            "temp_min_c":         src_forecast[j]["temp_min_c"],
                            "soil_moisture":      (src_forecast[j]["soil_moisture"]
                                                   if src_forecast[j].get("soil_moisture") is not None
                                                   else last_surface),
                            "soil_moisture_deep": (src_forecast[j].get("soil_moisture_deep")
                                                   if src_forecast[j].get("soil_moisture_deep") is not None
                                                   else last_deep),
                        }
                        for j in range(_i)
                    ]
                hist   = _make_confirmed(forecast,     history)
                hist_n = _make_confirmed(forecast_nbm, history_nbm)
                hist_e = _make_confirmed(forecast_ens, history_ens)

            prior = prior_report_for_day(recent_report, i)

            label_date = date.fromisoformat(fday["date"])
            if i == 0:
                day_name = "Today"
            elif i == 1:
                day_name = "Tomorrow"
            else:
                day_name = label_date.strftime("%a %b %-d")

            # Ensemble drives all UX signals/features
            feats_ens = build_features(hist_e, fday_ens, trail["trail_id"], prior)

            precip_2d = round(sum(r["precip_mm"] for r in hist[-2:]), 1) if len(hist) >= 2 else 0.0
            day_entry = {
                "date": fday["date"],
                "day_name": day_name,
                "signal": weather_signal(fday_ens, feats_ens),
                "forecast_precip_mm": fday_ens["precip_mm"],
                "signals": {
                    "forecast_precip_mm": round(fday_ens["precip_mm"] or 0.0, 1),
                    "precip_2d_mm": precip_2d,
                    "soil_moisture": round(feats_ens["soil_moisture"], 3),
                    "soil_moisture_deep": round(feats_ens["soil_moisture_deep"], 3),
                    "sunrise": fday_ens.get("sunrise"),
                    "sunset": fday_ens.get("sunset"),
                },
                "features": {k: round(v, 4) if isinstance(v, float) else v
                             for k, v in feats_ens.items() if k in FEATURE_COLUMNS},
            }

            # Intraday slots for all 7 forecast days
            hourly_for_day     = hourly_by_date.get(fday["date"], [])
            hourly_for_day_nbm = hourly_by_date_nbm.get(fday["date"], [])
            hourly_for_day_ens = hourly_by_date_ens.get(fday["date"], [])

            if intraday_model_ens and hourly_for_day_ens:
                ph_ens = _prev_hourly(label_date, all_hourly_ens)
                ph_ifs = _prev_hourly(label_date, all_hourly)
                ph_nbm = _prev_hourly(label_date, all_hourly_nbm)

                slots_ens = predict_slots(
                    intraday_model_ens, hist_e, fday_ens, hourly_for_day_ens,
                    trail["trail_id"], prior, ph_ens,
                )
                slots_ifs = (
                    predict_slots(intraday_model, hist, fday, hourly_for_day,
                                  trail["trail_id"], prior, ph_ifs)
                    if intraday_model and hourly_for_day else None
                )
                slots_nbm = (
                    predict_slots(intraday_model_nbm, hist_n, fday_nbm, hourly_for_day_nbm,
                                  trail["trail_id"], prior, ph_nbm)
                    if intraday_model_nbm and hourly_for_day_nbm else None
                )

                # Annotate each slot with all 3 scores
                for j, slot in enumerate(slots_ens):
                    slot["model_scores"] = {
                        "ifs":      slots_ifs[j]["score"] if slots_ifs else None,
                        "nbm":      slots_nbm[j]["score"] if slots_nbm else None,
                        "ensemble": slot["score"],
                    }

                # Write-once snapshot uses ensemble features
                for hour in TIME_SLOTS:
                    snap_key = f"{key}:{fday['date']}:{hour}"
                    if snap_key not in snapshots["intraday"]:
                        ifeats = build_intraday_features(
                            hist_e, fday_ens, hourly_for_day_ens, hour,
                            trail["trail_id"], prior, ph_ens,
                        )
                        snapshots["intraday"][snap_key] = {c: ifeats[c] for c in INTRADAY_FEATURE_COLUMNS}

                ens_day_score = round(sum(s["score"] for s in slots_ens) / len(slots_ens), 3)
                ifs_day_score = (round(sum(s["score"] for s in slots_ifs) / len(slots_ifs), 3)
                                 if slots_ifs else None)
                nbm_day_score = (round(sum(s["score"] for s in slots_nbm) / len(slots_nbm), 3)
                                 if slots_nbm else None)

                day_entry["slots"] = slots_ens
                day_entry["score"] = ens_day_score          # ensemble is primary
                day_entry["model_scores"] = {
                    "ifs":      ifs_day_score,
                    "nbm":      nbm_day_score,
                    "ensemble": ens_day_score,
                }

            elif intraday_model and hourly_for_day:
                # Ensemble model unavailable — fall back to IFS only
                ph_ifs = _prev_hourly(label_date, all_hourly)
                slots_ifs = predict_slots(
                    intraday_model, hist, fday, hourly_for_day,
                    trail["trail_id"], prior, ph_ifs,
                )
                for j, slot in enumerate(slots_ifs):
                    slot["model_scores"] = {"ifs": slot["score"], "nbm": None, "ensemble": None}

                for hour in TIME_SLOTS:
                    snap_key = f"{key}:{fday['date']}:{hour}"
                    if snap_key not in snapshots["intraday"]:
                        ifeats = build_intraday_features(
                            hist, fday, hourly_for_day, hour,
                            trail["trail_id"], prior, ph_ifs,
                        )
                        snapshots["intraday"][snap_key] = {c: ifeats[c] for c in INTRADAY_FEATURE_COLUMNS}

                ifs_day_score = round(sum(s["score"] for s in slots_ifs) / len(slots_ifs), 3)
                day_entry["slots"]        = slots_ifs
                day_entry["score"]        = ifs_day_score
                day_entry["model_scores"] = {"ifs": ifs_day_score, "nbm": None, "ensemble": None}

            else:
                day_entry["score"] = None

            days.append(day_entry)

        results[key] = {
            "trail_name": trail["name"],
            "days": days,
            "recent_report": recent_report,
        }

    output = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "trails": results,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    save_snapshots(snapshots)
    print(f"\nWrote {OUTPUT_PATH}")
    for key, trail_data in results.items():
        print(f"\n{trail_data['trail_name']}:")
        for d in trail_data["days"]:
            ms = d.get("model_scores") or {}
            if d.get("score") is not None:
                bar = "█" * int(d["score"] * 10)
                ifs_s = f" IFS:{ms['ifs']:.0%}" if ms.get("ifs") is not None else ""
                nbm_s = f" NBM:{ms['nbm']:.0%}" if ms.get("nbm") is not None else ""
                print(f"  {d['day_name']:12s} {bar:10s} {d['score']:.0%}  {d['signal']}{ifs_s}{nbm_s}")
            else:
                print(f"  {d['day_name']:12s} {'no data':10s}  {d['signal']}")
            if "slots" in d:
                for slot in d["slots"]:
                    sms = slot.get("model_scores") or {}
                    ifs_s = f" ifs:{sms['ifs']:.0%}" if sms.get("ifs") is not None else ""
                    nbm_s = f" nbm:{sms['nbm']:.0%}" if sms.get("nbm") is not None else ""
                    print(f"    {slot['slot']:6s} {slot['score']:.0%}{ifs_s}{nbm_s}")
        rr = trail_data["recent_report"]
        if rr:
            trust = " (TRUSTED)" if rr["trusted"] else ""
            print(f"  Latest report: {rr['color']} by {rr['username']}{trust} on {rr['date']}")


if __name__ == "__main__":
    main()
