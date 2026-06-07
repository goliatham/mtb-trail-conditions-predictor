"""Daily inference: fetch weather + recent reports, run model, write predictions.json."""

import csv
import json
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
from weather import get_forecast, get_historical, get_hourly_forecast_day

MODEL_PATH = Path(__file__).parent.parent / "model" / "model.joblib"
INTRADAY_MODEL_PATH = Path(__file__).parent.parent / "model" / "model_intraday.joblib"
OUTPUT_PATH = Path(__file__).parent.parent / "docs" / "predictions.json"
DATA_PATH = Path(__file__).parent.parent / "data" / "training_raw.csv"
_TRUSTED_PATH = Path(__file__).parent.parent / "config" / "trusted_users.txt"

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
    if hm:
        hours_ago = int(hm.group(1))
        date_str = date.today().isoformat()
    elif re.search(r"(\d+)\s+minute", text):
        hours_ago = 0
        date_str = date.today().isoformat()

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


def good_score(proba):
    """Weighted good-conditions score: green=1.0, yellow=0.5, red=0.0."""
    return round(proba[2] + 0.5 * proba[1], 3)


def weather_signal(fday, feats):
    if fday["precip_mm"] > 5:
        return f'{fday["precip_mm"]:.1f}mm rain forecast'
    elif feats["precip_1d_mm"] > 5:
        return f'{feats["precip_1d_mm"]:.1f}mm rain yesterday'
    elif feats["consecutive_dry_days"] >= 3:
        return f'{feats["consecutive_dry_days"]} dry days'
    return f'Soil {feats["soil_moisture"]:.2f}'


def predict_slots(intraday_model, history, fday, hourly, trail_id, prior_report=None):
    """Return 4 time-slot predictions for a single day."""
    slots = []
    for hour in TIME_SLOTS:
        ifeats = build_intraday_features(history, fday, hourly, hour, trail_id, prior_report)
        X = pd.DataFrame([[ifeats[col] for col in INTRADAY_FEATURE_COLUMNS]],
                         columns=INTRADAY_FEATURE_COLUMNS)
        proba = intraday_model.predict_proba(X)[0].tolist()
        slots.append({
            "slot": SLOT_LABELS[hour],
            "hour": hour,
            "score": good_score(proba),
            "proba": [round(p, 3) for p in proba],
        })
    return slots


def append_to_training(trail_key: str, trail_id: int, report: dict):
    """Append a live-fetched report to training_raw.csv if not already present."""
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


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    daily_model = joblib.load(MODEL_PATH)
    intraday_model = joblib.load(INTRADAY_MODEL_PATH) if INTRADAY_MODEL_PATH.exists() else None

    today = date.today()
    forecast = get_forecast()  # 7 days starting today
    history = get_historical(today - timedelta(days=14), today - timedelta(days=1))

    # Fetch hourly forecast for today + tomorrow
    hourly_today = get_hourly_forecast_day(today) if intraday_model else []
    tomorrow = today + timedelta(days=1)
    hourly_tomorrow = get_hourly_forecast_day(tomorrow) if intraday_model else []

    results = {}

    for key, trail in TRAILS.items():
        recent_report = fetch_recent_report(trail["id"])
        append_to_training(key, trail["trail_id"], recent_report)
        time.sleep(0.5)

        days = []
        for i, fday in enumerate(forecast):
            if i == 0:
                hist = history
            else:
                confirmed = [
                    {
                        "date": forecast[j]["date"],
                        "precip_mm": forecast[j]["precip_mm"],
                        "temp_max_c": forecast[j]["temp_max_c"],
                        "temp_min_c": forecast[j]["temp_min_c"],
                        "soil_moisture": None,
                    }
                    for j in range(i)
                ]
                hist = history + confirmed

            prior = prior_report_for_day(recent_report, i)
            feats = build_features(hist, fday, trail["trail_id"], prior)

            # Daily prediction (all 7 days)
            X_d = pd.DataFrame([[feats[col] for col in FEATURE_COLUMNS]],
                               columns=FEATURE_COLUMNS)
            proba = daily_model.predict_proba(X_d)[0].tolist()
            score = good_score(proba)

            label_date = date.fromisoformat(fday["date"])
            if i == 0:
                day_name = "Today"
            elif i == 1:
                day_name = "Tomorrow"
            else:
                day_name = label_date.strftime("%a %b %-d")

            day_entry = {
                "date": fday["date"],
                "day_name": day_name,
                "score": score,
                "proba": [round(p, 3) for p in proba],
                "signal": weather_signal(fday, feats),
                "forecast_precip_mm": fday["precip_mm"],
            }

            # Add intraday slots for today and tomorrow
            if intraday_model and i == 0 and hourly_today:
                day_entry["slots"] = predict_slots(
                    intraday_model, hist, fday, hourly_today, trail["trail_id"], prior
                )
            elif intraday_model and i == 1 and hourly_tomorrow:
                day_entry["slots"] = predict_slots(
                    intraday_model, hist, fday, hourly_tomorrow, trail["trail_id"], prior
                )

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

    print(f"Wrote {OUTPUT_PATH}")
    for key, trail_data in results.items():
        print(f"\n{trail_data['trail_name']}:")
        for d in trail_data["days"]:
            bar = "█" * int(d["score"] * 10)
            print(f"  {d['day_name']:12s} {bar:10s} {d['score']:.0%}  {d['signal']}")
            if "slots" in d:
                for slot in d["slots"]:
                    print(f"    {slot['slot']:6s} {slot['score']:.0%}")
        rr = trail_data["recent_report"]
        if rr:
            trust = " (TRUSTED)" if rr["trusted"] else ""
            print(f"  Latest report: {rr['color']} by {rr['username']}{trust} on {rr['date']}")


if __name__ == "__main__":
    main()
