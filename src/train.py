"""Join scraped condition labels with historical weather, train daily + intraday models."""

import csv
import json
from datetime import date, timedelta
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from features import (
    FEATURE_COLUMNS,
    INTRADAY_FEATURE_COLUMNS,
    TIME_SLOTS,
    assign_intraday_label,
    build_features,
    build_intraday_features,
)
from weather import get_historical, get_hourly_day

DATA_PATH = Path(__file__).parent.parent / "data" / "training_raw.csv"
FEEDBACK_PATH = Path(__file__).parent.parent / "data" / "user_feedback.json"
MODEL_PATH = Path(__file__).parent.parent / "model" / "model.joblib"
INTRADAY_MODEL_PATH = Path(__file__).parent.parent / "model" / "model_intraday.joblib"
HISTORY_DAYS = 14

# Reports are almost always posted in the afternoon. Same-day labels are
# therefore less reliable for morning slots where conditions may differ.
MORNING_SLOT_WEIGHT_MULT = 0.2   # applied to MTBProject labels for 7am/11am slots
MORNING_SLOTS = {7, 11}


def load_labels():
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_feedback():
    """Load user slot feedback exported from the browser UI."""
    if not FEEDBACK_PATH.exists():
        return []
    with open(FEEDBACK_PATH) as f:
        return json.load(f)


def _get_history(label_date, cache):
    start = label_date - timedelta(days=HISTORY_DAYS)
    end = label_date - timedelta(days=1)
    key = ("hist", start.isoformat())
    if key not in cache:
        cache[key] = get_historical(start, end)
    return cache[key]


def _get_day_weather(label_date, cache):
    key = ("day", label_date.isoformat())
    if key not in cache:
        result = get_historical(label_date, label_date)
        cache[key] = result[0] if result else None
    return cache[key]


def _get_hourly(label_date, cache):
    key = ("hourly", label_date.isoformat())
    if key not in cache:
        try:
            cache[key] = get_hourly_day(label_date)
        except Exception:
            cache[key] = []
    return cache[key]


def main():
    MODEL_PATH.parent.mkdir(exist_ok=True)

    labels = load_labels()
    print(f"Loaded {len(labels)} labeled records")

    daily_rows, daily_targets, daily_weights = [], [], []
    intraday_rows, intraday_targets, intraday_weights = [], [], []
    cache = {}
    skipped = 0

    for i, rec in enumerate(labels):
        try:
            label_date = date.fromisoformat(rec["date"])
        except ValueError:
            skipped += 1
            continue

        trail_id = int(rec["trail_id"])
        day_label = int(rec["label"])
        weight = 1.0 if rec["trusted"] == "True" else 0.4

        history = _get_history(label_date, cache)
        day_weather = _get_day_weather(label_date, cache)
        if day_weather is None:
            skipped += 1
            continue

        # Daily row
        feats = build_features(history, day_weather, trail_id)
        daily_rows.append([feats[col] for col in FEATURE_COLUMNS])
        daily_targets.append(day_label)
        daily_weights.append(weight)

        # Intraday rows — 4 slots per date using historical hourly precip
        hourly = _get_hourly(label_date, cache)
        if hourly:
            for hour in TIME_SLOTS:
                ifeats = build_intraday_features(history, day_weather, hourly, hour, trail_id)
                slot_label = assign_intraday_label(day_label, hourly, hour)
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

            history = _get_history(fb_date, cache)
            day_weather = _get_day_weather(fb_date, cache)
            if day_weather is None:
                fb_skipped += 1
                continue

            hourly = _get_hourly(fb_date, cache)
            if not hourly:
                fb_skipped += 1
                continue

            ifeats = build_intraday_features(history, day_weather, hourly, hour, trail_id)
            intraday_rows.append([ifeats[col] for col in INTRADAY_FEATURE_COLUMNS])
            intraday_targets.append(slot_label)
            intraday_weights.append(1.5)  # direct observation — no morning discount

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

    X_d = pd.DataFrame(daily_rows, columns=FEATURE_COLUMNS)
    train_model(X_d, pd.Series(daily_targets), pd.Series(daily_weights),
                FEATURE_COLUMNS, "Daily model", MODEL_PATH)

    if intraday_rows:
        X_i = pd.DataFrame(intraday_rows, columns=INTRADAY_FEATURE_COLUMNS)
        train_model(X_i, pd.Series(intraday_targets), pd.Series(intraday_weights),
                    INTRADAY_FEATURE_COLUMNS, "Intraday model", INTRADAY_MODEL_PATH)
    else:
        print("\nNo hourly data available — intraday model not trained")


if __name__ == "__main__":
    main()
