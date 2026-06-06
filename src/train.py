"""Join scraped condition labels with historical weather, train daily + intraday models."""

import csv
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
MODEL_PATH = Path(__file__).parent.parent / "model" / "model.joblib"
INTRADAY_MODEL_PATH = Path(__file__).parent.parent / "model" / "model_intraday.joblib"
HISTORY_DAYS = 14


def load_labels():
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
        feats = build_features(history, day_weather)
        feats["trail_id"] = trail_id
        daily_rows.append([feats[col] for col in FEATURE_COLUMNS])
        daily_targets.append(day_label)
        daily_weights.append(weight)

        # Intraday rows — 4 slots per date using historical hourly precip
        hourly = _get_hourly(label_date, cache)
        if hourly:
            for hour in TIME_SLOTS:
                ifeats = build_intraday_features(history, day_weather, hourly, hour)
                ifeats["trail_id"] = trail_id
                slot_label = assign_intraday_label(day_label, hourly, hour)
                intraday_rows.append([ifeats[col] for col in INTRADAY_FEATURE_COLUMNS])
                intraday_targets.append(slot_label)
                intraday_weights.append(weight)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(labels)}...")

    print(f"Skipped {skipped} records (missing date or weather)")

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
