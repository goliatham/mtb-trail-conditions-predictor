"""Feature engineering: transform weather history + forecast day into model inputs."""

import math
from datetime import date


RAIN_THRESHOLD_MM = 5.0  # significant rain event


TRAIL_DRYING_DAYS = {0: 3, 1: 2}  # P1 drains slower than P2

PRIOR_REPORT_TAU = 7.0          # decay constant in days (half-life ≈ 4.85 days)
_LABEL_MAP = {0: -1.0, 1: 0.0, 2: 1.0}


def build_features(history: list[dict], forecast_day: dict,
                   trail_id: int = 0, prior_report: dict = None) -> dict:
    """
    Build a feature dict for a single prediction day.

    history: ordered list of daily weather dicts (oldest first) for days
             BEFORE the prediction day
    forecast_day: weather dict for the prediction day itself
    trail_id: 0 = Phase 1, 1 = Phase 2
    prior_report: dict with keys 'label' (0/1/2) and 'days_ago' (int),
                  or None for no known prior report
    """
    precip_history = [r["precip_mm"] for r in history]
    soil_values = [r["soil_moisture"] for r in history if r["soil_moisture"] is not None]
    soil_deep_values = [r["soil_moisture_deep"] for r in history if r.get("soil_moisture_deep") is not None]

    precip_1d = sum(precip_history[-1:])
    precip_3d = sum(precip_history[-3:])
    precip_7d = sum(precip_history[-7:])

    days_since_rain = _days_since_last_rain(precip_history)
    consecutive_dry = _consecutive_dry_days(precip_history)

    soil_moisture = soil_values[-1] if soil_values else 0.2
    soil_moisture_deep = soil_deep_values[-1] if soil_deep_values else 0.25

    pred_date = date.fromisoformat(forecast_day["date"])

    drying_threshold = TRAIL_DRYING_DAYS.get(trail_id, 3)
    dry_surplus = consecutive_dry - drying_threshold  # negative = not dry enough yet

    if prior_report:
        label_val = _LABEL_MAP.get(prior_report["label"], 0.0)
        days = min(prior_report.get("days_ago", 30), 30)
        prior_report_score = label_val * math.exp(-days / PRIOR_REPORT_TAU)
    else:
        prior_report_score = 0.0

    raw_prob = forecast_day.get("precip_prob_pct")
    return {
        "precip_1d_mm": precip_1d,
        "precip_3d_mm": precip_3d,
        "precip_7d_mm": precip_7d,
        "soil_moisture": soil_moisture,
        "temp_max_c": forecast_day["temp_max_c"] or 20.0,
        "temp_min_c": forecast_day["temp_min_c"] or 10.0,
        "days_since_rain": days_since_rain,
        "consecutive_dry_days": consecutive_dry,
        "dry_surplus": dry_surplus,
        "month": pred_date.month,
        "forecast_precip_mm": forecast_day["precip_mm"] or 0.0,
        "precip_prob_pct": raw_prob if raw_prob is not None else 50.0,
        "soil_moisture_deep": soil_moisture_deep,
        "prior_report_score": prior_report_score,
        "trail_id": trail_id,
    }


def _days_since_last_rain(precip_history: list[float]) -> int:
    for i, p in enumerate(reversed(precip_history)):
        if p >= RAIN_THRESHOLD_MM:
            return i
    return len(precip_history)


def _consecutive_dry_days(precip_history: list[float]) -> int:
    count = 0
    for p in reversed(precip_history):
        if p < 1.0:
            count += 1
        else:
            break
    return count


def _hours_since_rain(hourly: list[dict], hour: int, days_since_rain: int) -> float:
    """Hours since last rain ≥1mm before this slot.

    Scans today's hourly data first; falls back to days_since_rain × 24 + slot_hour
    for the case where rain was on a prior day (no hour precision available).
    """
    rain_hours = [r["hour"] for r in hourly if r["hour"] < hour and r["precip_mm"] >= 1.0]
    if rain_hours:
        return float(hour - max(rain_hours))
    return float(days_since_rain * 24 + hour)


TIME_SLOTS = [7, 11, 15, 19]  # hours: 7am, 11am, 3pm, 7pm


def build_intraday_features(history: list[dict], forecast_day: dict,
                            hourly: list[dict], hour: int,
                            trail_id: int = 0, prior_report: dict = None) -> dict:
    """Build features for a specific time slot within a day."""
    base = build_features(history, forecast_day, trail_id, prior_report)
    precip_to_slot = sum(r["precip_mm"] for r in hourly if r["hour"] < hour)
    precip_3h = sum(r["precip_mm"] for r in hourly if hour - 3 <= r["hour"] < hour)
    base["precip_midnight_to_slot_mm"] = precip_to_slot
    base["precip_3h_before_slot_mm"] = precip_3h
    base["hour"] = hour
    base["hours_since_rain"] = _hours_since_rain(hourly, hour, base["days_since_rain"])
    return base


def assign_intraday_label(day_label: int, hourly: list[dict], hour: int) -> int:
    """
    Infer the condition label for a specific time slot from a daily label + hourly rain.

    day_label: 0=red, 1=yellow, 2=green
    Returns: 0/1/2
    """
    precip_before = sum(r["precip_mm"] for r in hourly if r["hour"] < hour)
    precip_total = sum(r["precip_mm"] for r in hourly)

    if day_label == 2:  # green day
        # If it rained heavily before this slot, downgrade
        if precip_before > 8:
            return 1
        return 2

    if day_label == 0:  # red/closed day
        # If rain hadn't started yet before this slot, slot might have been rideable
        if precip_before < 2 and precip_total > 8:
            return 1  # yellow — probably rideable before the rain hit
        return 0

    # yellow — stays yellow across all slots
    return 1


FEATURE_COLUMNS = [
    "precip_1d_mm",
    "precip_3d_mm",
    "precip_7d_mm",
    "soil_moisture",
    "soil_moisture_deep",
    "temp_max_c",
    "temp_min_c",
    "days_since_rain",
    "consecutive_dry_days",
    "dry_surplus",
    # "month",  # ablation: testing whether month suppresses soil_moisture
    "forecast_precip_mm",
    "prior_report_score",
    "trail_id",
]

INTRADAY_FEATURE_COLUMNS = FEATURE_COLUMNS + [
    "precip_midnight_to_slot_mm",
    "precip_3h_before_slot_mm",
    "hour",
    "hours_since_rain",
]
