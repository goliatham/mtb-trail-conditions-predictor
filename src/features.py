"""Feature engineering: transform weather history + forecast day into model inputs."""

from datetime import date


RAIN_THRESHOLD_MM = 5.0  # significant rain event


def build_features(history: list[dict], forecast_day: dict) -> dict:
    """
    Build a feature dict for a single prediction day.

    history: ordered list of daily weather dicts (oldest first) for days
             BEFORE the prediction day (from weather.get_window or similar)
    forecast_day: weather dict for the prediction day itself
    """
    precip_history = [r["precip_mm"] for r in history]
    soil_values = [r["soil_moisture"] for r in history if r["soil_moisture"] is not None]

    precip_1d = sum(precip_history[-1:])
    precip_3d = sum(precip_history[-3:])
    precip_7d = sum(precip_history[-7:])

    days_since_rain = _days_since_last_rain(precip_history)
    consecutive_dry = _consecutive_dry_days(precip_history)

    soil_moisture = soil_values[-1] if soil_values else 0.2  # fallback to moderate

    pred_date = date.fromisoformat(forecast_day["date"])

    return {
        "precip_1d_mm": precip_1d,
        "precip_3d_mm": precip_3d,
        "precip_7d_mm": precip_7d,
        "soil_moisture": soil_moisture,
        "temp_max_c": forecast_day["temp_max_c"] or 20.0,
        "temp_min_c": forecast_day["temp_min_c"] or 10.0,
        "days_since_rain": days_since_rain,
        "consecutive_dry_days": consecutive_dry,
        "month": pred_date.month,
        "forecast_precip_mm": forecast_day["precip_mm"] or 0.0,
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


TIME_SLOTS = [7, 11, 15, 19]  # hours: 7am, 11am, 3pm, 7pm


def build_intraday_features(history: list[dict], forecast_day: dict,
                            hourly: list[dict], hour: int) -> dict:
    """Build features for a specific time slot within a day."""
    base = build_features(history, forecast_day)
    precip_to_slot = sum(r["precip_mm"] for r in hourly if r["hour"] < hour)
    precip_3h = sum(r["precip_mm"] for r in hourly if hour - 3 <= r["hour"] < hour)
    precip_after_slot = sum(r["precip_mm"] for r in hourly if r["hour"] >= hour)
    base["precip_midnight_to_slot_mm"] = precip_to_slot
    base["precip_3h_before_slot_mm"] = precip_3h
    base["precip_slot_to_midnight_mm"] = precip_after_slot
    base["hour"] = hour
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
    "temp_max_c",
    "temp_min_c",
    "days_since_rain",
    "consecutive_dry_days",
    "month",
    "forecast_precip_mm",
    "trail_id",  # 0 = Phase 1, 1 = Phase 2
]

INTRADAY_FEATURE_COLUMNS = FEATURE_COLUMNS + [
    "precip_midnight_to_slot_mm",
    "precip_3h_before_slot_mm",
    "precip_slot_to_midnight_mm",
    "hour",
]
