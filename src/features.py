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
    rain_history   = [r.get("rain_mm", r["precip_mm"]) for r in history]
    snow_history   = [r.get("snow_cm", 0.0) for r in history]
    creek_history  = [r.get("creek_peak_ft") for r in history]

    soil_values      = [r["soil_moisture"]       for r in history if r["soil_moisture"] is not None]
    soil_deep_values = [r["soil_moisture_deep"]   for r in history if r.get("soil_moisture_deep") is not None]
    soil_t0_values   = [r["soil_temp_0cm"]        for r in history if r.get("soil_temp_0cm")  is not None]
    soil_t6_values   = [r["soil_temp_6cm"]        for r in history if r.get("soil_temp_6cm")  is not None]
    soil_t18_values  = [r["soil_temp_18cm"]       for r in history if r.get("soil_temp_18cm") is not None]

    precip_1d = sum(precip_history[-1:])
    precip_3d = sum(precip_history[-3:])
    precip_7d = sum(precip_history[-7:])

    rain_1d = sum(rain_history[-1:])
    rain_2d = sum(rain_history[-2:])
    rain_3d = sum(rain_history[-3:])
    rain_7d = sum(rain_history[-7:])

    snow_1d = sum(snow_history[-1:])
    snow_2d = sum(snow_history[-2:])
    snow_3d = sum(snow_history[-3:])
    snow_7d = sum(snow_history[-7:])

    days_since_rain = _days_since_last_rain(precip_history)
    consecutive_dry = _consecutive_dry_days(precip_history)

    def _creek_max(window):
        vals = [v for v in window if v is not None]
        return max(vals) if vals else 0.0

    creek_peak_1d  = _creek_max(creek_history[-1:])
    creek_peak_7d  = _creek_max(creek_history[-7:])
    creek_peak_14d = _creek_max(creek_history[-14:])

    soil_moisture      = soil_values[-1]     if soil_values      else 0.2
    soil_moisture_deep = soil_deep_values[-1] if soil_deep_values else 0.25
    soil_temp_0cm      = soil_t0_values[-1]  if soil_t0_values   else 10.0
    soil_temp_6cm      = soil_t6_values[-1]  if soil_t6_values   else 10.0
    soil_temp_18cm     = soil_t18_values[-1] if soil_t18_values  else 10.0

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
        "precip_1d_mm":       precip_1d,
        "precip_3d_mm":       precip_3d,
        "precip_7d_mm":       precip_7d,
        "rain_1d_mm":         rain_1d,
        "rain_2d_mm":         rain_2d,
        "rain_3d_mm":         rain_3d,
        "rain_7d_mm":         rain_7d,
        "snow_1d_cm":         snow_1d,
        "snow_2d_cm":         snow_2d,
        "snow_3d_cm":         snow_3d,
        "snow_7d_cm":         snow_7d,
        "soil_moisture":      soil_moisture,
        "soil_moisture_deep": soil_moisture_deep,
        "soil_temp_0cm":      soil_temp_0cm,
        "soil_temp_6cm":      soil_temp_6cm,
        "soil_temp_18cm":     soil_temp_18cm,
        "temp_max_c":         forecast_day["temp_max_c"] or 20.0,
        "temp_min_c":         forecast_day["temp_min_c"] or 10.0,
        "days_since_rain":    days_since_rain,
        "consecutive_dry_days": consecutive_dry,
        "dry_surplus":        dry_surplus,
        "month":              pred_date.month,
        "forecast_precip_mm": forecast_day["precip_mm"] or 0.0,
        "precip_prob_pct":    raw_prob if raw_prob is not None else 50.0,
        "prior_report_score": prior_report_score,
        "trail_id":           trail_id,
        "creek_peak_1d_ft":   creek_peak_1d,
        "creek_peak_7d_ft":   creek_peak_7d,
        "creek_peak_14d_ft":  creek_peak_14d,
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


def _hours_since_rain(hourly: list[dict], hour: int, days_since_rain: int,
                       history_hourly: list[list[dict]] = None) -> float:
    """Hours since last rain ≥1mm before this slot.

    Scans today's hourly first, then history_hourly (newest day first) for
    exact hour precision.  Falls back to days_since_rain × 24 + slot_hour
    when no hourly history is available.
    """
    rain_hours = [r["hour"] for r in hourly if r["hour"] < hour and r["precip_mm"] >= 1.0]
    if rain_hours:
        return float(hour - max(rain_hours))
    if history_hourly:
        for days_back, day_hourly in enumerate(history_hourly, 1):
            rain_in_day = [r["hour"] for r in day_hourly if r["precip_mm"] >= 1.0]
            if rain_in_day:
                last_hour = max(rain_in_day)
                # hours from last_hour on days_back days ago to current slot
                return float((days_back - 1) * 24 + (24 - last_hour) + hour)
        return float(len(history_hourly) * 24 + hour)
    return float(days_since_rain * 24 + hour)


TIME_SLOTS = [7, 11, 15, 19]  # hours: 7am, 11am, 3pm, 7pm


def build_intraday_features(history: list[dict], forecast_day: dict,
                            hourly: list[dict], hour: int,
                            trail_id: int = 0, prior_report: dict = None,
                            history_hourly: list[list[dict]] = None) -> dict:
    """Build features for a specific time slot within a day.

    history_hourly: daily hourly arrays for recent past days, newest first.
    When provided, hours_since_rain uses actual rain timestamps instead of
    falling back to whole-day granularity.
    """
    base = build_features(history, forecast_day, trail_id, prior_report)
    precip_to_slot = sum(r["precip_mm"] for r in hourly if r["hour"] < hour)
    precip_3h = sum(r["precip_mm"] for r in hourly if hour - 3 <= r["hour"] < hour)

    # Temperature at slot hour from hourly; interpolate from daily min/max if absent
    temp_at_slot = next((r["temp_c"] for r in hourly if r.get("hour") == hour and r.get("temp_c") is not None), None)
    if temp_at_slot is None:
        t_min = forecast_day.get("temp_min_c") or 10.0
        t_max = forecast_day.get("temp_max_c") or 20.0
        # rough diurnal: min at 6am, max at 2pm
        frac = max(0.0, min(1.0, (hour - 6) / 8.0)) if hour <= 14 else max(0.0, 1.0 - (hour - 14) / 10.0)
        temp_at_slot = t_min + frac * (t_max - t_min)

    base["precip_midnight_to_slot_mm"] = precip_to_slot
    base["precip_3h_before_slot_mm"]   = precip_3h
    base["hour"]                       = hour
    base["hours_since_rain"]           = _hours_since_rain(hourly, hour, base["days_since_rain"], history_hourly)
    base["temp_at_slot_c"]             = temp_at_slot
    return base


FEATURE_COLUMNS = [
    "precip_1d_mm",
    "precip_3d_mm",
    "precip_7d_mm",
    "rain_1d_mm",
    "rain_2d_mm",
    "rain_3d_mm",
    "rain_7d_mm",
    "snow_1d_cm",
    "snow_2d_cm",
    "snow_3d_cm",
    "snow_7d_cm",
    "soil_moisture",
    "soil_moisture_deep",
    "soil_temp_0cm",
    "soil_temp_6cm",
    "soil_temp_18cm",
    "temp_max_c",
    "temp_min_c",
    "days_since_rain",
    "consecutive_dry_days",
    "dry_surplus",
    # "month",  # ablation: testing whether month suppresses soil_moisture
    "forecast_precip_mm",
    "prior_report_score",
    "trail_id",
    "creek_peak_1d_ft",
    "creek_peak_7d_ft",
    "creek_peak_14d_ft",
]

INTRADAY_FEATURE_COLUMNS = [
    c for c in FEATURE_COLUMNS if c != "days_since_rain"
] + [
    "precip_midnight_to_slot_mm",
    "precip_3h_before_slot_mm",
    "hour",
    "hours_since_rain",
    "temp_at_slot_c",
]
