"""Open-Meteo client for Alum Creek historical weather and 7-day forecast."""

import time
import requests
from datetime import date, timedelta


def _get(url, params, timeouts=(30, 60, 90), retry_delays=(10, 30)):
    """GET with retries, escalating timeouts, and longer delays for SSL/connection errors."""
    last_err = None
    for i, timeout in enumerate(timeouts):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.SSLError) as e:
            last_err = e
            if i < len(timeouts) - 1:
                time.sleep(retry_delays[min(i, len(retry_delays) - 1)])
    raise last_err

# Alum Creek State Park, Delaware OH
LAT = 40.201876
LON = -82.938115

DAILY_VARS = [
    "precipitation_sum",
    "temperature_2m_max",
    "temperature_2m_min",
    "soil_moisture_0_to_7cm_mean",
    "soil_moisture_7_to_28cm_mean",
]


def get_historical(start: date, end: date) -> list[dict]:
    """Fetch daily historical weather between start and end dates (inclusive)."""
    resp = _get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": LAT,
            "longitude": LON,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "daily": ",".join(DAILY_VARS),
            "timezone": "America/New_York",
        },
    )
    return _parse_daily(resp.json())


def get_forecast() -> tuple[list[dict], dict[str, list[dict]]]:
    """Fetch 7-day daily forecast + hourly precip in one call.

    Returns (daily_list, hourly_by_date) where hourly_by_date is keyed by
    date string — avoids multiple round-trips to api.open-meteo.com.
    """
    resp = _get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": LAT,
            "longitude": LON,
            "daily": ",".join(DAILY_VARS) + ",sunrise,sunset,precipitation_probability_max",
            "hourly": (
                "precipitation"
                ",soil_moisture_0_to_1cm,soil_moisture_1_to_3cm"
                ",soil_moisture_3_to_9cm,soil_moisture_9_to_27cm"
            ),
            "timezone": "America/New_York",
            "forecast_days": 7,
        },
    )
    data = resp.json()
    daily = _parse_daily(data)
    for i, d in enumerate(daily):
        d["sunrise"] = data["daily"]["sunrise"][i][11:16]  # "HH:MM"
        d["sunset"] = data["daily"]["sunset"][i][11:16]

    hourly = data["hourly"]
    sm_0_1  = hourly.get("soil_moisture_0_to_1cm",  [])
    sm_1_3  = hourly.get("soil_moisture_1_to_3cm",  [])
    sm_3_9  = hourly.get("soil_moisture_3_to_9cm",  [])
    sm_9_27 = hourly.get("soil_moisture_9_to_27cm", [])

    midnight_soil: dict[str, tuple] = {}
    hourly_by_date: dict[str, list[dict]] = {}
    for i, t in enumerate(hourly["time"]):
        d = t[:10]
        hour = int(t[11:13])
        hourly_by_date.setdefault(d, []).append({
            "hour": hour,
            "precip_mm": hourly["precipitation"][i] or 0.0,
        })
        if hour == 0 and d not in midnight_soil:
            s01, s13, s39, s927 = (
                sm_0_1[i] if i < len(sm_0_1) else None,
                sm_1_3[i] if i < len(sm_1_3) else None,
                sm_3_9[i] if i < len(sm_3_9) else None,
                sm_9_27[i] if i < len(sm_9_27) else None,
            )
            surf = (1*s01 + 2*s13 + 4*s39) / 7 if all(v is not None for v in (s01, s13, s39)) else None
            deep = (2*s39 + 18*s927) / 20 if all(v is not None for v in (s39, s927)) else None
            midnight_soil[d] = (surf, deep)

    # Overwrite daily soil moisture with start-of-day (midnight) forecast values.
    # The daily forecast API always returns None for soil moisture; hourly is the real source.
    for day_entry in daily:
        ms = midnight_soil.get(day_entry["date"])
        if ms:
            if ms[0] is not None:
                day_entry["soil_moisture"] = ms[0]
            if ms[1] is not None:
                day_entry["soil_moisture_deep"] = ms[1]

    return daily, hourly_by_date


def get_window(target: date, history_days: int = 14) -> list[dict]:
    """Return historical days leading up to (but not including) target date."""
    start = target - timedelta(days=history_days)
    end = target - timedelta(days=1)
    return get_historical(start, end)


def _parse_daily(payload: dict) -> list[dict]:
    daily = payload["daily"]
    dates = daily["time"]
    prob = daily.get("precipitation_probability_max")
    rows = []
    for i, d in enumerate(dates):
        rows.append({
            "date": d,
            "precip_mm": daily["precipitation_sum"][i] or 0.0,
            "temp_max_c": daily["temperature_2m_max"][i],
            "temp_min_c": daily["temperature_2m_min"][i],
            "soil_moisture": daily["soil_moisture_0_to_7cm_mean"][i],
            "soil_moisture_deep": daily["soil_moisture_7_to_28cm_mean"][i],
            "precip_prob_pct": prob[i] if prob is not None else None,
        })
    return rows


def get_hourly_range(start: date, end: date) -> dict[str, list[dict]]:
    """Fetch hourly precip for a date range. Returns dict keyed by date string."""
    resp = _get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": LAT,
            "longitude": LON,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": "precipitation,temperature_2m",
            "timezone": "America/New_York",
        },
    )
    hourly = resp.json()["hourly"]
    by_date: dict[str, list[dict]] = {}
    for i, t in enumerate(hourly["time"]):
        d = t[:10]
        by_date.setdefault(d, []).append({
            "hour": int(t[11:13]),
            "precip_mm": hourly["precipitation"][i] or 0.0,
        })
    return by_date


def get_hourly_day(target: date) -> list[dict]:
    """Fetch hourly precip and temp for a single date."""
    resp = _get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": LAT,
            "longitude": LON,
            "start_date": target.isoformat(),
            "end_date": target.isoformat(),
            "hourly": "precipitation,temperature_2m",
            "timezone": "America/New_York",
        },
    )
    hourly = resp.json()["hourly"]
    return [
        {"hour": int(t.split("T")[1][:2]), "precip_mm": hourly["precipitation"][i] or 0.0}
        for i, t in enumerate(hourly["time"])
    ]


def get_hourly_forecast_day(target: date) -> list[dict]:
    """Fetch hourly precip forecast for a single date (today or tomorrow)."""
    resp = _get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": LAT,
            "longitude": LON,
            "hourly": "precipitation",
            "timezone": "America/New_York",
            "forecast_days": 2,
        },
    )
    hourly = resp.json()["hourly"]
    return [
        {"hour": int(t.split("T")[1][:2]), "precip_mm": hourly["precipitation"][i] or 0.0}
        for i, t in enumerate(hourly["time"])
        if t.startswith(target.isoformat())
    ]


if __name__ == "__main__":
    today = date.today()
    print("=== Last 7 days ===")
    hist = get_historical(today - timedelta(days=7), today - timedelta(days=1))
    for r in hist:
        print(r)
    print("\n=== 7-day forecast ===")
    fcast = get_forecast()
    for r in fcast:
        print(r)
