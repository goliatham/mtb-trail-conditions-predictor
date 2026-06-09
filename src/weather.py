"""Open-Meteo client for Alum Creek historical weather and 7-day forecast."""

import time
import requests
from datetime import date, timedelta


def _get(url, params, timeouts=(30, 60, 90)):
    """GET with retries and escalating timeouts."""
    last_err = None
    for i, timeout in enumerate(timeouts):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            if i < len(timeouts) - 1:
                time.sleep(2 ** i)
    raise last_err

# Alum Creek State Park, Delaware OH
LAT = 40.35
LON = -82.98

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


def get_forecast() -> list[dict]:
    """Fetch 7-day daily forecast starting today."""
    resp = _get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": LAT,
            "longitude": LON,
            "daily": ",".join(DAILY_VARS),
            "timezone": "America/New_York",
            "forecast_days": 7,
        },
    )
    return _parse_daily(resp.json())


def get_window(target: date, history_days: int = 14) -> list[dict]:
    """Return historical days leading up to (but not including) target date."""
    start = target - timedelta(days=history_days)
    end = target - timedelta(days=1)
    return get_historical(start, end)


def _parse_daily(payload: dict) -> list[dict]:
    daily = payload["daily"]
    dates = daily["time"]
    rows = []
    for i, d in enumerate(dates):
        rows.append({
            "date": d,
            "precip_mm": daily["precipitation_sum"][i] or 0.0,
            "temp_max_c": daily["temperature_2m_max"][i],
            "temp_min_c": daily["temperature_2m_min"][i],
            "soil_moisture": daily["soil_moisture_0_to_7cm_mean"][i],
            "soil_moisture_deep": daily["soil_moisture_7_to_28cm_mean"][i],
        })
    return rows


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
