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
    "rain_sum",
    "snowfall_sum",
    "temperature_2m_max",
    "temperature_2m_min",
    "soil_moisture_0_to_7cm_mean",
    "soil_moisture_7_to_28cm_mean",
]

_HOURLY_VARS = (
    "precipitation,rain,snowfall,temperature_2m"
    ",soil_moisture_0_to_1cm,soil_moisture_1_to_3cm"
    ",soil_moisture_3_to_9cm,soil_moisture_9_to_27cm"
    ",soil_temperature_0cm,soil_temperature_6cm,soil_temperature_18cm"
)


def _build_hourly_and_soil(hourly: dict) -> tuple[dict[str, list[dict]], dict[str, tuple], dict[str, tuple]]:
    """Parse hourly section into (hourly_by_date, midnight_soil_by_date, midnight_soil_temp_by_date).

    midnight_soil values are (surface_0_7cm, deep_7_28cm) aggregated from
    the 4 IFS soil layers at hour=0 (start of day, before any rain).
    midnight_soil_temp values are (temp_0cm, temp_6cm, temp_18cm) at hour=0.
    hourly records include temp_c alongside precip_mm.
    """
    sm_0_1  = hourly.get("soil_moisture_0_to_1cm",  [])
    sm_1_3  = hourly.get("soil_moisture_1_to_3cm",  [])
    sm_3_9  = hourly.get("soil_moisture_3_to_9cm",  [])
    sm_9_27 = hourly.get("soil_moisture_9_to_27cm", [])
    st_0    = hourly.get("soil_temperature_0cm",    [])
    st_6    = hourly.get("soil_temperature_6cm",    [])
    st_18   = hourly.get("soil_temperature_18cm",   [])
    temps   = hourly.get("temperature_2m",          [])

    hourly_by_date: dict[str, list[dict]]  = {}
    midnight_soil:      dict[str, tuple]   = {}
    midnight_soil_temp: dict[str, tuple]   = {}

    for i, t in enumerate(hourly["time"]):
        d    = t[:10]
        hour = int(t[11:13])
        temp_c = temps[i] if i < len(temps) else None
        hourly_by_date.setdefault(d, []).append({
            "hour":      hour,
            "precip_mm": hourly["precipitation"][i] or 0.0,
            "temp_c":    temp_c,
        })
        if hour == 0 and d not in midnight_soil:
            s01  = sm_0_1[i]  if i < len(sm_0_1)  else None
            s13  = sm_1_3[i]  if i < len(sm_1_3)  else None
            s39  = sm_3_9[i]  if i < len(sm_3_9)  else None
            s927 = sm_9_27[i] if i < len(sm_9_27) else None
            surf = (1*s01 + 2*s13 + 4*s39) / 7 if all(v is not None for v in (s01, s13, s39)) else None
            deep = (2*s39 + 18*s927) / 20      if all(v is not None for v in (s39, s927))     else None
            midnight_soil[d] = (surf, deep)

            t0  = st_0[i]  if i < len(st_0)  else None
            t6  = st_6[i]  if i < len(st_6)  else None
            t18 = st_18[i] if i < len(st_18) else None
            midnight_soil_temp[d] = (t0, t6, t18)

    return hourly_by_date, midnight_soil, midnight_soil_temp


def _apply_midnight_soil(daily: list[dict], midnight_soil: dict[str, tuple]) -> None:
    """Overwrite daily soil entries with midnight (start-of-day) values from hourly."""
    for entry in daily:
        ms = midnight_soil.get(entry["date"])
        if ms:
            if ms[0] is not None:
                entry["soil_moisture"] = ms[0]
            if ms[1] is not None:
                entry["soil_moisture_deep"] = ms[1]


def _apply_midnight_soil_temps(daily: list[dict], midnight_soil_temp: dict[str, tuple]) -> None:
    """Write midnight soil temperature readings onto daily entries."""
    for entry in daily:
        mt = midnight_soil_temp.get(entry["date"])
        if mt:
            if mt[0] is not None:
                entry["soil_temp_0cm"]  = mt[0]
            if mt[1] is not None:
                entry["soil_temp_6cm"]  = mt[1]
            if mt[2] is not None:
                entry["soil_temp_18cm"] = mt[2]


def get_forecast(model: str = "best_match") -> tuple[list[dict], dict[str, list[dict]]]:
    """Fetch 7-day daily forecast + hourly precip/soil in one call.

    Returns (daily_list, hourly_by_date).  Daily soil moisture is set from
    the midnight (hour=0) hourly reading — same basis as get_historical_forecast().
    """
    resp = _get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude":     LAT,
            "longitude":    LON,
            "daily":        ",".join(DAILY_VARS) + ",sunrise,sunset,precipitation_probability_max",
            "hourly":       _HOURLY_VARS,
            "timezone":     "America/New_York",
            "forecast_days": 7,
            "models":       model,
        },
    )
    data  = resp.json()
    daily = _parse_daily(data)
    for i, d in enumerate(daily):
        d["sunrise"] = data["daily"]["sunrise"][i][11:16]
        d["sunset"]  = data["daily"]["sunset"][i][11:16]
    hourly_by_date, midnight_soil, midnight_soil_temp = _build_hourly_and_soil(data["hourly"])
    _apply_midnight_soil(daily, midnight_soil)
    _apply_midnight_soil_temps(daily, midnight_soil_temp)
    return daily, hourly_by_date


def get_historical_forecast(start: date, end: date, model: str = "best_match") -> tuple[list[dict], dict[str, list[dict]]]:
    """Fetch daily weather + hourly precip/soil from the historical forecast API.

    Uses the same IFS model as get_forecast() so training and inference see
    identical data source and scale.  Replaces get_historical() + get_hourly_range().

    Returns (daily_list, hourly_by_date).
    """
    resp = _get(
        "https://historical-forecast-api.open-meteo.com/v1/forecast",
        params={
            "latitude":   LAT,
            "longitude":  LON,
            "start_date": start.isoformat(),
            "end_date":   end.isoformat(),
            "daily":      ",".join(DAILY_VARS),
            "hourly":     _HOURLY_VARS,
            "timezone":   "America/New_York",
            "models":     model,
        },
    )
    data  = resp.json()
    daily = _parse_daily(data)
    hourly_by_date, midnight_soil, midnight_soil_temp = _build_hourly_and_soil(data["hourly"])
    _apply_midnight_soil(daily, midnight_soil)
    _apply_midnight_soil_temps(daily, midnight_soil_temp)
    return daily, hourly_by_date


def _parse_daily(payload: dict) -> list[dict]:
    daily = payload["daily"]
    dates = daily["time"]
    prob  = daily.get("precipitation_probability_max")
    rain  = daily.get("rain_sum")
    snow  = daily.get("snowfall_sum")
    rows  = []
    for i, d in enumerate(dates):
        rows.append({
            "date":               d,
            "precip_mm":          daily["precipitation_sum"][i] or 0.0,
            "rain_mm":            (rain[i] or 0.0) if rain is not None else 0.0,
            "snow_cm":            (snow[i] or 0.0) if snow is not None else 0.0,
            "temp_max_c":         daily["temperature_2m_max"][i],
            "temp_min_c":         daily["temperature_2m_min"][i],
            "soil_moisture":      daily["soil_moisture_0_to_7cm_mean"][i],
            "soil_moisture_deep": daily["soil_moisture_7_to_28cm_mean"][i],
            "precip_prob_pct":    prob[i] if prob is not None else None,
        })
    return rows


def get_hourly_day(target: date) -> list[dict]:
    """Fetch hourly precip and temp for a single historical date."""
    resp = _get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude":   LAT,
            "longitude":  LON,
            "start_date": target.isoformat(),
            "end_date":   target.isoformat(),
            "hourly":     "precipitation,temperature_2m",
            "timezone":   "America/New_York",
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
            "latitude":     LAT,
            "longitude":    LON,
            "hourly":       "precipitation",
            "timezone":     "America/New_York",
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
    print("=== Last 7 days (historical forecast) ===")
    hist, hourly = get_historical_forecast(today - timedelta(days=7), today - timedelta(days=1))
    for r in hist:
        print(r)
    print("\nSample hourly record:", hourly.get((today - timedelta(days=1)).isoformat(), [{}])[0])
    print("\n=== 7-day forecast ===")
    fcast, _ = get_forecast()
    for r in fcast:
        print(r)
