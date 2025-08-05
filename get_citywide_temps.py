import httpx
from pprint import pprint
from datetime import datetime, date, timedelta
from matplotlib import pyplot as plt
import pandas as pd
import math
import numpy as np
import model
import os

plt.rcParams.update(
    {
        "font.size": 18,
        "font.family": "sans-serif",
        "font.sans-serif": "Arial",
        "font.weight": "bold",
        "axes.labelweight": "bold",
    }
)


headers = {
    "User-Agent": "vrbow4EbquY4wxNtd7wznDQqUvWuXiUvoQse9zZA9FXSgJwZ"
}  # random string for user agent, TODO: change later, not important

# station_lat = 40.764629
# station_long = -73.966113
# gtfs_id = "B08"
# title = "Lexington Av/63 St"
station_lat = 40.616622
station_long = -74.030876
gtfs_id = "R45"
title = "Bay Ridge-95 St"


## CREDIT TO https://api.weather.gov
# get nearest station
url = f"https://api.weather.gov/points/{station_lat:.4f},{station_long:.4f}"
print("Requesting ", url)
data = httpx.get(url, headers=headers).json()


forecastHourlyUrl = data["properties"]["forecastHourly"]

print(forecastHourlyUrl)


def calculate_heat_index(T, RH):
    T = float(T)
    RH = float(RH)

    if T <= 80:  # no need to use heat index if temp is below 80F
        return T

    # T in Farenheit, RH as a percentage
    HI = (
        -42.379
        + 2.04901523 * T
        + 10.14333127 * RH
        - 0.22475541 * T * RH
        - 0.00683783 * T * T
        - 0.05481717 * RH * RH
        + 0.00122874 * T * T * RH
        + 0.00085282 * T * RH * RH
        - 0.00000199 * T * T * RH * RH
    )
    if RH < 13 and T > 80 and T < 112:
        HI -= ((13 - RH) / 4) * math.sqrt((17 - abs(T - 95.0)) / 17)
    if RH > 85 and T > 80 and T < 87:
        HI += ((RH - 85) / 10) * ((87 - T) / 5)
    if HI < 80:
        # use simpler formula
        return 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (RH * 0.094))
    return HI


def fToC(temp: float):
    return (temp - 32) * (5 / 9)


def get_min_or_max_from_hourly_preds(hourly_preds, date_to_filter: date, min_or_max):
    best = None

    just_this_date = list(
        filter(
            lambda pred: pred["startTime"].date() == date_to_filter,
            hourly_preds,
        )
    )

    for pred in just_this_date:
        if min_or_max == "max":
            if not best or pred["temperature"] > best:
                best = pred["temperature"]
        elif min_or_max == "min":
            if not best or pred["temperature"] < best:
                best = pred["temperature"]

    return best


# hourly data
hourlyData = httpx.get(forecastHourlyUrl, headers=headers).json()

city_hourly_preds = hourlyData["properties"]["periods"]  # across days

# convert to datetime objects
for pred in city_hourly_preds:  # modifies list
    pred["startTime"] = datetime.fromisoformat(pred["startTime"])
    pred["temperature"] = float(pred["temperature"])

times = []
city_temps = []
plat_temps = []
platform_heat_indexes = []


for pred in city_hourly_preds[12:]:
    pred_time = pred["startTime"]
    times.append(pred_time)

    pred_temp = float(pred["temperature"])
    city_temps.append(fToC(pred_temp))

    hour = pred_time.hour
    month = pred_time.month
    day_of_week = pred_time.weekday()

    low_temp = get_min_or_max_from_hourly_preds(
        city_hourly_preds, pred_time.date(), "min"
    )
    high_temp = get_min_or_max_from_hourly_preds(
        city_hourly_preds, pred_time.date(), "max"
    )

    prev_low = get_min_or_max_from_hourly_preds(
        city_hourly_preds, (pred_time - timedelta(days=1)).date(), "min"
    )
    prev_high = get_min_or_max_from_hourly_preds(
        city_hourly_preds, (pred_time - timedelta(days=1)).date(), "max"
    )

    station_encoded = model.le_station.transform([gtfs_id])[0]

    humidity_input = pd.DataFrame(
        {
            "High Temp (°F)": [high_temp],
            "Low Temp (°F)": [low_temp],
            "Day_of_Week": [day_of_week],
            "Station_encoded": [station_encoded],
            "Hour": [hour],
            "Month": [month],
        }
    )
    humidity_input = humidity_input.reindex(
        columns=model.humidity_feature_cols, fill_value=0
    )
    predicted_humidity = model.humidity_model.predict(humidity_input)[0]

    offset_input_df = pd.DataFrame(
        {
            "Station_encoded": [station_encoded],
            "Hour": [hour],
            "Day_of_Week": [day_of_week],
            "Street level relative humidity": [predicted_humidity],
            "High Temp (°F)": [high_temp],
            "Low Temp (°F)": [low_temp],
            "Month": [month],
        }
    )
    offset_input_df = offset_input_df.reindex(
        columns=model.offset_model.feature_names_in_, fill_value=0
    )
    offset_pred = model.offset_model.predict(offset_input_df)[0]
    offset_pred_clipped = np.clip(offset_pred, -3, 15)
    street_level_temp_pred = high_temp + offset_pred_clipped

    platform_offset_input_df = pd.DataFrame(
        {
            "Station_encoded": [station_encoded],
            "Hour": [hour],
            "Day_of_Week": [day_of_week],
            "Street level air temperature": [street_level_temp_pred],
            "Prev_High": [prev_high],
            "Prev_Low": [prev_low],
        }
    )
    platform_offset_input_df = platform_offset_input_df.reindex(
        columns=model.platform_offset_model.feature_names_in_, fill_value=0
    )
    platform_offset_pred = model.platform_offset_model.predict(
        platform_offset_input_df
    )[0]
    platform_temp_pred = street_level_temp_pred + platform_offset_pred

    # Platform level relative humidity
    platform_humidity_input_df = pd.DataFrame(
        {
            "Platform level air temperature": [platform_temp_pred],
            "Station_encoded": [station_encoded],
            "Hour": [hour],
            "Day_of_Week": [day_of_week],
            "Prev_High": [prev_high],
            "Prev_Low": [prev_low],
        }
    ).reindex(columns=model.platform_humidity_model.feature_names_in_, fill_value=0)
    oredicted_plat_humidity = model.platform_humidity_model.predict(
        platform_humidity_input_df
    )[0]

    platform_level_heat_index = calculate_heat_index(
        platform_temp_pred, oredicted_plat_humidity
    )  # this is with street level predicted humidity

    if prev_low == None:  # if this is TODAY
        plat_temps.append(None)
        platform_heat_indexes.append(None)
    else:
        plat_temps.append(fToC(platform_temp_pred))
        platform_heat_indexes.append(fToC(platform_level_heat_index))

    print(
        f"{pred_time} | {low_temp}, {high_temp} | {platform_temp_pred} | (plat){oredicted_plat_humidity} | Plat heat index: {platform_level_heat_index}"
    )

CHART_OUT_DIR = "charts_out"

if not os.path.exists(CHART_OUT_DIR):
    os.makedirs(CHART_OUT_DIR)


fig, ax = plt.subplots(figsize=(20, 10))

plt.axhspan(0, 26.5, color="#3da935", alpha=0.3)
plt.axhspan(26.5, 32.5, color="#fecc00", alpha=0.3)
plt.axhspan(32.5, 40.5, color="#f7a92c", alpha=0.3)
plt.axhspan(40.5, 51.5, color="#e54e1f", alpha=1)

plt.plot(
    times,
    platform_heat_indexes,
    label="Platform Level Heat Index",
    linewidth=6,
)
plt.plot(times, city_temps, label="Citywide Air Temperature", linewidth=6)
plt.plot(times, plat_temps, label="Platform Level Air Temperature", linewidth=6)

# for bay ridge
if gtfs_id == "R45":
    plt.text(
        0.71,
        0.945,
        "DANGER!",
        fontsize=36,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )
    plt.arrow(
        0.78,
        0.945,
        0.024,
        -0.01,
        width=0.01,
        head_length=0.013,
        transform=ax.transAxes,
        color="#000",
    )

plt.xlabel("Time")
plt.ylabel("°C")
plt.title(title, fontweight="bold")
plt.legend()
plt.tight_layout()

plt.ylim(17, 43)

plt.savefig(
    os.path.join(CHART_OUT_DIR, f"out_{datetime.now()}.png"),
    bbox_inches="tight",
    dpi=400,
    transparent=False,
)

plt.show()
