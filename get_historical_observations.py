import httpx
from pprint import pprint
from datetime import datetime, date, timedelta, timezone
from matplotlib import pyplot as plt
import pandas as pd
import math
import numpy as np
import os
import json

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

# TODO: get zone id by station (check: are they all in the same zone?)
zone_id = "NYZ072"


url = f"https://api.weather.gov/points/{station_lat:.4f},{station_long:.4f}"
print("Requesting ", url)
data = httpx.get(url, headers=headers).json()

start_time = datetime(2024, 6, 20).astimezone(timezone.utc)
end_time = datetime(2024, 6, 29).astimezone(timezone.utc)
params = {
    "start": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "end": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
}
pprint(params)
historical_url = f"https://api.weather.gov/zones/forecast/{zone_id}/observations"
historical_data = httpx.get(historical_url, headers=headers, params=params).json()

pprint(historical_data)
observation_groups = []
for feature in historical_data["features"][:15]:
    observation = feature["properties"]

    observation["timestamp"] = datetime.fromisoformat(observation["timestamp"])
    # if we need a new array
    if len(observation_groups) == 0:
        observation_groups.append([])

    if (
        len(observation_groups[-1]) == 0
        or observation["timestamp"] == observation_groups[-1][0]["timestamp"]
    ):  # part of the same group
        observation_groups[-1].append(observation)
    elif (
        observation["timestamp"] != observation_groups[-1][0]["timestamp"]
    ):  # unique group
        observation_groups.append([observation])

print("\nData groups:")

for observation_group in observation_groups:
    for observation in observation_group:
        station_id = observation["stationId"]
        temp = observation["temperature"]["value"]
        unitCode = observation["temperature"]["unitCode"][-4:]
        relativeHumidity = observation["relativeHumidity"]["value"]
        timestamp = observation["timestamp"]
        if relativeHumidity is None or temp is None:
            continue

        print(
            f"At {timestamp}, temp was {temp}{unitCode} and relative humidity was {relativeHumidity:.2f} at {station_id}"
        )
    print()
