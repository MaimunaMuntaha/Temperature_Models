import httpx
from pprint import pprint
from datetime import datetime
from matplotlib import pyplot as plt

plt.rcParams.update(
    {"font.size": 18, "font.family": "sans-serif", "font.sans-serif": "Arial"}
)


headers = {"User-Agent": "vrbow4EbquY4wxNtd7wznDQqUvWuXiUvoQse9zZA9FXSgJwZ"}

station_lat = 40.7358  # up to 4 decimal places
station_long = -73.9905

# get nearest station
url = f"https://api.weather.gov/points/{station_lat:.4f},{station_long:.4f}"
print("Requesting ", url)
data = httpx.get(url, headers=headers).json()


forecastHourlyUrl = data["properties"]["forecastHourly"]

print(forecastHourlyUrl)


hourlyData = httpx.get(forecastHourlyUrl, headers=headers).json()

hourly_preds = hourlyData["properties"]["periods"]  # across days

times = []
temps = []

for pred in hourly_preds:
    pred_time = datetime.fromisoformat(pred["startTime"])
    times.append(pred_time)

    pred_temp = float(pred["temperature"])
    temps.append(pred_temp)

plt.plot(times, temps)
plt.xlabel("Time")
plt.ylabel("Temp")
plt.title("Time vs Temp @ UnSq")
plt.show()
