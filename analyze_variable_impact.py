import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pprint import pprint
import seaborn as sns
import os


plt.rcParams.update(
    {
        "font.size": 20,
        "font.family": "sans-serif",
        "font.sans-serif": "Arial",
        "font.weight": "bold",
        "axes.labelweight": "bold",
    }
)

CHART_OUT_DIR = "paper_charts"

if not os.path.exists(CHART_OUT_DIR):
    os.makedirs(CHART_OUT_DIR)


def f_to_c(f):
    return (f - 32) * (5 / 9)


cuny_df = pd.read_csv("CUNY_MTA.csv")
cuny_df["platform_collection_timestamp"] = pd.to_datetime(
    cuny_df["platform_collection_timestamp"], errors="coerce"
)

# cuny_df = cuny_df[cuny_df["Platform level heat index"] > 80]
cuny_df = cuny_df[cuny_df["platform_collection_timestamp"].dt.month == 7]  # only july

cuny_df["Number of services"] = cuny_df.apply(
    lambda row: len(row["Services"]),  # "EFMR" = 4
    axis=1,
)

# Merge in data from https://github.com/gregfeliu/NYC-Subway-Frequencies/tree/main

hourly_route_freq_df = pd.read_csv("hourly_route_trip_freq.csv")


# add column with calculated tph at a station
def calc_tph(row):
    total_tph = 0
    for service in row["Services"]:
        match_service = service
        # Fix shuttle outliers
        if match_service == "S":
            if row["gtfs_stop_id"] == "902" or row["gtfs_stop_id"] == "901":
                match_service = "42nd St. Shuttle"
        if match_service == "Z":
            return 6  # Z runs 6 tph during rush

        # match service to time and station
        this_service_tph = hourly_route_freq_df[
            hourly_route_freq_df["Service"] == match_service
        ]

        # get hour of this timestamp
        hour = row["platform_collection_timestamp"].hour
        weekday = row["platform_collection_timestamp"].weekday()

        str_weekday = None
        if weekday < 5:  # 0,1,2,3,4 are M-F
            str_weekday = "Weekday"
        elif weekday == 5:
            str_weekday = "Saturday"
        elif weekday == 6:
            str_weekday = "Sunday"

        # print(f"Pre TPH at {str_weekday}, hour {hour}, on {match_service} line: ")
        # print(row)
        tph_rows = this_service_tph[
            (this_service_tph["Hour"] == hour)
            & (this_service_tph["Day of Week"] == str_weekday)
        ]["TPH"]

        if len(tph_rows) == 0:
            tph = 0
        else:
            tph = tph_rows.item()
        # print(f"TPH at {str_weekday}, hour {hour}, on {match_service} line: ", tph)
        total_tph += tph
    return total_tph


cuny_df["Total TPH"] = cuny_df.apply(
    calc_tph,
    axis=1,
)
cuny_df["Platform level heat index (°C)"] = cuny_df.apply(
    lambda row: f_to_c(row["Platform level heat index"]),
    axis=1,
)

# -------------------------------------------------------------------------------------------------

# Box Plots for Number of services
axes = cuny_df.boxplot(
    column="Platform level heat index (°C)",
    by="Number of services",
    grid=False,
    vert=True,
    patch_artist=True,
    figsize=(14, 7),
    return_type="axes",
    medianprops=dict(color="red", linewidth=2),
    whiskerprops=dict(color="black", linewidth=3),
    capprops=dict(color="black", linewidth=3),
    showmeans=False,
)

axes[0].set_title("")
plt.ylim(20, 50)
plt.xlabel("Number of Services")
plt.ylabel("Platform Level Heat Index (°C)")
plt.suptitle("July 2024 and July 2025 Platform Level Heat Index", fontweight="bold")
plt.tight_layout()

plt.savefig(
    os.path.join(CHART_OUT_DIR, "box_plot_platform_heat_index_by_service.jpg"),
    bbox_inches="tight",
    dpi=400,
)

# -------------------------------------------------------------------------------------------------

# Stacked histogram plots
# fig, axs = plt.subplots(cuny_df["Number of services"].max())
fig, ax = plt.subplots()  # 1 subplot
fig.set_figwidth(14)
fig.set_figheight(7)
fig.suptitle("July 2024 and July 2025", fontweight="bold")

for number_of_services in range(1, cuny_df["Number of services"].max() + 1):
    cuny_df_only_this_number_of_services = cuny_df[
        cuny_df["Number of services"] == number_of_services
    ]

    sns.kdeplot(
        cuny_df_only_this_number_of_services["Platform level heat index (°C)"],
        ax=ax,
        label=f"{number_of_services} service{"" if number_of_services == 1 else "s"}",
        linewidth=4,
    )
    # Median heat index
    median_heat_index = np.median(
        cuny_df_only_this_number_of_services["Platform level heat index (°C)"]
    )
    plt.axvline(
        x=median_heat_index,
        color=ax.get_lines()[-1].get_color(),
        linestyle="--",
        linewidth=3,
    )

ax.set_xlabel("Platform Level Heat Index (°C)")
ax.set_ylabel("Density")
ax.legend()
ax.set_xlim(22, 48)
fig.tight_layout()
plt.savefig(
    os.path.join(CHART_OUT_DIR, "histogram_platform_heat_index_by_service.jpg"),
    bbox_inches="tight",
    dpi=400,
)

# -------------------------------------------------------------------------------------------------

# Scatter plot for TPH
plt.figure("TPH vs Platform Level Heat Index")
plt.scatter(cuny_df["Total TPH"], cuny_df["Platform level heat index (°C)"])
plt.xlabel("Total TPH")
plt.ylabel("Platform Level Heat Index (°C)")

# -------------------------------------------------------------------------------------------------


# Plot for citywide temps

citywide_df = pd.read_csv("citywide.csv")
citywide_df["Date"] = pd.to_datetime(citywide_df["Date"], errors="coerce")

plt.figure("Citywide Temperature vs Platform Heat Index", (14, 7))

years = [2024, 2025]
marker_colors = ["blue", "orange"]

for index in range(len(years)):
    year = years[index]
    marker_color = marker_colors[index]
    START_DATE = f"{year}-07-01"
    END_DATE = f"{year}-07-31"

    citywide_df_in_date_range = citywide_df[
        (citywide_df["Date"] >= START_DATE) & (citywide_df["Date"] <= END_DATE)
    ]
    plt.scatter(
        citywide_df_in_date_range["Date"].dt.strftime(r"%-d"),
        f_to_c(citywide_df_in_date_range["High Temp (°F)"]),
        label="Citywide High",
        s=64,
        color=marker_color,
        marker="o",
    )

    filtered_cuny_df = cuny_df[
        (cuny_df["platform_collection_timestamp"] >= START_DATE)
        & (cuny_df["platform_collection_timestamp"] <= END_DATE)
    ]

    # Take mean of each days
    xs = []
    ys = []

    for day in list(pd.date_range(start=START_DATE, end=END_DATE, freq=f"1D")):
        this_day_cuny_df = cuny_df[
            cuny_df["platform_collection_timestamp"].dt.date == day.date()
        ]
        this_day_cuny_df = this_day_cuny_df[
            (this_day_cuny_df["platform_collection_timestamp"].dt.hour >= 12)
            & (this_day_cuny_df["platform_collection_timestamp"].dt.hour < 14)
        ]
        mean_var = this_day_cuny_df["Platform level air temperature"].mean()
        xs.append(day.strftime(r"%-d"))
        ys.append(f_to_c(mean_var))

    plt.scatter(
        xs,
        ys,
        label=f"{year} Platform Mean",
        s=256,
        color=marker_color,
        marker="*",
    )
plt.grid(True)
# plt.xticks(rotation=45)
plt.xlabel("Day in July")
plt.ylabel("Air Temperature (°C)")
plt.title(
    "Platform Level Air Temperature [12:00 to 14:00] vs. Citywide High Air Temperature",
    fontweight="bold",
)
plt.legend()
plt.savefig(
    os.path.join(CHART_OUT_DIR, "time_series_citywide_to_mean_air_temp.jpg"),
    bbox_inches="tight",
    dpi=400,
)
# -------------------------------------------------------------------------------------------------
# helper program to take july means for arcgis program
mta_df = pd.read_csv("all_mta_station_data.csv")

output_mean_df_rows = []

## Print every station
for gtfs_id in cuny_df["gtfs_stop_id"].unique():

    gtfs_id_df = cuny_df[cuny_df["gtfs_stop_id"] == gtfs_id]

    gtfs_id_df = gtfs_id_df[
        gtfs_id_df["platform_collection_timestamp"].dt.month == 7
    ]  # already filtered to only july
    # print(gtfs_id_df["platform_collection_timestamp"])

    platform_mean_air_temp = gtfs_id_df["Platform level air temperature"].mean()
    platform_mean_heat_index = gtfs_id_df["Platform level heat index"].mean()
    platform_mean_humidity = gtfs_id_df["Platform level relative humidity"].mean()
    gtfs_lat = mta_df[mta_df["GTFS Stop ID"] == gtfs_id]["GTFS Latitude"].item()
    gtfs_long = mta_df[mta_df["GTFS Stop ID"] == gtfs_id]["GTFS Longitude"].item()

    output_mean_df_rows.append(
        {
            "gtfs_id": gtfs_id,
            "platform_mean_air_temp": platform_mean_air_temp,
            "platform_mean_heat_index": platform_mean_heat_index,
            "platform_mean_humidity": platform_mean_humidity,
            "gtfs_lat": gtfs_lat,
            "gtfs_long": gtfs_long,
        }
    )

# assemble new df
mean_df = pd.DataFrame(output_mean_df_rows)

print(mean_df)  # save to csv for arcgis
