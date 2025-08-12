import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pprint import pprint
import seaborn as sns


plt.rcParams.update(
    {
        "font.size": 18,
        "font.family": "sans-serif",
        "font.sans-serif": "Arial",
        "font.weight": "bold",
        "axes.labelweight": "bold",
    }
)

cuny_df = pd.read_csv("CUNY_MTA.csv")
cuny_df["Timestamp"] = pd.to_datetime(cuny_df["Timestamp"], errors="coerce")

# cuny_df = cuny_df[cuny_df["Platform level heat index"] > 80]
cuny_df = cuny_df[cuny_df["Timestamp"].dt.month == 7]  # only july

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
        hour = row["Timestamp"].hour
        weekday = row["Timestamp"].weekday()

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

print(cuny_df)

# Plots for Number of services
cuny_df.boxplot(
    column="Platform level heat index",
    by="Number of services",
    grid=False,
    vert=True,
    patch_artist=True,
    return_type="dict",
)
plt.xlabel("Number of Services")
plt.ylabel("Platform Level Heat Index (°F)")
plt.suptitle("July 2024 and July 2025", fontweight="bold")

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
        cuny_df_only_this_number_of_services["Platform level heat index"],
        ax=ax,
        label=f"{number_of_services} service{"" if number_of_services == 1 else "s"}",
        linewidth=2,
    )
    # Median heat index
    median_heat_index = np.median(
        cuny_df_only_this_number_of_services["Platform level heat index"]
    )
    plt.axvline(
        x=median_heat_index,
        color=ax.get_lines()[-1].get_color(),
        linestyle="--",
        linewidth=1.5,
    )

    ax.set_xlabel("Platform Level Heat Index")
    ax.set_ylabel("Density")
ax.legend()
ax.set_xlim(75, 125)
fig.tight_layout()
plt.savefig(f"test.jpg", bbox_inches="tight", dpi=400)

# Scatter plot for TPH
plt.figure("TPH vs Platform Level Heat Index")
plt.scatter(cuny_df["Total TPH"], cuny_df["Platform level heat index"])
plt.xlabel("Total TPH")
plt.ylabel("Platform Level Heat Index")

# Final show
plt.show()
