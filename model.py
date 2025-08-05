import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score

@st.cache_data
def load_data():
    citywide_df = pd.read_csv("citywide.csv")
    cuny_df = pd.read_csv("CUNY_MTA.csv")
    return citywide_df, cuny_df


def adjusted_r2(r2, n, k):
    return 1 - (1 - r2) * ((n - 1) / (n - k - 1))

def compute_metrics(y_true, y_pred, n, k):
    r2 = r2_score(y_true, y_pred)
    adj_r2_val = adjusted_r2(r2, n, k)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, adj_r2_val, mae, rmse


@st.cache_resource
def train_model(citywide_df, cuny_df):
    cuny_df["Date"] = pd.to_datetime(cuny_df["Date"], errors="coerce")
    citywide_df["Date"] = pd.to_datetime(citywide_df["Date"], errors="coerce")

    cuny_df["Platform level relative humidity"] = pd.to_numeric(cuny_df["Platform level relative humidity"], errors="coerce")
    cuny_df["Street level air temperature"] = pd.to_numeric(cuny_df["Street level air temperature"], errors="coerce")
    cuny_df["Street level air temperature - Sunny conditions (complete only if there is no shady spot near the subway entrance)"] = pd.to_numeric(
        cuny_df["Street level air temperature - Sunny conditions (complete only if there is no shady spot near the subway entrance)"], errors="coerce")
    cuny_df["Platform level air temperature"] = pd.to_numeric(cuny_df["Platform level air temperature"], errors="coerce")
    cuny_df["Street level relative humidity"] = pd.to_numeric(cuny_df["Street level relative humidity"], errors="coerce")

    cuny_df["Street level air temperature"] = cuny_df["Street level air temperature"].combine_first(
        cuny_df["Street level air temperature - Sunny conditions (complete only if there is no shady spot near the subway entrance)"])

    cuny_df = cuny_df.dropna(subset=[
        "Street level air temperature",
        "Street level relative humidity",
        "Platform level relative humidity",
        "Platform level air temperature"
    ])

    merged_df = pd.merge(cuny_df, citywide_df, on="Date", how="inner")
    merged_df["Hour"] = pd.to_datetime(merged_df["Time for street level data collection"], format="%I:%M:%S %p", errors="coerce").dt.hour
    merged_df["Day_of_Week"] = merged_df["Date"].dt.dayofweek
    merged_df["Month"] = merged_df["Date"].dt.month
    le_station = LabelEncoder()
    merged_df["Station_encoded"] = le_station.fit_transform(merged_df["gtfs_stop_id"])

    station_to_gtfs = (
        merged_df[["Station name", "gtfs_stop_id"]]
        .drop_duplicates()
        .set_index("Station name")["gtfs_stop_id"]
        .to_dict()
    )

    # Street-Level Humidity Model
    humidity_features = merged_df[["High Temp (°F)", "Low Temp (°F)", "Day_of_Week", "Station_encoded", "Hour", "Month"]].copy()
    humidity_target = merged_df["Street level relative humidity"]
    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(humidity_features, humidity_target, test_size=0.2, random_state=42)
    humidity_model = RandomForestRegressor(n_estimators=100, random_state=42)
    humidity_model.fit(X_train_h, y_train_h)
    y_pred_h = humidity_model.predict(X_test_h)
    r2_h, adj_r2_h, mae_h, rmse_h = compute_metrics(y_test_h, y_pred_h, X_test_h.shape[0], X_test_h.shape[1])


    # Street-Level Temperature Offset Model
    merged_df["Offset"] = merged_df["Street level air temperature"] - merged_df["High Temp (°F)"]
    merged_df = merged_df[(merged_df["Offset"] > -5) & (merged_df["Offset"] < 20)]
    offset_features = merged_df[
        ["Station_encoded", "Hour", "Day_of_Week", "Street level relative humidity", "High Temp (°F)", "Low Temp (°F)", "Month"]
    ].copy()
    offset_target = merged_df["Offset"]
    X_train_offset, X_test_offset, y_train_offset, y_test_offset = train_test_split(offset_features, offset_target, test_size=0.2, random_state=42)
    offset_model = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42)
    offset_model.fit(X_train_offset, y_train_offset)
    y_pred_offset = offset_model.predict(X_test_offset)
    r2_offset, adj_r2_offset, mae_offset, rmse_offset = compute_metrics(y_test_offset, y_pred_offset, X_test_offset.shape[0], X_test_offset.shape[1])

    # Merging dfs
    citywide_df["Prev_Date"] = citywide_df["Date"] + pd.Timedelta(days=1)
    prev_weather = citywide_df[["Date", "High Temp (°F)", "Low Temp (°F)"]].rename(
        columns={"High Temp (°F)": "Prev_High", "Low Temp (°F)": "Prev_Low", "Date": "Prev_Date"})

    platform_daily = merged_df.groupby(["Date", "gtfs_stop_id"])[
        ["Platform level air temperature", "Platform level relative humidity"]
    ].mean().reset_index()
    platform_daily.rename(columns={
        "Platform level air temperature": "Prev_Platform_Temp",
        "Platform level relative humidity": "Prev_Platform_Humidity"
    }, inplace=True)
    platform_daily["Prev_Date"] = platform_daily["Date"] + pd.Timedelta(days=1)
    prev_platform = platform_daily[["gtfs_stop_id", "Prev_Date", "Prev_Platform_Temp", "Prev_Platform_Humidity"]]

    platform_df = pd.merge(merged_df, prev_weather, left_on="Date", right_on="Prev_Date", how="left")
    platform_df = pd.merge(platform_df, prev_platform, left_on=["Date", "gtfs_stop_id"],
                           right_on=["Prev_Date", "gtfs_stop_id"], how="left", suffixes=("", "_prev_platform"))
    platform_df = platform_df.dropna(subset=["Platform level air temperature", "Platform level relative humidity"])

    # Platform Temperature Offset Model
    platform_df["Platform_Offset"] = platform_df["Platform level air temperature"] - platform_df["Street level air temperature"]
    platform_offset_features = platform_df[
        ["Station_encoded", "Hour", "Day_of_Week", "Street level air temperature", "Prev_High", "Prev_Low"]
    ].copy()
    platform_offset_target = platform_df["Platform_Offset"]
    X_train_pf_off, X_test_pf_off, y_train_pf_off, y_test_pf_off = train_test_split(platform_offset_features, platform_offset_target, test_size=0.2, random_state=42)
    platform_offset_model = RandomForestRegressor(n_estimators=200, random_state=42)
    platform_offset_model.fit(X_train_pf_off, y_train_pf_off)
    y_pred_pf_off = platform_offset_model.predict(X_test_pf_off) 
    r2_pf_off, adj_r2_pf_off, mae_pf_off, rmse_pf_off = compute_metrics(y_test_pf_off, y_pred_pf_off, X_test_pf_off.shape[0], X_test_pf_off.shape[1])

    # Platform Humidity Model
    platform_humidity_features = platform_df[
        ["Platform level air temperature", "Station_encoded", "Hour", "Day_of_Week", "Prev_High", "Prev_Low" ] #"Prev_Platform_Temp"
    ].copy()
    platform_humidity_target = platform_df["Platform level relative humidity"] 
    X_train_ph, X_test_ph, y_train_ph, y_test_ph = train_test_split(platform_humidity_features, platform_humidity_target, test_size=0.2, random_state=42)
    platform_humidity_model = RandomForestRegressor(n_estimators=200, random_state=42)
    platform_humidity_model.fit(X_train_ph, y_train_ph)
    y_pred_ph = platform_humidity_model.predict(X_test_ph)
    r2_ph, adj_r2_ph, mae_ph, rmse_ph = compute_metrics(y_test_ph, y_pred_ph, X_test_ph.shape[0], X_test_ph.shape[1])

    return (
        humidity_model,
        offset_model,
        platform_offset_model,
        platform_humidity_model,
        le_station,
        station_to_gtfs,
        humidity_features.columns,
        (r2_h, adj_r2_h, mae_h, rmse_h),
        (r2_offset, adj_r2_offset, mae_offset, rmse_offset),
        (r2_pf_off, adj_r2_pf_off, mae_pf_off, rmse_pf_off),
        (r2_ph, adj_r2_ph, mae_ph, rmse_ph),
        platform_daily,
    )


st.title("NYC MTA Temperature & Humidity Forecast")

try:
    citywide_df, cuny_df = load_data()
    (
        humidity_model,
        offset_model,
        platform_offset_model,
        platform_humidity_model,
        le_station,
        station_to_gtfs,
        humidity_feature_cols,
        (r2_h, adj_r2_h, mae_h, rmse_h),
        (r2_offset, adj_r2_offset, mae_offset, rmse_offset),
        (r2_pf_off, adj_r2_pf_off, mae_pf_off, rmse_pf_off),
        (r2_ph, adj_r2_ph, mae_ph, rmse_ph),
        platform_daily
    ) = train_model(citywide_df, cuny_df)


    station_name = st.selectbox("Select Station", sorted(cuny_df["Station name"].dropna().unique()))
    date = st.date_input("Select Date", value=datetime.date.today())
    time = st.time_input("Select Time", value=datetime.time(12, 0))
    high_temp = st.number_input("Citywide High Temp (°F)", value=85.0)
    low_temp = st.number_input("Citywide Low Temp (°F)", value=70.0)
    prev_high = st.number_input("Previous Day High Temp (°F)", value=85.0)
    prev_low = st.number_input("Previous Day Low Temp (°F)", value=70.0) 

    if st.button("Predict All"):
        hour = pd.to_datetime(time.strftime("%H:%M:%S")).hour
        day_of_week = date.weekday()
        month = date.month
        gtfs_id = station_to_gtfs.get(station_name)
        station_encoded = le_station.transform([gtfs_id])[0]

        # 1. Predict Street Relative Humidity
        humidity_input = pd.DataFrame({
            "High Temp (°F)": [high_temp],
            "Low Temp (°F)": [low_temp],
            "Day_of_Week": [day_of_week],
            "Station_encoded": [station_encoded],
            "Hour": [hour],
            "Month": [month],
        }).reindex(columns=humidity_feature_cols, fill_value=0)
        predicted_humidity = humidity_model.predict(humidity_input)[0]

        # 2. Predict Street Temp
        offset_input_df = pd.DataFrame({
            "Station_encoded": [station_encoded],
            "Hour": [hour],
            "Day_of_Week": [day_of_week],
            "Street level relative humidity": [predicted_humidity],
            "High Temp (°F)": [high_temp],
            "Low Temp (°F)": [low_temp],
            "Month": [month],
        }).reindex(columns=offset_model.feature_names_in_, fill_value=0)
        offset_pred = offset_model.predict(offset_input_df)[0]
        street_temp_pred = high_temp + np.clip(offset_pred, -3, 15)

        # 3. Predict Platform Temp
        platform_offset_input_df = pd.DataFrame({
            "Station_encoded": [station_encoded],
            "Hour": [hour],
            "Day_of_Week": [day_of_week],
            "Street level air temperature": [street_temp_pred],
            "Prev_High": [prev_high],
            "Prev_Low": [prev_low],
        }).reindex(columns=platform_offset_model.feature_names_in_, fill_value=0)
        platform_offset_pred = platform_offset_model.predict(platform_offset_input_df)[0]
        platform_temp_pred = street_temp_pred + platform_offset_pred

        # 4. Predict Platform Relative Humidity  
        platform_history = platform_daily[
            (platform_daily["gtfs_stop_id"] == gtfs_id)
            & (platform_daily["Prev_Date"] == pd.to_datetime(date))
        ]
        # if not platform_history.empty:
        #     prev_platform_temp = platform_history["Prev_Platform_Temp"].values[0]
        #     prev_platform_humidity = platform_history["Prev_Platform_Humidity"].values[0]
        # else:
        #     prev_platform_temp = platform_temp_pred  # fallback
        #     prev_platform_humidity = 60.0  # fallback

        platform_humidity_input_df = pd.DataFrame({
            "Platform level air temperature": [platform_temp_pred],
            "Station_encoded": [station_encoded],
            "Hour": [hour],
            "Day_of_Week": [day_of_week],
            "Prev_High": [prev_high],
            "Prev_Low": [prev_low],
            # "Prev_Platform_Temp": [prev_platform_temp],
            # "Prev_Platform_Humidity": [prev_platform_humidity],
        }).reindex(columns=platform_humidity_model.feature_names_in_, fill_value=0)
        platform_rh_pred = platform_humidity_model.predict(platform_humidity_input_df)[0]

        # Prediction
        st.success(f"Predicted Street-Level Relative Humidity: {predicted_humidity:.1f}%")
        st.success(f"Predicted Street-Level Temperature: {street_temp_pred:.2f} °F")
        st.info(f"Predicted Platform-Level Temperature: {platform_temp_pred:.2f} °F")
        st.warning(f"Predicted Platform-Level Relative Humidity: {platform_rh_pred:.1f}%")

        # Model's metrics -- r^2, adj r^2, mae, rmse
        st.sidebar.header("Model Performance")

        def display_metrics(name, r2, adj_r2, mae, rmse):
            st.sidebar.subheader(name)
            st.sidebar.write(f"R²: {r2:.3f}")
            st.sidebar.write(f"Adjusted R²: {adj_r2:.3f}")
            st.sidebar.write(f"MAE: {mae:.2f}")
            st.sidebar.write(f"RMSE: {rmse:.2f}")

        display_metrics("Street Relative Humidity", r2_h, adj_r2_h, mae_h, rmse_h)
        display_metrics("Street Temp", r2_offset, adj_r2_offset, mae_offset, rmse_offset)
        display_metrics("Platform Temp", r2_pf_off, adj_r2_pf_off, mae_pf_off, rmse_pf_off)
        display_metrics("Platform Relative Humidity", r2_ph, adj_r2_ph, mae_ph, rmse_ph)

except Exception as e:
    st.error(f"Unexpected error: {e}")
