import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

@st.cache_data
def load_data():
    citywide_df = pd.read_csv("citywide.csv")
    cuny_df = pd.read_csv("CUNY_MTA.csv")
    return citywide_df, cuny_df

def adjusted_r2(r2, n, k):
    return 1 - (1 - r2) * ((n - 1) / (n - k - 1))

@st.cache_resource
#MODEL to predict platform AND street level is from lines 21-139
def train_model(citywide_df, cuny_df):
    cuny_df['Date'] = pd.to_datetime(cuny_df['Date'], errors='coerce')
    citywide_df['Date'] = pd.to_datetime(citywide_df['Date'], errors='coerce')
    # clean sheet data to be embedded for model params easily
    cuny_df['Street level air temperature'] = pd.to_numeric(cuny_df['Street level air temperature'], errors='coerce')
    cuny_df['Street level air temperature - Sunny conditions (complete only if there is no shady spot near the subway entrance)'] = pd.to_numeric(
        cuny_df['Street level air temperature - Sunny conditions (complete only if there is no shady spot near the subway entrance)'], errors='coerce'
    )
    cuny_df['Platform level air temperature'] = pd.to_numeric(cuny_df['Platform level air temperature'], errors='coerce')
    cuny_df['Street level relative humidity'] = pd.to_numeric(cuny_df['Street level relative humidity'], errors='coerce')

    cuny_df['Street level air temperature'] = cuny_df['Street level air temperature'].combine_first(
        cuny_df['Street level air temperature - Sunny conditions (complete only if there is no shady spot near the subway entrance)']
    )

    cuny_df = cuny_df.dropna(subset=['Street level air temperature', 'Street level relative humidity'])

    merged_df = pd.merge(cuny_df, citywide_df, on='Date', how='inner')
    merged_df['Hour'] = pd.to_datetime(
        merged_df['Time for street level data collection'], format='%I:%M:%S %p', errors='coerce'
    ).dt.hour 
    merged_df['Day_of_Week'] = merged_df['Date'].dt.dayofweek
    merged_df['Month'] = merged_df['Date'].dt.month
    le_station = LabelEncoder()
    merged_df['Station_encoded'] = le_station.fit_transform(merged_df['gtfs_stop_id'])

    station_to_gtfs = merged_df[['Station name', 'gtfs_stop_id']].drop_duplicates().set_index('Station name')['gtfs_stop_id'].to_dict()
    
    #train humidity model
    humidity_features = merged_df[['High Temp (°F)', 'Low Temp (°F)', 'Day_of_Week', 'Station_encoded', 'Hour', 'Month']].copy()
    humidity_target = merged_df['Street level relative humidity']
    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(humidity_features, humidity_target, test_size=0.2, random_state=42)
    humidity_model = RandomForestRegressor(n_estimators=100, random_state=42)
    humidity_model.fit(X_train_h, y_train_h)

    r2_h = humidity_model.score(X_test_h, y_test_h)
    adj_r2_h = adjusted_r2(r2_h, X_test_h.shape[0], X_test_h.shape[1])

    merged_df['Offset'] = merged_df['Street level air temperature'] - merged_df['High Temp (°F)']
    merged_df = merged_df[(merged_df['Offset'] > -5) & (merged_df['Offset'] < 20)]
    offset_features = merged_df[['Station_encoded', 'Hour', 'Day_of_Week', 'Street level relative humidity', 'High Temp (°F)', 'Low Temp (°F)', 'Month']].copy()
    offset_target = merged_df['Offset']

    X_train_offset, X_test_offset, y_train_offset, y_test_offset = train_test_split(offset_features, offset_target, test_size=0.2, random_state=42)
    offset_model = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42)
    offset_model.fit(X_train_offset, y_train_offset)

    r2_offset = offset_model.score(X_test_offset, y_test_offset)
    adj_r2_offset = adjusted_r2(r2_offset, X_test_offset.shape[0], X_test_offset.shape[1])
    y_pred_offset = offset_model.predict(X_test_offset)

    citywide_df['Prev_Date'] = citywide_df['Date'] + pd.Timedelta(days=1)
    prev_weather = citywide_df[['Date', 'High Temp (°F)', 'Low Temp (°F)']].rename(
        columns={
            'High Temp (°F)': 'Prev_High',
            'Low Temp (°F)': 'Prev_Low',
            'Date': 'Prev_Date'
        }
    )

    platform_daily = merged_df.groupby(['Date', 'gtfs_stop_id'])['Platform level air temperature'].mean().reset_index()
    platform_daily.rename(columns={'Platform level air temperature': 'Prev_Platform_Temp'}, inplace=True)
    platform_daily['Prev_Date'] = platform_daily['Date'] + pd.Timedelta(days=1)
    prev_platform = platform_daily[['gtfs_stop_id', 'Prev_Date', 'Prev_Platform_Temp']]

    platform_df = pd.merge(merged_df, prev_weather, left_on='Date', right_on='Prev_Date', how='left')
    platform_df = pd.merge(platform_df, prev_platform, left_on=['Date', 'gtfs_stop_id'], right_on=['Prev_Date', 'gtfs_stop_id'], how='left', suffixes=('', '_prev_platform'))
    platform_df = platform_df.dropna(subset=['Platform level air temperature'])

    platform_features = platform_df[[
        'Street level air temperature',
        'Station_encoded',
        'Hour',
        'Day_of_Week',
        'Prev_High',
        'Prev_Low',
        'Prev_Platform_Temp'
    ]].copy()

    platform_features['Prev_Platform_Temp'] = platform_features['Prev_Platform_Temp'].fillna(platform_features['Street level air temperature'])
    platform_target = platform_df['Platform level air temperature']

    X_train_pf, X_test_pf, y_train_pf, y_test_pf = train_test_split(platform_features, platform_target, test_size=0.2, random_state=42)
    platform_model = RandomForestRegressor(n_estimators=200, random_state=42)
    platform_model.fit(X_train_pf, y_train_pf)

    r2_p = platform_model.score(X_test_pf, y_test_pf)
    adj_r2_p = adjusted_r2(r2_p, X_test_pf.shape[0], X_test_pf.shape[1])

    platform_df['Platform_Offset'] = platform_df['Platform level air temperature'] - platform_df['Street level air temperature']

    platform_offset_features = platform_df[[
        'Station_encoded', 
        'Hour', 
        'Day_of_Week', 
        'Street level air temperature', 
        'Prev_High', 
        'Prev_Low',
        'Prev_Platform_Temp'
    ]].copy()

    platform_offset_features['Prev_Platform_Temp'] = platform_offset_features['Prev_Platform_Temp'].fillna(platform_offset_features['Street level air temperature'])

    platform_offset_target = platform_df['Platform_Offset']
    X_train_pf_off, X_test_pf_off, y_train_pf_off, y_test_pf_off = train_test_split(
        platform_offset_features, platform_offset_target, test_size=0.2, random_state=42)
    platform_offset_model = RandomForestRegressor(n_estimators=200, random_state=42)
    platform_offset_model.fit(X_train_pf_off, y_train_pf_off)
    r2_pf_off = platform_offset_model.score(X_test_pf_off, y_test_pf_off)

    adj_r2_pf_off = adjusted_r2(r2_pf_off, X_test_pf_off.shape[0], X_test_pf_off.shape[1])
    y_pred_pf_off = platform_offset_model.predict(X_test_pf_off)

    return (offset_model, le_station, platform_model, 
            humidity_model, humidity_features.columns,
            adj_r2_h, adj_r2_offset, adj_r2_p,
            y_test_offset, y_pred_offset, X_test_offset,
            platform_offset_model, adj_r2_pf_off, y_test_pf_off, y_pred_pf_off, X_test_pf_off,
            platform_daily, station_to_gtfs)
    
# STREAMLIT -- not necessary for website -- only model above
st.title("NYC MTA Temperature Forecast")

try:
    citywide_df, cuny_df = load_data()
    (offset_model, le_station, platform_model, 
    humidity_model, humidity_feature_cols,
    adj_r2_h, adj_r2_offset, adj_r2_p,
    y_test_offset, y_pred_offset, X_test_offset,
    platform_offset_model, adj_r2_pf_off, y_test_pf_off, y_pred_pf_off, X_test_pf_off,
    platform_daily, station_to_gtfs) = train_model(citywide_df, cuny_df)

    station_name = st.selectbox("Select Station", sorted(cuny_df['Station name'].dropna().unique()))
    date = st.date_input("Select Date", value=datetime.date.today())
    time = st.time_input("Select Time", value=datetime.time(12, 0))
    high_temp = st.number_input("Citywide High Temp (°F)", value=85.0)
    low_temp = st.number_input("Citywide Low Temp (°F)", value=70.0)
    prev_high = st.number_input("Previous Day High Temp (°F)", value=85.0)
    prev_low = st.number_input("Previous Day Low Temp (°F)", value=70.0)

    if st.button("Predict Platform and Street Temperature"):
        try:
            hour = pd.to_datetime(time.strftime('%H:%M:%S')).hour
            day_of_week = date.weekday()
            month = date.month
            gtfs_id = station_to_gtfs.get(station_name)
            if gtfs_id is None:
                st.error("Selected station does not have a GTFS Stop ID mapping.")
                st.stop()
            station_encoded = le_station.transform([gtfs_id])[0]

            humidity_input = pd.DataFrame({
                'High Temp (°F)': [high_temp],
                'Low Temp (°F)': [low_temp],
                'Day_of_Week': [day_of_week],
                'Station_encoded': [station_encoded],
                'Hour': [hour],
                'Month': [month]
            })
            humidity_input = humidity_input.reindex(columns=humidity_feature_cols, fill_value=0)
            predicted_humidity = humidity_model.predict(humidity_input)[0]

            offset_input_df = pd.DataFrame({
                'Station_encoded': [station_encoded],
                'Hour': [hour],
                'Day_of_Week': [day_of_week],
                'Street level relative humidity': [predicted_humidity],
                'High Temp (°F)': [high_temp],
                'Low Temp (°F)': [low_temp],
                'Month': [month]
            })
            offset_input_df = offset_input_df.reindex(columns=offset_model.feature_names_in_, fill_value=0)
            offset_pred = offset_model.predict(offset_input_df)[0]
            offset_pred_clipped = np.clip(offset_pred, -3, 15)
            street_level_temp_pred = high_temp + offset_pred_clipped
            st.success(f"Predicted Street-Level Temperature: {street_level_temp_pred:.2f} °F\n(Predicted Offset: {offset_pred_clipped:+.2f} °F, Predicted Humidity: {predicted_humidity:.1f}%)")

            prev_platform_temp = high_temp + 2.0
            platform_offset_input_df = pd.DataFrame({
                'Station_encoded': [station_encoded],
                'Hour': [hour],
                'Day_of_Week': [day_of_week],
                'Street level air temperature': [street_level_temp_pred],
                'Prev_High': [prev_high],
                'Prev_Low': [prev_low],
                'Prev_Platform_Temp': [prev_platform_temp]
            })
            platform_offset_input_df = platform_offset_input_df.reindex(columns=platform_offset_model.feature_names_in_, fill_value=0)
            platform_offset_pred = platform_offset_model.predict(platform_offset_input_df)[0]
            platform_temp_pred_offset = street_level_temp_pred + platform_offset_pred
            st.info(f"Predicted Platform-Level Temperature (Offset Model): {platform_temp_pred_offset:.2f} °F (Predicted Offset: {platform_offset_pred:+.2f} °F)")

            st.sidebar.header("Model Performance (Adjusted R²)")
            st.sidebar.write(f"Humidity model: {adj_r2_h:.3f}")
            st.sidebar.write(f"Street-level Temp model: {adj_r2_offset:.3f}")
            st.sidebar.write(f"Platform-level Offset model: {adj_r2_pf_off:.3f}")
            st.sidebar.subheader("Model Error Metrics")
            st.sidebar.write(f"Street Temp RMSE: {rmse_offset:.2f} °F")
            st.sidebar.write(f"Street Temp MAE: {mae_offset:.2f} °F")
            st.sidebar.write(f"Platform Temp RMSE: {rmse_pf:.2f} °F")
            st.sidebar.write(f"Platform Temp MAE: {mae_pf:.2f} °F")

            st.subheader("True vs Predicted (Street-Level Temp) Plot")
            high_temp_test = np.full_like(y_test_offset, high_temp)
            true_temp = y_test_offset + high_temp_test
            pred_temp = y_pred_offset + high_temp_test

            # Street-level actual and predicted full temps
            true_temp = y_test_offset + np.full_like(y_test_offset, high_temp)
            pred_temp = y_pred_offset + np.full_like(y_test_offset, high_temp)
            
            rmse_offset = mean_squared_error(true_temp, pred_temp, squared=False)
            mae_offset = mean_absolute_error(true_temp, pred_temp)

            true_pf_temp = y_test_pf_off + X_test_pf_off['Street level air temperature'].values
            pred_pf_temp = y_pred_pf_off + X_test_pf_off['Street level air temperature'].values
            
            rmse_pf = mean_squared_error(true_pf_temp, pred_pf_temp, squared=False)
            mae_pf = mean_absolute_error(true_pf_temp, pred_pf_temp)

            
            fig, ax = plt.subplots()
            ax.scatter(true_temp, pred_temp, alpha=0.5)
            ax.plot([true_temp.min(), true_temp.max()], [true_temp.min(), true_temp.max()], 'r--')
            ax.set_xlabel("True Temp (°F)")
            ax.set_ylabel("Predicted Temp (°F)")
            ax.set_title("Street-Level Temperature Prediction")
            st.pyplot(fig) 
            st.subheader("True vs Predicted (Platform Offset) Plot")
            street_temp_test = X_test_pf_off['Street level air temperature'].values
            true_pf_temp = y_test_pf_off + street_temp_test
            pred_pf_temp = y_pred_pf_off + street_temp_test
            fig2, ax2 = plt.subplots()
            ax2.scatter(true_pf_temp, pred_pf_temp, alpha=0.5)
            ax2.plot([true_pf_temp.min(), true_pf_temp.max()], [true_pf_temp.min(), true_pf_temp.max()], 'r--')
            ax2.set_xlabel("True Platform Temp (°F)")
            ax2.set_ylabel("Predicted Platform Temp (°F)")
            ax2.set_title("Platform Temperature Offset Model")
            st.pyplot(fig2)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
except ValueError as ve:
    st.error(str(ve))
except Exception as e:
    st.error(f"Unexpected error: {e}")
