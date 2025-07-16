import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import datetime

@st.cache_data
def load_data():
    citywide_df = pd.read_csv("citywide_weather.csv")
    cuny_df = pd.read_excel("CUNY_MTA.xlsx")
    return citywide_df, cuny_df

@st.cache_resource
@st.cache_resource
def train_model(citywide_df, cuny_df):
    cuny_df['Date'] = pd.to_datetime(cuny_df['Date'], errors='coerce')
    citywide_df['Date'] = pd.to_datetime(citywide_df['Date'])

    # Convert temperatures to numeric
    cuny_df['Street level air temperature'] = pd.to_numeric(cuny_df['Street level air temperature'], errors='coerce')
    cuny_df['Platform level air temperature'] = pd.to_numeric(cuny_df['Platform level air temperature'], errors='coerce')
    cuny_df = cuny_df.dropna(subset=['Street level air temperature'])

    # Merge main weather data given the Date
    merged_df = pd.merge(cuny_df, citywide_df, on='Date', how='inner')

    merged_df['Hour'] = pd.to_datetime(
        merged_df['Time for street level data collection'], format='%I:%M:%S %p', errors='coerce'
    ).dt.hour
    # Categorize hour into a time period 
    def time_category(hour):
        if pd.isna(hour): return np.nan
        if hour < 11:
            return 'morning'
        elif 11 <= hour < 16:
            return 'afternoon'
        elif 16 <= hour <= 23:
            return 'evening'
        return 'night'

    merged_df['Time_Category'] = merged_df['Hour'].apply(time_category)
    merged_df['Day_of_Week'] = merged_df['Date'].dt.dayofweek
    le_station = LabelEncoder()
    merged_df['Station_encoded'] = le_station.fit_transform(merged_df['Station name'])
 
    # --- Train street-level model --- 
    model_df = merged_df.copy()
    street_features = pd.get_dummies(model_df[['High Temp (°F)', 'Low Temp (°F)', 'Day_of_Week', 'Station_encoded', 'Time_Category']], drop_first=True)
    street_target = model_df['Street level air temperature']
    X_train_street, X_test_street, y_train_street, y_test_street = train_test_split(
        street_features, street_target, test_size=0.2, random_state=42
    )
    street_model = RandomForestRegressor(n_estimators=200, random_state=42)
    street_model.fit(X_train_street, y_train_street)
 
    # --- Train platform-level model --- 

    # Merge previous day's weather with current day's
    citywide_df['Prev_Date'] = citywide_df['Date'] + pd.Timedelta(days=1)
    prev_weather = citywide_df[['Date', 'High Temp (°F)', 'Low Temp (°F)']].rename(
        columns={
            'High Temp (°F)': 'Prev_High',
            'Low Temp (°F)': 'Prev_Low',
            'Date': 'Prev_Date'
        }
    )

    platform_df = pd.merge(merged_df, prev_weather, left_on='Date', right_on='Prev_Date', how='left')

    platform_df = platform_df.dropna(subset=['Platform level air temperature'])

    platform_features = platform_df[[
        'Street level air temperature',
        'Station_encoded',
        'Time_Category',
        'Day_of_Week',
        'Prev_High',
        'Prev_Low'
    ]]
    platform_features = pd.get_dummies(platform_features, columns=['Time_Category'], drop_first=True)
    platform_target = platform_df['Platform level air temperature']

    X_train_pf, X_test_pf, y_train_pf, y_test_pf = train_test_split(platform_features, platform_target, test_size=0.2, random_state=42)
    platform_model = RandomForestRegressor(n_estimators=200, random_state=42)
    platform_model.fit(X_train_pf, y_train_pf)

    return street_model, le_station, platform_model

# GUI STreamlit
st.title("NYC MTA Temperature Forecast")

try:
    citywide_df, cuny_df = load_data()
    model, le_station, platform_model = train_model(citywide_df, cuny_df) 
    st.subheader("Forecast Street and Subway Platform Temperature")
    station_name = st.selectbox("Select Station", sorted(cuny_df['Station name'].dropna().unique()))
    date = st.date_input("Select Date", value=datetime.date.today())
    time = st.time_input("Select Time", value=datetime.time(12, 0))

    high_temp = st.number_input("Citywide High Temp (°F)", value=85.0)
    low_temp = st.number_input("Citywide Low Temp (°F)", value=70.0)

    if st.button("Predict Platform and Street Temperature"):
        try:
            hour = pd.to_datetime(time.strftime('%H:%M:%S')).hour
            day_of_week = date.weekday()
            station_encoded = le_station.transform([station_name])[0]

            def time_category(hour):
                if hour < 11:
                    return 'morning'
                elif 11 <= hour < 16:
                    return 'afternoon'
                elif 16 <= hour <= 23:
                    return 'evening'
                return 'night'

            time_cat = time_category(hour)

            input_df = pd.DataFrame({
                'High Temp (°F)': [high_temp],
                'Low Temp (°F)': [low_temp],
                'Day_of_Week': [day_of_week],
                'Station_encoded': [station_encoded],
                'Time_Category': [time_cat]
            })

            input_features = pd.get_dummies(input_df, drop_first=True).reindex(columns=model.feature_names_in_, fill_value=0)
            prediction = model.predict(input_features)[0]
            st.success(f"Predicted Street-Level Temperature: {prediction:.2f} °F")

            # ---- Platform Temperature Prediction ----
            platform_input_df = pd.DataFrame({
                'Street level air temperature': [prediction],
                'Station_encoded': [station_encoded],
                'Day_of_Week': [day_of_week],
                'Time_Category': [time_cat]
            })

            platform_input_df = pd.get_dummies(platform_input_df, columns=['Time_Category'], drop_first=True)
            for col in ['Time_Category_afternoon', 'Time_Category_evening']:
                if col not in platform_input_df.columns:
                    platform_input_df[col] = 0
            platform_input_df = platform_input_df.reindex(columns=platform_model.feature_names_in_, fill_value=0)

            platform_prediction = platform_model.predict(platform_input_df)[0]
            st.info(f"Predicted Platform-Level Temperature: {platform_prediction:.2f} °F")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
except ValueError as ve:
    st.error(str(ve))
except Exception as e:
    st.error(f"Unexpected error: {e}")
