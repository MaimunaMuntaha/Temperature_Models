import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load and merge data
@st.cache_data
def load_and_prepare_data():
    weather_df = pd.read_csv('citywide_weather.csv')
    weather_df['Date'] = pd.to_datetime(weather_df['Date'])

    station_df = pd.read_excel('CUNY_MTA.xlsx')
    station_df['Date'] = pd.to_datetime(station_df['Date'], errors='coerce')
    station_df = station_df.dropna(subset=[
        'Date', 'Street level air temperature', 'Platform level air temperature', 'Station name'
    ])

    for col in ['Street level air temperature', 'Platform level air temperature']:
        station_df[col] = (
            station_df[col].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)
        )

    merged_df = pd.merge(station_df, weather_df, on='Date', how='inner')
    return merged_df

# Feature engineering
def create_features(df):
    df['Hour'] = pd.to_datetime(df['Time for street level data collection'], format='%H:%M:%S', errors='coerce').dt.hour
    df['Platform_Hour'] = pd.to_datetime(df['Time for platform level data collection'], format='%I:%M:%S %p', errors='coerce').dt.hour
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    df['Day_of_Month'] = df['Date'].dt.day
    df['High Temp (°F)'] = pd.to_numeric(df['High Temp (°F)'], errors='coerce')
    df['Platform RH'] = pd.to_numeric(df['Platform level relative humidity'], errors='coerce')
    df['Sunny_Street'] = df['Street level air temperature - Sunny conditions (complete only if there is no shady spot near the subway entrance)'].notna().astype(int)
    df['Crowdedness'] = df['How crowded is the platform?'].map({'Empty': 0, 'Light': 1, 'Medium': 2, 'Heavy': 3})

    le_station = LabelEncoder()
    df['Station_Encoded'] = le_station.fit_transform(df['Station name'])

    # Interaction features
    df['Temp_Hour_Interaction'] = df['Street level air temperature'] * df['Platform_Hour']
    df['Crowdedness_Interaction'] = df['Crowdedness'] * df['Street level air temperature']

    df = df.dropna(subset=[
        'Street level air temperature', 'Platform level air temperature', 'High Temp (°F)',
        'Station_Encoded', 'Platform_Hour', 'Platform RH', 'Crowdedness',
        'Temp_Hour_Interaction', 'Crowdedness_Interaction'
    ])

    return df, le_station

# Platform model
class PlatformTempPredictor:
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.station_encoder = None

    def fit(self, df, station_encoder):
        self.station_encoder = station_encoder
        X = df[[
            'Street level air temperature', 'Station_Encoded', 'Platform_Hour',
            'Day_of_Week', 'Day_of_Month', 'Platform RH', 'Crowdedness', 'Sunny_Street',
            'Temp_Hour_Interaction', 'Crowdedness_Interaction'
        ]]
        y = df['Platform level air temperature']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        st.write("### Platform Temp Model Performance (GradientBoosting):")
        st.write(f"- R² Score: {r2_score(y_test, y_pred):.3f}")
        st.write(f"- MAE: {mean_absolute_error(y_test, y_pred):.2f}°F")

    def predict(self, features_dict):
        try:
            station_encoded = self.station_encoder.transform([features_dict['station']])[0]
        except:
            station_encoded = 0

        features = np.array([[
            features_dict['street_temp'], station_encoded, features_dict['platform_hour'],
            features_dict['day_of_week'], features_dict['day_of_month'], features_dict['platform_rh'],
            features_dict['crowdedness'], features_dict['sunny_street'],
            features_dict['street_temp'] * features_dict['platform_hour'],
            features_dict['crowdedness'] * features_dict['street_temp']
        ]])
        return round(self.model.predict(features)[0], 1)

# Streamlit app
st.title("🔥 Enhanced MTA Platform Temperature Predictor")

df = load_and_prepare_data()
df, station_encoder = create_features(df)

predictor = PlatformTempPredictor()
predictor.fit(df, station_encoder)

station = st.selectbox("Station", sorted(df['Station name'].dropna().unique()))
street_temp = st.number_input("Observed Street Temperature (°F)", 50.0, 120.0, step=0.1)
platform_hour = st.slider("Platform Observation Hour", 0, 23, 12)
day = st.date_input("Date")
day_of_week = day.weekday()
day_of_month = day.day
platform_rh = st.slider("Platform Relative Humidity (%)", 10.0, 100.0, 50.0)
crowdedness = st.selectbox("Crowdedness Level", ['Empty', 'Light', 'Medium', 'Heavy'])
crowdedness_encoded = {'Empty': 0, 'Light': 1, 'Medium': 2, 'Heavy': 3}[crowdedness]
sunny_street = st.checkbox("Sunny at Street Level?", value=True)

if st.button("Predict Platform Temperature"):
    result = predictor.predict({
        'station': station,
        'street_temp': street_temp,
        'platform_hour': platform_hour,
        'day_of_week': day_of_week,
        'day_of_month': day_of_month,
        'platform_rh': platform_rh,
        'crowdedness': crowdedness_encoded,
        'sunny_street': int(sunny_street)
    })

    st.success(f"Predicted Platform Temperature: {result}°F")
    st.write(f"Difference from street: {round(result - street_temp, 1)}°F")
