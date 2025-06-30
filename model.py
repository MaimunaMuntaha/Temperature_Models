import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

#Function to load data
@st.cache_data
def load_and_prepare_data():
    weather_df = pd.read_csv('citywide_weather.csv')
    weather_df['Date'] = pd.to_datetime(weather_df['Date'])

    station_df = pd.read_excel('CUNY_MTA.xlsx')
    station_df['Date'] = pd.to_datetime(station_df['Date'], errors='coerce')
    station_df = station_df.dropna(subset=['Date', 'Street level air temperature', 'Station name'])

    merged_df = pd.merge(station_df, weather_df, on='Date', how='inner')
    return merged_df
    
#Create the embeddings for training the model e.g., Street Level Ta, Citywide High temp, Staton name, and Hour of day 
def create_features(df):
    df['Hour'] = pd.to_datetime(
        df['Time for street level data collection'],
        format='%H:%M:%S', errors='coerce'
    ).dt.hour
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    df['Day_of_Month'] = df['Date'].dt.day

    df['Street level air temperature'] = (
        df['Street level air temperature'].astype(str)
        .str.extract(r'(\d+\.?\d*)')[0].astype(float)
    )
    df['High Temp (°F)'] = pd.to_numeric(df['High Temp (°F)'], errors='coerce')

    df = df.dropna(subset=['Street level air temperature', 'High Temp (°F)', 'Station name', 'Hour'])

    le_station = LabelEncoder()
    df['Station_Encoded'] = le_station.fit_transform(df['Station name'])

    return df, le_station
    
# What the inputs will be to the GUI/Streamlit UI and how the model will learn
class StreetTempPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
        self.station_encoder = None

    def fit(self, df, station_encoder):
        self.station_encoder = station_encoder
        X = df[['High Temp (°F)', 'Station_Encoded', 'Hour', 'Day_of_Week', 'Day_of_Month']]
        y = df['Street level air temperature']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        st.write("### Model Performance:")
        st.write(f"- R² Score: {r2_score(y_test, y_pred):.3f}")
        st.write(f"- MAE: {mean_absolute_error(y_test, y_pred):.2f}°F")

    def predict(self, high_temp, station_name, hour, day_of_week, day_of_month):
        try:
            station_encoded = self.station_encoder.transform([station_name])[0]
        except:
            station_encoded = 0
        features = np.array([[high_temp, station_encoded, hour, day_of_week, day_of_month]])
        return round(self.model.predict(features)[0], 1)

# Load and train the CSV files
df = load_and_prepare_data()
df, station_encoder = create_features(df)
predictor = StreetTempPredictor()
predictor.fit(df, station_encoder)

# ACTUAL UI / Streamlit
st.title("🌡️ Street-Level Temperature Predictor 🌡️")

station = st.selectbox("Select a Station", sorted(df['Station name'].dropna().unique()))
high_temp = st.number_input("Citywide High Temp (°F)", min_value=50.0, max_value=120.0, step=0.1)
date_input = st.date_input("Date")
hour = st.slider("Hour of the Day", 0, 23, 12)

if st.button("Predict"):
    prediction = predictor.predict(
        high_temp=high_temp,
        station_name=station,
        hour=hour,
        day_of_week=date_input.weekday(),
        day_of_month=date_input.day
    )
    st.success(f"Predicted Street Temp: {prediction}°F")

