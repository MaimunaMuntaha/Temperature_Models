import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
from sklearn.model_selection import GridSearchCV
warnings.filterwarnings('ignore')

# Function to load data
@st.cache_data
def load_and_prepare_data():
    weather_df = pd.read_csv('citywide_weather.csv')
    weather_df['Date'] = pd.to_datetime(weather_df['Date'])

    station_df = pd.read_excel('CUNY_MTA.xlsx')
    station_df['Date'] = pd.to_datetime(station_df['Date'], errors='coerce')
    station_df = station_df.dropna(subset=['Date', 'Street level air temperature', 'Station name', 'Platform level air temperature'])
    
    # Clean temperature columns
    for col in ['Street level air temperature', 'Platform level air temperature']:
        station_df[col] = (
            station_df[col].astype(str)
            .str.extract(r'(\d+\.?\d*)')[0].astype(float)
        )
    
    merged_df = pd.merge(station_df, weather_df, on='Date', how='inner')
    return merged_df

# Feature engineering for both models
def create_features(df):
    # Time features
    df['Hour'] = pd.to_datetime(
        df['Time for street level data collection'],
        format='%H:%M:%S', errors='coerce'
    ).dt.hour
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    df['Day_of_Month'] = df['Date'].dt.day
    
    # Clean weather data
    df['High Temp (°F)'] = pd.to_numeric(df['High Temp (°F)'], errors='coerce')
    
    # Encode categorical features
    le_station = LabelEncoder()
    df['Station_Encoded'] = le_station.fit_transform(df['Station name'])
    
    # Crowd level encoding (if available)
    if 'How crowded is the platform?' in df.columns:
        crowd_mapping = {'Empty': 0, 'Light': 1, 'Medium': 2, 'Heavy': 3, 'Very Heavy': 4}
        df['Crowd_Encoded'] = df['How crowded is the platform?'].map(crowd_mapping).fillna(1)
    
    df = df.dropna(subset=['Street level air temperature', 'Platform level air temperature', 
                          'High Temp (°F)', 'Station_Encoded', 'Hour'])
    
    return df, le_station

# Street Temperature Predictor (same as before)
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
        st.write("### Street Temp Model Performance:")
        st.write(f"- R² Score: {r2_score(y_test, y_pred):.3f}")
        st.write(f"- MAE: {mean_absolute_error(y_test, y_pred):.2f}°F")

    def predict(self, high_temp, station_name, hour, day_of_week, day_of_month):
        try:
            station_encoded = self.station_encoder.transform([station_name])[0]
        except:
            station_encoded = 0
        features = np.array([[high_temp, station_encoded, hour, day_of_week, day_of_month]])
        return round(self.model.predict(features)[0], 1)

# New Platform Temperature Predictor
class PlatformTempPredictor:
    def __init__(self):
        # Parameters found through grid search
        self.model = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.station_encoder = None

    def fit(self, df, station_encoder):
        self.station_encoder = station_encoder
        
        # Features: street temp, station, hour, day, crowd level
        X = df[['Street level air temperature', 'Station_Encoded', 
                'Hour', 'Day_of_Week', 'Day_of_Month', 'Crowd_Encoded']]
        y = df['Platform level air temperature']

        # Grid search for best parameters (commented out after finding optimal)
        # param_grid = {
        #     'n_estimators': [100, 200, 300],
        #     'max_depth': [10, 15, 20],
        #     'min_samples_split': [2, 5, 10],
        #     'min_samples_leaf': [1, 2, 4]
        # }
        # grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, 
        #                          cv=3, n_jobs=-1, verbose=2)
        # grid_search.fit(X, y)
        # self.model = grid_search.best_estimator_
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        st.write("### Platform Temp Model Performance:")
        st.write(f"- R² Score: {r2_score(y_test, y_pred):.3f}")
        st.write(f"- MAE: {mean_absolute_error(y_test, y_pred):.2f}°F")

    def predict(self, street_temp, station_name, hour, day_of_week, day_of_month, crowd_level):
        try:
            station_encoded = self.station_encoder.transform([station_name])[0]
        except:
            station_encoded = 0
            
        crowd_mapping = {'Empty': 0, 'Light': 1, 'Medium': 2, 'Heavy': 3, 'Very Heavy': 4}
        crowd_encoded = crowd_mapping.get(crowd_level, 1)
        
        features = np.array([[street_temp, station_encoded, hour, 
                             day_of_week, day_of_month, crowd_encoded]])
        return round(self.model.predict(features)[0], 1)

# Load and prepare data
df = load_and_prepare_data()
df, station_encoder = create_features(df)

# Train both models
street_predictor = StreetTempPredictor()
street_predictor.fit(df, station_encoder)

platform_predictor = PlatformTempPredictor()
platform_predictor.fit(df, station_encoder)

# Streamlit UI
st.title("🌡️ MTA Temperature Prediction System 🌡️")

station = st.selectbox("Select a Station", sorted(df['Station name'].dropna().unique()))
high_temp = st.number_input("Citywide High Temp (°F)", min_value=50.0, max_value=120.0, step=0.1, value=75.0)
date_input = st.date_input("Date")
hour = st.slider("Hour of the Day", 0, 23, 12)
crowd_level = st.selectbox("Platform Crowd Level", ['Empty', 'Light', 'Medium', 'Heavy', 'Very Heavy'], index=1)

if st.button("Predict Temperatures"):
    # First predict street temperature
    street_temp = street_predictor.predict(
        high_temp=high_temp,
        station_name=station,
        hour=hour,
        day_of_week=date_input.weekday(),
        day_of_month=date_input.day
    )
    
    # Then use street temp to predict platform temp
    platform_temp = platform_predictor.predict(
        street_temp=street_temp,
        station_name=station,
        hour=hour,
        day_of_week=date_input.weekday(),
        day_of_month=date_input.day,
        crowd_level=crowd_level
    )
    
    st.success(f"""
    - Predicted Street Temperature: {street_temp}°F
    - Predicted Platform Temperature: {platform_temp}°F
    - Temperature Difference: {round(platform_temp - street_temp, 1)}°F
    """)
    
    # Additional insights
    st.write("### Insights:")
    if platform_temp - street_temp > 5:
        st.warning("⚠️ Significant heat island effect detected - platform is much hotter than street level")
    elif platform_temp - street_temp < -2:
        st.info("❄️ Platform is cooler than street level - possible ventilation or depth factors")
    else:
        st.success("✅ Normal temperature variation between street and platform")
