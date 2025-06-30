import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Function to load data
@st.cache_data
def load_and_prepare_data():
    weather_df = pd.read_csv('citywide_weather.csv')
    weather_df['Date'] = pd.to_datetime(weather_df['Date'])
    station_df = pd.read_excel('CUNY_MTA.xlsx')
    station_df['Date'] = pd.to_datetime(station_df['Date'], errors='coerce')
    station_df = station_df.dropna(subset=['Date', 'Street level air temperature', 'Station name'])
    merged_df = pd.merge(station_df, weather_df, on='Date', how='inner')
    return merged_df

# Create the embeddings for training the model
def create_features(df):
    df['Hour'] = pd.to_datetime(
        df['Time for street level data collection'],
        format='%H:%M:%S', errors='coerce'
    ).dt.hour
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    df['Day_of_Month'] = df['Date'].dt.day
    
    # Clean temperature data
    df['Street level air temperature'] = (
        df['Street level air temperature'].astype(str)
        .str.extract(r'(\d+\.?\d*)')[0].astype(float)
    )
    df['High Temp (°F)'] = pd.to_numeric(df['High Temp (°F)'], errors='coerce')
    
    # Clean platform temperature data
    df['Platform level air temperature'] = (
        df['Platform level air temperature'].astype(str)
        .str.extract(r'(\d+\.?\d*)')[0].astype(float)
    )
    
    # Clean crowding data
    if 'How crowded is the platform' in df.columns:
        df['Platform_Crowding'] = df['How crowded is the platform'].fillna('Unknown')
    else:
        df['Platform_Crowding'] = 'Unknown'
    
    df = df.dropna(subset=['Street level air temperature', 'High Temp (°F)', 'Station name', 'Hour'])
    
    # Encode categorical variables
    le_station = LabelEncoder()
    df['Station_Encoded'] = le_station.fit_transform(df['Station name'])
    
    le_crowding = LabelEncoder()
    df['Crowding_Encoded'] = le_crowding.fit_transform(df['Platform_Crowding'])
    
    return df, le_station, le_crowding

# Street-level temperature predictor
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
        self.r2_score = r2_score(y_test, y_pred)
        self.mae = mean_absolute_error(y_test, y_pred)
        
        return self.r2_score, self.mae
    
    def predict(self, high_temp, station_name, hour, day_of_week, day_of_month):
        try:
            station_encoded = self.station_encoder.transform([station_name])[0]
        except:
            station_encoded = 0
        
        features = np.array([[high_temp, station_encoded, hour, day_of_week, day_of_month]])
        return round(self.model.predict(features)[0], 1)

# Platform-level temperature predictor
class PlatformTempPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
        self.station_encoder = None
        self.crowding_encoder = None
    
    def fit(self, df, station_encoder, crowding_encoder):
        self.station_encoder = station_encoder
        self.crowding_encoder = crowding_encoder
        
        # Filter data where platform temperature is available
        platform_df = df.dropna(subset=['Platform level air temperature'])
        
        if len(platform_df) == 0:
            st.warning("No platform temperature data available for training.")
            return None, None
        
        X = platform_df[['Street level air temperature', 'Station_Encoded', 'Hour', 
                        'Day_of_Week', 'Day_of_Month', 'Crowding_Encoded']]
        y = platform_df['Platform level air temperature']
        
        if len(X) < 10:  # Need minimum data for training
            st.warning("Insufficient platform temperature data for reliable training.")
            return None, None
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        self.r2_score = r2_score(y_test, y_pred)
        self.mae = mean_absolute_error(y_test, y_pred)
        
        return self.r2_score, self.mae
    
    def predict(self, street_temp, station_name, hour, day_of_week, day_of_month, crowding='Unknown'):
        try:
            station_encoded = self.station_encoder.transform([station_name])[0]
        except:
            station_encoded = 0
        
        try:
            crowding_encoded = self.crowding_encoder.transform([crowding])[0]
        except:
            crowding_encoded = 0
        
        features = np.array([[street_temp, station_encoded, hour, day_of_week, day_of_month, crowding_encoded]])
        return round(self.model.predict(features)[0], 1)

# Main Streamlit App
def main():
    st.title("🌡️ NYC Temperature Prediction System 🌡️")
    st.markdown("### Two-Stage Temperature Prediction: Citywide → Street-Level → Platform-Level")
    
    # Load and prepare data
    with st.spinner("Loading and preparing data..."):
        df = load_and_prepare_data()
        df, station_encoder, crowding_encoder = create_features(df)
    
    # Train models
    with st.spinner("Training models..."):
        # Street-level predictor
        street_predictor = StreetTempPredictor()
        street_r2, street_mae = street_predictor.fit(df, station_encoder)
        
        # Platform-level predictor
        platform_predictor = PlatformTempPredictor()
        platform_r2, platform_mae = platform_predictor.fit(df, station_encoder, crowding_encoder)
    
    # Display model performance
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏠 Street-Level Model Performance")
        st.write(f"**R² Score:** {street_r2:.3f}")
        st.write(f"**MAE:** {street_mae:.2f}°F")
    
    with col2:
        st.subheader("🚇 Platform-Level Model Performance")
        if platform_r2 is not None:
            st.write(f"**R² Score:** {platform_r2:.3f}")
            st.write(f"**MAE:** {platform_mae:.2f}°F")
        else:
            st.write("**Model not available** (insufficient data)")
    
    st.divider()
    
    # User inputs
    st.subheader("🎯 Make Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        station = st.selectbox("Select a Station", sorted(df['Station name'].dropna().unique()))
        high_temp = st.number_input("Citywide High Temp (°F)", min_value=20.0, max_value=120.0, value=75.0, step=0.1)
        date_input = st.date_input("Date")
    
    with col2:
        hour = st.slider("Hour of the Day", 0, 23, 12)
        if 'Platform_Crowding' in df.columns:
            crowding_options = sorted(df['Platform_Crowding'].unique())
            crowding = st.selectbox("Platform Crowding Level", crowding_options)
        else:
            crowding = 'Unknown'
            st.info("Platform crowding data not available")
    
    # Prediction button
    if st.button("🔮 Predict Temperatures", type="primary"):
        # Stage 1: Predict street-level temperature
        street_temp = street_predictor.predict(
            high_temp=high_temp,
            station_name=station,
            hour=hour,
            day_of_week=date_input.weekday(),
            day_of_month=date_input.day
        )
        
        # Display street-level prediction
        st.success(f"🏠 **Predicted Street-Level Temperature:** {street_temp}°F")
        
        # Stage 2: Predict platform-level temperature
        if platform_r2 is not None:
            platform_temp = platform_predictor.predict(
                street_temp=street_temp,
                station_name=station,
                hour=hour,
                day_of_week=date_input.weekday(),
                day_of_month=date_input.day,
                crowding=crowding
            )
            
            st.success(f"🚇 **Predicted Platform-Level Temperature:** {platform_temp}°F")
            
            # Temperature difference analysis
            temp_diff = platform_temp - street_temp
            if temp_diff > 0:
                st.info(f"📊 Platform is **{abs(temp_diff):.1f}°F warmer** than street level")
            elif temp_diff < 0:
                st.info(f"📊 Platform is **{abs(temp_diff):.1f}°F cooler** than street level")
            else:
                st.info("📊 Platform and street temperatures are **the same**")
        else:
            st.warning("🚇 Platform-level prediction not available due to insufficient training data")
    
    # Additional insights
    st.divider()
    st.subheader("📈 Data Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", len(df))
    
    with col2:
        platform_records = len(df.dropna(subset=['Platform level air temperature']))
        st.metric("Platform Records", platform_records)
    
    with col3:
        st.metric("Unique Stations", df['Station name'].nunique())

if __name__ == "__main__":
    main()
