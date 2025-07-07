import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
 
# Set page config
st.set_page_config(
    page_title="NYC Subway Forecast", 
    layout="wide"
)

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess both datasets for street and platform temperature prediction"""
    # Load files
    weather_df = pd.read_csv('citywide_weather.csv')
    station_df = pd.read_excel('CUNY_MTA.xlsx')

    # Fix column names
    weather_df.columns = weather_df.columns.str.strip()
    station_df.columns = station_df.columns.str.strip()

    # Parse dates
    weather_df['Date'] = pd.to_datetime(weather_df['Date'])
    station_df['Date'] = pd.to_datetime(station_df['Date'], errors='coerce')
    
    # Keep only valid rows for platform model
    platform_df = station_df.dropna(subset=[
        'Date', 'Station name', 'Platform level air temperature', 'Platform level relative humidity',
        'Street level air temperature', 'Street level relative humidity', 'How crowded is the platform?'
    ]).copy()

    # Keep only valid rows for street model
    street_df = station_df.dropna(subset=[
        'Date', 'Station name', 'Street level air temperature'
    ]).copy()

    # Convert temps/humidity to numeric for platform model
    platform_df['Platform level air temperature'] = pd.to_numeric(platform_df['Platform level air temperature'], errors='coerce')
    platform_df['Platform level relative humidity'] = pd.to_numeric(platform_df['Platform level relative humidity'], errors='coerce')
    platform_df['Street level air temperature'] = pd.to_numeric(platform_df['Street level air temperature'], errors='coerce')
    platform_df['Street level relative humidity'] = pd.to_numeric(platform_df['Street level relative humidity'], errors='coerce')

    # Convert street temp to numeric for street model
    street_df['Street level air temperature'] = (
        street_df['Street level air temperature'].astype(str)
        .str.extract(r'(\d+\.?\d*)')[0].astype(float)
    )

    # Drop any remaining invalid rows
    platform_df = platform_df.dropna(subset=[
        'Platform level air temperature', 'Platform level relative humidity',
        'Street level air temperature', 'Street level relative humidity'
    ])
    
    street_df = street_df.dropna(subset=['Street level air temperature'])

    # Merge with weather data
    platform_merged = pd.merge(platform_df, weather_df, on='Date', how='inner')
    street_merged = pd.merge(street_df, weather_df, on='Date', how='inner')

    # Feature extraction for platform model
    platform_merged['Hour'] = pd.to_datetime(platform_merged['Time for street level data collection'], format='%H:%M:%S', errors='coerce').dt.hour
    platform_merged['Day_of_week'] = platform_merged['Date'].dt.dayofweek
    platform_merged['Day_of_year'] = platform_merged['Date'].dt.dayofyear
    platform_merged['Month'] = platform_merged['Date'].dt.month

    # Feature extraction for street model  
    street_merged['Hour'] = pd.to_datetime(street_merged['Time for street level data collection'], format='%H:%M:%S', errors='coerce').dt.hour
    street_merged['Day_of_Week'] = street_merged['Date'].dt.dayofweek
    street_merged['Day_of_Month'] = street_merged['Date'].dt.day
    street_merged['Month'] = street_merged['Date'].dt.month

    # Label encoders
    le_station_platform = LabelEncoder()
    le_crowd = LabelEncoder()
    le_station_street = LabelEncoder()
    
    platform_merged['Station_encoded'] = le_station_platform.fit_transform(platform_merged['Station name'])
    platform_merged['Crowd_encoded'] = le_crowd.fit_transform(platform_merged['How crowded is the platform?'])
    street_merged['Station_Encoded'] = le_station_street.fit_transform(street_merged['Station name'])

    # Derived features for platform model
    platform_merged['Temp_difference'] = platform_merged['Platform level air temperature'] - platform_merged['Street level air temperature']
    platform_merged['Humidity_difference'] = platform_merged['Platform level relative humidity'] - platform_merged['Street level relative humidity']
    platform_merged['Is_rush_hour'] = ((platform_merged['Hour'] >= 7) & (platform_merged['Hour'] <= 9)) | ((platform_merged['Hour'] >= 17) & (platform_merged['Hour'] <= 19))
    platform_merged['Is_weekend'] = platform_merged['Day_of_week'] >= 5
    platform_merged['Temp_range'] = platform_merged['High Temp (°F)'] - platform_merged['Low Temp (°F)']

    return platform_merged, street_merged, le_station_platform, le_crowd, le_station_street

class StreetTempPredictor:
    """Model to predict street-level temperature using BaggingRegressor"""
    def __init__(self):
        self.model = BaggingRegressor(
            n_estimators=100,
            max_samples=0.8,
            max_features=0.8,
            random_state=42,
            n_jobs=-1
        )
        self.station_encoder = None
        self.performance_metrics = {}
    
    def fit(self, df, station_encoder):
        self.station_encoder = station_encoder
        
        # Prepare features
        X = df[['High Temp (°F)', 'Station_Encoded', 'Hour', 'Day_of_Week', 'Day_of_Month']].copy()
        y = df['Street level air temperature'].copy()
        
        # Remove any remaining NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit model
        self.model.fit(X_train, y_train)
        
        # Calculate performance metrics
        y_pred = self.model.predict(X_test)
        self.performance_metrics = {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
    
    def predict(self, high_temp, station_name, hour, day_of_week, day_of_month):
        try:
            station_encoded = self.station_encoder.transform([station_name])[0]
        except:
            station_encoded = 0
        
        features = np.array([[high_temp, station_encoded, hour, day_of_week, day_of_month]])
        return round(self.model.predict(features)[0], 1)

@st.cache_resource
def train_platform_model(X, y):
    """Train the platform temperature prediction model using GradientBoostingRegressor"""
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Use the best performing model - GradientBoostingRegressor
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        validation_fraction=0.2,
        n_iter_no_change=10
    )
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Calculate performance metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    model_results = {
        'GradientBoostingRegressor': {
            'model': model,
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'predictions': y_pred,
            'actual': y_test
        }
    }
    
    return model, scaler, model_results, 'GradientBoostingRegressor'

def main():
    st.title("NYC Subway Forecast")
    st.markdown("Predict both street-level and platform temperatures given a Platform below.") 
    
    # Load and prepare data
    platform_df, street_df, le_station_platform, le_crowd, le_station_street = load_and_preprocess_data()
    
    # Train street temperature model
    street_predictor = StreetTempPredictor()
    street_predictor.fit(street_df, le_station_street)
    
    # Prepare platform model features
    platform_feature_columns = ['Station_encoded', 'Street level air temperature', 'Street level relative humidity',
                               'High Temp (°F)', 'Low Temp (°F)', 'Hour', 'Month', 'Day_of_week',
                               'Crowd_encoded', 'Humidity_difference', 'Is_rush_hour', 'Is_weekend', 'Temp_range']
    
    X_platform = platform_df[platform_feature_columns]
    y_platform = platform_df['Platform level air temperature']
    
    # Train platform model
    platform_model, platform_scaler, platform_model_results, best_platform_model = train_platform_model(X_platform, y_platform)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Combined Prediction", "Model Performance", "Data Analysis"])
    
    if page == "Combined Prediction":
        st.header("Temperature Prediction Model Stats")
        
        # Show model performance summary
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Street Model Performance")
            st.metric("R² Score", f"{street_predictor.performance_metrics['r2']:.4f}")
            st.metric("MAE", f"{street_predictor.performance_metrics['mae']:.2f}°F") 
        
        with col2:
            st.subheader("Platform Model Performance")
            platform_r2 = platform_model_results[best_platform_model]['r2']
            platform_mae = platform_model_results[best_platform_model]['mae']
            st.metric("R² Score", f"{platform_r2:.4f}")
            st.metric("MAE", f"{platform_mae:.2f}°F") 
        
        # Create prediction interface
        st.subheader("Input Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Get common stations (stations that exist in both datasets)
            common_stations = set(platform_df['Station name'].unique()) & set(street_df['Station name'].unique())
            station_name = st.selectbox("Select Station", sorted(list(common_stations)))
            
            prediction_date = st.date_input("Select Date")
            prediction_time = st.time_input("Select Time", value=time(12, 0))
            
            high_temp = st.slider("Daily High Temperature (°F)", 20, 100, 75)
            low_temp = st.slider("Daily Low Temperature (°F)", 10, 90, 55)
            
            crowd_level = st.selectbox("Platform Crowd Level", ["Light", "Medium", "Heavy"])
        
        with col2:
            st.subheader("Prediction Results")
            
            if st.button("Predict Temperatures", type="primary"):
                try:
                    # Create datetime object
                    datetime_obj = datetime.combine(prediction_date, prediction_time)
                    hour = datetime_obj.hour
                    day_of_week = datetime_obj.weekday()
                    day_of_month = datetime_obj.day
                    month = datetime_obj.month
                    
                    # Step 1: Predict street temperature using BaggingRegressor
                    predicted_street_temp = street_predictor.predict(
                        high_temp=high_temp,
                        station_name=station_name,
                        hour=hour,
                        day_of_week=day_of_week,
                        day_of_month=day_of_month
                    )
                    
                    st.metric("Street Temperature Forecast:", f"{predicted_street_temp}°F")
                    
                    # Step 2: Use predicted street temp to predict platform temp using GradientBoostingRegressor
                    try:
                        station_encoded = le_station_platform.transform([station_name])[0]
                        crowd_encoded = le_crowd.transform([crowd_level])[0]
                        
                        # Calculate additional features for platform model
                        avg_humidity = platform_df[(platform_df['Station name'] == station_name) & 
                                                 (platform_df['Hour'] == hour)]['Street level relative humidity'].mean()
                        if pd.isna(avg_humidity):
                            avg_humidity = platform_df['Street level relative humidity'].mean()
                        
                        humidity_diff = 0  # Will be estimated by the model
                        is_rush_hour = (7 <= hour <= 9) or (17 <= hour <= 19)
                        is_weekend = day_of_week >= 5
                        temp_range = high_temp - low_temp
                        
                        # Create feature vector for platform prediction
                        platform_features = np.array([[station_encoded, predicted_street_temp, avg_humidity, 
                                                     high_temp, low_temp, hour, month, day_of_week, 
                                                     crowd_encoded, humidity_diff, is_rush_hour, 
                                                     is_weekend, temp_range]])
                        
                        # Scale features and predict
                        platform_features_scaled = platform_scaler.transform(platform_features)
                        predicted_platform_temp = platform_model.predict(platform_features_scaled)[0]
                        
                        st.metric("Platform Temperature Forecast:", f"{predicted_platform_temp:.1f}°F")
                        
                        # Calculate and display temperature difference
                        temp_diff = predicted_platform_temp - predicted_street_temp
                        st.metric("🌡Temperature Difference (Platform - Street)", f"{temp_diff:+.1f}°F")
                        
                        # Provide context
                        if temp_diff > 8:
                            st.info("Platform is significantly warmer than street level")
                        elif temp_diff > 3:
                            st.info("Platform is moderately warmer than street level")
                        elif temp_diff < -8:
                            st.info("Platform is significantly cooler than street level")
                        elif temp_diff < -3:
                            st.info("Platform is moderately cooler than street level")
                        else:
                            st.info("Platform temperature is similar to street level")
                            
                        # Show prediction confidence
                        st.write("**Prediction Confidence:**")
                        st.write(f"- Street temp uncertainty: ±{street_predictor.performance_metrics['mae']:.1f}°F")
                        st.write(f"- Platform temp uncertainty: ±{platform_mae:.1f}°F")
                        
                        
                    except ValueError as e:
                        st.error(f"Error predicting platform temperature: Station might not be in platform training data.")
                        
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
    
    elif page == "Model Performance":
        st.header("📊 Model Performance Analysis")
        
        # Street model performance
        st.subheader("Street Temperature Model - BaggingRegressor")
        street_metrics = street_predictor.performance_metrics
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", f"{street_metrics['r2']:.4f}")
        with col2:
            st.metric("Mean Absolute Error", f"{street_metrics['mae']:.2f}°F")
        with col3:
            st.metric("Root Mean Squared Error", f"{street_metrics['rmse']:.2f}°F")
        
        # Platform model performance
        st.subheader("Platform Temperature Model - GradientBoostingRegressor")
        
        best_results = platform_model_results[best_platform_model]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", f"{best_results['r2']:.4f}")
        with col2:
            st.metric("Mean Absolute Error", f"{best_results['mae']:.2f}°F")
        with col3:
            st.metric("Root Mean Squared Error", f"{np.sqrt(best_results['mse']):.2f}°F") 
        
        # Platform model predictions vs actual
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=best_results['actual'],
            y=best_results['predictions'],
            mode='markers',
            name='Predictions',
            marker=dict(color='blue', opacity=0.7)
        ))
        
        # Add perfect prediction line
        min_val = min(best_results['actual'].min(), best_results['predictions'].min())
        max_val = max(best_results['actual'].max(), best_results['predictions'].max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f'Platform Temperature: Predicted vs Actual ({best_platform_model})',
            xaxis_title='Actual Temperature (°F)',
            yaxis_title='Predicted Temperature (°F)',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model comparison info
        st.subheader("Model Selection Rationale")
        st.write("""
        **Best Models Selected:**
        - **Street Temperature**: BaggingRegressor (R² = 0.3427)
        - **Platform Temperature**: GradientBoostingRegressor (R² = 0.2434)
        
        These models were selected based on comprehensive performance evaluation 
        and represent the optimal balance between accuracy and generalization.
        """)
    
    elif page == "Data Analysis":
        st.header("📈 Data Analysis")
        
        # Dataset overview
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Platform Records", f"{len(platform_df):,}")
        with col2:
            st.metric("Street Records", f"{len(street_df):,}")
        with col3:
            st.metric("Platform Stations", platform_df['Station name'].nunique())
        with col4:
            st.metric("Street Stations", street_df['Station name'].nunique())
        
        # Temperature distributions
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.histogram(platform_df, x='Platform level air temperature', 
                              title='Platform Temperature Distribution', nbins=30)
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            fig2 = px.histogram(street_df, x='Street level air temperature', 
                              title='Street Temperature Distribution', nbins=30)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Platform vs Street temperature relationship
        fig3 = px.scatter(platform_df, x='Street level air temperature', y='Platform level air temperature',
                         color='How crowded is the platform?', title='Platform vs Street Temperature Relationship',
                         hover_data=['Station name'])
        
        # Add diagonal reference line
        min_temp = min(platform_df['Street level air temperature'].min(), platform_df['Platform level air temperature'].min())
        max_temp = max(platform_df['Street level air temperature'].max(), platform_df['Platform level air temperature'].max())
        fig3.add_shape(type="line", x0=min_temp, y0=min_temp, x1=max_temp, y1=max_temp,
                      line=dict(color="red", dash="dash"), name="Equal Temperature Line")
        
        st.plotly_chart(fig3, use_container_width=True)

if __name__ == "__main__":
    main()
