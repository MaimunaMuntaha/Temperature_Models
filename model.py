import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import datetime, matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import time


USE_HYPERPARAM_SEARCH = False

st.set_page_config(page_title="Subway Heat Forecast")


@st.cache_data
def load_data():
    citywideData = pd.read_csv("citywide.csv")
    cunyMtaData = pd.read_csv("CUNY_MTA.csv")
    citywideData["Date"] = pd.to_datetime(citywideData["Date"], errors="coerce")
    cunyMtaData["Date"] = pd.to_datetime(cunyMtaData["Date"], errors="coerce")
    cunyMtaData["Platform level air temperature"] = pd.to_numeric(
        cunyMtaData["Platform level air temperature"], errors="coerce"
    )
    cunyMtaData = cunyMtaData.dropna(
        subset=["Platform level air temperature", "Station name"]
    )
    return citywideData, cunyMtaData


def adjusted_r2(rSquared, sampleCount, featureCount):
    return (
        1 - (1 - rSquared) * ((sampleCount - 1) / (sampleCount - featureCount - 1))
        if sampleCount > featureCount + 1
        else np.nan
    )


def hyper_param_search_and_train_model(X_train, y_train):
    # n_estimators = [int(x) for x in np.linspace(start = 10, stop = 2000, num = 10)]
    n_estimators = [10, 20, 50, 100, 200, 500, 1000, 1200, 1500, 1800, 1900, 2000]
    # Number of features to consider at every split
    max_features = [
        "log2",
        "sqrt",
        1.0,
    ]  # all in the following array are fast but log2 is fastest ['log2', 'sqrt', 1.0]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    # max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = list(range(1, 4))
    # Minimum number of samples required at each leaf node
    min_samples_leaf = list(range(1, 4))

    # Define the parameter grid for RandomizedSearchCV
    param_grid = {
        "n_estimators": n_estimators,
        # 'max_depth': max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        # 'max_features': max_features,
    }

    # Initialize the Random Forest model
    rf_model = RandomForestRegressor()  # 2 for mse

    time_start = time.time()
    print(
        "Searching for best hyperparamaters using a random search and checking by fitting model..."
    )

    # Initialize the RandomizedSearchCV with 5-fold cross-validation
    random_search = GridSearchCV(  # GridSearchCV is more exhuastive
        rf_model,
        param_grid,
        cv=5,
        # scoring='r2',
        scoring="neg_mean_absolute_error",  # neg_mean_absolute_error = MAE
        n_jobs=-1,
    )

    # Fit the RandomizedSearchCV to find the best hyperparameters (by randomly trying combos within param grid and fitting and testing accordingly)
    random_search.fit(X_train, y_train)
    time_end = time.time()
    time_diff = time_end - time_start
    print(f"Random search & Fit finished, elapsed {time_diff} seconds")

    # Get the best parameters
    best_params = random_search.best_params_
    best_model = random_search.best_estimator_
    return best_params, best_model


@st.cache_resource
def train_model(citywideData, cunyMtaData):
    mergedData = cunyMtaData.merge(citywideData, on="Date", how="inner")
    # Time/date
    mergedData["Hour"] = pd.to_datetime(
        mergedData["Time for street level data collection"],
        format="%I:%M:%S %p",
        errors="coerce",
    ).dt.hour

    # unique stations
    stationEncoder = LabelEncoder()
    mergedData["StationEncoded"] = stationEncoder.fit_transform(
        mergedData["Station name"]
    )

    # MOST IMPORTANT CHANGE: Platform vs citywide relationships
    mergedData["PlatformVsHigh"] = (
        mergedData["Platform level air temperature"] - mergedData["High Temp (°F)"]
    )

    stationStats = mergedData.groupby("Station name").agg(
        {
            "PlatformVsHigh": ["mean"],
        }
    )
    stationStats.columns = [
        f"{firstPart}{secondPart.capitalize()}"
        for firstPart, secondPart in stationStats.columns
    ]
    stationStats = stationStats.reset_index()

    mergedData = mergedData.merge(stationStats, on="Station name", how="left")

    # Remove outliers
    mergedData = mergedData[
        (mergedData["Platform level air temperature"] > 30)
        & (mergedData["Platform level air temperature"] < 130)
    ]

    # FEATURE LIST-- what we're training model on now
    featureList = [
        "High Temp (°F)",
        "Low Temp (°F)",
        "StationEncoded",
        "Hour",
        "PlatformVsHighMean",
    ]
    mergedData = mergedData.dropna(
        subset=featureList + ["Platform level air temperature"]
    )
    X, y = mergedData[featureList], mergedData["Platform level air temperature"]

    XTrain, XTest, yTrain, yTest = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    if USE_HYPERPARAM_SEARCH:
        model = hyper_param_search_and_train_model(XTrain, yTrain)
    else:
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            max_features="sqrt",
            random_state=42,
        )
        model.fit(XTrain, yTrain)

    predictions = model.predict(XTest)
    rSquared = r2_score(yTest, predictions)
    meanAbsError = mean_absolute_error(yTest, predictions)
    rootMeanSquaredError = root_mean_squared_error(yTest, predictions)
    adjustedRSquared = adjusted_r2(rSquared, len(yTest), XTest.shape[1])

    # Feature importance
    feature_importances = model.feature_importances_
    fig, ax = plt.subplots()
    indices = np.argsort(feature_importances)
    features = X.columns
    ax.barh(features, feature_importances, color="skyblue", ec="black")
    st.pyplot(fig)

    return (
        model,
        stationEncoder,
        featureList,
        (rSquared, adjustedRSquared, meanAbsError, rootMeanSquaredError),
        stationStats,
    )


# streamlit
st.title("Subway Heat Forecast")

try:
    citywideData, cunyMtaData = load_data()
    (
        model,
        stationEncoder,
        featureList,
        (rSquared, adjustedRSquared, meanAbsError, rootMeanSquaredError),
        stationStats,
    ) = train_model(citywideData, cunyMtaData)

    st.subheader("Make a Prediction")
    column1, column2 = st.columns(2)
    with column1:
        selectedStation = st.selectbox(
            "Station", sorted(cunyMtaData["Station name"].dropna().unique())
        )
        selectedDate = st.date_input("Date", value=datetime.date.today())
        selectedTime = st.time_input("Time", value=datetime.time(12, 0))
    with column2:
        cityHigh = st.number_input("Citywide High (°F)", value=85.0)
        cityLow = st.number_input("Citywide Low (°F)", value=70.0)

    if st.button("Predict Platform Temp"):
        hourValue = pd.to_datetime(selectedTime.strftime("%H:%M:%S")).hour

        try:
            stationEncodedValue = stationEncoder.transform([selectedStation])[0]
        except Exception:
            st.error("Station not in training data.")
            st.stop()

        predictionValues = {
            "High Temp (°F)": cityHigh,
            "Low Temp (°F)": cityLow,
            "StationEncoded": stationEncodedValue,
            "Hour": hourValue,
        }

        XInput = pd.DataFrame([predictionValues]).reindex(
            columns=featureList, fill_value=0
        )
        predictedValue = model.predict(XInput)[0]
        st.success(f"Predicted Platform Temperature: {predictedValue:.1f} °F")
        st.metric(
            "The Difference from Citywide High", f"{predictedValue - cityHigh:+.1f}°F"
        )

    # Sidebar performance
    st.sidebar.header("Model Performance")
    st.sidebar.write(f"R²: {rSquared:.3f}")
    st.sidebar.write(f"Adj R²: {adjustedRSquared:.3f}")
    st.sidebar.write(f"MAE: {meanAbsError:.2f} °F")
    st.sidebar.write(f"RMSE: {rootMeanSquaredError:.2f} °F")


except Exception as error:
    st.error(f"Error: {error}")
