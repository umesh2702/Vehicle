import os
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .features import add_derived_features

DATA_DIR = os.path.join(os.getcwd(), "data")
MERGED_FILE = os.path.join(DATA_DIR, "merged_sensors.csv")
MODEL_PATH = os.path.join(DATA_DIR, "anomaly_model.pkl")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.pkl")


def clean_numeric_column(series: pd.Series) -> pd.Series:
    """
    Clean numeric columns: remove units, fix malformed scientific notation,
    convert to float safely.
    """
    if series.dtype == object:
        series = series.astype(str)
        # Remove any non-numeric chars except - . e E
        series = series.str.replace(r"[^0-9eE\.\-]", "", regex=True)
        # Fix 1.23-10 -> 1.23e-10
        series = series.str.replace(r"([0-9])\-([0-9]+)$", r"\1e-\2", regex=True)
    # Convert to numeric, invalid parsing becomes NaN
    return pd.to_numeric(series, errors="coerce")


def train_anomaly_detector(X: pd.DataFrame):
    """Train anomaly detection model using IsolationForest"""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(Xs)

    # Save the trained model + scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")
    print(f"✅ Scaler saved to {SCALER_PATH}")


if __name__ == "__main__":
    if not os.path.exists(MERGED_FILE):
        raise FileNotFoundError(f"{MERGED_FILE} not found. Run data_processing first.")

    df = pd.read_csv(MERGED_FILE, low_memory=False)

    # Add derived features (rolling averages etc.)
    df = add_derived_features(df)

    # Select numeric features
    feature_cols = [
        "soc", "soh", "battery_temp", "motor_rpm", "motor_torque",
        "soc_ma_5", "soc_ma_10",
        "soh_ma_5", "soh_ma_10",
        "battery_temp_ma_5", "battery_temp_ma_10",
        "motor_rpm_ma_5", "motor_rpm_ma_10",
        "motor_torque_ma_5", "motor_torque_ma_10",
    ]

    # Clean columns safely
    for c in feature_cols:
        if c in df.columns:
            df[c] = clean_numeric_column(df[c])

    # Replace NaNs with 0
    X = df[feature_cols].fillna(0)
             
    print("🚀 Training anomaly detector...")
    train_anomaly_detector(X)

def predict_vehicle_health(rpm, speed, temp, dtc_code, dtc_lookup):
    response = {
        "RPM": rpm,
        "Speed": speed,
        "Temperature": temp,
        "DTC": dtc_code,
        "message": ""
    }

    if dtc_code and dtc_code in dtc_lookup:
        response["message"] = f"⚠️ DTC {dtc_code}: {dtc_lookup[dtc_code]}"
    elif dtc_code:
        response["message"] = f"❓ DTC {dtc_code} not found in database."
    else:
        response["message"] = "✅ No diagnostic trouble code detected."

    if temp and temp > 100:
        response["message"] += " Engine temperature is too high!"
    if rpm and rpm > 5000:
        response["message"] += " Engine RPM unusually high."
    if speed and speed > 120:
        response["message"] += " Speed exceeds safe highway limits."

    return response
