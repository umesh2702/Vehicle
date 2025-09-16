import pandas as pd
import re

def clean_numeric_column(series: pd.Series) -> pd.Series:
    """
    Cleans a numeric column that may contain strings with units (e.g., '943RPM').
    Extracts numbers and converts to float.
    """
    return (
        series.astype(str)
        .apply(lambda x: re.sub(r"[^0-9\.-]", "", x))  # keep only digits, minus, dot
        .replace("", "0")  # replace empty with 0
        .astype(float)
    )

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds derived features such as moving averages to the dataset.
    Ensures numeric columns are cleaned before applying rolling functions.
    """

    # Columns we expect to be numeric
    numeric_cols = [
        "soc",           # state of charge
        "soh",           # state of health
        "battery_temp",  # battery temperature
        "motor_rpm",     # motor speed
        "motor_torque",  # torque
    ]

    for c in numeric_cols:
        if c in df.columns:
            # Clean values first (remove units like RPM, °C, etc.)
            df[c] = clean_numeric_column(df[c])

            # Add moving averages
            df[f"{c}_ma_5"] = df[c].rolling(window=5, min_periods=1).mean()
            df[f"{c}_ma_10"] = df[c].rolling(window=10, min_periods=1).mean()

    return df
