import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "merged_sensors.csv")

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a dataset from CSV or Excel into a pandas DataFrame.
    Handles messy CSVs gracefully.
    """
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".csv":
        try:
            df = pd.read_csv(file_path, on_bad_lines="skip")  # pandas >=1.3
        except TypeError:
            # fallback for pandas <1.3
            df = pd.read_csv(file_path, error_bad_lines=False)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # Replace completely empty values with NaN
    df = df.dropna(how="all")

    return df



def merge_datasets(files):
    """
    Merge multiple datasets on timestamp (outer join).
    Handles duplicate columns with suffixes and removes true duplicates.
    """
    merged = None

    for i, f in enumerate(files):
        print(f"Processing {f} ...")
        df = load_dataset(f)

        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()

        # Ensure timestamp column exists
        if "timestamp" not in df.columns:
            df["timestamp"] = pd.RangeIndex(start=0, stop=len(df), step=1)

        df = df.set_index("timestamp")

        if merged is None:
            merged = df
        else:
            merged = merged.join(df, how="outer", rsuffix=f"_{i}")

    # Drop exact duplicate columns
    merged = merged.loc[:, ~merged.columns.duplicated()]

    return merged.reset_index()


def main():
    # Collect all dataset files
    files = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.endswith((".csv", ".xls", ".xlsx"))
    ]

    if not files:
        raise FileNotFoundError(f"No datasets found in {DATA_DIR}")

    print(f"Found {len(files)} dataset files.")
    merged = merge_datasets(files)

    print(f"Final merged shape: {merged.shape}")
    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Merged dataset saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
