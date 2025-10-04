import os
import pandas as pd

# ========= CONFIG =========
INPUT_FOLDER = "Training_2018"
CHUNK_SIZE = 1_500_000  # For big files

# ========== MAIN SCRIPT ===========
for filename in os.listdir(INPUT_FOLDER):
    if not filename.endswith(".csv"):
        continue

    file_path = os.path.join(INPUT_FOLDER, filename)
    print(f"\nProcessing file: {filename}")

    # Load CSV in chunks if large
    dfs = []
    for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, dtype=str):
        dfs.append(chunk)
    df = pd.concat(dfs, ignore_index=True)
    del dfs  # free RAM

    # Column count
    choice = input("Do you want to see the column count? (y/n): ")
    if choice.lower() == "y":
        print(f"Number of columns: {df.shape[1]}")

    # Row count
    choice = input("Do you want to see the row count? (y/n): ")
    if choice.lower() == "y":
        print(f"Number of rows: {df.shape[0]}")

    # Duplicate column names
    choice = input("Check for duplicate column names? (y/n): ")
    if choice.lower() == "y":
        col_counts = pd.Series(df.columns).value_counts()
        duplicate_cols = col_counts[col_counts > 1]
        if not duplicate_cols.empty:
            print(f"Duplicate columns: {list(duplicate_cols.index)}")
            remove_cols = input("Remove duplicate columns and save updated CSV? (y/n): ")
            if remove_cols.lower() == "y":
                cols = pd.Series(df.columns)
                _, idx = pd.unique(cols, return_index=True)
                unique_cols = cols[idx].tolist()
                df = df[unique_cols]
                out_path = os.path.join(INPUT_FOLDER, f"{os.path.splitext(filename)[0]}_nodupcol.csv")
                df.to_csv(out_path, index=False)
                print(f"Saved file without duplicate columns: {out_path}")
        else:
            print("No duplicate column names.")

    # Duplicate rows
    choice = input("Check for duplicate rows? (y/n): ")
    if choice.lower() == "y":
        dup_rows = df.duplicated().sum()
        print(f"Duplicate rows: {dup_rows}")
        if dup_rows > 0:
            remove_rows = input("Remove duplicate rows and save updated CSV? (y/n): ")
            if remove_rows.lower() == "y":
                rows_before = df.shape[0]
                df = df.drop_duplicates()
                out_path = os.path.join(INPUT_FOLDER, f"{os.path.splitext(filename)[0]}_nodup.csv")
                df.to_csv(out_path, index=False)
                print(f"Saved file without duplicate rows: {out_path}")

    # Missing values
    choice = input("Check for missing values? (y/n): ")
    if choice.lower() == "y":
        missing = df.isnull().sum()
        missing_dict = {col: n for col, n in missing.items() if n > 0}
        if missing_dict:
            print("Missing values per column:")
            for col, count in missing_dict.items():
                print(f"  {col}: {count}")
        else:
            print("No missing values found.")
