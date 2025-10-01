import os
import pandas as pd
from collections import Counter, defaultdict

# Parent folder containing CSVs
parent_folder = "Downscale_Csv_2018"

# Parameters
chunk_size = 1_000_000  # adjust based on your RAM
low_variance_threshold = 3  # max unique values to consider "low-variance"

# Iterate through all CSVs in the parent folder
for root, dirs, files in os.walk(parent_folder):
    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(root, file)
            print(f"\nProcessing file: {file}")

            # Track unique values per column
            col_unique_values = {}
            # Track label distributions for each value in each column
            col_value_label_counts = defaultdict(lambda: defaultdict(Counter))

            try:
                for chunk in pd.read_csv(file_path, chunksize=chunk_size, dtype=str, low_memory=False):
                    if "label" not in chunk.columns:
                        raise ValueError("No 'label' column found in dataset.")

                    for col in chunk.columns:
                        if col == "label":
                            continue

                        vals = chunk[col].dropna().unique()
                        if col not in col_unique_values:
                            col_unique_values[col] = set(vals)
                        else:
                            col_unique_values[col].update(vals)

                        # Count label distribution for each column value
                        for val, lbl in zip(chunk[col], chunk["label"]):
                            if pd.notna(val) and pd.notna(lbl):
                                col_value_label_counts[col][val][lbl] += 1

                # After all chunks processed, check for constant or low-variance columns
                constant_cols = {col: list(vals)[0] for col, vals in col_unique_values.items() if len(vals) == 1}
                low_variance_cols = {col: list(vals) for col, vals in col_unique_values.items()
                                     if 2 <= len(vals) <= low_variance_threshold}

                # Print results
                if constant_cols:
                    print("  Constant Columns:")
                    for col, val in constant_cols.items():
                        print(f"    {col} = {val}")
                else:
                    print("  No constant columns found.")

                if low_variance_cols:
                    print("  Low-Variance Columns (with label distribution):")
                    for col, vals in low_variance_cols.items():
                        print(f"    {col}: {vals}")
                        for v in vals:
                            dist = dict(col_value_label_counts[col][v])
                            print(f"      {v} -> {dist}")
                else:
                    print("  No low-variance columns found.")

            except Exception as e:
                print(f"Error processing {file}: {e}")
