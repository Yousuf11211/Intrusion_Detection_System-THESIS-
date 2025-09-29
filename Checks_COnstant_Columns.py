import os
import pandas as pd

# Parent folder containing CSVs
parent_folder = "Raw_Data_2018"

# Parameters
chunk_size = 5_000_000  # adjust based on your RAM
low_variance_threshold = 3  # max unique values to consider "low-variance"

# Iterate through all CSVs in the parent folder
for root, dirs, files in os.walk(parent_folder):
    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(root, file)
            print(f"\nProcessing file: {file}")

            # Track unique values per column across chunks
            col_unique_values = {}

            try:
                for chunk in pd.read_csv(file_path, chunksize=chunk_size, dtype=str, low_memory=False):
                    for col in chunk.columns:
                        vals = chunk[col].dropna().unique()
                        if col not in col_unique_values:
                            col_unique_values[col] = set(vals)
                        else:
                            col_unique_values[col].update(vals)

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
                    print("  Low-Variance Columns:")
                    for col, vals in low_variance_cols.items():
                        print(f"    {col} -> {vals}")
                else:
                    print("  No low-variance columns found.")

            except Exception as e:
                print(f"Error processing {file}: {e}")
