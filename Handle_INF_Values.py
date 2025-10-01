import os
import pandas as pd
import numpy as np

# --- 1. Configuration ---
input_folder = "Downscale_Csv_2018"

# List the columns to process
columns_to_process = [
    'delta_start',
    'handshake_duration',
    'cov_bwd_header_bytes_delta_len'
    # Add any other columns with potential 'inf' values
]

chunk_size = 500_000

print(f"Starting analysis in folder: {input_folder}")

for dirpath, _, filenames in os.walk(input_folder):
    for filename in filenames:
        if filename.endswith('.csv') and not filename.endswith('_imputed.csv'):

            input_csv_path = os.path.join(dirpath, filename)
            print(f"\n--- Processing file: {input_csv_path} ---")

            medians = {}

            # --- Phase 1: Calculate medians ---
            try:
                csv_cols = pd.read_csv(input_csv_path, nrows=0).columns.tolist()

                for col in columns_to_process:
                    if col in csv_cols:
                        series = pd.read_csv(input_csv_path, usecols=[col], low_memory=False).squeeze("columns")
                        numeric_series = pd.to_numeric(series, errors='coerce')
                        median_val = numeric_series.replace([np.inf, -np.inf], np.nan).median()
                        medians[col] = median_val
                        print(f"Column '{col}': Median (ignoring inf) = {median_val}")

            except Exception as e:
                print(f"Error during median calculation: {e}")
                continue

            # --- Phase 2: Impute 'inf' and save ---
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}_imputed.csv"
            output_csv_path = os.path.join(dirpath, output_filename)

            is_first_chunk = True

            try:
                for chunk in pd.read_csv(input_csv_path, chunksize=chunk_size, low_memory=False):
                    for col, median_val in medians.items():
                        if col in chunk.columns:
                            chunk[col].replace([np.inf, -np.inf], np.nan, inplace=True)
                            chunk[col].fillna(median_val, inplace=True)

                    if is_first_chunk:
                        chunk.to_csv(output_csv_path, index=False, mode='w')
                        is_first_chunk = False
                    else:
                        chunk.to_csv(output_csv_path, index=False, mode='a', header=False)

                print(f"----> Successfully created '{output_filename}' in '{dirpath}'")

            except Exception as e:
                print(f"Error while creating new file: {e}")

print("\nAll files processed.")
