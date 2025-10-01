import os
import pandas as pd
import numpy as np

# --- Configuration ---
input_folder = "Downscale_Csv_2018"
cols_to_fix = ['delta_start', 'handshake_duration']
chunk_size = 2_000_000

# Process each CSV in the input folder
for filename in os.listdir(input_folder):
    if not filename.endswith(".csv") or filename.endswith("_imputed.csv"):
        continue

    input_csv = os.path.join(input_folder, filename)
    print(f"\n--- Processing file: {input_csv} ---")

    # --- Phase 1: Calculate medians safely ---
    medians = {}
    for col in cols_to_fix:
        series = pd.read_csv(input_csv, usecols=[col], low_memory=False).squeeze("columns")
        numeric_series = pd.to_numeric(series, errors='coerce')
        medians[col] = numeric_series.dropna().median()
        print(f"Median of {col} = {medians[col]}")

    # --- Phase 2: Process in chunks ---
    base_name = os.path.splitext(filename)[0]
    output_csv = os.path.join(input_folder, f"{base_name}_imputed.csv")
    is_first_chunk = True

    for chunk in pd.read_csv(input_csv, chunksize=chunk_size, low_memory=False):
        # Preview before processing
        if is_first_chunk:
            print("\n--- Before processing (first 5 rows) ---")
            print(chunk.head())

        # Flag invalid handshakes
        chunk['handshake_incomplete'] = 0
        for col in cols_to_fix:
            invalid_mask = chunk[col].astype(str).str.lower() == "not a complete handshake"
            chunk['handshake_incomplete'] = chunk['handshake_incomplete'] | invalid_mask.astype(int)
            # Fill invalid strings with median
            chunk.loc[invalid_mask, col] = medians[col]

        # Preview after processing
        if is_first_chunk:
            print("\n--- After processing (first 5 rows) ---")
            print(chunk.head())
            is_first_chunk = False

        # Save chunk to new CSV
        if is_first_chunk:
            chunk.to_csv(output_csv, index=False, mode='w')
            is_first_chunk = False
        else:
            chunk.to_csv(output_csv, index=False, mode='a', header=False)

    print(f"\nProcessing complete. New CSV saved as: {output_csv}")
