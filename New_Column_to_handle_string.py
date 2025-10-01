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
    first_chunk = True

    for chunk in pd.read_csv(input_csv, chunksize=chunk_size, low_memory=False):
        # Flag invalid handshakes
        incomplete_mask = (chunk[cols_to_fix[0]].astype(str).str.lower() == "not a complete handshake") | \
                          (chunk[cols_to_fix[1]].astype(str).str.lower() == "not a complete handshake")
        chunk['handshake_incomplete'] = incomplete_mask.astype(int)

        # Fill invalid strings with median
        for col in cols_to_fix:
            mask = chunk[col].astype(str).str.lower() == "not a complete handshake"
            chunk.loc[mask, col] = medians[col]

        # Reorder columns: place new column just after delta_start and handshake_duration
        cols = list(chunk.columns)
        for col in cols_to_fix:
            cols.remove(col)
        # Place delta_start, handshake_duration, handshake_incomplete first
        new_order = [cols_to_fix[0], cols_to_fix[1], 'handshake_incomplete'] + cols
        chunk = chunk[new_order]

        # Preview first 5 affected rows before/after
        affected_rows = chunk['handshake_incomplete'] == 1
        if affected_rows.any() and first_chunk:
            print("\n--- First 5 affected rows (after filling) ---")
            print(chunk.loc[affected_rows, ['delta_start', 'handshake_duration', 'handshake_incomplete']].head())
            first_chunk = False

        # Save to CSV with headers only in the first chunk
        if first_chunk:
            chunk.to_csv(output_csv, index=False, mode='w')
            first_chunk = False
        else:
            chunk.to_csv(output_csv, index=False, mode='a', header=False)

    print(f"\nProcessing complete. New CSV saved as: {output_csv}")
