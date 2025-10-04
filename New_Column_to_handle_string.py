import os
import pandas as pd
import numpy as np

# --- Configuration ---
input_folder = "Downscale_Csv_2018"
input_filename = "Cleaned.csv"
input_csv = os.path.join(input_folder, input_filename)

output_filename = "Cleaned_filled.csv"
output_csv = os.path.join(input_folder, output_filename)

chunk_size = 1_500_000

cols_to_fix = ['delta_start', 'handshake_duration']

print(f"Processing file: {input_csv}")

# --- Phase 1: Calculate medians ---
medians = {}
for col in cols_to_fix:
    series = pd.read_csv(input_csv, usecols=[col], low_memory=False).squeeze("columns")
    numeric_series = pd.to_numeric(series, errors='coerce')
    median_val = numeric_series.dropna().median()
    medians[col] = median_val
    print(f"Median of {col} = {median_val}")

# --- Phase 2: Process in chunks ---
first_chunk_csv = True
first_chunk_preview = True

for chunk in pd.read_csv(input_csv, chunksize=chunk_size, low_memory=False):
    # Identify invalid rows
    affected_rows = chunk[cols_to_fix[0]].astype(str).str.contains('not a complete handshake') | \
                    chunk[cols_to_fix[1]].astype(str).str.contains('not a complete handshake')

    # Create new column
    chunk.insert(loc=chunk.columns.get_loc(cols_to_fix[1]) + 1,
                 column='handshake_incomplete',
                 value=affected_rows.astype(int))

    # Fill invalid strings with median
    for col in cols_to_fix:
        chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
        chunk[col].fillna(medians[col], inplace=True)

    # --- Preview first 5 affected rows ---
    if affected_rows.any() and first_chunk_preview:
        print("\n--- First 5 affected rows (after filling) ---")
        preview_cols = cols_to_fix + ['handshake_incomplete']
        print(chunk.loc[affected_rows, preview_cols].head())
        first_chunk_preview = False

    # --- Save chunk to CSV ---
    if first_chunk_csv:
        chunk.to_csv(output_csv, index=False, mode='w')
        first_chunk_csv = False
    else:
        chunk.to_csv(output_csv, index=False, mode='a', header=False)

print(f"\nProcessing complete. Filled CSV saved as: {output_csv}")
