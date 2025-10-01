import pandas as pd
import numpy as np

csv_file = "Downscale_Csv_2018/Cleaned.csv"
chunk_size = 1_500_000

# Store suspicious values
string_values = {}

for chunk_idx, chunk in enumerate(pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False)):
    # Look only at object dtype columns (pandas thinks they contain strings)
    for col in chunk.select_dtypes(include=['object']).columns:
        # Try to convert to numeric
        converted = pd.to_numeric(chunk[col], errors='coerce')

        # Mask: rows that failed conversion (so they are actual strings or bad values)
        mask = converted.isna() & chunk[col].notna()

        if mask.any():
            # Initialize dict for column if not yet
            if col not in string_values:
                string_values[col] = []

            # Collect up to 20 suspicious values with row numbers
            bad_rows = chunk.loc[mask, col].head(20)
            for i, val in bad_rows.items():
                string_values[col].append((chunk_idx * chunk_size + i, val))

# --- Print results ---
if not string_values:
    print("No actual string values found, columns are safe to convert to numeric.")
else:
    print("Found string values in numeric-looking columns:")
    for col, entries in string_values.items():
        print(f"\nColumn: {col}")
        for row, val in entries:
            print(f"   Row {row}: {repr(val)}")
