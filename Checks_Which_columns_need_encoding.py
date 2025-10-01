import pandas as pd
import random

csv_file = "Downscale_Csv_2018/Cleaned.csv"
chunk_size = 1_500_000

# Dictionary to hold string values with row indices
string_values = {}

row_offset = 0  # keeps track of row numbers across chunks

for chunk in pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False):
    cat_in_chunk = chunk.select_dtypes(include=['object']).columns.tolist()

    for col in cat_in_chunk:
        if col not in string_values:
            string_values[col] = []

        # Find string values and their row indices (relative to whole file)
        for idx, val in chunk[col].dropna().items():
            if isinstance(val, str):
                if len(string_values[col]) < 10:  # only collect first 10
                    string_values[col].append((idx + row_offset, val))

    row_offset += len(chunk)  # move row index base for next chunk

# Print results
print("Sample string values with row numbers:\n")
for col, samples in string_values.items():
    print(f"- Column: {col}")
    for row_num, value in samples:
        print(f"   Row {row_num}: {value}")
