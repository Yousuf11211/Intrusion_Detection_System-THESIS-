import os
import pandas as pd

# Folder containing CSV files
input_folder = "Processed_Data_2017_Dask"

# Chunk size
chunk_size = 100000  # adjust as needed

# Dictionary to track seen rows (by hash)
seen_rows = set()
duplicate_count = 0

# Loop through all CSV files
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_folder, filename)
        print(f"Processing {filename}...")

        # Read file in chunks
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, dtype=str):
            # Convert each row to tuple so it can be hashed
            row_tuples = [tuple(x) for x in chunk.values]
            for row in row_tuples:
                if row in seen_rows:
                    duplicate_count += 1
                else:
                    seen_rows.add(row)

print(f"Total duplicate rows across all CSVs: {duplicate_count}")
