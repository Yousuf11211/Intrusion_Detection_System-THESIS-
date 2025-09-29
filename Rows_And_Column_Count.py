import os
import pandas as pd

# Folder containing CSV files
folder_path = "Processed_Data_2017_Dask1"

# Initialize totals
total_rows = 0
total_columns = None  # will set after reading first file

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)

        file_rows = 0
        file_columns = None

        # Read in chunks
        chunk_size = 100000
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
            file_rows += len(chunk)
            if file_columns is None:
                file_columns = len(chunk.columns)

        total_rows += file_rows
        if total_columns is None:
            total_columns = file_columns

        print(f"{filename}: Rows = {file_rows}, Columns = {file_columns}")

print("\n--- Summary ---")
print(f"Total rows across all CSV files: {total_rows}")
print(f"Number of columns (consistent across files): {total_columns}")
