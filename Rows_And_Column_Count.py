import os
import pandas as pd

# Folder containing CSV files
folder_path = "Processed_Data_2017_Dask1"

# Initialize counters
total_rows = 0
total_columns = None  # will set after reading first file

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)

        # Get column count (just from header)
        if total_columns is None:
            total_columns = len(pd.read_csv(file_path, nrows=0).columns)

        # Count rows in chunks
        chunk_size = 100000  # adjust based on memory
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            total_rows += len(chunk)

print(f"Total rows across all CSV files: {total_rows}")
print(f"Number of columns: {total_columns}")
