import os
import pandas as pd
import numpy as np

# Parent folder containing all CSVs
parent_folder = "Raw_Data_2017"

# Output folder for merged and shuffled CSV
output_folder = "Processed_Data_2017"
os.makedirs(output_folder, exist_ok=True)

# Temporary list to store file paths
all_files = []

# Count total rows before merging
total_rows_before = 0

for root, dirs, files in os.walk(parent_folder):
    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(root, file)
            all_files.append(file_path)

            # Count rows in this CSV
            try:
                for chunk in pd.read_csv(file_path, chunksize=500000):
                    total_rows_before += len(chunk)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

print(f"Found {len(all_files)} CSV files with a total of {total_rows_before} rows before merging.")

# Step 1: Merge all CSVs in chunks, append to a new file
merged_file = os.path.join(output_folder, "Merged.csv")
total_rows_merged = 0  # counter for merged file

for i, file_path in enumerate(all_files):
    print(f"Merging file {i + 1}/{len(all_files)}: {file_path}")
    try:
        for chunk in pd.read_csv(file_path, chunksize=500000):
            if not os.path.exists(merged_file):
                chunk.to_csv(merged_file, mode="w", index=False)
            else:
                chunk.to_csv(merged_file, mode="a", index=False, header=False)
            total_rows_merged += len(chunk)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        continue

print(f"Merging completed. Merged file saved as {merged_file}")
print(f"Total rows in merged file: {total_rows_merged}")

# Step 2: Shuffle the merged CSV
shuffled_file = os.path.join(output_folder, "Merged_Shuffled.csv")

chunks = []
for chunk in pd.read_csv(merged_file, chunksize=100000):
    chunk["_rand"] = np.random.rand(len(chunk))
    chunks.append(chunk)

df = pd.concat(chunks)
df = df.sample(frac=1, random_state=42)  # shuffle
df.drop(columns=["_rand"], inplace=True)

df.to_csv(shuffled_file, index=False)
print(f"Shuffled merged CSV saved as {shuffled_file}")
print(f"Total rows in shuffled file: {len(df)}")
