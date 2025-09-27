import os
import warnings
import dask.dataframe as dd

# Suppress warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# Folders
parent_folder = "Raw_Data_2017"
output_folder = "Processed_Data_2011_Dask"
os.makedirs(output_folder, exist_ok=True)

# Step 1: Load all CSVs with Dask
print("Loading all CSVs with Dask...")
df = dd.read_csv(os.path.join(parent_folder, "*.csv"), dtype=str, assume_missing=True)

# Step 2: Optional - get total rows
total_rows = df.shape[0].compute()
print(f"Total rows across all CSVs: {total_rows:,}")

# Step 3: Add random column for shuffle
print("Shuffling all rows globally (Dask)...")
df = df.assign(_rand=df.map_partitions(lambda d: d.index.to_series().rank().astype("float")))
df_shuffled = df.set_index("_rand", shuffle="tasks", drop=True)

# Step 4: Write shuffled data to multiple CSV partitions
shuffled_file_pattern = os.path.join(output_folder, "Merged_Shuffled_*.csv")
print(f"Writing shuffled data to multiple CSVs: {shuffled_file_pattern}")
df_shuffled.to_csv(shuffled_file_pattern, index=False)

# Step 5: Concatenate all partitioned CSVs into a single file
final_file = os.path.join(output_folder, "Merged_Shuffled_All.csv")
print(f"Concatenating all partitioned CSVs into {final_file} ...")

with open(final_file, "w", encoding="utf-8") as outfile:
    first_file = True
    for fname in sorted(os.listdir(output_folder)):
        if fname.startswith("Merged_Shuffled_") and fname.endswith(".csv"):
            file_path = os.path.join(output_folder, fname)
            with open(file_path, "r", encoding="utf-8") as infile:
                if first_file:
                    # Write header + data
                    outfile.write(infile.read())
                    first_file = False
                else:
                    # Skip header (first line)
                    next(infile)
                    outfile.write(infile.read())

print(f"Final single shuffled CSV saved as {final_file}")
