import os
import warnings
import dask.dataframe as dd

# Suppress warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# Folders
parent_folder = "Raw_Data_2018"
output_folder = "Processed_Data_2018"
os.makedirs(output_folder, exist_ok=True)

# Step 1: Load all CSVs with Dask
print("Loading all CSVs with Dask...")
df = dd.read_csv(os.path.join(parent_folder, "*.csv"), dtype=str, assume_missing=True)

# Step 2: Optional - check total rows (triggers computation)
total_rows = df.shape[0].compute()
print(f"Total rows across all CSVs: {total_rows:,}")

# Step 3: Shuffle globally
print("Shuffling all rows globally (Dask)...")
df_shuffled = df.sample(frac=1, random_state=42)

# Step 3b: Repartition to ~20 partitions → ~20 CSV files
print("Repartitioning to 20 partitions for memory-efficient CSV output...")
df_shuffled = df_shuffled.repartition(npartitions=20)

# Step 4: Write shuffled data to partitioned CSVs
shuffled_file_pattern = os.path.join(output_folder, "Merged_Shuffled_*.csv")
print(f"Writing shuffled data to multiple CSVs: {shuffled_file_pattern}")
df_shuffled.to_csv(shuffled_file_pattern, index=False)

print("Shuffling and writing completed successfully.")
