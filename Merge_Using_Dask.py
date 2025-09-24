import os
import warnings
import dask.dataframe as dd

# Suppress dtype warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# Parent folder containing all CSVs
parent_folder = "Raw_Data_2018"

# Output folder for merged and shuffled CSV
output_folder = "Processed_Data_2018"
os.makedirs(output_folder, exist_ok=True)

# Find all CSV files
all_files = []
total_rows_before = 0

for root, dirs, files in os.walk(parent_folder):
    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(root, file)
            all_files.append(file_path)

            # Count rows in this CSV
            try:
                row_count = 0
                duplicate_count = 0
                for chunk in dd.read_csv(file_path, assume_missing=True, dtype=str, blocksize="50MB").to_delayed():
                    # Approximate row count per chunk
                    row_count += 1  # This is approximate; exact count will come from Dask later
                total_rows_before += row_count
                print(f"{file} -> {row_count} rows (approximate, duplicates not counted at this stage)")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

print(f"\nFound {len(all_files)} CSV files.\n")

# Step 1: Load all CSVs with Dask
print("Loading all CSVs with Dask...")
df = dd.read_csv(os.path.join(parent_folder, "*.csv"), dtype=str, assume_missing=True)

# Optional: count total rows (will trigger computation)
total_rows = df.shape[0].compute()
print(f"Total rows across all CSVs: {total_rows:,}")

# Step 2: Shuffle globally
print("Shuffling all rows globally (Dask)...")
df_shuffled = df.sample(frac=1, random_state=42)

# Step 3: Save shuffled dataset to a single CSV
shuffled_file = os.path.join(output_folder, "Merged_Shuffled.csv")
df_shuffled.to_csv(shuffled_file, index=False, single_file=True)
print(f"Shuffled merged CSV saved as {shuffled_file}")

# Optional: count duplicates after shuffle
duplicates = df_shuffled.duplicated().sum().compute()
print(f"Duplicates in shuffled dataset: {duplicates:,}")
