import os
import pandas as pd

# Input and output folders
parent_folder = "Downscale_Csv_2018"
output_folder = "D_No_Constant_Column"
os.makedirs(output_folder, exist_ok=True)

# Chunk size for reading large files
chunk_size = 500000

# Manually provided columns to check
manual_columns = [
    "protocol",
    "payload_bytes_min",
    "fwd_payload_bytes_min",
    "bwd_payload_bytes_min",
    "urg_flag_counts",
    "fwd_urg_flag_counts",
    "bwd_urg_flag_counts",
    "urg_flag_percentage_in_total",
    "fwd_urg_flag_percentage_in_total",
    "bwd_urg_flag_percentage_in_total",
    "fwd_urg_flag_percentage_in_fwd_packets",
    "bwd_urg_flag_percentage_in_bwd_packets",
]


def process_csv(file_path, output_path):
    print(f"\nProcessing: {os.path.basename(file_path)}")

    # Step 1: Check manual columns
    manual_constants = {}
    available_columns = pd.read_csv(file_path, nrows=0).columns.tolist()

    for col in manual_columns:
        if col in available_columns:
            unique_vals = set()
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False, usecols=[col]):
                unique_vals.update(chunk[col].dropna().unique())
                if len(unique_vals) > 1:
                    break
            if len(unique_vals) == 1:
                manual_constants[col] = list(unique_vals)[0]
        else:
            print(f"  Column '{col}' not found in {os.path.basename(file_path)}, skipping.")

    # Step 2: Print constants found
    if manual_constants:
        print("Constant columns (from manual list):")
        for col, val in manual_constants.items():
            print(f"  {col} = {val}")
    else:
        print("No constant columns found from manual list.")

    # Step 3: Drop them and save cleaned CSV
    drop_cols = list(manual_constants.keys())
    cleaned_chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
        cleaned_chunks.append(chunk.drop(columns=drop_cols, errors="ignore"))

    cleaned_df = pd.concat(cleaned_chunks, ignore_index=True)
    cleaned_df.to_csv(output_path, index=False)
    print(f"Saved cleaned file to: {output_path}")


# Process all CSVs in parent folder
for root, dirs, files in os.walk(parent_folder):
    for file in files:
        if file.endswith(".csv"):
            input_path = os.path.join(root, file)
            output_path = os.path.join(output_folder, file)
            process_csv(input_path, output_path)

print("\nAll files processed successfully.")
