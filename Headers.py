import os
import pandas as pd

# Main folder with all raw datasets
main_folder = "Raw_Data"

# Output file where we'll save headers
output_file = "all_headers.csv"

# Store rows here
rows = []

for root, dirs, files in os.walk(main_folder):
    for file in files:
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(root, file)
        print(f"Reading headers from {file_path} ...")

        try:
            # Read only headers (no data)
            df = pd.read_csv(file_path, nrows=0)
            headers = list(df.columns)

            # Build one row: filename + headers
            row = [file] + headers
            rows.append(row)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

# Find the max number of headers across all files
max_len = max(len(r) for r in rows)

# Pad rows with empty strings so all have the same length
rows_padded = [r + [""] * (max_len - len(r)) for r in rows]

# Create DataFrame
columns = ["filename"] + [f"title_{i}" for i in range(1, max_len)]
headers_df = pd.DataFrame(rows_padded, columns=columns)

# Save to CSV
headers_df.to_csv(output_file, index=False)
print(f"\nSaved all headers (row-wise) to {output_file}")
