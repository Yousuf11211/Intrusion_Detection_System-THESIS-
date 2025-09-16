import os
import pandas as pd

# Main folder with all raw datasets
main_folder = "Raw_Data"

# Output file where we’ll save all headers
output_file = "all_headers.csv"

# Store results here
all_headers = []

for root, dirs, files in os.walk(main_folder):
    for file in files:
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(root, file)
        print(f"Reading headers from {file_path} ...")

        try:
            # Only read the header (first row) instead of full file
            df = pd.read_csv(file_path, nrows=0)
            headers = list(df.columns)

            # Add each header with filename
            for col in headers:
                all_headers.append({"filename": file, "column": col})

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

# Convert list to DataFrame
headers_df = pd.DataFrame(all_headers)

# Save to CSV
headers_df.to_csv(output_file, index=False)
print(f"\nSaved all headers to {output_file}")
