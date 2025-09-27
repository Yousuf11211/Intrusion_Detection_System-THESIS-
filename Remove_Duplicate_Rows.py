import os
import pandas as pd

# Input and output folders
input_folder = "Processed_Data_2017_Dask"
output_folder = "Processed_Data_2017_Dask1"
os.makedirs(output_folder, exist_ok=True)

# Chunk size
chunk_size = 100000  # adjust based on your memory

# Process each CSV file
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        print(f"Processing {filename}...")

        # Initialize set to track seen rows
        seen_rows = set()

        # Open output file for writing
        with open(output_path, "w", encoding="utf-8") as out_file:
            first_chunk = True

            # Read input CSV in chunks
            for chunk in pd.read_csv(input_path, chunksize=chunk_size, dtype=str):
                # Convert each row to tuple to check for duplicates
                row_tuples = [tuple(x) for x in chunk.values]

                # Filter out duplicates
                unique_rows = []
                for row in row_tuples:
                    if row not in seen_rows:
                        seen_rows.add(row)
                        unique_rows.append(row)

                # Write unique rows to output CSV
                if unique_rows:
                    df_unique = pd.DataFrame(unique_rows, columns=chunk.columns)
                    df_unique.to_csv(out_file, index=False, header=first_chunk, mode="a")
                    first_chunk = False

print("Duplicate removal completed. Clean files saved in:", output_folder)
