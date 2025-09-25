import os
import pandas as pd

# Input folder (will overwrite CSVs here)
input_folder = "Processed_Data_2017_Dask"

# Chunk size
chunk_size = 100000  # adjust based on memory

# Process each CSV file
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        input_path = os.path.join(input_folder, filename)
        temp_path = os.path.join(input_folder, f"temp_{filename}")
        print(f"Processing {filename}...")

        seen_rows = set()

        with open(temp_path, "w", encoding="utf-8") as out_file:
            first_chunk = True

            for chunk in pd.read_csv(input_path, chunksize=chunk_size, dtype=str):
                row_tuples = [tuple(x) for x in chunk.values]

                unique_rows = []
                for row in row_tuples:
                    if row not in seen_rows:
                        seen_rows.add(row)
                        unique_rows.append(row)

                if unique_rows:
                    df_unique = pd.DataFrame(unique_rows, columns=chunk.columns)
                    df_unique.to_csv(out_file, index=False, header=first_chunk, mode="a")
                    first_chunk = False

        # Replace original file with the cleaned file
        os.replace(temp_path, input_path)

print("Duplicate removal completed. Original CSVs overwritten.")
