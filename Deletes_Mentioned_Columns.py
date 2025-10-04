import os
import pandas as pd

# --- Step 1: Set your folder path and the columns you want to remove ---

# The folder where your original CSV files are located.
# Example for Windows: "C:\\Users\\YourUser\\Desktop\\MyData"
# Example for Mac/Linux: "/home/youruser/Desktop/MyData"
input_folder = "Downscale_Csv_2018"

# A list of the column names you want to remove.
# The names must be an exact match.
columns_to_remove = [
    'flow_id',
    'src_ip',
    'dst_ip',
    'timestamp',
    # Add any other columns you want to remove here
]

# --- Step 2: The rest of the script processes the files ---

# Set a chunk size for reading large files to manage memory usage
chunk_size = 100_0000

print(f"Searching for CSV files in: {input_folder}\n")

# Loop through every file in the specified folder
for filename in os.listdir(input_folder):
    # Check if the file is a CSV and not one we've already created
    if filename.endswith('.csv') and not filename.endswith('_columns_removed.csv'):

        input_csv_path = os.path.join(input_folder, filename)

        # Create the new filename for the output file
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}_columns_removed.csv"
        output_csv_path = os.path.join(input_folder, output_filename)

        print(f"Processing '{filename}'...")

        # A flag to ensure we write the header only for the first chunk
        is_first_chunk = True

        # Read the input CSV in manageable chunks
        for chunk in pd.read_csv(input_csv_path, chunksize=chunk_size, low_memory=False):

            # Drop the specified columns from the chunk
            # 'errors='ignore'' prevents a crash if a column doesn't exist
            chunk.drop(columns=columns_to_remove, inplace=True, errors='ignore')

            # If this is the first chunk, create a new file and write the header
            if is_first_chunk:
                chunk.to_csv(output_csv_path, index=False, mode='w')
                is_first_chunk = False
            # For all other chunks, append to the existing file without the header
            else:
                chunk.to_csv(output_csv_path, index=False, mode='a', header=False)

        print(f"----> Successfully created '{output_filename}'\n")

print("All files processed.")