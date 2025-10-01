import os
import pandas as pd
import numpy as np

# --- 1. Configuration ---
# Set the main folder path. The script will search inside this folder and all its subfolders.
# Example for Windows: "C:\\Users\\YourUser\\Desktop\\MyData"
# Example for Mac/Linux: "/home/youruser/Desktop/MyData"
input_folder = "Dowmscale_Csv_2018"

# Set the threshold for deleting a column (e.g., 0.30 for 30%)
threshold = 0.30

# Set a chunk size for reading large files to manage memory
chunk_size = 2_000_000

# --- 2. Main Script ---
print(f"Starting recursive analysis in folder: {input_folder}")
print(f"Threshold for deletion is set to {threshold:.0%}\n")

# Use os.walk() to go through the main folder and all its subfolders
for dirpath, _, filenames in os.walk(input_folder):
    for filename in filenames:
        # Process only if it's a CSV and not a file we've already cleaned
        if filename.endswith('.csv') and not filename.endswith('_cleaned.csv'):

            input_csv_path = os.path.join(dirpath, filename)
            print(f"--- Processing file: {input_csv_path} ---")

            # --- Pass 1: Analyze the file to find problematic columns ---
            print("Phase 1: Analyzing columns for 'inf' values...")

            inf_counts = pd.Series(dtype=int)
            total_rows = 0

            try:
                # Read file in chunks to get 'inf' counts
                for chunk in pd.read_csv(input_csv_path, chunksize=chunk_size, low_memory=False):
                    total_rows += len(chunk)
                    # Efficiently count 'inf' values in all columns of the chunk
                    inf_in_chunk = chunk.apply(pd.to_numeric, errors='coerce').pipe(np.isinf).sum()
                    inf_counts = inf_counts.add(inf_in_chunk, fill_value=0)

                # Calculate the percentage and find columns to delete
                inf_percentages = inf_counts / total_rows
                columns_to_delete = inf_percentages[inf_percentages > threshold].index.tolist()

            except Exception as e:
                print(f"An error occurred while analyzing '{filename}': {e}")
                continue  # Skip to the next file

            # --- Report findings and ask for confirmation ---
            if not columns_to_delete:
                print("Result: No columns exceeded the threshold. No action needed.\n")
                continue

            print(
                f"\nAnalysis complete. Found {len(columns_to_delete)} columns with more than {threshold:.0%} 'inf' values:")
            for col in columns_to_delete:
                print(f"  - '{col}' ({inf_percentages[col]:.2%} inf)")

            print("-" * 20)

            # Ask for user confirmation
            user_input = input("Are you sure you want to permanently delete these columns? (yes/no): ").lower()

            # --- Pass 2: Delete the columns if confirmed ---
            if user_input in ['yes', 'y']:
                print("\nPhase 2: Deleting columns and creating new file...")

                base_name = os.path.splitext(filename)[0]
                output_filename = f"{base_name}_cleaned.csv"
                output_csv_path = os.path.join(dirpath, output_filename)  # Save in the same directory

                is_first_chunk = True

                try:
                    # Re-read the file in chunks to perform the deletion
                    for chunk in pd.read_csv(input_csv_path, chunksize=chunk_size, low_memory=False):
                        # Drop the identified columns
                        chunk.drop(columns=columns_to_delete, inplace=True, errors='ignore')

                        # Write to a new CSV file
                        if is_first_chunk:
                            chunk.to_csv(output_csv_path, index=False, mode='w')
                            is_first_chunk = False
                        else:
                            chunk.to_csv(output_csv_path, index=False, mode='a', header=False)

                    print(f"----> Successfully created '{output_filename}' in '{dirpath}'\n")

                except Exception as e:
                    print(f"An error occurred while creating the new file: {e}\n")

            else:
                print("Operation cancelled. No columns were deleted.\n")


print("All files processed.")
