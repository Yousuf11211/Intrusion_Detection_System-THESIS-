import os
import pandas as pd
from collections import defaultdict

# --- 1. Configuration ---
# Set the folder where your original CSV files are located.
PARENT_FOLDER = "Downscale_Csv_2018"

# Set the folder where the new, cleaned CSV files will be saved.
OUTPUT_FOLDER = "D_Cleaned_CSVs"

# Define the number of rows to read at a time. Adjust this based on your
# computer's RAM. A smaller number uses less memory.
CHUNK_SIZE = 1_000_000


# --- 2. Helper Functions ---

def get_user_yes_no(prompt):
    """A simple function to get a 'yes' or 'no' answer from the user."""
    while True:
        response = input(f"{prompt} (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


# --- 3. Main Processing Function ---

def analyze_and_clean_csv(file_path, output_path):
    """
    Analyzes a CSV for constant/low-variance columns and optionally removes them.
    """
    print("-" * 70)
    print(f"Processing file: {os.path.basename(file_path)}")

    try:
        # Step A: Analyze the file to find unique values for all columns.
        # This is done first and only once per file for efficiency.
        print("  Analyzing columns... (this may take a moment for large files)")

        # defaultdict(set) is a special dictionary. If you try to access a key
        # that doesn't exist, it automatically creates an empty set for it.
        col_unique_values = defaultdict(set)

        # We read the CSV in chunks to avoid loading the entire large file into memory.
        # dtype=str tells pandas to treat all data as text, which is faster and
        # prevents errors from mixed data types in a column.
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, dtype=str, low_memory=False):
            for col in chunk.columns:
                # .dropna() removes any missing values (NaNs) before finding unique ones.
                # .update() adds all unique items from the chunk to our master set for that column.
                col_unique_values[col].update(chunk[col].dropna().unique())
        print("  Analysis complete.")

        # Step B: Ask the user what to look for and report the findings.
        columns_to_drop = []

        # Check for constant columns if the user says 'y'.
        if get_user_yes_no("  Do you want to find constant columns?"):
            # This is a dictionary comprehension. It builds a dictionary by looping through
            # col_unique_values and keeping only the columns where the number of unique values is 1.
            constant_cols = {col: list(vals)[0] for col, vals in col_unique_values.items() if len(vals) == 1}

            if constant_cols:
                print("\n  [RESULT] Found Constant Columns:")
                for col, val in constant_cols.items():
                    print(f"    - Column '{col}' has only one value: {val}")
                columns_to_drop.extend(constant_cols.keys())
            else:
                print("\n  [RESULT] No constant columns were found.")

        # Check for low-variance columns if the user says 'y'.
        if get_user_yes_no("  Do you want to find low-variance columns?"):
            while True:  # Loop until a valid number is entered.
                try:
                    threshold = int(input("    Enter the maximum number of unique values (e.g., 3): "))
                    break
                except ValueError:
                    print("    That wasn't a valid number. Please enter an integer.")

            # Find columns with a unique value count between 2 and the threshold.
            low_variance_cols = {col: list(vals) for col, vals in col_unique_values.items()
                                 if 2 <= len(vals) <= threshold}

            if low_variance_cols:
                print(f"\n  [RESULT] Found Low-Variance Columns (up to {threshold} unique values):")
                for col, vals in low_variance_cols.items():
                    print(f"    - Column '{col}' has values: {vals}")
                # Add columns to the drop list, avoiding duplicates.
                new_cols_to_add = [col for col in low_variance_cols if col not in columns_to_drop]
                columns_to_drop.extend(new_cols_to_add)
            else:
                print(f"\n  [RESULT] No low-variance columns found with the specified threshold.")

        # Step C: If we found columns to drop, ask the user for confirmation to delete.
        if not columns_to_drop:
            print("\nNo columns were selected for removal. Moving to the next file.")
            return

        # Use set() to get a unique list of columns to remove, then sort it for clean printing.
        final_drop_list = sorted(list(set(columns_to_drop)))
        print("\nColumns identified for removal:", final_drop_list)

        if get_user_yes_no("Do you want to remove these columns and save a new, cleaned file?"):
            print(f"  Removing {len(final_drop_list)} columns and saving new file...")

            # Create another iterator to read the file chunk-by-chunk for writing.
            chunk_iterator = pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False)

            # Write the first chunk to the new file with headers.
            first_chunk = next(chunk_iterator)
            first_chunk.drop(columns=final_drop_list, errors="ignore").to_csv(output_path, index=False)

            # Loop through the rest of the chunks and append them to the same file without headers.
            for chunk in chunk_iterator:
                chunk.drop(columns=final_drop_list, errors="ignore").to_csv(output_path, mode='a', header=False,
                                                                            index=False)

            print(f"  Successfully saved cleaned file to: {output_path}")
        else:
            print("  Skipping file modification as requested.")

    except Exception as e:
        print(f"ERROR: An unexpected error occurred while processing {os.path.basename(file_path)}.")
        print(f"DETAILS: {e}")


# --- 4. Script Execution ---

# This standard Python construct ensures the code inside only runs when
# the script is executed directly (not when imported as a module).
if __name__ == "__main__":
    # Create the output directory if it doesn't already exist.
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # os.walk goes through all folders and files in the parent directory.
    for root, dirs, files in os.walk(PARENT_FOLDER):
        for file in files:
            if file.endswith(".csv"):
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(OUTPUT_FOLDER, file)
                analyze_and_clean_csv(input_file_path, output_file_path)

    print("\n" + "-" * 70)
    print("All files have been processed.")