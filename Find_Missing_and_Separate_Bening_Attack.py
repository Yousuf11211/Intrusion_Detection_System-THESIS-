import os
import pandas as pd
from collections import defaultdict

# --- 1. Global Configuration ---
# Use this section to easily configure the script's behavior.

# The folder containing your original, unsorted CSV files.
INPUT_FOLDER = "Raw_Data_2018"

# The main folder where all cleaned and separated files will be saved.
OUTPUT_FOLDER = "Cleaned_Separated_Data"

# The number of rows to process at a time (adjust based on your RAM).
CHUNK_SIZE = 500_000

# The default maximum number of rows for ATTACK files before splitting.
ATTACK_ROWS_PER_FILE = 1_000_000

# The name of the column that contains the labels (e.g., 'Label'). This is case-sensitive!
LABEL_COLUMN = 'Label'

# The value representing the "normal" or "benign" traffic.
BENIGN_LABEL_VALUE = 'BENIGN'


# --- 2. Core Functions ---

def analyze_file(file_path):
    """
    Reads a CSV file in chunks to analyze its contents. (Pass 1)

    This function counts the total rows for each label and also counts how
    many of those rows have missing values.

    Returns:
        A tuple containing two dictionaries:
        - total_counts_by_label: {'LABEL': count, ...}
        - missing_counts_by_label: {'LABEL': count, ...}
    """
    print(f"\nAnalyzing file: {os.path.basename(file_path)}...")

    # defaultdict is a special dictionary. If you access a key that doesn't
    # exist, it automatically creates it with a default value (0 for int).
    total_counts_by_label = defaultdict(int)
    missing_counts_by_label = defaultdict(int)

    try:
        # First, check if the label column even exists to avoid errors.
        header_df = pd.read_csv(file_path, nrows=0, low_memory=False)
        if LABEL_COLUMN not in header_df.columns:
            print(f"  ERROR: Label column '{LABEL_COLUMN}' not found. Skipping file.")
            return None, None  # Return None to indicate an error

        # Read the large CSV file in smaller, memory-safe chunks.
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False):
            # Count total rows for each label in this chunk.
            total_counts = chunk[LABEL_COLUMN].value_counts()
            for label, count in total_counts.items():
                total_counts_by_label[label] += count

            # Find rows that have at least one empty cell.
            rows_with_missing = chunk[chunk.isnull().any(axis=1)]
            if not rows_with_missing.empty:
                # Count how many of those incomplete rows belong to each label.
                missing_counts = rows_with_missing[LABEL_COLUMN].value_counts()
                for label, count in missing_counts.items():
                    missing_counts_by_label[label] += count

        print("  Analysis complete.")
        return total_counts_by_label, missing_counts_by_label

    except Exception as e:
        print(f"  ERROR during analysis: {e}. Skipping this file.")
        return None, None


def get_user_instructions(total_counts, missing_counts):
    """
    Shows the analysis reports to the user and asks for instructions
    on how to proceed with cleaning and saving the data.
    """
    # Report 1: Total Row Counts
    print("\n--- Total Row Count Report ---")
    for label, count in sorted(total_counts.items()):
        print(f"  - {label}: {count:,} total rows.")
    print("------------------------------")

    # Prompt 1: Get the row limit for Benign files
    benign_rows_per_file = 0
    if BENIGN_LABEL_VALUE in total_counts:
        while True:  # Loop until we get a valid number
            try:
                prompt = f"\nEnter the max rows per Benign file (press Enter for default: {ATTACK_ROWS_PER_FILE:,}): "
                user_input = input(prompt).strip()
                if not user_input:  # User pressed Enter
                    benign_rows_per_file = ATTACK_ROWS_PER_FILE
                    break
                val = int(user_input)
                if val > 0:
                    benign_rows_per_file = val
                    break
                else:
                    print("  Please enter a positive number.")
            except ValueError:
                print("  Invalid input. Please enter a whole number.")

    # Prompt 2: Get which labels to clean
    labels_to_delete = set()
    if not missing_counts:
        print("\nThis file has no rows with missing values. No cleaning is needed.")
    else:
        print("\n--- Missing Value Report ---")
        for label, count in sorted(missing_counts.items()):
            print(f"  - {label}: {count:,} rows have missing values.")
        print("----------------------------")

        user_input = input(
            "Enter labels to clean (e.g., BENIGN,DoS), or type 'all' or 'none'.\n> "
        ).strip()

        if user_input.lower() == 'all':
            labels_to_delete = set(missing_counts.keys())
        elif user_input.lower() not in ['none', '']:
            labels_to_delete = {label.strip() for label in user_input.split(',')}

    return labels_to_delete, benign_rows_per_file


def process_and_save_file(file_path, labels_to_delete, benign_rows_per_file):
    """
    Reads the CSV file again, cleans it based on user instructions,
    separates it into Benign/Attack, and saves the output in chunks. (Pass 2)
    """
    if labels_to_delete:
        print(f"\nStarting cleanup for labels: {', '.join(labels_to_delete)}")
    else:
        print("\nStarting separation process (no cleaning requested).")

    # --- Setup for saving ---
    benign_output_path = os.path.join(OUTPUT_FOLDER, "Benign")
    attacks_output_path = os.path.join(OUTPUT_FOLDER, "Attacks")
    os.makedirs(benign_output_path, exist_ok=True)
    os.makedirs(attacks_output_path, exist_ok=True)

    # Buffers hold DataFrames in memory until they are large enough to save.
    benign_buffer = []
    attack_buffers = defaultdict(list)

    # Counters keep track of file parts (e.g., part_1, part_2).
    benign_part_count = 1
    attack_part_counts = defaultdict(lambda: 1)

    try:
        # --- Main processing loop ---
        for chunk_num, chunk in enumerate(pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False)):
            print(f"  Processing chunk {chunk_num + 1}...")

            # 1. Clean: Remove rows with missing values if they match the user's selection.
            chunk_cleaned = chunk
            if labels_to_delete:
                rows_to_drop_mask = (chunk[LABEL_COLUMN].isin(labels_to_delete)) & (chunk.isnull().any(axis=1))
                chunk_cleaned = chunk[~rows_to_drop_mask]

            # 2. Separate: Split the cleaned data into Benign and Attack categories.
            benign_data = chunk_cleaned[chunk_cleaned[LABEL_COLUMN] == BENIGN_LABEL_VALUE]
            attack_data = chunk_cleaned[chunk_cleaned[LABEL_COLUMN] != BENIGN_LABEL_VALUE]

            # 3. Buffer: Add the separated data to our temporary holding lists.
            if not benign_data.empty:
                benign_buffer.append(benign_data)
            if not attack_data.empty:
                for label, group in attack_data.groupby(LABEL_COLUMN):
                    attack_buffers[label].append(group)

            # 4. Save (if buffers are full):
            # Check Benign buffer against the user-defined size.
            if benign_rows_per_file > 0 and sum(len(df) for df in benign_buffer) >= benign_rows_per_file:
                df_to_save = pd.concat(benign_buffer, ignore_index=True)
                filename = os.path.join(benign_output_path, f"Benign_part_{benign_part_count}.csv")
                df_to_save.to_csv(filename, index=False)
                print(f"    Saved {len(df_to_save):,} benign rows to {os.path.basename(filename)}")
                benign_buffer.clear()
                benign_part_count += 1

            # Check each Attack buffer against the default size.
            for label, buffer_list in list(attack_buffers.items()):
                if sum(len(df) for df in buffer_list) >= ATTACK_ROWS_PER_FILE:
                    df_to_save = pd.concat(buffer_list, ignore_index=True)
                    safe_name = "".join(c for c in label if c.isalnum() or c in ('-', '_'))
                    filename = os.path.join(attacks_output_path, f"{safe_name}_part_{attack_part_counts[label]}.csv")
                    df_to_save.to_csv(filename, index=False)
                    print(f"    Saved {len(df_to_save):,} '{label}' rows to {os.path.basename(filename)}")
                    attack_buffers[label].clear()
                    attack_part_counts[label] += 1

        # --- Final save ---
        # After the loop, save any data remaining in the buffers.
        print("\n  Saving remaining data...")
        if benign_buffer:
            df_to_save = pd.concat(benign_buffer, ignore_index=True)
            filename = os.path.join(benign_output_path, f"Benign_part_{benign_part_count}.csv")
            df_to_save.to_csv(filename, index=False)
            print(f"    Saved {len(df_to_save):,} benign rows to {os.path.basename(filename)}")

        for label, buffer_list in attack_buffers.items():
            if buffer_list:
                df_to_save = pd.concat(buffer_list, ignore_index=True)
                safe_name = "".join(c for c in label if c.isalnum() or c in ('-', '_'))
                filename = os.path.join(attacks_output_path, f"{safe_name}_part_{attack_part_counts[label]}.csv")
                df_to_save.to_csv(filename, index=False)
                print(f"    Saved {len(df_to_save):,} '{label}' rows to {os.path.basename(filename)}")

        print(f"\nFinished processing {os.path.basename(file_path)}.")

    except Exception as e:
        print(f"  FATAL ERROR during processing: {e}.")


def main():
    """
    The main function to orchestrate the entire process.
    """
    print("Starting the CSV Cleaning and Separation Process...")

    # Find all CSV files to be processed.
    all_csv_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(INPUT_FOLDER)
        for file in files if file.endswith(".csv")
    ]

    if not all_csv_files:
        print(f"No CSV files found in '{INPUT_FOLDER}'. Exiting.")
        return

    # Process each file one by one.
    for file_path in all_csv_files:
        print("=" * 80)

        # Pass 1: Analyze the file
        total_counts, missing_counts = analyze_file(file_path)
        if total_counts is None:  # An error occurred in analysis
            continue

        # Get user instructions based on the analysis
        labels_to_delete, benign_rows = get_user_instructions(total_counts, missing_counts)

        # Pass 2: Process and save the file according to instructions
        process_and_save_file(file_path, labels_to_delete, benign_rows)

    print("\n" + "=" * 80)
    print("All files have been processed successfully!")
    print("=" * 80)


# --- 3. Run the script ---
# This part ensures the main() function is called only when you run the script directly.
if __name__ == "__main__":
    main()