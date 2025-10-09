import os
import pandas as pd
from collections import defaultdict
import numpy as np

# --- 1. Global Configuration ---
INPUT_FOLDER = "Raw_Data_2018"
OUTPUT_FOLDER = "Cleaned_Shuffled_Data_V2"
CHUNK_SIZE = 500_000
DEFAULT_ROWS_PER_FILE = 1_000_000
LABEL_COLUMN = 'Label'  # Case-insensitive
BENIGN_LABEL_VALUE = 'BENIGN'


# --- 2. Core Functions ---

def analyze_all_files(all_files):
    """
    Analyzes all CSV files to get aggregated counts for a comprehensive report.
    This combines the analysis from your original file-by-file loop.
    """
    print("--- Phase 1: Analyzing all source files ---")
    grand_total_counts = defaultdict(int)
    grand_missing_counts = defaultdict(int)
    first_file_label_col = None

    for file_path in all_files:
        print(f"  Scanning: {os.path.basename(file_path)}...")
        actual_label_col_name = None
        try:
            # This is your original case-insensitive column finding logic
            header_df = pd.read_csv(file_path, nrows=0, low_memory=False)
            for col_name in header_df.columns:
                if col_name.lower() == LABEL_COLUMN.lower():
                    actual_label_col_name = col_name
                    if first_file_label_col is None:
                        first_file_label_col = actual_label_col_name
                    break

            if not actual_label_col_name:
                print(f"    Warning: Label column '{LABEL_COLUMN}' not found. Skipping.")
                continue

            # This is your original counting logic for total and missing rows
            for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False):
                total_counts = chunk[actual_label_col_name].value_counts()
                for label, count in total_counts.items():
                    grand_total_counts[label] += count
                rows_with_missing = chunk[chunk.isnull().any(axis=1)]
                if not rows_with_missing.empty:
                    missing_counts = rows_with_missing[actual_label_col_name].value_counts()
                    for label, count in missing_counts.items():
                        grand_missing_counts[label] += count
        except Exception as e:
            print(f"    Error analyzing {os.path.basename(file_path)}: {e}")

    print("--- Analysis complete ---")
    return grand_total_counts, grand_missing_counts, first_file_label_col


def get_user_instructions(total_counts, missing_counts):
    """
    This function contains ALL of your original interactive prompts, plus the new shuffle prompt.
    """
    print("\n--- Total Row Count Report (from all files) ---")
    for label, count in sorted(total_counts.items()):
        print(f"  - {label}: {count:,} total rows.")
    print("-------------------------------------------------")

    # Prompt 1: Benign row limit (Original logic)
    benign_rows_per_file = 0
    if BENIGN_LABEL_VALUE in total_counts:
        while True:
            try:
                prompt = f"\nEnter max rows per Benign file (default: {DEFAULT_ROWS_PER_FILE:,}): "
                user_input = input(prompt).strip()
                benign_rows_per_file = int(user_input) if user_input else DEFAULT_ROWS_PER_FILE
                if benign_rows_per_file > 0:
                    break
                else:
                    print("  Please enter a positive number.")
            except ValueError:
                print("  Invalid input. Please enter a whole number.")

    # Prompt 2: Attack row limit (Original logic)
    attack_rows_per_file = 0
    if any(label != BENIGN_LABEL_VALUE for label in total_counts):
        while True:
            try:
                prompt = f"Enter max rows per Attack file (default: {DEFAULT_ROWS_PER_FILE:,}): "
                user_input = input(prompt).strip()
                attack_rows_per_file = int(user_input) if user_input else DEFAULT_ROWS_PER_FILE
                if attack_rows_per_file > 0:
                    break
                else:
                    print("  Please enter a positive number.")
            except ValueError:
                print("  Invalid input. Please enter a whole number.")

    # New Prompt: Shuffle option
    should_shuffle = input("Do you want to shuffle the final output files? (y/n): ").strip().lower() in ['y', 'yes']

    # Prompt 3: Cleaning rules (Original logic)
    labels_to_delete = set()
    if not missing_counts:
        print("\nNo rows with missing values found across all files. No cleaning needed.")
    else:
        print("\n--- Missing Value Report (from all files) ---")
        for label, count in sorted(missing_counts.items()):
            print(f"  - {label}: {count:,} rows have missing values.")
        print("---------------------------------------------")
        user_input = input("Enter labels to clean (e.g., BENIGN,DoS), or 'all' or 'none'.\n> ").strip()
        if user_input.lower() == 'all':
            labels_to_delete = set(missing_counts.keys())
        elif user_input.lower() not in ['none', '']:
            labels_to_delete = {label.strip() for label in user_input.split(',')}

    # Prompt 4 & 5: Advanced separation (Original logic)
    separate_by_missing_status = False
    separation_scope = 'none'
    print("-" * 30)
    if input("Separate output into 'NoMissing' and 'Missing' folders? (y/n): ").strip().lower() in ['y', 'yes']:
        separate_by_missing_status = True
        while True:
            scope_input = input("Apply to [B]enign, [A]ttacks, or [Bo]th?: ").strip().lower()
            if scope_input in ['b', 'benign']:
                separation_scope = 'benign'; break
            elif scope_input in ['a', 'attacks']:
                separation_scope = 'attacks'; break
            elif scope_input in ['bo', 'both']:
                separation_scope = 'both'; break
            else:
                print("Invalid input. Please enter 'b', 'a', or 'bo'.")

    return labels_to_delete, benign_rows_per_file, attack_rows_per_file, separate_by_missing_status, separation_scope, should_shuffle


def process_all_files(all_files, actual_label_col, instructions):
    """
    Pools, cleans, shuffles, and saves data from all files based on user instructions.
    """
    print("\n--- Phase 2: Pooling and Cleaning Data ---")
    labels_to_delete, benign_rows_per_file, attack_rows_per_file, separate_by_missing, scope, should_shuffle = instructions

    data_pools = defaultdict(lambda: defaultdict(list))

    for file_path in all_files:
        print(f"  Processing {os.path.basename(file_path)}...")
        try:
            for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False):
                # This is your original cleaning logic, applied during the pooling stage
                if labels_to_delete:
                    rows_to_drop_mask = (chunk[actual_label_col].isin(labels_to_delete)) & (chunk.isnull().any(axis=1))
                    chunk = chunk[~rows_to_drop_mask]
                if chunk.empty: continue

                # This is your original separation logic, now used to sort data into pools
                chunk_missing = chunk[chunk.isnull().any(axis=1)]
                chunk_no_missing = chunk.dropna()

                for label, group in chunk_no_missing.groupby(actual_label_col):
                    data_pools[label]['NoMissing'].append(group)
                for label, group in chunk_missing.groupby(actual_label_col):
                    data_pools[label]['Missing'].append(group)
        except Exception as e:
            print(f"    Warning: Could not process {os.path.basename(file_path)}. Error: {e}")

    print("\n--- Phase 3: Shuffling and Saving Final Files ---")

    # Create output directories based on the original logic
    os.makedirs(os.path.join(OUTPUT_FOLDER, "Benign"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "Attacks"), exist_ok=True)
    if separate_by_missing:
        os.makedirs(os.path.join(OUTPUT_FOLDER, "NoMissing", "Benign"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_FOLDER, "NoMissing", "Attacks"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_FOLDER, "Missing", "Benign"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_FOLDER, "Missing", "Attacks"), exist_ok=True)

    for label, status_pools in data_pools.items():
        for status, df_list in status_pools.items():
            print(f"\nProcessing: {label} ({status})")
            if not df_list: continue

            full_df = pd.concat(df_list, ignore_index=True)

            # Here is the new conditional shuffling logic
            if should_shuffle:
                print(f"  Shuffling {len(full_df):,} rows...")
                processed_df = full_df.sample(frac=1).reset_index(drop=True)
            else:
                print("  Skipping shuffling as requested.")
                processed_df = full_df

            # The rest of the saving logic combines your original rules with the new "split" method
            is_benign = (label == BENIGN_LABEL_VALUE)
            row_limit = benign_rows_per_file if is_benign else attack_rows_per_file
            if row_limit <= 0: continue

            should_this_pool_be_separated = (separate_by_missing and (
                        scope == 'both' or (scope == 'benign' and is_benign) or (scope == 'attacks' and not is_benign)))

            num_files = int(np.ceil(len(processed_df) / row_limit))
            print(f"  Splitting into {num_files} file(s)...")
            for i in range(num_files):
                start_row, end_row = i * row_limit, (i + 1) * row_limit
                df_part = processed_df.iloc[start_row:end_row]

                safe_name = "".join(c for c in label if c.isalnum() or c in ('-', '_'))
                subfolder = "Benign" if is_benign else "Attacks"

                if should_this_pool_be_separated:
                    path = os.path.join(OUTPUT_FOLDER, status, subfolder)
                else:
                    path = os.path.join(OUTPUT_FOLDER, subfolder)

                output_filename = os.path.join(path, f"{safe_name}_part_{i + 1}.csv")
                df_part.to_csv(output_filename, index=False)
                print(f"    Saved {os.path.relpath(output_filename)}")


def main():
    """ The main function orchestrates the new, more efficient workflow. """
    print("Starting the CSV Cleaning and Separation Process...")
    all_csv_files = [os.path.join(root, file) for root, _, files in os.walk(INPUT_FOLDER) for file in files if
                     file.endswith(".csv")]

    if not all_csv_files:
        print(f"No CSV files found in '{INPUT_FOLDER}'. Exiting.")
        return

    # Stage 1: Analyze all files to get one complete report
    grand_total_counts, grand_missing_counts, actual_label_col = analyze_all_files(all_csv_files)
    if not actual_label_col:
        print("\nCould not find the label column in any files. Exiting.")
        return

    # Stage 2: Get all user instructions once
    instructions = get_user_instructions(grand_total_counts, grand_missing_counts)

    # Stage 3: Run the main processing workflow with all instructions
    process_all_files(all_csv_files, actual_label_col, instructions)

    print("\n" + "=" * 80 + "\nAll files have been processed successfully!\n" + "=" * 80)


if __name__ == "__main__":
    main()