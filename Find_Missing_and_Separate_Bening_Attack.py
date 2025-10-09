import os
import pandas as pd
from collections import defaultdict

# --- 1. Global Configuration ---
INPUT_FOLDER = "Raw_Data_2018"
OUTPUT_FOLDER = "Cleaned_Separated_Data"
CHUNK_SIZE = 500_000
DEFAULT_ROWS_PER_FILE = 1_000_000
# This is now case-insensitive! You can set it to 'label', 'Label', etc.
LABEL_COLUMN = 'Label'
BENIGN_LABEL_VALUE = 'BENIGN'


# --- 2. Core Functions ---

def analyze_file(file_path):
    """
    Reads a CSV file to analyze its contents, now with case-insensitive label detection.
    Returns counts and the actual, correct name of the label column found.
    """
    print(f"\nAnalyzing file: {os.path.basename(file_path)}...")
    total_counts_by_label = defaultdict(int)
    missing_counts_by_label = defaultdict(int)

    # --- MODIFICATION: Case-insensitive column finding logic ---
    actual_label_col_name = None
    try:
        header_df = pd.read_csv(file_path, nrows=0, low_memory=False)
        for col_name in header_df.columns:
            if col_name.lower() == LABEL_COLUMN.lower():
                actual_label_col_name = col_name
                print(f"  Detected label column as: '{actual_label_col_name}'")
                break  # Found it, no need to look further

        if actual_label_col_name is None:
            print(f"  ERROR: A label column like '{LABEL_COLUMN}' not found. Skipping file.")
            return None, None, None

        for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False):
            total_counts = chunk[actual_label_col_name].value_counts()
            for label, count in total_counts.items():
                total_counts_by_label[label] += count

            rows_with_missing = chunk[chunk.isnull().any(axis=1)]
            if not rows_with_missing.empty:
                missing_counts = rows_with_missing[actual_label_col_name].value_counts()
                for label, count in missing_counts.items():
                    missing_counts_by_label[label] += count

        print("  Analysis complete.")
        return total_counts_by_label, missing_counts_by_label, actual_label_col_name

    except Exception as e:
        print(f"  ERROR during analysis: {e}. Skipping this file.")
        return None, None, None


def get_user_instructions(total_counts, missing_counts):
    """
    Shows analysis reports to the user and asks for instructions.
    (This function is unchanged)
    """
    print("\n--- Total Row Count Report ---")
    for label, count in sorted(total_counts.items()):
        print(f"  - {label}: {count:,} total rows.")
    print("------------------------------")

    benign_rows_per_file = 0
    if BENIGN_LABEL_VALUE in total_counts:
        while True:
            try:
                prompt = f"\nEnter the max rows per Benign file (press Enter for default: {DEFAULT_ROWS_PER_FILE:,}): "
                user_input = input(prompt).strip()
                if not user_input:
                    benign_rows_per_file = DEFAULT_ROWS_PER_FILE
                    break
                val = int(user_input)
                if val > 0:
                    benign_rows_per_file = val
                    break
                else:
                    print("  Please enter a positive number.")
            except ValueError:
                print("  Invalid input. Please enter a whole number.")

    attack_rows_per_file = 0
    if any(label != BENIGN_LABEL_VALUE for label in total_counts):
        while True:
            try:
                prompt = f"Enter the max rows per Attack file (press Enter for default: {DEFAULT_ROWS_PER_FILE:,}): "
                user_input = input(prompt).strip()
                if not user_input:
                    attack_rows_per_file = DEFAULT_ROWS_PER_FILE
                    break
                val = int(user_input)
                if val > 0:
                    attack_rows_per_file = val
                    break
                else:
                    print("  Please enter a positive number.")
            except ValueError:
                print("  Invalid input. Please enter a whole number.")

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

    separate_by_missing_status = False
    separation_scope = 'none'

    print("-" * 30)
    user_wants_separation = input(
        "Do you want to separate output into 'NoMissing' and 'Missing' folders? (y/n): ").strip().lower()
    if user_wants_separation in ['y', 'yes']:
        separate_by_missing_status = True
        while True:
            scope_input = input("Apply this to [B]enign, [A]ttacks, or [Bo]th?: ").strip().lower()
            if scope_input in ['b', 'benign']:
                separation_scope = 'benign'
                break
            elif scope_input in ['a', 'attacks']:
                separation_scope = 'attacks'
                break
            elif scope_input in ['bo', 'both']:
                separation_scope = 'both'
                break
            else:
                print("Invalid input. Please enter 'b', 'a', or 'bo'.")

    return labels_to_delete, benign_rows_per_file, attack_rows_per_file, separate_by_missing_status, separation_scope


# --- MODIFICATION: Added 'actual_label_col' parameter ---
def process_and_save_file(file_path, actual_label_col, labels_to_delete, benign_rows_per_file, attack_rows_per_file,
                          separate_by_missing, scope):
    """
    Reads the CSV again, cleans it, and saves it using the chosen method.
    """
    if labels_to_delete:
        print(f"\nStarting cleanup for labels: {', '.join(labels_to_delete)}")
    else:
        print("\nStarting separation process (no cleaning requested).")

    # (Folder creation logic is unchanged)
    if separate_by_missing:
        for status in ["NoMissing", "Missing"]:
            for category in ["Benign", "Attacks"]:
                os.makedirs(os.path.join(OUTPUT_FOLDER, status, category), exist_ok=True)
    else:
        os.makedirs(os.path.join(OUTPUT_FOLDER, "Benign"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_FOLDER, "Attacks"), exist_ok=True)

    buffers = defaultdict(lambda: defaultdict(list))
    part_counts = defaultdict(lambda: defaultdict(lambda: 1))

    try:
        for chunk_num, chunk in enumerate(pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False)):
            print(f"  Processing chunk {chunk_num + 1}...")

            chunk_cleaned = chunk
            if labels_to_delete:
                # --- MODIFICATION: Use the actual, detected column name ---
                rows_to_drop_mask = (chunk[actual_label_col].isin(labels_to_delete)) & (chunk.isnull().any(axis=1))
                chunk_cleaned = chunk[~rows_to_drop_mask]

            # --- MODIFICATION: Use the actual, detected column name ---
            benign_data = chunk_cleaned[chunk_cleaned[actual_label_col] == BENIGN_LABEL_VALUE]
            attack_data = chunk_cleaned[chunk_cleaned[actual_label_col] != BENIGN_LABEL_VALUE]

            if not benign_data.empty:
                if separate_by_missing and scope in ['benign', 'both']:
                    buffers['Benign']['NoMissing'].append(benign_data.dropna())
                    buffers['Benign']['Missing'].append(benign_data[benign_data.isnull().any(axis=1)])
                else:
                    buffers['Benign']['default'].append(benign_data)

            if not attack_data.empty:
                if separate_by_missing and scope in ['attacks', 'both']:
                    # --- MODIFICATION: Use the actual, detected column name ---
                    for label, group in attack_data.groupby(actual_label_col):
                        buffers[label]['NoMissing'].append(group.dropna())
                        buffers[label]['Missing'].append(group[group.isnull().any(axis=1)])
                else:
                    # --- MODIFICATION: Use the actual, detected column name ---
                    for label, group in attack_data.groupby(actual_label_col):
                        buffers[label]['default'].append(group)

            # (Saving logic is unchanged)
            for category, status_buffers in list(buffers.items()):
                for status, buffer_list in list(status_buffers.items()):
                    is_benign = (category == BENIGN_LABEL_VALUE)
                    row_limit = benign_rows_per_file if is_benign else attack_rows_per_file
                    if row_limit > 0 and sum(len(df) for df in buffer_list) >= row_limit:
                        df_to_save = pd.concat(buffer_list, ignore_index=True)
                        if df_to_save.empty: continue
                        safe_name = "".join(c for c in category if c.isalnum() or c in ('-', '_'))
                        part_num = part_counts[category][status]
                        if separate_by_missing:
                            subfolder = "Benign" if is_benign else "Attacks"
                            path = os.path.join(OUTPUT_FOLDER, status, subfolder)
                        else:
                            path = os.path.join(OUTPUT_FOLDER, "Benign" if is_benign else "Attacks")
                        filename = os.path.join(path, f"{safe_name}_part_{part_num}.csv")
                        df_to_save.to_csv(filename, index=False)
                        print(f"    Saved {len(df_to_save):,} rows to {os.path.relpath(filename)}")
                        buffer_list.clear()
                        part_counts[category][status] += 1

        print("\n  Saving remaining data...")
        for category, status_buffers in buffers.items():
            for status, buffer_list in status_buffers.items():
                if buffer_list:
                    df_to_save = pd.concat(buffer_list, ignore_index=True)
                    if df_to_save.empty: continue
                    is_benign = (category == BENIGN_LABEL_VALUE)
                    safe_name = "".join(c for c in category if c.isalnum() or c in ('-', '_'))
                    part_num = part_counts[category][status]
                    if separate_by_missing:
                        subfolder = "Benign" if is_benign else "Attacks"
                        path = os.path.join(OUTPUT_FOLDER, status, subfolder)
                    else:
                        path = os.path.join(OUTPUT_FOLDER, "Benign" if is_benign else "Attacks")
                    filename = os.path.join(path, f"{safe_name}_part_{part_num}.csv")
                    df_to_save.to_csv(filename, index=False)
                    print(f"    Saved {len(df_to_save):,} rows to {os.path.relpath(filename)}")

        print(f"\nFinished processing {os.path.basename(file_path)}.")

    except Exception as e:
        print(f"  FATAL ERROR during processing: {e}.")


def main():
    """
    The main function to orchestrate the entire process.
    """
    print("Starting the CSV Cleaning and Separation Process...")
    all_csv_files = [os.path.join(root, file) for root, _, files in os.walk(INPUT_FOLDER) for file in files if
                     file.endswith(".csv")]

    if not all_csv_files:
        print(f"No CSV files found in '{INPUT_FOLDER}'. Exiting.")
        return

    for file_path in all_csv_files:
        print("=" * 80)
        # --- MODIFICATION: Get the actual label column name from the analysis ---
        total_counts, missing_counts, actual_label_col = analyze_file(file_path)
        if total_counts is None: continue

        labels_to_delete, benign_rows, attack_rows, separate_by_missing, scope = get_user_instructions(total_counts,
                                                                                                       missing_counts)

        # --- MODIFICATION: Pass the actual column name to the processing function ---
        process_and_save_file(file_path, actual_label_col, labels_to_delete, benign_rows, attack_rows,
                              separate_by_missing, scope)

    print("\n" + "=" * 80)
    print("All files have been processed successfully!")
    print("=" * 80)


# --- 3. Run the script ---
if __name__ == "__main__":
    main()