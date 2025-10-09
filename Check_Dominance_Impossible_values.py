import os
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

# --- GLOBAL CONFIGURATION VARIABLES ---
# Folder containing your CSV files
INPUT_FOLDER = "Downscale_Csv_2018"
# Adjust chunk size based on your system's RAM
CHUNK_SIZE = 1_000_000

# Parameters for Dominance Report
DOMINANCE_RANGES = [
    (0.95, 1.01, "95-100%"),
    (0.90, 0.95, "90-95%"),
    (0.80, 0.90, "80-90%"),
    (0.70, 0.80, "70-80%"),
    (0.60, 0.70, "60-70%"),
    (0.50, 0.60, "50-60%"),
]

# Parameters for Data Validation (Row Removal)
NEVER_NEGATIVE_KEYWORDS = [
    'port', 'duration', 'count', 'bytes', 'size', 'rate', 'percentage',
    'variance', 'std', 'total', 'max', 'min', 'median', 'mode', 'mean',
    'iat', 'active', 'idle', 'bulk', 'handshake', 'subflow'
]
CAN_BE_NEGATIVE_KEYWORDS = ['skew', 'cov', 'delta']
PORT_COLUMNS = ['src_port', 'dst_port']

# Parameters for 'inf' Column Removal
INF_THRESHOLD = 0.30  # Deletes columns if they are over 30% 'inf'


# ==============================================================================
# TASK 1: DOMINANCE REPORT LOGIC
# ==============================================================================

def generate_dominance_report(file_path):
    """Analyzes a CSV for value dominance and creates a report."""
    print(f"\nGenerating Dominance Report for: {os.path.basename(file_path)}")
    # ... (The rest of this function is unchanged)
    col_counters = defaultdict(Counter)
    total_counts = Counter()
    label_counter = Counter()
    col_value_label_counter = defaultdict(lambda: defaultdict(Counter))

    try:
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, dtype=str, low_memory=False):
            labels = None
            if "Label" in chunk.columns:
                labels = chunk["Label"]
            elif "label" in chunk.columns:
                labels = chunk["label"]

            for col in chunk.columns:
                values = chunk[col].dropna()
                col_counters[col].update(values)
                total_counts[col] += len(values)

                if labels is not None and col.lower() != "label":
                    for v, lbl in zip(chunk[col], labels):
                        if pd.notna(v) and pd.notna(lbl):
                            col_value_label_counter[col][v][lbl] += 1

            if labels is not None:
                label_counter.update(labels.dropna())

        bucketed = {label: [] for _, _, label in DOMINANCE_RANGES}
        for col, counts in col_counters.items():
            if total_counts[col] == 0:
                continue
            most_common_val, most_common_count = counts.most_common(1)[0]
            ratio = most_common_count / total_counts[col]

            for low, high, label in DOMINANCE_RANGES:
                if low <= ratio < high:
                    bucketed[label].append((col, counts, total_counts[col]))
                    break

        report_path = f"{os.path.splitext(file_path)[0]}_dominance_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Dominance Report for {os.path.basename(file_path)}\n")
            f.write("=" * 60 + "\n\n")

            if label_counter:
                total_labels = sum(label_counter.values())
                f.write("Global Label Distribution:\n" + "-" * 40 + "\n")
                for lbl, count in label_counter.most_common():
                    perc = (count / total_labels) * 100
                    f.write(f"  {lbl}: {count:,} ({perc:.2f}%)\n")
                f.write("\n")

            for label in bucketed:
                f.write(f"\nColumns in {label} range:\n" + "-" * 40 + "\n")
                if not bucketed[label]:
                    f.write("  None\n")
                else:
                    for col, counts, total in bucketed[label]:
                        f.write(f"\nColumn: {col}\n")
                        for val, count in counts.most_common():
                            ratio = count / total
                            f.write(f"  Value '{val}': {count:,} ({ratio * 100:.2f}%)")
                            if col in col_value_label_counter and val in col_value_label_counter[col]:
                                lbl_counts = col_value_label_counter[col][val]
                                breakdown = ", ".join(f"{lbl}: {c:,}" for lbl, c in lbl_counts.most_common())
                                f.write(f" -> Labels: [{breakdown}]")
                            f.write("\n")
        print(f"Report saved to {report_path}")
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")


# ==============================================================================
# TASK 2: DATA VALIDATION & CLEANING LOGIC (ROW REMOVAL)
# ==============================================================================

def is_column_never_negative(column_name):
    """Checks if a column name suggests its values should never be negative."""
    lower = column_name.lower()
    if any(kw in lower for kw in CAN_BE_NEGATIVE_KEYWORDS):
        return False
    return any(kw in lower for kw in NEVER_NEGATIVE_KEYWORDS)


def validate_data_quality(df):
    """Performs several data quality checks on the dataframe."""
    results = {'negative_issues': {}, 'port_issues': {}, 'percentage_issues': {}}
    if 'Label' in df.columns and 'label' not in df.columns:
        df = df.rename(columns={'Label': 'label'})
    if 'label' not in df.columns:
        print("Warning: 'label' column not found. Cannot provide label breakdowns for issues.")
        df['label'] = 'Unknown'

    print("--- Running Validation Checks ---")
    for col in df.columns:
        if is_column_never_negative(col):
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            invalid_mask = numeric_col < 0
            if invalid_mask.sum() > 0:
                results['negative_issues'][col] = {
                    'count': invalid_mask.sum(), 'rows': list(df[invalid_mask].index),
                    'labels': df.loc[invalid_mask, 'label'].value_counts().to_dict()
                }
    for col in PORT_COLUMNS:
        if col in df.columns:
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            invalid_mask = ~numeric_col.between(0, 65535, inclusive='both')
            if invalid_mask.sum() > 0:
                results['port_issues'][col] = {
                    'count': invalid_mask.sum(), 'rows': list(df[invalid_mask].index),
                    'labels': df.loc[invalid_mask, 'label'].value_counts().to_dict()
                }
    for col in [c for c in df.columns if 'percentage' in c.lower()]:
        numeric_col = pd.to_numeric(df[col], errors='coerce')
        invalid_mask = ~numeric_col.between(0, 100, inclusive='both')
        if invalid_mask.sum() > 0:
            results['percentage_issues'][col] = {
                'count': invalid_mask.sum(), 'rows': list(df[invalid_mask].index),
                'labels': df.loc[invalid_mask, 'label'].value_counts().to_dict()
            }

    print("\n--- VALIDATION SUMMARY ---")
    for issue_type, issues in results.items():
        title = issue_type.replace('_', ' ').title()
        if issues:
            print(f"Found {len(issues)} columns with {title}:")
            for col, info in issues.items():
                print(f"  - {col}: {info['count']} invalid values. Label breakdown: {info['labels']}")
        else:
            print(f"No issues found for: {title}")
    return results


def clean_data_if_needed(df, results):
    """Asks user if they want to clean the data and saves a new file if yes."""
    invalid_indices = set()
    for issue_group in results.values():
        for col_info in issue_group.values():
            invalid_indices.update(col_info['rows'])
    if not invalid_indices:
        print("\nNo invalid rows to clean.")
        return
    print(f"\nFound {len(invalid_indices)} unique rows with at least one invalid value.")
    choice = input("Do you want to remove all invalid rows and save a new file? (y/n): ")
    if choice.lower() != 'y':
        print("Skipping data cleaning.")
        return
    print("Cleaning data...")
    original_rows = len(df)
    df_clean = df.drop(index=list(invalid_indices)).copy()
    rows_removed = original_rows - len(df_clean)
    print(f"Removed {rows_removed} invalid rows ({rows_removed / original_rows * 100:.2f}%)")
    return df_clean


def run_data_validation(file_path):
    """Loads a CSV and runs the full validation and cleaning pipeline."""
    print(f"\nValidating and Cleaning: {os.path.basename(file_path)}")
    try:
        df = pd.concat([chunk for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE)])
        print(f"Loaded {len(df)} rows.")
        validation_results = validate_data_quality(df)
        df_clean = clean_data_if_needed(df, validation_results)
        if df_clean is not None:
            clean_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_validated.csv"
            output_path = os.path.join(os.path.dirname(file_path), clean_filename)
            df_clean.to_csv(output_path, index=False)
            print(f"Saved clean data to: {output_path}")
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")


# ==============================================================================
# TASK 3: 'INF' COLUMN REMOVAL LOGIC
# ==============================================================================

def run_inf_column_removal(file_path):
    """Analyzes and removes columns with a high percentage of 'inf' values."""
    print(f"\n--- Processing file for 'inf' columns: {os.path.basename(file_path)} ---")

    # --- Pass 1: Analyze the file to find problematic columns ---
    print(f"Phase 1: Analyzing columns for 'inf' values (Threshold: {INF_THRESHOLD:.0%})...")
    inf_counts = pd.Series(dtype=int)
    total_rows = 0
    try:
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False):
            total_rows += len(chunk)
            inf_in_chunk = chunk.apply(pd.to_numeric, errors='coerce').pipe(np.isinf).sum()
            inf_counts = inf_counts.add(inf_in_chunk, fill_value=0)

        inf_percentages = inf_counts / total_rows
        columns_to_delete = inf_percentages[inf_percentages > INF_THRESHOLD].index.tolist()
    except Exception as e:
        print(f"An error occurred while analyzing '{os.path.basename(file_path)}': {e}")
        return

    # --- Report findings and ask for confirmation ---
    if not columns_to_delete:
        print("Result: No columns with 'inf' values exceeded the threshold. No action needed.")
        return

    print(f"\nAnalysis complete. Found {len(columns_to_delete)} columns to remove:")
    for col in columns_to_delete:
        print(f"  - '{col}' ({inf_percentages[col]:.2%} inf)")

    user_input = input("Are you sure you want to permanently delete these columns? (yes/no): ").lower()

    # --- Pass 2: Delete the columns if confirmed ---
    if user_input in ['yes', 'y']:
        print("\nPhase 2: Deleting columns and creating new file...")
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_filename = f"{base_name}_cleaned.csv"
        output_csv_path = os.path.join(os.path.dirname(file_path), output_filename)
        is_first_chunk = True
        try:
            for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False):
                chunk.drop(columns=columns_to_delete, inplace=True, errors='ignore')
                if is_first_chunk:
                    chunk.to_csv(output_csv_path, index=False, mode='w')
                    is_first_chunk = False
                else:
                    chunk.to_csv(output_csv_path, index=False, mode='a', header=False)
            print(f"Successfully created '{output_filename}'")
        except Exception as e:
            print(f"An error occurred while creating the new file: {e}")
    else:
        print("Operation cancelled. No columns were deleted.")


# ==============================================================================
# MAIN DRIVER
# ==============================================================================
def main():
    """Main function to prompt user and run the selected task."""
    print("--- Data Analysis and Validation Tool ---")
    print(f"Target folder: {INPUT_FOLDER}")
    print("\nPlease choose a task to perform:")
    print("  1: Generate Dominance Report (finds high-frequency values)")
    print("  2: Validate and Clean Data (Row Removal)")
    print("  3: Remove Columns with High 'inf' Values")

    choice = input("Enter your choice (1, 2, or 3): ")

    if choice not in ['1', '2', '3']:
        print("Invalid choice. Please run the script again.")
        return

    if not os.path.isdir(INPUT_FOLDER):
        print(f"Error: Input folder not found at '{INPUT_FOLDER}'")
        return

    print("\nStarting process...")
    for filename in os.listdir(INPUT_FOLDER):
        # Process only CSVs, and skip files that are already outputs of this script
        if filename.endswith(".csv") and not filename.endswith(("_validated.csv", "_cleaned.csv")):
            file_path = os.path.join(INPUT_FOLDER, filename)

            if choice == '1':
                generate_dominance_report(file_path)
            elif choice == '2':
                run_data_validation(file_path)
            elif choice == '3':
                run_inf_column_removal(file_path)

            print("-" * 60)

    print("\nAll files processed.")


if __name__ == "__main__":
    main()