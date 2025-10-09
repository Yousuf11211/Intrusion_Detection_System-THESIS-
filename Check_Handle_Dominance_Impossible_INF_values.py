import os
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

# --- GLOBAL CONFIGURATION VARIABLES ---
INPUT_FOLDER = "Downscale_Csv_2018"
CHUNK_SIZE = 1_000_000
DOMINANCE_RANGES = [
    (0.95, 1.01, "95-100%"), (0.90, 0.95, "90-95%"),
    (0.80, 0.90, "80-90%"), (0.70, 0.80, "70-80%"),
    (0.60, 0.70, "60-70%"), (0.50, 0.60, "50-60%"),
]
NEVER_NEGATIVE_KEYWORDS = [
    'port', 'duration', 'count', 'bytes', 'size', 'rate', 'percentage',
    'variance', 'std', 'total', 'max', 'min', 'median', 'mode', 'mean',
    'iat', 'active', 'idle', 'bulk', 'handshake', 'subflow'
]
CAN_BE_NEGATIVE_KEYWORDS = ['skew', 'cov', 'delta']
PORT_COLUMNS = ['src_port', 'dst_port']
INF_THRESHOLD = 0.30


# ==============================================================================
# TASK 1: DOMINANCE REPORT LOGIC
# ==============================================================================
def generate_dominance_report(file_path):
    """Analyzes a CSV for value dominance and creates a report."""
    print(f"\nGenerating Dominance Report for: {os.path.basename(file_path)}")
    # (This function's logic remains unchanged)
    col_counters = defaultdict(Counter)
    total_counts = Counter()
    label_counter = Counter()
    col_value_label_counter = defaultdict(lambda: defaultdict(Counter))
    try:
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, dtype=str, low_memory=False):
            labels = chunk.get("Label") or chunk.get("label")
            if labels is not None:
                label_counter.update(labels.dropna())
            for col in chunk.columns:
                values = chunk[col].dropna()
                col_counters[col].update(values)
                total_counts[col] += len(values)
                if labels is not None and col.lower() != "label":
                    for v, lbl in zip(chunk[col], labels):
                        if pd.notna(v) and pd.notna(lbl):
                            col_value_label_counter[col][v][lbl] += 1
        bucketed = {label: [] for _, _, label in DOMINANCE_RANGES}
        for col, counts in col_counters.items():
            if total_counts[col] == 0: continue
            _, most_common_count = counts.most_common(1)[0]
            ratio = most_common_count / total_counts[col]
            for low, high, label in DOMINANCE_RANGES:
                if low <= ratio < high:
                    bucketed[label].append((col, counts, total_counts[col]))
                    break
        report_path = f"{os.path.splitext(file_path)[0]}_dominance_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Dominance Report for {os.path.basename(file_path)}\n" + "=" * 60 + "\n\n")
            if label_counter:
                total_labels = sum(label_counter.values())
                f.write("Global Label Distribution:\n" + "-" * 40 + "\n")
                for lbl, count in label_counter.most_common():
                    f.write(f"  {lbl}: {count:,} ({(count / total_labels) * 100:.2f}%)\n")
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
                            if val in col_value_label_counter.get(col, {}):
                                lbl_counts = col_value_label_counter[col][val]
                                breakdown = ", ".join(f"{lbl}: {c:,}" for lbl, c in lbl_counts.most_common())
                                f.write(f" -> Labels: [{breakdown}]")
                            f.write("\n")
        print(f"Report saved to {report_path}")
    except Exception as e:
        print(f"Error during dominance report: {e}")


# ==============================================================================
# TASK 2: DATA VALIDATION & CLEANING LOGIC (ROW REMOVAL)
# ==============================================================================
def run_data_validation(file_path):
    """Loads a CSV and runs the full validation and cleaning pipeline."""
    print(f"\nValidating and Cleaning: {os.path.basename(file_path)}")
    # (This function's logic remains unchanged)
    try:
        df = pd.concat([chunk for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE)])
        print(f"Loaded {len(df)} rows.")
        results = {'negative_issues': {}, 'port_issues': {}, 'percentage_issues': {}}
        if 'Label' in df.columns and 'label' not in df.columns:
            df = df.rename(columns={'Label': 'label'})
        if 'label' not in df.columns:
            print("Warning: 'label' column not found.")
            df['label'] = 'Unknown'
        for col in df.columns:
            if any(kw in col.lower() for kw in CAN_BE_NEGATIVE_KEYWORDS): continue
            if any(kw in col.lower() for kw in NEVER_NEGATIVE_KEYWORDS):
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                if (numeric_col < 0).sum() > 0:
                    mask = numeric_col < 0
                    results['negative_issues'][col] = {'count': mask.sum(), 'rows': list(df[mask].index),
                                                       'labels': df.loc[mask, 'label'].value_counts().to_dict()}
        for col in PORT_COLUMNS:
            if col in df.columns:
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                if (~numeric_col.between(0, 65535)).sum() > 0:
                    mask = ~numeric_col.between(0, 65535)
                    results['port_issues'][col] = {'count': mask.sum(), 'rows': list(df[mask].index),
                                                   'labels': df.loc[mask, 'label'].value_counts().to_dict()}
        invalid_indices = set()
        for group in results.values():
            for info in group.values():
                invalid_indices.update(info['rows'])
        if not invalid_indices:
            print("\nNo invalid rows to clean.")
            return
        print(f"\nFound {len(invalid_indices)} unique rows with invalid values.")
        if input("Remove invalid rows and save new file? (y/n): ").lower() == 'y':
            df_clean = df.drop(index=list(invalid_indices)).copy()
            clean_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_validated.csv"
            output_path = os.path.join(os.path.dirname(file_path), clean_filename)
            df_clean.to_csv(output_path, index=False)
            print(f"Saved clean data to: {output_path}")
        else:
            print("Skipping data cleaning.")
    except Exception as e:
        print(f"Error during data validation: {e}")


# ==============================================================================
# TASK 3: 'INF' COLUMN REMOVAL LOGIC
# ==============================================================================
def run_inf_column_removal(file_path):
    """Analyzes and removes columns with a high percentage of 'inf' values."""
    print(f"\n--- Processing file for 'inf' columns: {os.path.basename(file_path)} ---")
    print(f"Phase 1: Analyzing columns (Threshold: {INF_THRESHOLD:.0%})...")
    inf_counts = pd.Series(dtype=int)
    total_rows = 0
    try:
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False):
            total_rows += len(chunk)
            inf_counts = inf_counts.add(chunk.apply(pd.to_numeric, errors='coerce').pipe(np.isinf).sum(), fill_value=0)
        if total_rows == 0:
            print("File is empty. Skipping.")
            return
        inf_percentages = inf_counts / total_rows
        columns_to_delete = inf_percentages[inf_percentages > INF_THRESHOLD].index.tolist()
    except Exception as e:
        print(f"Error during analysis: {e}")
        return

    if not columns_to_delete:
        print("Result: No columns exceeded the 'inf' threshold.")
        # *** NEW: ASK TO IMPUTE EVEN IF NOTHING WAS REMOVED ***
        if (inf_counts > 0).any():
            if input(
                    "Some 'inf' values were found below the threshold. Handle them with imputation? (y/n): ").lower() == 'y':
                run_inf_imputation(file_path)
        return

    print(f"\nFound {len(columns_to_delete)} columns to remove:")
    for col in columns_to_delete:
        print(f"  - '{col}' ({inf_percentages[col]:.2%} inf)")

    if input("Permanently delete these columns? (y/n): ").lower() not in ['yes', 'y']:
        print("Operation cancelled.")
        return

    print("\nPhase 2: Deleting columns and creating new file...")
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_filename = f"{base_name}_cleaned.csv"
    output_csv_path = os.path.join(os.path.dirname(file_path), output_filename)
    try:
        is_first_chunk = True
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False):
            chunk.drop(columns=columns_to_delete, inplace=True, errors='ignore')
            if is_first_chunk:
                chunk.to_csv(output_csv_path, index=False, mode='w')
                is_first_chunk = False
            else:
                chunk.to_csv(output_csv_path, index=False, mode='a', header=False)
        print(f"Successfully created '{output_filename}'")

        # --- NEW: POST-CLEANING WORKFLOW ---
        print("\n--- Next Steps for the Cleaned File ---")
        print("What would you like to do now?")
        print("  1: Re-analyze the cleaned file for remaining 'inf' values")
        print("  2: Handle remaining 'inf' values with median imputation")
        print("  3: Do nothing / Continue to next file")
        choice = input("Enter your choice (1, 2, or 3): ")

        if choice == '1':
            report_remaining_inf(output_csv_path)
        elif choice == '2':
            run_inf_imputation(output_csv_path)
        else:
            print("Continuing to the next file.")

    except Exception as e:
        print(f"Error during file creation: {e}")


# ==============================================================================
# TASK 4: REPORT REMAINING 'INF' VALUES (New Helper Function)
# ==============================================================================
def report_remaining_inf(file_path):
    """A simple analysis pass to report, but not act on, 'inf' values."""
    print(f"\n--- Re-analyzing for remaining 'inf' in {os.path.basename(file_path)} ---")
    inf_counts = pd.Series(dtype=int)
    total_rows = 0
    try:
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False):
            total_rows += len(chunk)
            inf_counts = inf_counts.add(chunk.apply(pd.to_numeric, errors='coerce').pipe(np.isinf).sum(), fill_value=0)
        if total_rows == 0: return

        inf_percentages = inf_counts / total_rows
        remaining_inf_cols = inf_percentages[inf_percentages > 0].index.tolist()

        if not remaining_inf_cols:
            print("No remaining 'inf' values found.")
        else:
            print("Found remaining 'inf' values in the following columns:")
            for col in remaining_inf_cols:
                print(f"  - '{col}': {inf_counts[col]} values ({inf_percentages[col]:.4f}%)")
    except Exception as e:
        print(f"Error during re-analysis: {e}")


# ==============================================================================
# TASK 5: 'INF' IMPUTATION LOGIC (New Function)
# ==============================================================================
def run_inf_imputation(file_path):
    """Finds all 'inf' values and replaces them with the column median."""
    print(f"\n--- Imputing 'inf' values in {os.path.basename(file_path)} ---")
    medians = {}
    try:
        print("Phase 1: Calculating medians for columns with 'inf' values...")
        # First, find which columns have 'inf' values
        inf_counts = pd.Series(dtype=int)
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False):
            inf_counts = inf_counts.add(chunk.apply(pd.to_numeric, errors='coerce').pipe(np.isinf).sum(), fill_value=0)
        cols_to_process = inf_counts[inf_counts > 0].index.tolist()

        if not cols_to_process:
            print("No 'inf' values found to impute.")
            return

        # Now calculate medians for only those columns
        for col in cols_to_process:
            series = pd.read_csv(file_path, usecols=[col], low_memory=False).squeeze("columns")
            median_val = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).median()
            medians[col] = median_val
            print(f"  - Column '{col}': Median is {median_val}")

        print("\nPhase 2: Replacing 'inf' values and saving new file...")
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_filename = f"{base_name}_imputed.csv"
        output_csv_path = os.path.join(os.path.dirname(file_path), output_filename)
        is_first_chunk = True
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False):
            for col, median_val in medians.items():
                if col in chunk.columns:
                    chunk[col] = pd.to_numeric(chunk[col], errors='coerce').replace([np.inf, -np.inf], median_val)
            if is_first_chunk:
                chunk.to_csv(output_csv_path, index=False, mode='w')
                is_first_chunk = False
            else:
                chunk.to_csv(output_csv_path, index=False, mode='a', header=False)
        print(f"Successfully created '{output_filename}'")
    except Exception as e:
        print(f"Error during imputation: {e}")


# ==============================================================================
# MAIN DRIVER
# ==============================================================================
def main():
    """Main function to prompt user and run the selected task."""
    print("--- Data Analysis and Validation Tool ---")
    print(f"Target folder: {INPUT_FOLDER}")
    print("\nPlease choose a task to perform:")
    print("  1: Generate Dominance Report")
    print("  2: Validate and Clean Data (Row Removal)")
    print("  3: Remove Columns with High 'inf' Values (with imputation option)")
    choice = input("Enter your choice (1, 2, or 3): ")

    if choice not in ['1', '2', '3']:
        print("Invalid choice. Please run the script again.")
        return
    if not os.path.isdir(INPUT_FOLDER):
        print(f"Error: Input folder not found at '{INPUT_FOLDER}'")
        return

    print("\nStarting process...")
    for filename in os.listdir(INPUT_FOLDER):
        output_suffixes = ("_validated.csv", "_cleaned.csv", "_imputed.csv")
        if filename.endswith(".csv") and not filename.endswith(output_suffixes):
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