import pandas as pd
import os

INPUT_FOLDER = "Training_2018"
CHUNK_SIZE = 500_000

NEVER_NEGATIVE_KEYWORDS = [
    'port', 'duration', 'count', 'bytes', 'size', 'rate', 'percentage',
    'variance', 'std', 'total', 'max', 'min', 'median', 'mode', 'mean',
    'iat', 'active', 'idle', 'bulk', 'handshake', 'subflow'
]
CAN_BE_NEGATIVE_KEYWORDS = [
    'skew', 'cov', 'delta'
]
PORT_COLUMNS = ['src_port', 'dst_port']

def is_column_never_negative(column_name):
    lower = column_name.lower()
    if any(kw in lower for kw in CAN_BE_NEGATIVE_KEYWORDS):
        return False
    if any(kw in lower for kw in NEVER_NEGATIVE_KEYWORDS):
        return True
    return False

def validate_data_quality(df, filename):
    print("\nValidating:", filename)
    results = {}

    percentage_cols = [col for col in df.columns if 'percentage' in col.lower()]

    negative_issues = {}
    analyzed_cols = []
    skipped_cols = []

    # For aggregation of invalid row labels
    invalid_row_label_info = {}

    for col in df.columns:
        if col.lower() == 'label':
            skipped_cols.append(col)
            continue
        if is_column_never_negative(col):
            analyzed_cols.append(col)
            try:
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                invalid_mask = numeric_col < 0
                negative_count = invalid_mask.sum()
                if negative_count > 0:
                    row_idx = df[invalid_mask].index
                    unique_row_count = len(row_idx)
                    label_counts = df.loc[invalid_mask, 'label'].value_counts().to_dict()
                    negative_issues[col] = {
                        'negative_count': negative_count,
                        'unique_rows': unique_row_count,
                        'row_indices': list(row_idx),
                        'labels': label_counts
                    }
            except Exception as e:
                print(f"Warning: Could not check column {col}: {e}")
        else:
            skipped_cols.append(col)
    results['negative_issues'] = negative_issues

    # Repeat for ports
    port_issues = {}
    for col in PORT_COLUMNS:
        if col in df.columns:
            analyzed_cols.append(col)
            try:
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                invalid_mask = (numeric_col < 0) | (numeric_col > 65535)
                invalid_count = invalid_mask.sum()
                if invalid_count > 0:
                    row_idx = df[invalid_mask].index
                    unique_row_count = len(row_idx)
                    label_counts = df.loc[invalid_mask, 'label'].value_counts().to_dict()
                    port_issues[col] = {
                        'invalid_count': invalid_count,
                        'unique_rows': unique_row_count,
                        'row_indices': list(row_idx),
                        'labels': label_counts
                    }
            except Exception as e:
                print(f"Warning: Could not check port column {col}: {e}")
    results['port_issues'] = port_issues

    percentage_issues = {}
    for col in percentage_cols:
        analyzed_cols.append(col)
        try:
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            invalid_mask = (numeric_col < 0) | (numeric_col > 100)
            invalid_count = invalid_mask.sum()
            if invalid_count > 0:
                row_idx = df[invalid_mask].index
                unique_row_count = len(row_idx)
                label_counts = df.loc[invalid_mask, 'label'].value_counts().to_dict()
                percentage_issues[col] = {
                    'invalid_count': invalid_count,
                    'unique_rows': unique_row_count,
                    'row_indices': list(row_idx),
                    'labels': label_counts
                }
        except Exception as e:
            print(f"Warning: Could not check percentage column {col}: {e}")
    results['percentage_issues'] = percentage_issues

    missing_values = df.isnull().sum()
    columns_with_missing = {col: count for col, count in missing_values.items() if count > 0}
    results['missing_values'] = columns_with_missing

    print("\n--- VALIDATION SUMMARY ---")
    print(f"Columns analyzed: {len(analyzed_cols)}")
    print(f"Columns skipped: {len(skipped_cols)}")

    # Print detailed output for negatives
    if negative_issues:
        print(f"Columns with invalid negative values ({len(negative_issues)}):")
        for col, info in negative_issues.items():
            print(f"  {col}: {info['negative_count']} negative values, {info['unique_rows']} unique rows affected")
            print(f"    Label counts in affected rows: {info['labels']}")
    else:
        print("No invalid negative values found")

    # Print port issues
    if port_issues:
        print(f"Columns with invalid port values ({len(port_issues)}):")
        for col, info in port_issues.items():
            print(f"  {col}: {info['invalid_count']} invalid values, {info['unique_rows']} unique rows affected")
            print(f"    Label counts in affected rows: {info['labels']}")
    else:
        print("All port values are valid (0-65535)")

    # Print percentage issues
    if percentage_issues:
        print(f"Columns with invalid percentage values ({len(percentage_issues)}):")
        for col, info in percentage_issues.items():
            print(f"  {col}: {info['invalid_count']} invalid values, {info['unique_rows']} unique rows affected")
            print(f"    Label counts in affected rows: {info['labels']}")
    else:
        print("All percentage values are valid (0-100)")

    if columns_with_missing:
        print(f"Columns with missing values ({len(columns_with_missing)}):")
        for col, count in list(columns_with_missing.items())[:5]:
            print(f"   {col}: {count} missing values")
        if len(columns_with_missing) > 5:
            print(f"   ... and {len(columns_with_missing) - 5} more columns")
    else:
        print("No missing values found")

    results['analyzed'] = analyzed_cols
    results['skipped'] = skipped_cols
    return results

def clean_data_if_needed(df, results, filename):
    invalid_indices = set()
    # Aggregate row indices from all types of issues
    for issue_group in ['negative_issues', 'port_issues', 'percentage_issues']:
        for col, info in results[issue_group].items():
            invalid_indices.update(info['row_indices'])

    if not invalid_indices:
        print("No data cleaning needed.")
        return df

    print(f"\nFound {len(invalid_indices)} unique rows with at least one invalid value.")
    choice = input("Do you want to remove all rows with any invalid value? (y/n): ")
    if choice.lower() != 'y':
        print("Skipping data cleaning.")
        return df

    print("Cleaning data...")
    original_rows = len(df)
    df_clean = df.drop(index=list(invalid_indices)).copy()
    rows_removed = original_rows - len(df_clean)
    print(f"Removed {rows_removed} invalid rows ({rows_removed/original_rows*100:.2f}%)")
    print(f"Clean dataset: {len(df_clean)} rows remaining")

    clean_filename = f"{os.path.splitext(filename)[0]}_validated.csv"
    output_path = os.path.join(INPUT_FOLDER, clean_filename)
    df_clean.to_csv(output_path, index=False)
    print(f"Saved clean data: {clean_filename}")

    return df_clean

def main():
    print("Starting automated data validation...")
    print(f"Processing files in: {INPUT_FOLDER}")

    for filename in os.listdir(INPUT_FOLDER):
        if not filename.endswith(".csv"):
            continue

        file_path = os.path.join(INPUT_FOLDER, filename)
        print(f"\nLoading: {filename}")
        dfs = []
        try:
            for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE):
                dfs.append(chunk)

            df = pd.concat(dfs, ignore_index=True)
            del dfs

            print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            validation_results = validate_data_quality(df, filename)
            df_clean = clean_data_if_needed(df, validation_results, filename)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

if __name__ == "__main__":
    main()
