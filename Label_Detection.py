import os
import pandas as pd
from collections import Counter, defaultdict
from math import floor

# =========================
# Global configuration
# =========================
PARENT_FOLDER = "Cleaned_Datasets"         # where source CSVs live
REPORTS_FOLDER = "Labelled_Reports"        # summary reports
TRAIN_FOLDER = "Training_2018"             # output train folder
TEST_FOLDER = "Test_2018"                  # output test folder
TRAIN_CSV_NAME = "training_2.csv"          # output train file name
TEST_CSV_NAME = "test_2.csv"               # output test file name
CHUNK_SIZE = 100_000                       # rows per read_csv chunk
LABEL_COLUMN = "Label"                     # label column name
OUTPUT_ENCODING = "utf-8"                  # output text encoding
SAVE_INTERMEDIATE_REPORTS = True           # set False to skip text reports

os.makedirs(REPORTS_FOLDER, exist_ok=True)
os.makedirs(TRAIN_FOLDER, exist_ok=True)
os.makedirs(TEST_FOLDER, exist_ok=True)

def safe_lower(x):
    try:
        return str(x).strip().lower()
    except Exception:
        return str(x)

def count_labels_first(file_path):
    """
    First pass: stream through the file to count labels only.
    Returns (label_counts: Counter, total_rows: int).
    """
    label_counts = Counter()
    total = 0
    for chunk in pd.read_csv(
        file_path,
        usecols=[LABEL_COLUMN],
        chunksize=CHUNK_SIZE,
        low_memory=True
    ):
        s = chunk[LABEL_COLUMN].dropna()
        label_counts.update(s)
        total += len(chunk)
    return label_counts, total

def plan_stratified_split(counts, train_ratio=0.6):
    """
    For each label, decide how many go to train vs test, preserving ratios.
    Returns two dicts: train_needed[label], test_needed[label].
    """
    train_needed = {}
    test_needed = {}
    for label, cnt in counts.items():
        t = int(round(cnt * train_ratio))
        # Ensure bounds
        t = max(0, min(cnt, t))
        train_needed[label] = t
        test_needed[label] = cnt - t
    return train_needed, test_needed

def build_report(file_path, total_samples, label_counts, train_counts, test_counts):
    """
    Create a human-readable report as text.
    """
    benign_key_variants = {k for k in label_counts.keys() if safe_lower(k) == "benign"}
    benign_total = sum(label_counts[k] for k in benign_key_variants) if benign_key_variants else 0
    attack_total = total_samples - benign_total

    lines = []
    lines.append(f"Report for {file_path}")
    lines.append("=" * 60)
    lines.append(f"Total samples: {total_samples}")
    lines.append(f"Benign: {benign_total}")
    lines.append(f"Attacks: {attack_total}")
    lines.append("")
    lines.append("Breakdown by label (full dataset):")
    lines.append("-" * 40)
    for label, cnt in label_counts.items():
        lines.append(f"{label:<25}: {cnt}")
    lines.append("")
    lines.append("Training split label counts (60% target):")
    lines.append("-" * 40)
    for label, cnt in sorted(train_counts.items(), key=lambda x: str(x[0]).lower()):
        lines.append(f"{label:<25}: {cnt}")
    lines.append("")
    lines.append("Test split label counts (40% target):")
    lines.append("-" * 40)
    for label, cnt in sorted(test_counts.items(), key=lambda x: str(x[0]).lower()):
        lines.append(f"{label:<25}: {cnt}")
    return "\n".join(lines)

def write_report_text(report_text, file_path):
    rel = os.path.relpath(file_path, PARENT_FOLDER)
    rel_name = rel.replace(os.sep, "_")
    out_path = os.path.join(REPORTS_FOLDER, f"{os.path.splitext(rel_name)[0]}.txt")
    with open(out_path, "w", encoding=OUTPUT_ENCODING) as f:
        f.write(report_text)
    print(f"Saved report to {out_path}")

def split_and_write(file_path):
    """
    Two-pass streaming approach:
      1) Count labels only.
      2) Stream rows again, writing each row to train/test according to per-label quotas.
    """
    # Pass 1: counts
    try:
        label_counts, total_rows = count_labels_first(file_path)
    except ValueError as e:
        print(f"Error: {e} in {file_path}. Maybe no '{LABEL_COLUMN}' column.")
        return
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    if total_rows == 0:
        print(f"No rows found in {file_path}, skipping.")
        return

    # Plan per-label quotas
    train_needed, test_needed = plan_stratified_split(label_counts, train_ratio=0.6)

    # Prepare output paths
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    train_path = os.path.join(TRAIN_FOLDER, TRAIN_CSV_NAME)
    test_path = os.path.join(TEST_FOLDER, TEST_CSV_NAME)

    # Initialize writers lazily to capture original column order and headers once
    train_writer = None
    test_writer = None

    # Track how many written per label to each split
    written_train = defaultdict(int)
    written_test = defaultdict(int)

    # Pass 2: assign rows
    for chunk in pd.read_csv(
        file_path,
        chunksize=CHUNK_SIZE,
        low_memory=True
    ):
        if LABEL_COLUMN not in chunk.columns:
            print(f"'{LABEL_COLUMN}' column not found in {file_path}, skipping this file.")
            return

        # Initialize the writers when first chunk arrives (preserve full columns)
        if train_writer is None:
            # Remove existing files to avoid appending to old runs
            if os.path.exists(train_path):
                os.remove(train_path)
            if os.path.exists(test_path):
                os.remove(test_path)

        # We'll build rows for train and test separately for this chunk
        train_rows = []
        test_rows = []

        # Iterate rows and assign based on remaining quotas
        for idx, row in chunk.iterrows():
            label = row[LABEL_COLUMN]
            # Safety: if unseen label appears, treat as new label with quotas 60/40 from remaining seen counts
            if label not in train_needed:
                # Set quotas on-the-fly (rare in well-formed data)
                train_needed[label] = 0
                test_needed[label] = 0

            # Decide destination
            if written_train[label] < train_needed[label]:
                train_rows.append(row)
                written_train[label] += 1
            elif written_test[label] < test_needed[label]:
                test_rows.append(row)
                written_test[label] += 1
            else:
                # If both quotas for this label are filled (rounding issues), send to set with fewer total rows for that label
                # This keeps split stable if counts slightly off due to rounding.
                if written_train[label] < written_test[label]:
                    train_rows.append(row)
                    written_train[label] += 1
                else:
                    test_rows.append(row)
                    written_test[label] += 1

        # Convert to DataFrames
        if train_rows:
            train_df = pd.DataFrame(train_rows)
            header = not os.path.exists(train_path)
            train_df.to_csv(train_path, mode="a", index=False, header=header)
        if test_rows:
            test_df = pd.DataFrame(test_rows)
            header = not os.path.exists(test_path)
            test_df.to_csv(test_path, mode="a", index=False, header=header)

    # Final counts for reporting
    final_train_counts = dict(written_train)
    final_test_counts = dict(written_test)

    # Optional: write a report per input file
    if SAVE_INTERMEDIATE_REPORTS:
        report_text = build_report(
            file_path=file_path,
            total_samples=total_rows,
            label_counts=label_counts,
            train_counts=final_train_counts,
            test_counts=final_test_counts
        )
        write_report_text(report_text, file_path)

    # Print quick console summary
    print(f"Done: {file_path}")
    print("Train counts per label:", final_train_counts)
    print("Test counts per label:", final_test_counts)

def main():
    for root, dirs, files in os.walk(PARENT_FOLDER):
        for file in files:
            if not file.endswith(".csv"):
                continue
            file_path = os.path.join(root, file)
            print(f"Processing file: {file_path}")
            split_and_write(file_path)

if __name__ == "__main__":
    main()
