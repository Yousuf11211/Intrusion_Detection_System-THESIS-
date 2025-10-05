import os
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

# ===== CONFIGURATION =====
INPUT_FOLDER = "Training_2018"
OUTPUT_FOLDER = "Balanced_Training_2018"


# ===== FUNCTIONS =====
def get_csv_files(folder):
    """Return all CSV files in the folder"""
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")]


def display_label_counts(y, le):
    """Display label counts"""
    counts = Counter(y)
    rev_mapping = {i: label for i, label in enumerate(le.classes_)}
    benign_key = next((k for k, v in rev_mapping.items() if v.lower() == "benign"), None)
    print("\nLabel distribution:")
    if benign_key is not None:
        print(f"  Benign: {counts.get(benign_key, 0):,}")
    for k in sorted(counts.keys()):
        if k != benign_key:
            print(f"  {rev_mapping[k]}: {counts.get(k, 0):,}")
    print(f"Total samples: {sum(counts.values()):,}")


def get_user_target_counts(y, le):
    """Ask user for target counts for each label"""
    counts = Counter(y)
    rev_mapping = {i: label for i, label in enumerate(le.classes_)}
    target_strategy = {}
    benign_key = next((k for k, v in rev_mapping.items() if v.lower() == "benign"), None)

    if benign_key is not None:
        current = counts.get(benign_key, 0)
        while True:
            try:
                target = int(input(f"Benign (current: {current:,}): "))
                if target > 0:
                    target_strategy[benign_key] = target
                    break
            except ValueError:
                print("Enter a valid positive number.")

    for k in sorted(counts.keys()):
        if k != benign_key:
            current = counts.get(k, 0)
            while True:
                try:
                    target = int(input(f"{rev_mapping[k]} (current: {current:,}): "))
                    if target > 0:
                        target_strategy[k] = target
                        break
                except ValueError:
                    print("Enter a valid positive number.")

    return target_strategy


def apply_resampling(X, y, target_strategy):
    """Apply undersampling and oversampling to reach target counts"""
    current_counts = Counter(y)
    undersample = {c: t for c, t in target_strategy.items() if current_counts[c] > t}
    oversample = {c: t for c, t in target_strategy.items() if current_counts[c] < t}

    X_res, y_res = X.copy(), y.copy()

    if undersample:
        rus = RandomUnderSampler(sampling_strategy=undersample, random_state=42)
        X_res, y_res = rus.fit_resample(X_res, y_res)

    if oversample:
        min_class_size = min(Counter(y_res).values())
        k_neighbors = max(1, min(min_class_size - 1, 5))
        smote = SMOTE(sampling_strategy=oversample, k_neighbors=k_neighbors, random_state=42)
        X_res, y_res = smote.fit_resample(X_res, y_res)

    return X_res, y_res


def balance_csv(df, target_strategy):
    """Balance a single CSV and return balanced DataFrame and LabelEncoder"""
    X = df.drop("label", axis=1)
    y = df["label"]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_bal, y_bal = apply_resampling(X, y_enc, target_strategy)
    df_bal = pd.DataFrame(X_bal, columns=X.columns)
    df_bal["label"] = le.inverse_transform(y_bal)
    return df_bal, le


# ===== MAIN SCRIPT =====
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    csv_files = get_csv_files(INPUT_FOLDER)

    if not csv_files:
        print("No CSV files found in the input folder!")
        return

    print(f"Found {len(csv_files)} CSV file(s).")

    # --- Interactive file selection ---
    files_to_process = []
    for file_path in csv_files:
        response = input(f"Process '{os.path.basename(file_path)}'? (y/n): ").lower()
        if response == 'y':
            files_to_process.append(file_path)
            print(f"  ✓ Added to processing list")
        else:
            print(f"  ✗ Skipped")

    if not files_to_process:
        print("No files selected for processing!")
        return

    # --- Similar files question ---
    similar_flag = input("Are all files similar (same rows, different columns)? (y/n): ").lower()

    if similar_flag == "y":
        # Use the CSV with most columns as base
        base_file = max(files_to_process, key=lambda f: pd.read_csv(f, nrows=1).shape[1])
        df_base = pd.read_csv(base_file)
        le_base = LabelEncoder()
        y_base = le_base.fit_transform(df_base["label"])
        display_label_counts(y_base, le_base)
        target_strategy = get_user_target_counts(y_base, le_base)
        df_bal, le_bal = balance_csv(df_base, target_strategy)
        print(f"Balanced base CSV: {os.path.basename(base_file)} -> shape {df_bal.shape}")

        # Apply column matching and save each file
        for f in files_to_process:
            df_current = pd.read_csv(f)
            cols_out = [c for c in df_current.columns if c in df_bal.columns]
            df_out = df_bal[cols_out + ["label"]]
            out_file = os.path.join(OUTPUT_FOLDER, os.path.basename(f).replace(".csv", "_balanced.csv"))
            df_out.to_csv(out_file, index=False)
            print(f"Saved balanced CSV: {os.path.basename(out_file)}")

    else:
        # Balance each file independently
        for f in files_to_process:
            df = pd.read_csv(f)
            le_file = LabelEncoder()
            y_file = le_file.fit_transform(df["label"])
            display_label_counts(y_file, le_file)
            target_strategy = get_user_target_counts(y_file, le_file)
            df_bal, _ = balance_csv(df, target_strategy)
            out_file = os.path.join(OUTPUT_FOLDER, os.path.basename(f).replace(".csv", "_balanced.csv"))
            df_bal.to_csv(out_file, index=False)
            print(f"Saved balanced CSV: {os.path.basename(out_file)}")

    print("\nAll files processed.")


if __name__ == "__main__":
    main()
