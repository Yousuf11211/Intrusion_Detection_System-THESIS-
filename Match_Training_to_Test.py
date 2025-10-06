import os
import pandas as pd

# --- CONFIG ---
train_folder = "Balanced_Training_2018"
test_base_folder = "Test_2018"
output_folder = "Balanced_Test_2018"
base_test_file = "test_2_validated.csv"

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

# --- LOAD BASE TEST ---
base_test_path = os.path.join(test_base_folder, base_test_file)
print(f"Loading base test file: {base_test_path}")
base_test = pd.read_csv(base_test_path, low_memory=False)
base_test.columns = base_test.columns.str.lower().str.strip()

# --- PROCESS EACH TRAINING FILE ---
train_files = [f for f in os.listdir(train_folder) if f.endswith(".csv")]
print(f"Found {len(train_files)} training files in {train_folder}.")

for train_file in train_files:
    train_path = os.path.join(train_folder, train_file)
    print(f"\nProcessing training file: {train_file}")

    # Load training dataset
    train_df = pd.read_csv(train_path, low_memory=False)
    train_df.columns = train_df.columns.str.lower().str.strip()

    # Get columns (exclude label)
    train_columns = [c for c in train_df.columns if c != "label"]

    # Copy base test to modify
    test_aligned = base_test.copy()

    # Add missing columns present in training but not in test
    missing_cols = [c for c in train_columns if c not in test_aligned.columns]
    for col in missing_cols:
        test_aligned[col] = 0

    # Drop extra columns not in training
    extra_cols = [c for c in test_aligned.columns if c not in train_columns]
    if extra_cols:
        test_aligned = test_aligned.drop(columns=extra_cols)

    # Reorder columns to match training exactly
    test_aligned = test_aligned[train_columns]

    # Save aligned test CSV with same name as training file
    output_path = os.path.join(output_folder, train_file)
    test_aligned.to_csv(output_path, index=False)

    print(f"Saved aligned test file as: {output_path}")
    print(f"Matched columns: {len(train_columns)}")
    if missing_cols:
        print(f"Added missing columns: {missing_cols}")
    if extra_cols:
        print(f"Removed extra columns: {extra_cols}")

print("\nAll test files successfully aligned and saved to Balanced_Test_2018.")
