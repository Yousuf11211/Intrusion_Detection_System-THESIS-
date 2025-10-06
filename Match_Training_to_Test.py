import os
import pandas as pd

# --- CONFIG ---
train_folder = "Balanced_Training_2018"
test_base_folder = "Test_2018"
output_folder = "Balanced_Test_2018"
base_test_file = "test_2_validated.csv"

# Create output folder
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

    # Load training CSV
    train_df = pd.read_csv(train_path, low_memory=False)
    train_df.columns = train_df.columns.str.lower().str.strip()

    # Get training feature columns (excluding label)
    train_columns = [c for c in train_df.columns if c != "label"]

    # Copy base test
    test_aligned = base_test.copy()

    # Add missing columns (present in training but not in test)
    for col in train_columns:
        if col not in test_aligned.columns:
            test_aligned[col] = 0

    # Remove extra columns not in training
    test_aligned = test_aligned[train_columns]

    # Save aligned test CSV
    output_name = f"test_for_{os.path.splitext(train_file)[0]}.csv"
    output_path = os.path.join(output_folder, output_name)
    test_aligned.to_csv(output_path, index=False)

    print(f"Saved aligned test file → {output_path}")
    print(f"Matched columns: {len(train_columns)}")

print("\nAll test sets have been aligned and saved successfully.")
