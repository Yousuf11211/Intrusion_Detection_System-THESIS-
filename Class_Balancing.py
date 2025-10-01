import os
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

# --- 1. Configuration ---
# Set the main folder path. The script will search inside this folder and all its subfolders.
input_folder = "Downscale_Csv_2018"


def process_csv_file(input_csv_path):
    """
    This function performs the full resampling process on a single CSV file.
    """
    print(f"\n--- Processing file: {input_csv_path} ---")

    try:
        # --- 2. Load and Prepare Data ---
        print("  Loading data...")
        df = pd.read_csv(input_csv_path)

        # Drop rows with any NaN values as a final cleaning step before splitting
        df.dropna(inplace=True)

        X = df.drop('label', axis=1)
        y = df['label']

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        label_mapping = {label: i for i, label in enumerate(le.classes_)}
        rev_label_mapping = {i: label for label, i in label_mapping.items()}

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42,
                                                            stratify=y_encoded)

        original_counts = Counter(y_train)
        print("  Original Training Set Counts:")
        for label_int, count in sorted(original_counts.items()):
            print(f"    {rev_label_mapping[label_int]}: {count}")

        # --- 3. Apply Custom Resampling to the Training Data ---
        print("  Applying custom resampling...")
        # Step 3.1: Undersample
        undersample_strategy = {}
        benign_label_int = label_mapping.get('Benign')
        if benign_label_int is None:
            print("  'Benign' class not found. Skipping undersampling.")
        else:
            benign_count = original_counts.get(benign_label_int, 0)
            for label_int, count in original_counts.items():
                if count > 1_000_000 and label_int != benign_label_int:
                    undersample_strategy[label_int] = benign_count

            rus = RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=42)
            X_inter, y_inter = rus.fit_resample(X_train, y_train)

        # Step 3.2: Oversample
        intermediate_counts = Counter(y_inter)
        oversample_strategy = {label: count for label, count in intermediate_counts.items()}
        for label_int, count in intermediate_counts.items():
            if label_int == benign_label_int: continue
            if 200_000 <= count < 300_000:
                oversample_strategy[label_int] = 600_000
            elif 100_000 <= count < 200_000:
                oversample_strategy[label_int] = 400_000
            elif 20_000 <= count < 30_000:
                oversample_strategy[label_int] = 200_000
            elif 10_000 <= count < 20_000:
                oversample_strategy[label_int] = 100_000
            elif 300 <= count < 400:
                oversample_strategy[label_int] = 90_000
            elif 200 <= count < 300:
                oversample_strategy[label_int] = 70_000
            elif 100 <= count < 200:
                oversample_strategy[label_int] = 50_000
            elif 40 <= count < 50:
                oversample_strategy[label_int] = 40_000
            elif 1 <= count < 10:
                oversample_strategy[label_int] = 30_000

        # Find the smallest class size to safely set k_neighbors
        min_class_size = min(intermediate_counts.values())
        k_neighbors = max(1, min_class_size - 1)

        smote = SMOTE(sampling_strategy=oversample_strategy, k_neighbors=k_neighbors, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_inter, y_inter)

        # --- 4. Combine and Save the New Datasets ---
        print("  Saving new files...")
        # Recreate the original directory and base name
        dir_name = os.path.dirname(input_csv_path)
        base_name = os.path.splitext(os.path.basename(input_csv_path))[0]

        # Save the balanced training set
        df_balanced_train = pd.DataFrame(X_resampled, columns=X.columns)
        df_balanced_train['label'] = le.inverse_transform(y_resampled)
        train_output_path = os.path.join(dir_name, f"{base_name}_balanced_train.csv")
        df_balanced_train.to_csv(train_output_path, index=False)
        print(f"    ----> Saved balanced training set to: {train_output_path}")

        # Save the untouched test set
        df_test = pd.DataFrame(X_test, columns=X.columns)
        df_test['label'] = le.inverse_transform(y_test)
        test_output_path = os.path.join(dir_name, f"{base_name}_test.csv")
        df_test.to_csv(test_output_path, index=False)
        print(f"    ----> Saved test set to: {test_output_path}")

    except Exception as e:
        print(f"Could not process file {input_csv_path}. Error: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    for dirpath, _, filenames in os.walk(input_folder):
        for filename in filenames:
            if filename.endswith('.csv') and not filename.endswith(('_balanced_train.csv', '_test.csv')):
                file_path = os.path.join(dirpath, filename)
                process_csv_file(file_path)
    print("\nAll files processed.")