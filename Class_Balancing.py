import os
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

# ======================
# CONFIGURATION
# ======================
INPUT_FOLDER = "Training_2018"
OUTPUT_FOLDER = "Balanced_Training_2018"


def get_csv_files(folder):
    """Find all CSV files in the folder (excluding already processed ones)"""
    csv_files = []
    for filename in os.listdir(folder):
        if (filename.endswith('.csv') and
                not filename.endswith(('_balanced_train.csv', '_test.csv'))):
            csv_files.append(os.path.join(folder, filename))
    return csv_files


def display_label_counts(y, le):
    """Display label counts in a user-friendly way"""
    counts = Counter(y)
    rev_mapping = {i: label for i, label in enumerate(le.classes_)}

    print("\nCurrent label distribution:")
    print("-" * 40)

    # Show Benign first if it exists
    benign_key = None
    for key, val in rev_mapping.items():
        if val.lower() == 'benign':
            benign_key = key
            print(f"  Benign: {counts.get(key, 0):,}")
            break

    # Show other labels
    for key in sorted(counts.keys()):
        if key != benign_key:
            label_name = rev_mapping[key]
            print(f"  {label_name}: {counts.get(key, 0):,}")

    print("-" * 40)
    print(f"Total samples: {sum(counts.values):,}")


def get_user_target_counts(y, le):
    """Ask user for target counts for each label"""
    counts = Counter(y)
    rev_mapping = {i: label for i, label in enumerate(le.classes_)}
    target_strategy = {}

    print("\n=== SET TARGET SAMPLE COUNTS ===")
    print("Enter the desired number of samples for each class.")
    print("(Current count shown in parentheses)")

    # Ask for Benign first
    benign_key = None
    for key, val in rev_mapping.items():
        if val.lower() == 'benign':
            benign_key = key
            current = counts.get(key, 0)
            while True:
                try:
                    target = input(f"\nBenign (current: {current:,}): ")
                    target = int(target)
                    if target > 0:
                        target_strategy[key] = target
                        break
                    else:
                        print("Please enter a positive number.")
                except ValueError:
                    print("Please enter a valid number.")
            break

    # Ask for other labels
    for key in sorted(counts.keys()):
        if key != benign_key:
            label_name = rev_mapping[key]
            current = counts.get(key, 0)
            while True:
                try:
                    target = input(f"{label_name} (current: {current:,}): ")
                    target = int(target)
                    if target > 0:
                        target_strategy[key] = target
                        break
                    else:
                        print("Please enter a positive number.")
                except ValueError:
                    print("Please enter a valid number.")

    return target_strategy


def apply_resampling(X_train, y_train, target_strategy):
    """Apply undersampling and/or oversampling to reach target counts"""
    current_counts = Counter(y_train)

    # Determine what needs undersampling and oversampling
    undersample_strategy = {}
    oversample_strategy = {}

    for label, target in target_strategy.items():
        current = current_counts.get(label, 0)
        if current > target:
            undersample_strategy[label] = target
        elif current < target:
            oversample_strategy[label] = target

    X_resampled, y_resampled = X_train.copy(), y_train.copy()

    # Apply undersampling if needed
    if undersample_strategy:
        print("  Applying undersampling...")
        rus = RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X_resampled, y_resampled)

    # Apply oversampling if needed
    if oversample_strategy:
        print("  Applying oversampling (SMOTE)...")
        # Calculate safe k_neighbors
        min_class_size = min(Counter(y_resampled).values())
        k_neighbors = max(1, min(min_class_size - 1, 5))

        smote = SMOTE(sampling_strategy=oversample_strategy, k_neighbors=k_neighbors, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_resampled, y_resampled)

    return X_resampled, y_resampled


def process_single_file(file_path, target_strategy, output_folder):
    """Process a single CSV file with given target strategy"""
    try:
        print(f"\nProcessing: {os.path.basename(file_path)}")

        # Load data
        df = pd.read_csv(file_path)
        df.dropna(inplace=True)

        X = df.drop('label', axis=1)
        y = df['label']

        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Apply resampling
        X_resampled, y_resampled = apply_resampling(X_train, y_train, target_strategy)

        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Save balanced training set
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        # Create balanced training DataFrame
        df_train = pd.DataFrame(X_resampled, columns=X.columns)
        df_train['label'] = le.inverse_transform(y_resampled)

        train_path = os.path.join(output_folder, f"{base_name}_balanced.csv")
        df_train.to_csv(train_path, index=False)

        # Save test set
        df_test = pd.DataFrame(X_test, columns=X.columns)
        df_test['label'] = le.inverse_transform(y_test)

        test_path = os.path.join(output_folder, f"{base_name}_test.csv")
        df_test.to_csv(test_path, index=False)

        print(f"  ✓ Saved: {os.path.basename(train_path)}")
        print(f"  ✓ Saved: {os.path.basename(test_path)}")

        # Show final counts
        final_counts = Counter(y_resampled)
        rev_mapping = {i: label for i, label in enumerate(le.classes_)}
        print("  Final training set counts:")
        for label_int, count in sorted(final_counts.items()):
            print(f"    {rev_mapping[label_int]}: {count:,}")

        return True

    except Exception as e:
        print(f"  ERROR processing {file_path}: {e}")
        return False


def main():
    print("=== CSV CLASS BALANCING TOOL ===")
    print(f"Input folder: {INPUT_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")

    # Find all CSV files
    csv_files = get_csv_files(INPUT_FOLDER)

    if not csv_files:
        print("No CSV files found in the input folder!")
        return

    print(f"\nFound {len(csv_files)} CSV file(s):")
    for i, file_path in enumerate(csv_files, 1):
        print(f"  {i}. {os.path.basename(file_path)}")

    # Ask user which files to process
    files_to_process = []
    print("\n=== FILE SELECTION ===")
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        response = input(f"\nProcess '{filename}'? (y/n): ").lower()
        if response == 'y':
            files_to_process.append(file_path)
            print(f"  ✓ Added to processing list")
        else:
            print(f"  ✗ Skipped")

    if not files_to_process:
        print("No files selected for processing!")
        return

    print(f"\nWill process {len(files_to_process)} file(s)")

    # Process first file to get target strategy
    first_file = files_to_process[0]
    print(f"\n=== ANALYZING FIRST FILE: {os.path.basename(first_file)} ===")

    # Load first file to show label distribution
    try:
        df_first = pd.read_csv(first_file)
        df_first.dropna(inplace=True)
        y_first = df_first['label']
        le_first = LabelEncoder()
        y_encoded_first = le_first.fit_transform(y_first)

        display_label_counts(y_encoded_first, le_first)

        # Get target counts from user
        target_strategy = get_user_target_counts(y_encoded_first, le_first)

    except Exception as e:
        print(f"Error analyzing first file: {e}")
        return

    # Ask if user wants to use same strategy for all files
    if len(files_to_process) > 1:
        print(f"\n=== APPLY TO ALL FILES? ===")
        same_strategy = input("Use the same target counts for all files? (y/n): ").lower()
        use_same_strategy = (same_strategy == 'y')
    else:
        use_same_strategy = True

    # Process all files
    print(f"\n=== PROCESSING FILES ===")
    successful = 0

    for i, file_path in enumerate(files_to_process):
        print(f"\n[{i + 1}/{len(files_to_process)}]")

        if not use_same_strategy and i > 0:
            # Ask for new strategy for this file
            print(f"Setting target counts for: {os.path.basename(file_path)}")
            try:
                df_current = pd.read_csv(file_path)
                df_current.dropna(inplace=True)
                y_current = df_current['label']
                le_current = LabelEncoder()
                y_encoded_current = le_current.fit_transform(y_current)

                display_label_counts(y_encoded_current, le_current)
                target_strategy = get_user_target_counts(y_encoded_current, le_current)

            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")
                continue

        # Process the file
        if process_single_file(file_path, target_strategy, OUTPUT_FOLDER):
            successful += 1

    print(f"\n=== SUMMARY ===")
    print(f"Successfully processed: {successful}/{len(files_to_process)} files")
    print(f"Output saved to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
