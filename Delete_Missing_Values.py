import os
import pandas as pd

# Main folder with all raw datasets
main_folder = "Raw_Data"

# Output folder for cleaned csv's
output_folder = "Cleaned_Datasets"
os.makedirs(output_folder, exist_ok=True)

for root, dirs, files in os.walk(main_folder):
    for file in files:
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(root, file)
        print(f"\nProcessing {file_path} ...")

        try:
            df = pd.read_csv(file_path, low_memory=False)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        # Row counts before cleaning
        before = len(df)

        # Drop rows with ANY missing values
        df = df.dropna()

        # Row counts after cleaning
        after = len(df)
        dropped = before - after
        perc = round((dropped / before) * 100, 2) if before > 0 else 0

        print(f"Rows before: {before}")
        print(f"Rows after : {after}")
        print(f"Dropped    : {dropped} rows ({perc}%)")

        # Save cleaned file (keep same subfolder structure under Cleaned_Datasets)
        relative_path = os.path.relpath(root, main_folder)
        cleaned_subfolder = os.path.join(output_folder, relative_path)
        os.makedirs(cleaned_subfolder, exist_ok=True)

        output_file = os.path.join(cleaned_subfolder, file)
        df.to_csv(output_file, index=False)

        print(f"Saved cleaned file to {output_file}")

        # Free memory
        del df
