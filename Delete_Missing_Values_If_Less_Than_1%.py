import os
import pandas as pd

# Main folder with all raw datasets
main_folder = "Raw_Data"

# Output folder for cleaned csv's
output_folder = "Cleaned_Datasets"
os.makedirs(output_folder, exist_ok=True)


# 1% threshold to delete rows
threshold = 0.01

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

        # Tracks which columns were cleaned
        cols_dropped_rows = []

        for col in df.columns:
            missing_fraction = df[col].isna().mean()  # fraction of missing values
            if 0 < missing_fraction < threshold:
                # Drop rows where this column is missing
                before = len(df)
                df = df.dropna(subset=[col])
                after = len(df)
                cols_dropped_rows.append((col, before - after, round(missing_fraction * 100, 4)))

        # Save cleaned file (keep same subfolder structure under Cleaned_Datasets)
        relative_path = os.path.relpath(root, main_folder)
        cleaned_subfolder = os.path.join(output_folder, relative_path)
        os.makedirs(cleaned_subfolder, exist_ok=True)

        output_file = os.path.join(cleaned_subfolder, file)
        df.to_csv(output_file, index=False)

        print(f"Saved cleaned file to {output_file}")
        if cols_dropped_rows:
            print("  Columns where rows were dropped (<1% missing):")
            for col, dropped, perc in cols_dropped_rows:
                print(f"    {col:<25} → Dropped {dropped} rows ({perc}%)")
        else:
            print("  No rows dropped (all missing >=1% or no missing).")

        # Free memory
        del df
