import os
import pandas as pd
from pygame.transform import threshold

from Check_Missing import dataset_folders, output_folder

#dataset folders
dataset_folders = ["Dataset_1","Dataset_2", "Dataset_3", "Dataset_4"]

#output folders for cleaned csv's
output_folder = "Cleaned_Datasets"
os.makedirs(output_folder,exist_ok=True)

#1% to delete
threshold = 0.01

for folder in dataset_folders:
    if not os.path.exists(folder):
        print(f"Folder {folder} not found")
        continue


    #subfolder for each dataset
    cleaned_subfolder = os.path.join(output_folder, folder)
    os.makedirs(cleaned_subfolder, exist_ok=True)

    for file in os.listdir(folder):
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(folder, file)
        print(f"\nProcessing {file_path} ...")

        try:
            df = pd.read_csv(file_path, low_memory=False)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        #Tracks which colums were cleaned
        cols_dropped_rows = []

        for col in df.columns:
            missing_fraction = df[col].isna().mean()  # fraction of missing values
            if 0 < missing_fraction < threshold:
                # Drop rows where this column is missing
                before = len(df)
                df = df.dropna(subset=[col])
                after = len(df)
                cols_dropped_rows.append((col, before - after, round(missing_fraction * 100, 4)))

        #Save cleaned file
        output_file = os.path.join(cleaned_subfolder, file)
        df.to_csv(output_file, index=False)

        print(f"Saved cleaned file to {output_file}")
        if cols_dropped_rows:
            print("  Columns where rows were dropped (<1% missing):")
            for col, dropped, perc in cols_dropped_rows:
                print(f"    {col:<25} → Dropped {dropped} rows ({perc}%)")
        else:
            print("  No rows dropped (all missing >=1% or no missing).")

        #free memory
        del df