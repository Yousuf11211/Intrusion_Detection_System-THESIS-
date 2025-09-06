import os
import pandas as pd

#Dataset folders
dataset_folders = ["Dataset_1", "Dataset_2", "Dataset_3", "Dataset_4"]

#Report folder to save missing value details
output_folder = "CSV_Scan_Reports"
os.makedirs(output_folder, exist_ok=True)

for folder in dataset_folders:
    if not os.path.exists(folder):
        print(f"Folder {folder} not found, skipping...")
        continue

    for file in os.listdir(folder):
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(folder, file)
        print(f"\nScanning {file_path} ...")

        try:
            df = pd.read_csv(file_path, low_memory=False)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        # Counts total rows and columns
        total_rows, total_cols = df.shape

        # Missing values per column
        missing_counts = df.isna().sum()
        missing_cols = {col: val for col, val in missing_counts.items() if val > 0}

        # Label column detection
        label_col = None
        for col in df.columns:
            if col.lower() == "label":
                label_col = col
                break

        unique_labels = []
        if label_col:
            unique_labels = df[label_col].dropna().unique()

        #Save report
        report_lines = []
        report_lines.append(f"Report for {folder}/{file}")
        report_lines.append("=" * 50)
        report_lines.append(f"Total rows: {total_rows}")
        report_lines.append(f"Total columns: {total_cols}")
        report_lines.append("")

        if not missing_cols:
            report_lines.append("No missing values found in any column.")
        else:
            report_lines.append("Columns with missing values:")
            for col, val in missing_cols.items():
                report_lines.append(f"{col:<25}: {val} missing")
        report_lines.append("")

        if label_col:
            report_lines.append(f"Label column: {label_col}")
            report_lines.append(f"Unique labels (up to 10): {list(unique_labels)[:10]}")
        else:
            report_lines.append("WARNING: No 'Label' column found.")

 
        output_file = os.path.join(output_folder, f"{folder}_{os.path.splitext(file)[0]}_scan.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        print(f"Saved scan report to {output_file}")

        #clear memory
        del df
