import os
import pandas as pd

# Datasets folder to scan
main_folder = "Raw_Data"

# Report folder to save missing value details
output_folder = "Missing_Value_Report"
os.makedirs(output_folder, exist_ok=True)

for root, dirs, files in os.walk(main_folder):
    for file in files:
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(root, file)
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

        # Build report content
        report_lines = []
        report_lines.append(f"Report for {file_path}")
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

        # Create same subfolder structure inside Missing_Value_Report
        relative_path = os.path.relpath(root, main_folder)
        report_subfolder = os.path.join(output_folder, relative_path)
        os.makedirs(report_subfolder, exist_ok=True)

        # Save report
        output_file = os.path.join(report_subfolder, f"{os.path.splitext(file)[0]}_report.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        print(f"Saved scan report to {output_file}")

        # Free memory
        del df
