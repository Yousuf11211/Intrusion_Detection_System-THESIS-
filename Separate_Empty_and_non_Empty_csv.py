import os
import pandas as pd

def analyze_csvs(parent_folder, chunksize=2_000_000, report_file="CSV_Report.csv"):
    results = []

    print(f"Analyzing CSVs inside: {parent_folder}\n")

    # Walk through all subfolders and files
    for root, dirs, files in os.walk(parent_folder):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)

                total_rows = 0
                complete_rows = 0

                print(f"Processing file: {file_path}")

                # Read CSV in chunks
                for chunk in pd.read_csv(file_path, chunksize=chunksize, dtype=str, low_memory=False):
                    total_rows += len(chunk)
                    complete_rows += chunk.dropna().shape[0]

                missing_rows = total_rows - complete_rows

                print(f"  Total rows: {total_rows:,}")
                print(f"  Rows without missing values: {complete_rows:,}")
                print(f"  Rows with missing values: {missing_rows:,}")
                print("-" * 60)

                # Append to results
                results.append({
                    "File Path": file_path,
                    "Total Rows": total_rows,
                    "Complete Rows": complete_rows,
                    "Missing Rows": missing_rows
                })

    # Save results to a summary CSV
    df_report = pd.DataFrame(results)
    df_report.to_csv(report_file, index=False)
    print(f"\nSummary report saved to: {report_file}")

# Example usage
parent_folder = "Raw_Data_2018"
analyze_csvs(parent_folder)
