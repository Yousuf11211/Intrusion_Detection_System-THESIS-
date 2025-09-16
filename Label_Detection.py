import os
import pandas as pd

# Parent folder containing all datasets
parent_folder = "Cleaned_Datasets"

# Folder to save label reports
output_folder = "Labelled_Reports"
os.makedirs(output_folder, exist_ok=True)

# Walk through all subfolders and files under parent_folder
for root, dirs, files in os.walk(parent_folder):
    for file in files:
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(root, file)
        print(f"Processing file: {file_path}")

        try:
            df = pd.read_csv(file_path, low_memory=False)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        # Find the label column
        label_col = None
        for col in df.columns:
            if col.lower() == "label":
                label_col = col
                break

        if label_col is None:
            print(f"No 'label' column found in {file_path}. Skipping file.")
            continue

        # Count labels
        label_counts = df[label_col].value_counts().to_dict()
        total_samples = len(df)
        benign_count = sum(v for k, v in label_counts.items() if str(k).lower() == "benign")
        attack_count = total_samples - benign_count

        # Build report content
        report_lines = []
        report_lines.append(f"Report for {file_path}")
        report_lines.append("=" * 50)
        report_lines.append(f"Total samples: {total_samples}")
        report_lines.append(f"Benign: {benign_count}")
        report_lines.append(f"Attacks: {attack_count}")
        report_lines.append("")
        report_lines.append("Breakdown by label:")
        report_lines.append("-" * 30)

        for label, count in label_counts.items():
            report_lines.append(f"{label:<25}: {count}")

        report_text = "\n".join(report_lines)

        # Create report filename (keep folder structure flattened)
        relative_path = os.path.relpath(file_path, parent_folder)
        relative_name = relative_path.replace(os.sep, "_")  # replace slashes with _
        output_file = os.path.join(output_folder, f"{os.path.splitext(relative_name)[0]}.txt")

        # Save report
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_text)

        print(f"Saved report to {output_file}")
