import os
import pandas as pd
from collections import Counter, defaultdict

# Parent folder
parent_folder = "Downscale_Csv_2018"

# Parameters
chunk_size = 1_000_000  # adjust based on RAM

# Define ranges for dominance
ranges = [
    (0.95, 1.01, "95-100%"),   # captures almost constant
    (0.90, 0.95, "90-95%"),
    (0.80, 0.90, "80-90%"),
    (0.70, 0.80, "70-80%"),
    (0.60, 0.70, "60-70%"),
    (0.50, 0.60, "50-60%"),
]

for root, dirs, files in os.walk(parent_folder):
    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(root, file)
            print(f"\nProcessing file: {file}")

            # Track counts for each column
            col_counters = defaultdict(Counter)
            total_counts = Counter()

            # Track label distribution (assuming column is named "Label" or "label")
            label_counter = Counter()

            try:
                for chunk in pd.read_csv(file_path, chunksize=chunk_size, dtype=str, low_memory=False):
                    for col in chunk.columns:
                        values = chunk[col].dropna()
                        col_counters[col].update(values)
                        total_counts[col] += len(values)

                    # Track labels
                    if "Label" in chunk.columns:
                        label_counter.update(chunk["Label"].dropna())
                    elif "label" in chunk.columns:
                        label_counter.update(chunk["label"].dropna())

                # Prepare report
                bucketed = {label: [] for _, _, label in ranges}

                for col, counts in col_counters.items():
                    if total_counts[col] == 0:
                        continue
                    most_common_val, most_common_count = counts.most_common(1)[0]
                    ratio = most_common_count / total_counts[col]

                    for low, high, label in ranges:
                        if low <= ratio < high:
                            bucketed[label].append((col, most_common_val, ratio, total_counts[col]))
                            break

                # Save report
                report_path = os.path.join(root, f"{os.path.splitext(file)[0]}_dominance_report.txt")
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(f"Dominance Report for {file}\n")
                    f.write("=" * 60 + "\n\n")

                    # Label distribution
                    if label_counter:
                        total_labels = sum(label_counter.values())
                        f.write("Label Distribution:\n")
                        f.write("-" * 40 + "\n")
                        for lbl, count in label_counter.most_common():
                            perc = (count / total_labels) * 100
                            f.write(f"  {lbl}: {count:,} ({perc:.2f}%)\n")
                        f.write("\n")

                    # Column dominance report
                    for label in bucketed:
                        f.write(f"\nColumns in {label} range:\n")
                        f.write("-" * 40 + "\n")
                        if not bucketed[label]:
                            f.write("  None\n")
                        else:
                            for col, val, ratio, total in bucketed[label]:
                                f.write(f"  {col}: '{val}' = {ratio*100:.2f}% (out of {total:,} rows)\n")

                print(f"Report saved to {report_path}")

            except Exception as e:
                print(f"Error processing {file}: {e}")