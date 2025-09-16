import os
import pandas as pd
import matplotlib.pyplot as plt

# Parent folder containing all CSVs
parent_folder = "Raw_Data_2017"

# Collect global label counts
overall_counts = {}

# Walk through all CSV files
for root, dirs, files in os.walk(parent_folder):
    for file in files:
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(root, file)
        print(f"Processing {file_path}...")

        try:
            df = pd.read_csv(file_path, low_memory=False)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        # Find label column
        label_col = None
        for col in df.columns:
            if col.lower() == "label":
                label_col = col
                break

        if label_col is None:
            print(f"No 'label' column in {file_path}, skipping.")
            continue

        # Count per label and add to overall
        file_counts = df[label_col].value_counts().to_dict()
        for lbl, cnt in file_counts.items():
            overall_counts[lbl] = overall_counts.get(lbl, 0) + cnt

# Convert to DataFrame for saving
summary_df = pd.DataFrame(list(overall_counts.items()), columns=["Label", "Count"])
summary_df.to_csv("Overall_Label_Distribution.csv", index=False)

# --- Visualization ---

# Bar chart
plt.figure(figsize=(10, 6))
summary_df.set_index("Label")["Count"].plot(kind="bar")
plt.title("Overall Class Distribution (Benign vs. Attack Types)")
plt.ylabel("Number of Samples")
plt.xlabel("Label")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Overall_Class_Distribution.png", dpi=300)
plt.show()

print("Saved: Overall_Label_Distribution.csv and Overall_Class_Distribution.png")
