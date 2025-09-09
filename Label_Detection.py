import os
import pandas as pd

#List of input folders containing datasets
Data_set = ["Dataset_1", "Dataset_2", "Dataset_3", "Dataset_4"]

#folder to save label details
Output_folder = "Labelled_Reports"
os.makedirs(Output_folder, exist_ok=True)

#Check if the folder exist
for folder in Data_set:
    if not os.path.exists(folder):
        print(f"Folder {folder} does not exist.")
        continue

    #Scanning CSV
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            file_path = os.path.join(folder, file)
            print(f"Processing file: {file_path}")


            #Read CSV file
            file_path = os.path.join(folder, file)
            print(f"Processing file: {file_path}")
            try:
                df = pd.read_csv(file_path, low_memory=False)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
            

            #Finds the label column
            label_col = None
            for col in df.columns:
                if col.lower() == "label":
                    label_col = col
                    break

            if label_col is None:
                print(f"No 'label' column found in {file_path}. Skipping file.")
                continue


            #count labels
            label_counts = df[label_col].value_counts().to_dict()
            total_samples = len(df)
            benign_count = sum(v for k, v in label_counts.items() if str(k).lower() == "benign")
            attack_count = total_samples - benign_count

            #saving Attack Report Details as txt
            report_lines =[]
            report_lines.append(f"Report for {folder}/{file}")
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
            output_file = os.path.join(Output_folder, f"{folder}_{os.path.splitext(file)[0]}.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report_text)

            print(f"Saved report to {output_file}")


