import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Parent folder containing all CSVs
parent_folder = "Raw_Data_2017"

# We'll store dataframes here temporarily
dfs = []

# Read CSVs one by one
for root, dirs, files in os.walk(parent_folder):
    for file in files:
        if not file.endswith(".csv"):
            continue
        file_path = os.path.join(root, file)
        print(f"Loading {file_path} ...")

        try:
            df = pd.read_csv(file_path, low_memory=False)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        # Check if 'label' column exists
        label_cols = [col for col in df.columns if col.lower() == 'label']
        if not label_cols:
            print(f"No label column in {file_path}, skipping.")
            continue

        # Standardize label column
        df.rename(columns={label_cols[0]: 'label'}, inplace=True)
        dfs.append(df)

# Combine all CSVs
data = pd.concat(dfs, ignore_index=True)
print(f"Combined dataset shape: {data.shape}")

# Separate features and target
y = LabelEncoder().fit_transform(data['label'])
X = data.drop(columns=['label'])

# Encode categorical features if any
for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rf.fit(X, y)

# Feature importance
importances = rf.feature_importances_
feat_imp_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance_pct': 100 * importances
}).sort_values(by='Importance_pct', ascending=False)

# --- Save TXT report ---
with open("Feature_Importance_Report.txt", "w", encoding="utf-8") as f:
    f.write("Feature Importance Report\n")
    f.write("="*50 + "\n")
    for idx, row in feat_imp_df.iterrows():
        f.write(f"{row['Feature']:<30}: {row['Importance_pct']:.4f}%\n")
print("Feature importance report saved as Feature_Importance_Report.txt")

# --- Save CSV ---
feat_imp_df.to_csv("Feature_Importance.csv", index=False)
print("Feature importance saved to Feature_Importance.csv")

# --- Plot Top 20 ---
plt.figure(figsize=(12,6))
feat_imp_df.head(20).plot.bar(x='Feature', y='Importance_pct', legend=False)
plt.title("Top 20 Feature Importances")
plt.ylabel("Importance (%)")
plt.tight_layout()
plt.savefig("Top20_Feature_Importance.png", dpi=300)
plt.show()
print("Top 20 features plot saved as Top20_Feature_Importance.png")
