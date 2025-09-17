import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Parent folder containing all CSVs
parent_folder = "Raw_Data_2017"

# We'll store chunks here to combine for training
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
        if 'label' not in df.columns.str.lower():
            print(f"No label column in {file_path}, skipping.")
            continue

        # Make sure column name is 'label'
        df.rename(columns={col: 'label' for col in df.columns if col.lower() == 'label'}, inplace=True)

        dfs.append(df)

# Combine all files into one DataFrame
data = pd.concat(dfs, ignore_index=True)
print(f"Combined dataset shape: {data.shape}")

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(data['label'])
X = data.drop(columns=['label'])

# Encode categorical features if any
for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rf.fit(X, y)

# Feature importance
importances = rf.feature_importances_
feature_names = X.columns

# Create a DataFrame for importance
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

# Save to CSV
feat_imp_df.to_csv("Feature_Importance.csv", index=False)
print("Feature importance saved to Feature_Importance.csv")

# Plot top 20 features
plt.figure(figsize=(12, 6))
feat_imp_df.head(20).plot.bar(x='Feature', y='Importance', legend=False)
plt.title("Top 20 Feature Importances")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("Top20_Feature_Importance.png", dpi=300)
plt.show()
