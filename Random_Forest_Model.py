import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Paths
csv_file = "Processed_Data_2017/Merged_Shuffled.csv"  # your combined CSV
model_folder = "Trained_Model"
os.makedirs(model_folder, exist_ok=True)

# Load data
data = pd.read_csv(csv_file, low_memory=False)

# Normalize column names
data.columns = data.columns.str.lower()

# Separate features and target
if 'label' not in data.columns:
    raise ValueError("CSV does not have 'label' column.")

X = data.drop(columns=['label'])
y_raw = data['label']

# Encode target labels (multi-class)
le = LabelEncoder()
y = le.fit_transform(y_raw)

# Save mapping of numeric labels to original
mapping_path = os.path.join(model_folder, "label_mapping.txt")
with open(mapping_path, "w", encoding="utf-8") as f:
    f.write("Label Encoding Mapping:\n")
    f.write("="*40 + "\n")
    for cls, num in zip(le.classes_, range(len(le.classes_))):
        f.write(f"{cls:<30}: {num}\n")
print(f"Label mapping saved to {mapping_path}")

# Encode categorical features if any
for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# Train Random Forest (multi-class)
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rf.fit(X, y)

# Save trained model
model_path = os.path.join(model_folder, "random_forest_model.pkl")
joblib.dump(rf, model_path)
print(f"Trained model saved to {model_path}")
