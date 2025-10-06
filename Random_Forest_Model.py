import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# --- CONFIG ---
csv_file = "Processed_Data_2017/Merged_Shuffled.csv"
model_folder = "Trained_Model_Completedata"
report_folder = "Testing_2017_report"
os.makedirs(model_folder, exist_ok=True)
os.makedirs(report_folder, exist_ok=True)

# Switch mode
train_full_data = False  # Set True = use ALL data, False = do 80/20 split

# --- LOAD DATA ---
print("Loading dataset...")
data = pd.read_csv(csv_file, low_memory=False)
data.columns = data.columns.str.lower()

if 'label' not in data.columns:
    raise ValueError("CSV does not have 'label' column.")

X = data.drop(columns=['label'])
y_raw = data['label']

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y_raw)

# Save label mapping
mapping_path = os.path.join(model_folder, "label_mapping.txt")
with open(mapping_path, "w", encoding="utf-8") as f:
    f.write("Label Encoding Mapping:\n")
    f.write("="*40 + "\n")
    for cls, num in zip(le.classes_, range(len(le.classes_))):
        f.write(f"{cls:<30}: {num}\n")
print(f"Label mapping saved to {mapping_path}")

# Encode categorical features
for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# --- TRAINING ---
if train_full_data:
    print("Training Random Forest on FULL dataset...")
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X, y)

    # Save model
    model_path = os.path.join(model_folder, "random_forest_full.pkl")
    joblib.dump(rf, model_path)
    print(f"Model trained on full dataset and saved to {model_path}")

else:
    print("Splitting data (80% train / 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    # Save model
    model_path = os.path.join(model_folder, "random_forest_split.pkl")
    joblib.dump(rf, model_path)
    print(f"Model trained with 80/20 split and saved to {model_path}")

    # --- EVALUATION ---
    print("Evaluating on test data...")
    y_pred = rf.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=le.classes_)
    print("\nClassification Report:\n")
    print(report)

    # Save report
    report_path = os.path.join(report_folder, "test_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Test Report\n")
        f.write("="*80 + "\n\n")
        f.write(report)
    print(f"Report saved to {report_path}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print("\nConfusion Matrix (rows=actual, cols=predicted):\n")
    print(cm_df)

    cm_path = os.path.join(report_folder, "confusion_matrix.csv")
    cm_df.to_csv(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
