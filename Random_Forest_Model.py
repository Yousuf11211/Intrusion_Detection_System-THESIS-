import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Paths
csv_file = "Processed_Data_2017/Merged_Shuffled.csv"  # your combined CSV
model_folder = "Trained_Model_Completedata"
report_folder = "Testing_2017_report"
os.makedirs(model_folder, exist_ok=True)
os.makedirs(report_folder, exist_ok=True)

# Load data
print("Loading dataset...")
data = pd.read_csv(csv_file, low_memory=False)

# Normalize column names
data.columns = data.columns.str.lower()

# Ensure label column exists
if 'label' not in data.columns:
    raise ValueError("CSV does not have 'label' column.")

# Separate features and target
X = data.drop(columns=['label'])
y_raw = data['label']

# Encode target labels
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

# Encode categorical features if any
for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# Train-test split
print("Splitting data (80% train / 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

# Save trained model
model_path = os.path.join(model_folder, "random_forest_model.pkl")
joblib.dump(rf, model_path)
print(f"Trained model saved to {model_path}")

# Evaluate model
print("Evaluating on test data...")
y_pred = rf.predict(X_test)

# Classification report
report = classification_report(y_test, y_pred, target_names=le.classes_)
print("\nClassification Report:\n")
print(report)

# Save classification report as first test report
report_path = os.path.join(report_folder, "test_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("Test Report\n")
    f.write("="*80 + "\n\n")
    f.write(report)
print(f"Test report saved to {report_path}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

print("\nConfusion Matrix (rows=actual, cols=predicted):\n")
print(cm_df)

# Save confusion matrix
cm_path = os.path.join(report_folder, "confusion_matrix.csv")
cm_df.to_csv(cm_path)
print(f"Confusion matrix saved to {cm_path}")
